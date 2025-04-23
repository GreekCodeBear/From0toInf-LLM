class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # 总的专家数
        self.top_k = config.num_experts_per_tok  # 每个 token 仅使用 top_k 个专家
        self.norm_topk_prob = config.norm_topk_prob  # 是否对选出的 top_k 概率进行归一化

        # gating: 用于给每个 token 计算对应到 num_experts 个专家上的 logits（路由分数）
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # experts: 每个专家都是一个独立的 MLP，这里用一个 ModuleList 存放
        self.experts = nn.ModuleList([Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)])


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # (batch_size, seq_len, hidden_dim) -> (batch_size*seq_len, hidden_dim)
    hidden_states = hidden_states.view(-1, hidden_dim)

    # ===================== 1) 计算 gating 得分 =====================
    # router_logits: (batch_size * sequence_length, num_experts)
    router_logits = self.gate(hidden_states)

    # 对路由分数做 softmax，得到对各专家的概率分布
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    # 只取最大的 top_k 个专家和相应概率
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

    # norm_topk_prob: 是否把选出来的 top_k 概率再做归一化，使它们之和为 1
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    # 将概率 cast 回输入相同的 dtype (如 FP16/BF16 等)
    routing_weights = routing_weights.to(hidden_states.dtype)

    # ===================== 2) 初始化输出张量
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    # ===================== 3) 构造 one-hot mask，标识每个 token 被分配给哪些专家
    # expert_mask: (num_experts, batch_size*sequence_length, top_k)
    #   - one_hot会先得到 (batch_size*sequence_length, num_experts) 的one_hot编码，然后我们选了 top_k 列
    #   - permute(2,1,0) 目的是把 expert 这个维度放在最前面，便于后面 for expert_idx in ... 的循环使用
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # ===================== 4) 分别计算每个专家的输出并累加
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]

        # idx, top_x 记录哪些 token 被分配给了 expert_idx
        # expert_mask[expert_idx] 是 (batch_size*sequence_length, top_k) 的 one-hot
        # torch.where 返回 (行index, 列index)
        # 其中 "top_x" 表示第几个 token 的行索引，"idx" 表示 top_k 中第几个列位置
        idx, top_x = torch.where(expert_mask[expert_idx])

        # 取出对应 token 的 hidden_states
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        # 进入该 expert 的 MLP，并乘以对该 expert 的 gating 权重
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # 把该 expert 的输出累加回 final_hidden_states
        # index_add_ 是原位操作，会把 current_hidden_states 累加到 final_hidden_states[top_x] 上
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    # 最后 reshape 回 (batch_size, sequence_length, hidden_dim) 形式
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    ...
    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = attention_mask[None, :, :, None, None].expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts)).reshape(-1, top_k, num_experts).to(compute_device)
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = attention_mask[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(compute_device)
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
