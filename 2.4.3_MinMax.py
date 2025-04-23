class MiniMaxText01LightningAttention(nn.Module):
    def __init__(self, config: MiniMaxText01Config, layer_idx: Optional[int] = None):
        ...
        self.out_proj = nn.Linear(self.head_dim * self.num_heads, self.hidden_size, bias=bias)
        self.act = get_activation_fn(config.hidden_act)
        self.norm = MiniMaxText01RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.head_dim * self.num_heads, bias=bias)
        self.output_gate = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias=bias)
        ...

    # 这个就是forward方法的实现
    def inference(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,  # (b, n)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        qkv = self.act(self.qkv_proj(x))
        ...
        # lightning attention 实现掠过
        ...
        output = F.sigmoid(self.output_gate(x)) * output
        # outproj
        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv
