class AttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionWithKVCache, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("key_cache", None, persistent=False)
        self.register_buffer("value_cache", None, persistent=False)

    def forward(self, x, past_kv_cache=None, save_cache=False):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if past_kv_cache is not None:
            past_key, past_value = past_kv_cache
            k = torch.cat([past_key, k], dim=2)  # Concatenate along sequence dimension
            v = torch.cat([past_value, v], dim=2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        if save_cache:
            self.key_cache = k
            self.value_cache = v
        return output, (k, v)
