class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output
