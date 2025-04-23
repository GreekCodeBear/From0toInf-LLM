hidden_dim = 768
vocal_size = 50257
n_layers = 12

hidden_dim_2 = 4 * hidden_dim
res = 0
res += hidden_dim * vocal_size
for i in range(n_layers):
    res += 4 * (hidden_dim * hidden_dim + hidden_dim)
    res += 2 * (2 * hidden_dim)
    res += hidden_dim * hidden_dim_2 + hidden_dim_2
    res += hidden_dim_2 * hidden_dim + hidden_dim
print(f"{res/10**9}B")
