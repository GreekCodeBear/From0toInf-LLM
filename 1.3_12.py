# core code
is_all_accept = True
n = prefix_len - 1
for i in range(gamma):
    if random_seed:
        torch.manual_seed(random_seed)
    r = torch.rand(1, device=p.device)
    j = x[:, prefix_len + i]

    if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
        # accept, and update n
        n += 1
    else:
        # reject
        print(f"reject {n+1}")
        t = sample(max_fn(p[:, n, :] - q[:, n, :]))
        is_all_accept = False
        break
