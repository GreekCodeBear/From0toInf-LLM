@torch.no_grad()
def blockwise_parallel_decoding(prefix: torch.Tensor, approx_model: torch.nn.Module, target_model: torch.nn.Module, max_len: int, gamma: int = 4) -> torch.Tensor:
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            x = prefix
            prefix_len = prefix.shape[1]
            x = approx_model.generate(x, max_length=prefix_len + gamma)
            y = target_model(x).logits.argmax(dim=2)
            n = prefix_len - 1
            for _ in range(gamma):
                if y[0][n] == x[0][n + 1]:
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    print(f"reject {n+1}")
                    x[0][n + 1] = y[0][n]
                    break
            prefix = x[:, : n + 2]
            pbar.update(n - pbar.n)
    return prefix
