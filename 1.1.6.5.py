import torch


def create_padding_mask(seq, pad_token=0):
    """
    seq: (batch_size, seq_len)的输入序列
    返回: (batch_size, 1, 1, seq_len)的mask张量
    """
    # 等于 pad_token 的位置标记为 True
    mask = (seq == pad_token).unsqueeze(1).unsqueeze(2)
    # 这里产生的mask形状是 (batch_size, 1, 1, seq_len)
    # 在广播时会扩展到 (batch_size, 1, seq_len_q, seq_len_k)
    return mask  # True 表示需要屏蔽


def create_look_ahead_mask(size):
    """
    size: seq_len
    返回: (size, size)的上三角矩阵，
         其中上三角(不含对角线)为True，表示需要mask
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask  # True 表示需要屏蔽


def apply_mask(scores, mask):
    """
    scores: (batch_size, heads, seq_len, seq_len)的注意力得分
    mask:   形状可广播到 scores 的 bool 张量
    """
    # 将需要mask的地方置为一个很大的负数
    scores = scores.masked_fill(mask, float("-1e9"))
    return scores


# 假设我们有一个输入序列 (batch_size=2, seq_len=5)
seq = torch.tensor([[5, 6, 7, 8, 0], [1, 2, 0, 0, 0]])  # 0表示padding

pad_mask = create_padding_mask(seq, pad_token=0)  # (2, 1, 1, 5)
look_ahead_mask = create_look_ahead_mask(size=5)  # (5, 5)

# 最终mask可合并（或）在一起
combined_mask = pad_mask | look_ahead_mask  # 注意形状的broadcast
