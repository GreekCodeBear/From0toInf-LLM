def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def user_multi_head_attention_forward(query, key, value, embd_dim, num_heads, in_proj_weight, out_proj_weight):
    w_q, w_k, w_v = in_proj_weight[:embd_dim, :], in_proj_weight[embd_dim : 2 * embd_dim, :], in_proj_weight[2 * embd_dim :, :]
    q = np.matmul(query, w_q)
    k = np.matmul(key, w_k)
    v = np.matmul(value, w_v)

    batch_size, seq_length, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    q = q.reshape(batch_size, seq_length, num_heads, head_dim)
    k = k.reshape(batch_size, seq_length, num_heads, head_dim)
    v = v.reshape(batch_size, seq_length, num_heads, head_dim)

    # Using np.einsum to compute the scores
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(head_dim)
    weights = softmax(scores)

    # Using np.einsum to compute the weighted sum of values
    att_output = np.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch_size, seq_length, embed_dim)

    output = np.matmul(att_output, out_proj_weight)
    return output


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def user_multi_head_attention_forward(query, key, value, embd_dim, num_heads, in_proj_weight, out_proj_weight):
    w_q, w_k, w_v = in_proj_weight[:embd_dim, :], in_proj_weight[embd_dim : 2 * embd_dim, :], in_proj_weight[2 * embd_dim :, :]
    q = np.matmul(query, w_q)
    k = np.matmul(key, w_k)
    v = np.matmul(value, w_v)

    batch_size, seq_length, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    q = q.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 3, 1)
    v = v.reshape(batch_size, seq_length, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = np.matmul(q, k) / np.sqrt(head_dim)
    weights = softmax(scores)
    att_output = np.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
    output = np.matmul(att_output, out_proj_weight)
    return output


import numpy as np
from typing import Callable
import torch

batch_size, seq_len, embd_dim, num_heads = [int(_) for _ in input().split()]

np.random.seed(batch_size + seq_len + embd_dim + num_heads)
query = np.random.randn(batch_size, seq_len, embd_dim).astype(np.float32)
key = np.random.randn(batch_size, seq_len, embd_dim).astype(np.float32)
value = np.random.randn(batch_size, seq_len, embd_dim).astype(np.float32)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def user_multi_head_attention_forward(query, key, value, embd_dim, num_heads, in_proj_weight, out_proj_weight):
    w_q, w_k, w_v = in_proj_weight[:embd_dim, :], in_proj_weight[embd_dim : 2 * embd_dim, :], in_proj_weight[2 * embd_dim :, :]
    q = np.matmul(query, w_q)
    k = np.matmul(key, w_k)
    v = np.matmul(value, w_v)

    batch_size, seq_length, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    q = q.reshape(batch_size, seq_length, num_heads, head_dim)
    k = k.reshape(batch_size, seq_length, num_heads, head_dim)
    v = v.reshape(batch_size, seq_length, num_heads, head_dim)

    # Using np.einsum to compute the scores
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(head_dim)
    weights = softmax(scores)

    # Using np.einsum to compute the weighted sum of values
    att_output = np.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch_size, seq_length, embed_dim)

    output = np.matmul(att_output, out_proj_weight)
    return output


# 以下为测试代码
def xavier_uniform_(tensor, gain=1.0):  # 方阵的 xavier 初始化
    a = gain * np.sqrt(3 / tensor.shape[0])
    res = np.random.uniform(low=-a, high=a, size=tensor.shape)
    return (res + res.T) / 2  # 把矩阵变成对称的，不用考虑实现时要不要转置


def test_multi_head_attention_forward(mha_impl_func: Callable, in_proj_weight: np.ndarray, out_proj_weight: np.ndarray):
    # Run mha
    user_impl_out = mha_impl_func(query, key, value, embd_dim, num_heads, in_proj_weight, out_proj_weight)
    import torch
    import torch.nn as nn

    # 可以和 torch 的实现做对比
    mha = nn.MultiheadAttention(embd_dim, num_heads, bias=False, dropout=0)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.from_numpy(in_proj_weight))
        mha.out_proj.weight.copy_(torch.from_numpy(out_proj_weight))
        mha_out, _ = mha(torch.from_numpy(query).transpose(0, 1), torch.from_numpy(key).transpose(0, 1), torch.from_numpy(value).transpose(0, 1))
        mha_out = mha_out.transpose(0, 1).numpy()

    assert np.allclose(user_impl_out, mha_out, atol=1e-6), "Outputs do not match!"


weights = [xavier_uniform_(np.random.randn(embd_dim, embd_dim)).astype(np.float32) for _ in range(4)]
in_proj_weight = np.concatenate(weights[:3], axis=0)
out_proj_weight = weights[-1]


# 为方便调试，你可以用下面这个测试函数在本地验证自己实现的 user_multi_head_attention_forward() 的正确性。
# 但在提交代码到系统时务必注释掉这行代码，否则会因为找不到 torch 模块而报错。
# test_multi_head_attention_forward(user_multi_head_attention_forward, in_proj_weight, out_proj_weight)

result = user_multi_head_attention_forward(query, key, value, embd_dim, num_heads, in_proj_weight, out_proj_weight)

print(f"{int(np.linalg.norm(result.ravel()) * 100_000)}")  # 把结果摊平成一维数组，求范数然后保留 5 位小数
