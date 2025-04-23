## 多查询注意力
import torch
from torch import nn


class MutiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        ## 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)  ###
        self.v_linear = nn.Linear(hidden_size, self.head_dim)  ###

        ## 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]

        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        query = self.split_head(query)
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)

        ## 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask != None:
            attention_scores += (1 - attention_mask) * -1e9

        ## 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        output = self.o_linear(output)
        return output

    def split_head(self, x, head_num=None):

        batch_size = x.size()[0]

        if head_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)
