import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization class"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# Example usage
features = 768  # Typical BERT hidden size
layer_norm = LayerNorm(features)

# Dummy input tensor (batch_size, seq_length, features)
x = torch.randn(10, 20, features)
normalized_x = layer_norm(x)
print(normalized_x)
