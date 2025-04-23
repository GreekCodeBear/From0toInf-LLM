import torch
from torch.nn import InstanceNorm2d

batch_size, seq_size, dim = 2, 4, 8
x = torch.randn(batch_size, seq_size, dim)

layer_norm = torch.nn.LayerNorm(dim, elementwise_affine=True)
ln_output = layer_norm(x)

instance_norm = torch.nn.InstanceNorm2d(seq_size, affine=True)
in_output = instance_norm(x.reshape(batch_size, seq_size, dim, 1)).reshape(batch_size, seq_size, dim)

print("LayerNorm Output:\n", ln_output)
print("InstanceNorm2d:\n", in_output)
print("Outputs close:", torch.allclose(ln_output, in_output))
# Outputs close: True
