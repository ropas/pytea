import torch

n = 3
x = torch.Tensor([2.0 / n])
y = torch.sqrt(x)
z = y.item()
