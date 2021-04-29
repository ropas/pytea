import torch

t1 = torch.rand(5, 6, 7)
t2 = torch.rand(5, 6, 7)
t3 = torch.rand(5, 6, 7)

t_ = torch.stack([t1, t2, t3], dim=-2)
print(t_.shape)
