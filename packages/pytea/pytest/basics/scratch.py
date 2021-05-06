import random
import torch

i = random.randint(0, 2)

t = torch.rand(3, 5, 6)
x = t.shape
y = x[i]
z = x[1]
