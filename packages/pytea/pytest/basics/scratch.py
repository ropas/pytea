import torch
import random

t = random.randint(3, 12)
a = torch.rand(3, 5, 7, 9)
d = a[0]

b = a[:, 1:3, ::2, 4:-1]

