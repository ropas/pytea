import random
import torch

i = random.randint(0, 2)
a = torch.Size(2, 3, 4)
b = a[2]

# t = torch.rand(3, 5, 6)
# x = t.shape
# # y = x[i]
# z = x[1]
# f = x[1:3]
