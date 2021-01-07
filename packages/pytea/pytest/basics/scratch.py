import torch


x = [1, 2, 3]
y = [4, 5]
z = tuple(y)
b = isinstance(z, tuple)

c = 0
for (i, j) in zip(x, y):
    c += i + j
