import torch

i = 5
j = 0
for i, p in zip(range(3), range(5)):
    j += i
    r = 10
k = i + r
print(i, j, k)
