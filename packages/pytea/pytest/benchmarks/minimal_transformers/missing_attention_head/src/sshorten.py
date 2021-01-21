import torch

def hello(*woosung, **kwargs):
    for jisoo in woosung:
        if woosung[0].shape[0] != jisoo.shape[0]:
            raise ValueError("Nope!")
    return torch.rand(5, 4, 3)

a = torch.rand(7, 2)
b = torch.rand(7, 1, 2)

c = hello(a, b)