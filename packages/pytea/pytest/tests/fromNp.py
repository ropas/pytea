import torch
import numpy as np

a = np.zeros((2, 3, 4))
t = torch.from_numpy(a)
adtype = a.dtype
tdtpye = t.dtype
print(a)
print(t)
print(adtype)
print(tdtpye)
