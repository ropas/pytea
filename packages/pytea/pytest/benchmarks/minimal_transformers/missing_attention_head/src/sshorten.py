import torch
import numpy as np


a = torch.rand(5, 4, 3).numpy()
b = torch.rand(5, 4, 3).numpy()

c = np.mean(a == b)