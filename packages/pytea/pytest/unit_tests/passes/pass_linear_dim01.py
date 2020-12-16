'''
pass_linear_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check for nn.Linear with multiple input ranks.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

fc = nn.Linear(1024, 512)

a = torch.rand(10, 1024)
b = torch.rand(1024)
c = torch.rand(10, 20, 30, 40, 1024)

x = fc(a)
y = fc(b)
z = fc(c)

# shape assertions
x + torch.rand(10, 512)
y + torch.rand(512)
z + torch.rand(10, 20, 30, 40, 512)