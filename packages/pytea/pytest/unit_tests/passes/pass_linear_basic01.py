'''
pass_linear_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for nn.Linear.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 1024)
fc = nn.Linear(1024, 512)

b = fc(a)

# shape assertion
b + torch.rand(10, 512)