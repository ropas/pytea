'''
pass_max_pool2d_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for F.max_pool2d.
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 28, 28)

b = F.max_pool2d(a, 2)

# shape assertion
b + torch.rand(10, 14, 14)