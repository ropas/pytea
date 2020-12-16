'''
pass_max_pool2d_padding01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check 'padding' option on F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 30, 30)

b = F.max_pool2d(a, 2, 2, 1)

# shape assertion
b + torch.rand(10, 16, 16)