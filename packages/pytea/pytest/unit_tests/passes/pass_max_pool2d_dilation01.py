'''
pass_max_pool2d_dilation01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check 'dilation' option on F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 32, 40)

b = F.max_pool2d(a, 4, 2, 1, 3)

# shape assertion
b + torch.rand(10, 13, 17)