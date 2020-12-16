'''
pass_max_pool2d_stride01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check 'stride' option on F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 40, 100)

b = F.max_pool2d(a, 2, 4)

# shape assertion
b + torch.rand(10, 10, 25)