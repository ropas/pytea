'''
pass_max_pool2d_ceil_mode01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check 'ceil_mode' option on F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(12, 32, 40)

b = F.max_pool2d(a, 4, 3, 1, 3, ceil_mode=True)

# shape assertion
b + torch.rand(12, 9, 12)
