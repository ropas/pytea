'''
pass_max_pool2d_tuple02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Tuple parameters in F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 20, 28, 28)

b = F.max_pool2d(a, (14, 14), (2, 2))

# shape assertion
a + torch.rand(10, 20, 1, 1)