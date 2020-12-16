'''
fail_max_pool2d_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

2d tensor input in F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(28, 28)

# max_pool2d only allows 3d or 4d inputs
F.max_pool2d(a, 2)