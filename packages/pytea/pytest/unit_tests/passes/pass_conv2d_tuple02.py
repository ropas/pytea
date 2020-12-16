'''
pass_conv2d_tuple02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Tuple parameters in nn.Conv2d.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 32, 28, 28)
conv = nn.Conv2d(32, 64, (3, 5), (2, 2))

b = conv(a)

# shape assertion
b + torch.rand(10, 64, 13, 12)
