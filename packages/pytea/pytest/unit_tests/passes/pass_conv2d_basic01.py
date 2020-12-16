'''
pass_conv2d_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for F.max_pool2d.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 32, 28, 28)
conv = nn.Conv2d(32, 64, 3)

b = conv(a)

# shape assertion
b + torch.rand(10, 64, 26, 26)