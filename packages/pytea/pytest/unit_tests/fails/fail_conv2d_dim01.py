'''
fail_conv2d_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Dimension error in conv2d; conv2d only receive 4d tensor as input.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(32, 28, 28)
conv = nn.Conv2d(32, 64, 3)

b = conv(a)
