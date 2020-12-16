'''
pass_conv2d_full01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Full parameters in conv2d
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 32, 28, 28)
w = torch.rand(30, 32, 3, 4)
b = torch.rand(30)

x = F.conv2d(a, w, b, 3, 1, (1, 2))

# shape assertion
x + torch.rand(10, 30, 10, 8)
