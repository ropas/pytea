'''
fail_conv2d_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Check common mistake of conv2d implementation.
(64, 32, 3) instead of (32, 64, 3).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 32, 28, 28)
conv = nn.Conv2d(64, 32, 3)

# common mistake; this should be an error
b = conv(a)