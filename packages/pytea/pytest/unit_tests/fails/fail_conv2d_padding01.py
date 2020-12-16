"""
fail_conv2d_padding01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

padding should be <= half of kernel_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(32, 28, 28)
w = torch.rand(36, 32, 3, 5)

b = F.conv2d(a, w, padding=(1, 3))
