"""
fail_max_pool2d_padding01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

padding should be <= half of kernel_size
"""

import torch
import torch.nn.functional as F

a = torch.rand(20, 28, 28)

# max_pool2d only allows 3d or 4d inputs
F.max_pool2d(a, 2, 4, 3)
