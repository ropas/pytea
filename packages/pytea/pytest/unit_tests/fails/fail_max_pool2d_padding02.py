"""
fail_max_pool2d_padding02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

padding should be <= half of kernel_size
"""

import torch
import torch.nn.functional as F

a = torch.rand(20, 28, 28)

# max_pool2d only allows 3d or 4d inputs
F.max_pool2d(a, (3, 4), (5, 5), (2, 1))
