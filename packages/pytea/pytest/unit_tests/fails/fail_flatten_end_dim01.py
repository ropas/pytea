'''
fail_flatten_end_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

When start_dim > end_dim for torch.flatten.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5, 6, 7)

b = torch.flatten(a, 3, 2)