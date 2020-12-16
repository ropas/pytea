'''
fail_nll_loss_weight01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Dimension mismatch from two input tensors.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)
b = torch.rand(2, 4, 5)#.long()
w = torch.rand(2)

y = F.nll_loss(a, b, w)