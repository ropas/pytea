'''
pass_nll_loss_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for F.nll_loss.
.long() is not supported yet, therefore this code is error in real python.
and .item() is not supported, therefore there is no way to assert it is ()-shaped
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)
b = torch.rand(2, 4, 5)#.long()

y = F.nll_loss(a, b)

# shape assertion
#x = 0
#x += y.item()