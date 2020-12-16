'''
pass_nll_loss_reduction01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

F.nll_loss with reduction parameter.
.long() is not supported yet, therefore this code is error in real python.
and .item() is not supported, therefore there is no way to assert it is ()-shaped
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)
b = torch.rand(2, 4, 5)#.long()

x = F.nll_loss(a, b, reduction='none')
y = F.nll_loss(a, b, reduction='mean')
z = F.nll_loss(a, b, reduction='sum')

# shape assertion
#x = 0
#x += y.item()