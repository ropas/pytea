'''
fail_nll_loss_reduction01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

F.nll_loss with invalid reduction parameter string.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)
b = torch.rand(2, 4, 5)#.long()

y = F.nll_loss(a, b, reduction='woosung')