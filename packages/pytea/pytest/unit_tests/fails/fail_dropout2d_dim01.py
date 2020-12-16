'''
fail_dropout_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Dropout2d supports input with rank >= 2.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(100)
drop = nn.Dropout2d()

# 1-dim tensor cannot be used as an input of Dropout2d
b = drop(a)