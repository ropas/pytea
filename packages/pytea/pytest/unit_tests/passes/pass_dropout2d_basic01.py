'''
pass_dropout_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for nn.Dropout.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 32, 28, 28)
drop = nn.Dropout2d(0.2)

b = drop(a)

# shape assertion
b + torch.rand(10, 32, 28, 28)