'''
pass_flatten_basic02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for torch.flatten.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5, 6, 7)
b = torch.flatten(a)

# shape assertion
b + torch.rand(7*6*5*4*3*2*1)