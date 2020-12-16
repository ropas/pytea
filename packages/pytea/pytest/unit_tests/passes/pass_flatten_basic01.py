'''
pass_flatten_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for torch.flatten.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 1024)
b = torch.flatten(a)

# shape assertion
b + torch.rand(10240)