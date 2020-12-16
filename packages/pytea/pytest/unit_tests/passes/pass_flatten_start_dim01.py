'''
pass_flatten_start_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Apply torch.flatten with start_dim parameter
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5, 6, 7)
b = torch.flatten(a, 3)

# shape assertion
b + torch.rand(2, 3, 4, 210)