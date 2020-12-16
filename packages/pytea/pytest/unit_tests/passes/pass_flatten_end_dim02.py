'''
pass_flatten_end_dim02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Apply torch.flatten with start_dim and end_dim parameter
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5, 6, 7)
b = torch.flatten(a, 2, 2)

# shape assertion
b + torch.rand(2, 3, 4, 5, 6, 7)