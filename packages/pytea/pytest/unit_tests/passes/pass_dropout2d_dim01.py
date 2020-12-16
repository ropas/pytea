'''
pass_dropout_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Dropout2d supports input with rank >= 2.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(1, 2)
b = torch.rand(1, 2, 3)
c = torch.rand(1, 2, 3, 4)
d = torch.rand(1, 2, 3, 4, 5)
drop = nn.Dropout2d()

# shape assertion
a = a + drop(a)
b = b + drop(b)
c = c + drop(c)
d = d + drop(d)