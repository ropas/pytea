'''
pass_dropout_basic02.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Default parameter for dropout
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 64, 28, 28)
drop = nn.Dropout2d()

b = drop(a)

# shape assertion
b + torch.rand(10, 64, 28, 28)