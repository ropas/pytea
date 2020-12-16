'''
pass_argmax_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Basic functionality check for torch.Tensor.argmax.
()-shaped tensor cannot be asserted yet since there is no shape operator.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)
m = a.argmax()

# shape assertion
#x = 0
#x += m.item()