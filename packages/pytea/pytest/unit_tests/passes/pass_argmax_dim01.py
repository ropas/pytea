'''
pass_argmax_dim01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

torch.Tensor.argmax with dim parameter.
! This is not available since maximum stack size exceeding error has been occured
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3)
#m = a.argmax(dim=1)

# shape assertion
#m + torch.rand(2, 4, 5)