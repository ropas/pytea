'''
fail_log_softmax_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Out-of-range dimension for F.log_softmax.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(2, 3, 4, 5)

y = F.log_softmax(a, 4)