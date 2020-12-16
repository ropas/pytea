'''
fail_linear_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Dimension mismatch for nn.Linear.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(10, 1023)
fc = nn.Linear(1024, 512)

x = fc(a)