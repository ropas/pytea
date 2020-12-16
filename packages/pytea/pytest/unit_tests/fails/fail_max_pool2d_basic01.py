'''
fail_max_pool2d_basic01.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Woo Sung Song

Shape diminishes to 0 by F.max_pool2d
'''

import torch
import torch.nn.functional as F

a = torch.rand(10, 28, 28)

# torch shape diminishes    
F.max_pool2d(a, 29)