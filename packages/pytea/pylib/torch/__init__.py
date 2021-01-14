import LibCall

# from .functional import *
from .tensor import Tensor
from .functional import *
from . import nn as nn
from . import optim as optim
from . import utils as utils
from . import distributions as distributions
from . import cuda as cuda
from . import onnx as onnx
from .autograd import no_grad, enable_grad


class bool:
    pass


class device:
    def __init__(self, type):
        self.type = type
