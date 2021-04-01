import LibCall

# from .functional import *
from .tensor import Tensor, tensor
from .functional import *
from . import nn as nn
from . import optim as optim
from . import utils as utils
from . import distributions as distributions
from . import cuda as cuda
from . import onnx as onnx
from . import backends as backends
from . import distributed as distributed
from .autograd import no_grad, enable_grad
from .dtype import *


class bool:
    pass


class device:
    def __init__(self, type):
        self.type = type
