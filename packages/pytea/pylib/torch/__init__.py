import LibCall

from .tensor import Tensor, Size, FloatTensor, DoubleTensor, LongTensor, IntTensor

from . import nn as nn
from . import optim as optim
from . import utils as utils
from . import distributions as distributions
from . import cuda as cuda
from . import onnx as onnx
from . import backends as backends
from . import distributed as distributed
from . import autograd as autograd

from .autograd import no_grad, enable_grad

from .functional import *
from .dtype import *


class device:
    def __init__(self, type):
        self.type = type
