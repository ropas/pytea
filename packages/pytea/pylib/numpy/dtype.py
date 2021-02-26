import torch
import numpy as np

class dtype:
    pass

class complex128(dtype):
    precedence = 10

class complex64(dtype):
    precedence = 9

class float64(dtype):
    precedence = 8

class float32(dtype):
    precedence = 7

class float16(dtype):
    precedence = 6

#class bfloat16(dtype):
#    pass

class int64(dtype):
    precedence = 5

class int32(dtype):
    precedence = 4

class int16(dtype):
    precedence = 3

class int8(dtype):
    precedence = 2

class uint8(dtype):
    precedence = 1

class bool(dtype):
    precedence = 0

float = float32
double = float64
cfloat = complex64
cdouble = complex128
half = float16
short = int16
int = int32
long = int64

floatTypes = [float64, float32, float16]
intTypes = [int64, int32, int16, int8, uint8]

floatDefault = float32
intDefault = int64

def maxDtype(*objs):
    maxPrecedence = -1
    dtype = None
    for obj in objs:
        if isinstance(obj, np.ndarray):    
            if tensor.dtype.precedence > maxPrecedence:
                maxPrecedence = tensor.dtype.precedence
                dtype = tensor.dtype
        elif isinstance(obj, np.dtype):
            if obj.precedence > maxPrecedence:
                maxPrecedence = obj.precedence
                dtype = obj
    return dtype

def toNpdtype(torchDtype):
    if torchDtype is torch.complex128:
        return complex128
    elif torchDtype is torch.complex64:
        return complex64
    elif torchDtype is torch.float64:
        return float64
    elif torchDtype is torch.float32:
        return float32
    elif torchDtype is torch.float16:
        return float16
    elif torchDtype is torch.int64:
        return int64
    elif torchDtype is torch.int32:
        return int32
    elif torchDtype is torch.int16:
        return int16
    elif torchDtype is torch.int8:
        return int8
    elif torchDtype is torch.uint8:
        return uint8
    elif torchDtype is torch.bool:
        return bool