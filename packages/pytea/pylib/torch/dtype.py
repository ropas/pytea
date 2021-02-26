class dtype:
    def __eq__(self, other):
        return self.__name__ == other.__name__


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


# class bfloat16(dtype):
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

floatDefault = float32
intDefault = int64


def maxDtype(*tensors):
    maxPrecedence = -1
    dtype = None
    for tensor in tensors:
        if isinstance(tensor.dtype, str):
            raise Exception("tensor.dtype is " + tensor.dtype)
        if tensor.dtype.precedence > maxPrecedence:
            maxPrecedence = tensor.dtype.precedence
            dtype = tensor.dtype
    return dtype
