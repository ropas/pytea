"""
    LibCall.py
    ~~~

    PTTyR LibCall wrapper for Python

    :copyright: Copyright 2020 Seoul National University.
    :license: MIT License.
    :author: Ho Young Jhoo
"""

import builtins as built


def getAttr(name, self, baseClass, bind):
    if hasattr(self, name):
        return getattr(self, name)
    elif hasattr(baseClass, name):
        func = getattr(baseClass, name)
        if bind:

            def boundFunc(*args, **kwargs):
                return func(self, *args, **kwargs)

            return boundFunc
        else:
            return func
    else:
        raise AttributeError(f"object has no attribute '${name}'")


def DEBUG(value):
    print(value)


def objectClass():
    return object


class builtins:
    @staticmethod
    def isinstance(value, type):
        return built.isinstance(value, type)

    @staticmethod
    def toInt(value):
        return int(value)

    @staticmethod
    def toFloat(value):
        return float(value)

    @staticmethod
    def setSize(value, shape):
        value.shape = shape


class TorchLibCallImpl:
    @staticmethod
    def tensorInit(self, args, kwargs):
        pass

    @staticmethod
    def broadcast(self, other):
        pass

    @staticmethod
    def repeat(self, sizes):
        pass

    @staticmethod
    def matmul(self, other):
        pass

    @staticmethod
    def mul(self, other):
        pass

    @staticmethod
    def normal_(self, args, kwargs):
        pass

    @staticmethod
    def item(self):
        pass

    @staticmethod
    def callTensor(dims):
        pass

    @staticmethod
    def copyOut(tensor, out):
        pass

    @staticmethod
    def transpose(tensor, dim0, dim1):
        pass

    @staticmethod
    def transpose(tensor, dim0, dim1):
        pass

    @staticmethod
    def identityShape(tensor, args, kwargs):
        pass

    @staticmethod
    def flatten(tensor, start_dim=0, end_dim=-1):
        pass


torch = TorchLibCallImpl()


class ArgparseImpl:
    @staticmethod
    def inject_argument(parsed, args, kwargs):
        pass


argparse = ArgparseImpl()
