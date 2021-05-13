import builtins as built


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
    def len(value):
        return len(value)

    @staticmethod
    def randInt(lo, hi, prefix):
        import random

        return random.randint(lo, hi)

    @staticmethod
    def randFloat(lo, hi, prefix):
        import random

        return random.random() * (hi - lo) + lo

    @staticmethod
    def exit():
        import sys

        sys.exit(-1)

    @staticmethod
    def warn(msg):
        return NotImplementedError(msg)


class TorchLibCallImpl:
    @staticmethod
    def tensorInit(self, args, SizeClass):
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
    def identityShape(tensor, args, kwargs):
        pass

    @staticmethod
    def flatten(tensor, start_dim=0, end_dim=-1):
        pass

    @staticmethod
    def embedding(tensor, weight):
        pass

    @staticmethod
    def layer_norm(tensor, norm_tensor, weight, bias):
        pass


torch = TorchLibCallImpl()


class ArgparseImpl:
    @staticmethod
    def inject_argument(parsed, args, kwargs):
        pass


argparse = ArgparseImpl()
