import LibCall


def calculate_gain(nonlinearity, param=None):
    if nonlinearity == "linear":
        return 1
    elif nonlinearity == "identity":
        return 1
    elif nonlinearity == "conv1d":
        return 1
    elif nonlinearity == "conv2d":
        return 1
    elif nonlinearity == "conv3d":
        return 1
    elif nonlinearity == "tanh":
        return 5 / 3
    elif nonlinearity == "relu":
        return LibCall.math.float_fun(2, "sqrt")
    elif nonlinearity == "leaky_relu":
        return LibCall.math.float_fun(2 / (1 + param * param), "sqrt")
    elif nonlinearity == "selu":
        return 3 / 4


def xavier_uniform_(tensor, gain=1.0):
    pass


def xavier_normal_(tensor, gain=1.0):
    pass


def kaiming_uniform(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    pass


def kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    pass


def orthogonal_(tensor, gain=1):
    rank = tensor.ndim
    LibCall.guard.require_lte(
        2, rank, "'torch.nn.functional.orthogonal_' requires rank 2 Tensor."
    )
    pass


def sparse_(tensor, sparsity, std=0.01):
    pass


def uniform_(tensor, a=0.0, b=1.0):
    pass


def normal_(tensor, mean=0.0, std=1.0):
    pass


def constant_(tensor, val):
    pass


def ones_(tensor):
    pass


def zeros_(tensor):
    pass


def eye_(tensor):
    rank = tensor.ndim
    LibCall.guard.require_eq(
        rank, 2, "'torch.nn.functional.eye_' requires rank 2 Tensor."
    )
    pass


def dirac_(tensor, groups=1):
    rank = tensor.ndim
    LibCall.guard.require_lte(
        3, rank, "'torch.nn.functional.dirac_' requires rank {1,2,3} Tensor."
    )
    LibCall.guard.require_lte(
        rank, 5, "'torch.nn.functional.dirac_' requires rank {1,2,3} Tensor."
    )
    pass
