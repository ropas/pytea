def floor(x):
    return LibCall.math.floor(x)


def ceil(x):
    return LibCall.math.ceil(x)


# LibCall.math.float_fun returns unconstrainted float symbol if it receive symbolic value.
def sqrt(x):
    return LibCall.math.float_fun(x, "sqrt")


def log(x):
    return LibCall.math.float_fun(x, "log")


def log2(x):
    return LibCall.math.float_fun(x, "log2")


def log10(x):
    return LibCall.math.float_fun(x, "log10")


def exp(x):
    return LibCall.math.float_fun(x, "exp")


def expm1(x):
    return LibCall.math.float_fun(x, "expm1")
