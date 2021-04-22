import LibCall


def random():
    return LibCall.builtins.randFloat(0, 1, "randfloat")


def uniform(a, b):
    return LibCall.builtins.randFloat(a, b, "uniform")


def randint(a, b):
    return LibCall.builtins.randInt(a, b, "randint")


def seed(a=None, version=2):
    pass