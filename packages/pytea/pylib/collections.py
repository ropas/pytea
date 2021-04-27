import LibCall


def namedtuple(typename, field_names, **kwargs):
    names = field_names
    if isinstance(field_names, str):
        names = []
        LibCall.builtins.namedtuple_pushField(names, field_names)

    rev_names = dict()
    for i, name in enumerate(names):
        rev_names[name] = i

    class __TempTuple(tuple):
        def __init__(self, *args, **kwargs):
            for i, value in enumerate(args):
                LibCall.builtins.setIndice(self, i, value)
                LibCall.builtins.setAttr(self, names[i], value)
            for key, value in kwargs.items():
                LibCall.builtins.setIndice(self, rev_names[key], value)
                LibCall.builtins.setAttr(self, key, value)

        def __len__(self):
            return len(names)

    __TempTuple.__name__ = typename

    return __TempTuple
