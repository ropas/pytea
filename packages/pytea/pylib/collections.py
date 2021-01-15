import LibCall

def namedtuple(typename, field_names, **kwargs):
    class __TempTuple(tuple):
        # TODO: kwargs
        def __init__(self, *args):
            for i, name in enumerate(field_names):
                LibCall.builtins.setNamedTupleAttr(self, i, name, args[i])

    __TempTuple.__name__ = typename

    return __TempTuple
