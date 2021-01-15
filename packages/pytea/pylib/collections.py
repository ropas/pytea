import LibCall

def namedtuple(typename, field_names, **kwargs):
    class __TempTuple(tuple):
        # TODO: kwargs
        def __init__(self, *args):
            for i, name in enumerate(field_names):
                LibCall.builtins.setIndice(self, i, args[i])
                LibCall.builtins.setAttr(self, name, args[i])

        def __len__(self):
            return len(field_names)

    __TempTuple.__name__ = typename

    return __TempTuple
