"""
PyTea implementation of python builtin.ts.
"""
from . import LibCall

# object class is created in interpreter or backend explicitly.
object = LibCall.objectClass()
NotImplemented = NotImplemented


def super(__self_class__, __self_object__=None):
    def super_getattr(attr):
        return LibCall.builtins.superGetAttr(__self_class__, __self_object__, attr)

    super_proxy = LibCall.rawObject()
    super_proxy.__getattr__ = super_getattr
    return super_proxy


def len(value):
    return LibCall.builtins.len(value)


def print(value):
    LibCall.DEBUG(value)


def isinstance(value, type):
    return LibCall.builtins.isinstance(value, type)


def exit(code=0):
    return LibCall.builtins.exit(code)


# temp
def round(number, digits):
    return number


class slice:
    def __init__(self, start=None, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step


class range:
    def __init__(self, start=None, stop=None, step=None):
        if stop is None:
            self.start = 0
            self.stop = start
            self.step = 1
        else:
            self.start = start
            self.stop = stop
            if step is None:
                self.step = 1
            else:
                self.step = step

        self._len = (self.stop - self.start) // self.step

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.start + index * self.step


class __Primitives:
    def __init__(self, type):
        self.type = type

    def __call__(self, value=None, **kwargs):
        return LibCall.builtins.cast(value, self.type, kwargs)


int = __Primitives(0)
int.__mro__ = (int, object)
int.__name__ = "int"
float = __Primitives(1)
float.__mro__ = (float, object)
float.__name__ = "float"
str = __Primitives(2)
str.__mro__ = (str, object)
str.__name__ = "str"
bool = __Primitives(3)
bool.__mro__ = (bool, object)
bool.__name__ = "bool"
tuple = __Primitives(4)
tuple.__mro__ = (tuple, object)
tuple.__name__ = "tuple"
list = __Primitives(5)
list.__mro__ = (list, object)
list.__name__ = "list"
dict = __Primitives(6)
dict.__mro__ = (dict, object)
dict.__name__ = "dict"
set = __Primitives(7)
set.__mro__ = (set, object)
set.__name__ = "set"
Ellipsis = __Primitives(8)
Ellipsis.__mro__ = (Ellipsis, object)
Ellipsis.__name__ = "Ellipsis"


def _tuple__getitem__(self, index):
    if isinstance(index, int):
        return LibCall.builtins.getItemByIndex(self, index)
    elif isinstance(index, slice):
        start, stop = None, None
        if index.start is not None:
            start = index.start if index.start >= 0 else len(self) + index.start
        if index.stop is not None:
            len_self = len(self)
            if index.stop >= len_self:
                stop = len_self
            elif index.stop >= 0:
                stop = index.stop
            else:
                stop = len(self) + index.stop

        if index.start is None:
            if index.stop is None:
                # index by [:]
                return self
            else:
                # index by [:stop]
                return (self[i] for i in range(0, stop))
        else:
            if index.stop is None:
                # index by [start:]
                return (self[i] for i in range(start, len(self)))
            else:
                # index by [start:stop]
                return (self[i] for i in range(start, stop))


tuple.__getitem__ = _tuple__getitem__


def _list_append(self, item):
    LibCall.builtins.list_append(self, item)


list.append = _list_append


def _list__getitem__(self, index):
    if isinstance(index, int):
        return LibCall.builtins.getItemByIndex(self, index)
    elif isinstance(index, slice):
        start, stop = None, None
        if index.start is not None:
            start = index.start if index.start >= 0 else len(self) + index.start
        else:
            start = None

        if index.stop is not None:
            stop = index.stop if index.stop >= 0 else len(self) + index.stop

        if index.start is None:
            if index.stop is None:
                # index by [:]
                return self
            else:
                # index by [:stop]
                return [self[i] for i in range(0, stop)]
        else:
            if index.stop is None:
                # index by [start:]
                return [self[i] for i in range(start, len(self))]
            else:
                # index by [start:stop]
                return [self[i] for i in range(start, stop)]
    else:
        # TODO: filter SVWarning
        raise IndexError("index is not an integer or slice")


list.__getitem__ = _list__getitem__


def _list__add__(self, items):
    ret = []
    for item in self:
        LibCall.builtins.list_append(ret, item)
    for item in items:
        LibCall.builtins.list_append(ret, item)
    return ret


list.__add__ = _list__add__


def _tuple__add__(self, items):
    ret = ()
    for item in self:
        LibCall.builtins.list_append(ret, item)
    for item in items:
        LibCall.builtins.list_append(ret, item)
    return ret


tuple.__add__ = _tuple__add__


def _dict_keys(self):
    return LibCall.builtins.dict_keys(self)


dict.keys = _dict_keys


def _dict_values(self):
    return LibCall.builtins.dict_values(self)


dict.values = _dict_values


def _dict_items(self):
    return LibCall.builtins.dict_items(self)


dict.items = _dict_items


def _dict_contains(self, key):
    return LibCall.builtins.has_key(self, key)


dict.__contains__ = _dict_contains


def _dict__setitem__(self, key, value):
    return LibCall.builtins.dict_setitem(self, key, value)


dict.__setitem__ = _dict__setitem__


def _dict__getitem__(self, key):
    return LibCall.builtins.dict_getitem(self, key)


dict.__getitem__ = _dict__getitem__


def _dict_pop(self, key, defaultVal):
    return LibCall.builtins.dict_pop(self, key, defaultVal)


dict.pop = _dict_pop


class _dict_keyiterator:
    def __init__(self, d):
        self.d = d
        self.keys = d.keys()
        self.values = d.values()
        self.idx = 0
        self.len = len(self.keys)

    def __next__(self):
        if self.idx < self.len:
            item = (self.keys[self.idx], self.values[self.idx])
            self.idx += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, i):
        return (self.keys[i], self.values[i])

    def __len__(self):
        return self.len


def _dict__iter__(self):
    return _dict_keyiterator(self)


dict.__iter__ = _dict__iter__


def _str_format(self, *args, **kwargs):
    # TODO: format it.
    return self


str.format = _str_format


def _str_replace(self, old, new, count=None):
    # TODO: replace it.
    return self


str.replace = _str_replace

# temp
def _str_ljust(self, length, character=" "):
    # TODO:
    return self


str.ljust = _str_ljust


def sum(values):
    a = 0
    for i in values:
        a += i
    return a


class map:
    def __init__(self, f, iterable):
        self.f = f
        self.iterable = iterable

    def __getitem__(self, index):
        return self.f(self.iterable[index])

    def __len__(self):
        return len(self.iterable)


class enumerate:
    def __init__(self, iterable, start=0):
        self.iterable = iterable
        self.start = start

    def __getitem__(self, index):
        return (self.start + index, self.iterable[index])

    def __len__(self):
        return len(self.iterable)


class zip:
    def __init__(self, *args):
        self.args = args
        self.len = 0

        found = True
        for l in args:
            len_l = len(l)
            if found or len_l < self.len:
                self.len = len_l
                found = False

    def __getitem__(self, index):
        value = []
        for l in self.args:
            value.append(l[index])

        return tuple(value)

    def __len__(self):
        return self.len


def iter(value):
    return value.__iter__()


class BaseException:
    def __init__(self, *args):
        self.args = args

    def with_traceback(self, *args, **kwargs):
        # ignore traceback
        return self


class Exception(BaseException):
    pass


class AssertionError(Exception):
    pass


class RuntimeError(Exception):
    pass


class TypeError(Exception):
    pass


class ValueError(Exception):
    pass


class IndexError(Exception):
    pass


class OSError(Exception):
    pass
