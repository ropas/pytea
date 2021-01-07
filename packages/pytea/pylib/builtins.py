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
float = __Primitives(1)
float.__mro__ = (float, object)
str = __Primitives(2)
str.__mro__ = (str, object)
bool = __Primitives(3)
bool.__mro__ = (bool, object)
tuple = __Primitives(4)  # TODO: make class
tuple.__mro__ = (tuple, object)
list = __Primitives(5)
list.__mro__ = (list, object)
dict = __Primitives(6)
dict.__mro__ = (dict, object)
set = __Primitives(7)
set.__mro__ = (set, object)
Ellipsis = __Primitives(8)
Ellipsis.__mro__ = (Ellipsis, object)


def _list_append(self, item):
    LibCall.builtins.list_append(self, item)


list.append = _list_append


def _str_format(self, *args, **kwargs):
    # TODO: format it.
    return self


str.format = _str_format


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
