from collections import namedtuple

TT = namedtuple('TT', ['x', 'y'])
a = TT(3, 5)
b = a[0]
c = a.y