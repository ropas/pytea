class A():
    def __init__(self, x):
        self.x = x

def test(x):
    x = A(x)
    return x

x = A(1)
y = test(2)
z = x