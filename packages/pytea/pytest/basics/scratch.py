class A:
    def __init__(self, x):
        self.x = x

    def __call__(self, y):
        self.x = self.x + y
        return 101


class B(A):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y


b = B(3, 2)
c = b(12)
