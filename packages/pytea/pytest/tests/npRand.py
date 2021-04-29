import numpy as np

x = np.random.rand(1_00, 3, 100, 100)
# y = np.random.randint(0, 2, 100)
print(x.shape)

a = np.random.randint(3, 4)
b = np.random.randint(3, 4, 5)
c = np.random.randint(3)
d = np.random.randint(3, size=(2, 3, 4))
print(a)
print(b)
print(c)
print(d)
