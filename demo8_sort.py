import numpy as np

a1 = np.array([1, 2, 3, 4])
print(a1.shape, a1)
a2 = np.reshape(a1, (2, 2))
print(a2.shape)
print(a2)

a1 *= 2
print(a1)
print(a2)
a3 = a2.T
print(a3)

a = np.zeros((10, 2))
b = a.T
print(a, a.shape)
print(b, b.shape)
c = np.reshape(b, (5, 4))
print(c, c.shape)
d = np.reshape(b, (20,))
print(d, d.shape)
e = np.reshape(b, (20, -1))
print(e, e.shape)
f = np.reshape(b, (-1, 20))
print(f, f.shape)
g = np.reshape(b, (1, 20))
print(g, g.shape)