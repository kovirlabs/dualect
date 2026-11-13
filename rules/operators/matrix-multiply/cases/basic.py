import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
D = A * B
s = 2 * A
y = A @ np.array([1, 1])
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
inner = u @ v
outer = np.outer(u, v)
