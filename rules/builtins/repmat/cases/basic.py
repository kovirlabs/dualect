import numpy as np

# Matrices, rows and columns tile differently in NumPy.
A = np.array([[1, 2], [3, 4]])
T = np.tile(A, (2, 3))  # 4x6
S = np.tile(A, (2, 2))  # 2x2 copies: 4x4
v = np.array([1, 2, 3])
w = np.tile(v, 3)  # a longer row
V = np.tile(v, (2, 1))  # two rows: a matrix
c = np.array([5, 6])
d = np.tile(c, 3)  # a longer column
W = np.tile(c[:, np.newaxis], (1, 3))  # three columns side by side
z = np.tile(7, 4)  # a scalar
E = np.tile(A, (0, 2))  # zero copies: empty
