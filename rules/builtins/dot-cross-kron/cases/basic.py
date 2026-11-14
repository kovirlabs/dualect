import numpy as np

# dot ignores orientation; kron needs it.
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
d = np.dot(u, v)  # 32: a row and a column are fine
w = np.cross([1, 0, 0], [0, 1, 0])  # [0 0 1]
A = np.array([[1, 2], [3, 4]])
K = np.kron(A, np.eye(2))  # 4x4 block matrix
r = np.array([1, 2])
c = np.array([10, 20])
M = np.outer(c, r)  # 2x2: the outer product
L = np.kron(r, [1, 1])  # two rows: a longer row
P = np.kron(c[:, np.newaxis], A)  # a column next to a matrix: 4x2
