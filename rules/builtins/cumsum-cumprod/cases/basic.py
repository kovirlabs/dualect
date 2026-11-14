import numpy as np

v = np.array([1, 2, 3, 4])
c = np.cumsum(v)  # 1 3 6 10
p = np.cumprod(v, dtype=float)  # 1 2 6 24
A = np.array([[1, 2], [3, 4], [5, 6]])
C = np.cumsum(A, axis=0)  # down each column
R = np.cumprod(A, axis=1, dtype=float)  # along each row
col = np.cumsum(np.array([1, 2, 3]))  # a column stays a column
e = np.cumsum(np.array([]))  # empty in, empty out
