import numpy as np

v = np.array([3, 1, 2, 1])
s = np.sort(v)
d = np.flip(np.sort(v))
A = np.array([[3, 1], [2, 5], [1, 4]])
S = np.sort(A, axis=0)  # each column
R = np.flip(np.sort(A, axis=1), axis=1)  # each row, largest first
c = np.sort(np.array([2, 3, 1]))  # a column stays a column
