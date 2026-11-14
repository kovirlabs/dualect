import numpy as np

v = np.array([1, 2, 3])
r = np.flip(v)  # 3 2 1
same_row = v  # a row has one row: unchanged
c = np.array([1, 2, 3])
rc = np.flip(c)  # 3; 2; 1
same_col = c  # a column has one column: unchanged
A = np.array([[1, 2, 3], [4, 5, 6]])
L = np.fliplr(A)  # columns reversed
U = np.flipud(A)  # rows reversed
F = np.flip(A, axis=0)  # first non-singleton dimension: the rows
G = np.fliplr(A)  # along dimension 2: the columns
w = np.flip(v)  # a vector, whichever way it points
