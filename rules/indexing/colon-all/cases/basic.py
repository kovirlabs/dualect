import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
v = A.ravel(order="F")  # 1 4 2 5 3 6
total = A.sum()
B = A.copy()
B[:] = 0
C = A.copy()
C[:] = np.arange(1, 7).reshape(C.shape, order="F")  # fills column by column
