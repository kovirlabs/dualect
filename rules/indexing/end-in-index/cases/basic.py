import numpy as np

x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
last = x[-1]
tail = x[-3:]
before_last = x[-2]
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
corner = A[-1, -1]
last_col = A[:, -1]
last_row_head = A[-1, :2]
