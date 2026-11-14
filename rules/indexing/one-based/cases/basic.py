import numpy as np

x = np.array([10, 20, 30, 40])
A = np.array([[1, 2, 3], [4, 5, 6]])
first = x[0]
last = x[3]
a23 = A[1, 2]
k4 = A.ravel(order="F")[3]  # linear index walks columns: this is A[1, 1]
i = 1
nxt = x[i + 1]
