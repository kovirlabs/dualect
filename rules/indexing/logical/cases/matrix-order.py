import numpy as np

A = np.array([[1, 5], [7, 2], [3, 9]])
B = A.T[A.T > 2]  # column-major order: 7 3 5 9 (A[A > 2] gives 5 7 3 9)
