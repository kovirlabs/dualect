import numpy as np

b = np.array([1, 2, 3])
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.column_stack((b, b))
C = np.column_stack((A, b))
D = np.column_stack((b, A, b))
