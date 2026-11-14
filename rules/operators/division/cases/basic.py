import numpy as np

q = 7 / 2  # 3.5: no integer division
v = np.array([2, 4, 8])
half = v / 2
recip = 1 / v
A = np.array([[4, 1], [2, 3]])
b = np.array([1, 2])
x = np.linalg.solve(A, b)  # never np.linalg.inv(A) @ b
B = np.array([[1, 2], [3, 4]])
R = np.linalg.solve(A.T, B.T).T  # R @ A = B
