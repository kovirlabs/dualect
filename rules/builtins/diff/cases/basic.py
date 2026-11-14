import numpy as np

x = np.array([1, 4, 9, 16, 25])
d = np.diff(x)  # 3 5 7 9: one element shorter
d2 = np.diff(x, n=2)  # 2 2 2
A = np.array([[1, 2], [4, 8], [9, 9]])
D = np.diff(A, axis=0)  # down each column: 3 6; 5 1
E = np.diff(A, axis=1)  # along each row: 1; 4; 0
m = np.array([1, 3, 2, 5]) > 2
steps = np.diff(m.astype(int))  # logical in, numbers out: 1 -1 1
