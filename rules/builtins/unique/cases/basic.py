import numpy as np

v = np.array([3, 1, 3, 2, 1])
u = np.unique(v)  # 1 2 3, a row
c = np.unique(np.array([5, 4, 5]))  # 4; 5, a column
A = np.array([[2, 1], [1, 3]])
w = np.unique(A)  # 1; 2; 3: a matrix gives a column
m = np.unique(np.array([1, 0, 1]) > 0)  # logical values stay logical
e = np.unique(np.array([]))  # empty
n = np.unique(v).size  # how many distinct values
