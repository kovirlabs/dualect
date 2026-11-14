import numpy as np

# Sample statistics divide by N - 1; a weight of 1 divides by N.
x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
s = x.std(ddof=1)  # N - 1: 2.1381
s_pop = x.std()  # N: 2
v = x.var(ddof=1)
A = np.array([[1, 2], [3, 5], [4, 9]])
col_std = A.std(axis=0, ddof=1)  # down each column
row_var = A.var(axis=1, ddof=1)  # the third argument is the dimension
row_pop = A.var(axis=1)
