import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
col_sums = A.sum(axis=0)
row_sums = A.sum(axis=1)
total = A.sum()
col_means = A.mean(axis=0)
row_means = A.mean(axis=1)
v = np.array([4, 9, 2, 9])
i = np.argmax(v)  # first occurrence, 0-based
mx = v[i]
j = np.argmin(v)
mn = v[j]
col_max = A.max(axis=0)
row_max = A.max(axis=1)
pairwise = np.maximum([1, 5, 3], [4, 2, 6])
