import numpy as np

# min skips NaN like max: the value, its first position, rows and pairs.
w = np.array([4, np.nan, 1, 1])
m = np.nanmin(w)  # 1: NaN is skipped
k = np.nanargmin(w)
m2 = w[k]  # k is the first of the tied minima: 3
A = np.array([[3, np.nan], [2, 5]])
col_min = np.nanmin(A, axis=0)  # 2 5
row_min = np.nanmin(A, axis=1)  # 3; 2
pair = np.fmin([1, np.nan, 3], [2, 2, np.nan])  # 1 2 3: NaN only where both are NaN
# dualect: index k
