import numpy as np

# The middle value; an even count averages the two middle ones.
x = np.array([7, 1, 3, 10])
m = np.median(x)  # 5: (3 + 7) / 2
odd = np.median([5, 2, 9])  # 5
A = np.array([[1, 8], [4, 2], [9, 3]])
col_m = np.median(A, axis=0)  # down each column: 4 3
row_m = np.median(A, axis=1)  # 4.5 3 6
with_nan = np.median([1, np.nan, 3])
