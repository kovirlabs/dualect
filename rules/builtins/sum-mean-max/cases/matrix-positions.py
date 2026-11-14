import numpy as np

# Of a matrix, [m, i] = max(A) gives each column's maximum and its row.
A = np.array([[3, 9, 2], [7, 1, 8], [7, 4, 5]])
# dualect: index row_of_max
row_of_max = np.argmax(A, axis=0)
col_max = A.max(axis=0)  # ties: the first row (column 1: row 2)
# dualect: index row_of_min
row_of_min = np.argmin(A, axis=0)
col_min = A.min(axis=0)
