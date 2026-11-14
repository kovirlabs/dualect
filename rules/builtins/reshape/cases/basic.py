import numpy as np

# Elements are taken and placed column by column.
B = np.reshape(np.arange(1, 7), (2, 3), order="F")  # [1 3 5; 2 4 6]
A = np.array([[1, 2, 3], [4, 5, 6]])
C = np.reshape(A, (3, 2), order="F")  # [1 5; 4 3; 2 6]
D = np.reshape(A, (-1, 2), order="F")  # the [] size is numel(A) / 2
E = np.reshape(A, (3, 2), order="F")  # the sizes as one vector
r = A.ravel(order="F")  # 1 4 2 5 3 6, a row
c = B.ravel(order="F")  # 1 2 3 4 5 6, a column
