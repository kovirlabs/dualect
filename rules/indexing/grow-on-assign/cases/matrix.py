import numpy as np

A = np.eye(2)
A = np.pad(A, ((0, 0), (0, 2)))
A[1, 3] = 1  # A becomes 2x4; the new column 3 is zeros
