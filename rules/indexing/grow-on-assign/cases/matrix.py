import numpy as np

A = np.pad(np.eye(2), ((0, 0), (0, 2)))  # two zero columns on the right
A[1, 3] = 1
