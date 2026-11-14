import numpy as np

# Writing into a reshaped array must not change the original.
v = np.arange(1, 7)
M = np.reshape(v, (2, 3), order="F").copy()
M[0, 0] = 100
first = v[0]  # still 1
