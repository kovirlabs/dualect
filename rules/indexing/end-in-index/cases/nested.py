import numpy as np

x = np.array([10, 20, 30, 40, 50, 60])
idx = np.array([1, 3, 5])  # positions, 0-based
v = x[idx[-1]]  # -1 refers to idx here
n = 2
w = x[-1 - n]
