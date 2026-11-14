import numpy as np

mask = np.array([1, 5, 3]) > 2
counts = mask.astype(float)  # 0 1 1, as numbers
n = float(True) + 1
flags = np.array([2, 0, -1]).astype(bool)  # nonzero is true
x = np.array([1, 2, 3])
y = x / 2  # already a double: no change
