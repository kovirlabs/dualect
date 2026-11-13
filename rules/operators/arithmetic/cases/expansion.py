# Implicit expansion: singleton dimensions stretch to match.
import numpy as np

r = np.array([1, 2, 3])
c = np.array([10, 20])
grid = c[:, np.newaxis] + r  # 1-D c + r would pair elements (or fail)
A = np.array([[1, 2, 3], [4, 5, 6]])
centered = A - A.mean(axis=0)
shifted = A + c[:, np.newaxis]
