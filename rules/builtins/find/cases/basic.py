import numpy as np

x = np.array([3, 0, 7, 1, 9])
idx = np.flatnonzero(x > 2)
first = np.flatnonzero(x > 2)[:1]
none = np.flatnonzero(x > 100)
vals = x[x > 2]  # no find needed to index with a mask
A = np.array([[0, 2], [3, 0], [0, 4]])
lin = np.flatnonzero(A.ravel(order="F"))  # column-major, like find
c, r = np.nonzero(A.T)
