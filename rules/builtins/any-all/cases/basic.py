import numpy as np

v = np.array([1, 5, 2])
hit = (v > 3).any()
none = (v > 3).all()
A = np.array([[1, 0, 2], [3, 4, 0]])
cols = A.any(axis=0)  # one answer per column: 1 1 1
cols_all = A.all(axis=0)  # 1 0 0
rows_all = A.all(axis=1)  # 0; 0
every = (A.ravel(order="F") > -1).all()  # every element
nonzero = np.array([0, 0, 7]).any()  # numbers: nonzero counts as true
