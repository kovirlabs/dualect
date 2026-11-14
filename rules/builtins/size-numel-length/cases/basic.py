import numpy as np

A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
r, c = A.shape
rows = A.shape[0]
cols = A.shape[1]
n = A.size
L = max(A.shape)  # len(A) would be 2
sz = A.shape
v = np.array([4, 5, 6])
nv = v.size
e = np.array([])
Le = max(e.shape)
ne = e.size
dA = A.ndim
dv = 2  # 2, although v is 1-D in NumPy
name = "beam"
nc = len(name)
