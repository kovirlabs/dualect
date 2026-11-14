import numpy as np

# Vector norms agree; the default matrix norm does not.
v = np.array([3, -4, 0])
n2 = np.linalg.norm(v)  # 5
taxi = np.linalg.norm(v, 1)  # 7
peak = np.linalg.norm(v, np.inf)  # 4
A = np.array([[1, 2], [3, 4]])
s = np.linalg.norm(A, 2)  # spectral norm (largest singular value)
f = np.linalg.norm(A)  # NumPy's default for a matrix
c1 = np.linalg.norm(A, 1)  # largest column sum
ri = np.linalg.norm(A, np.inf)  # largest row sum
