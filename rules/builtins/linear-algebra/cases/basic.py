import numpy as np

# Square matrices: the same LAPACK computations under NumPy names.
A = np.array([[4, 7], [2, 6]])
B = np.linalg.inv(A)
d = np.linalg.det(A)  # 10
t = np.trace(A)  # 10
S = np.array([[1, 2], [2, 4]])  # singular: the second row is twice the first
r = np.linalg.matrix_rank(S)  # 1
r_full = np.linalg.matrix_rank(A)  # 2
r_tol = np.linalg.matrix_rank(
    np.array([[1, 0], [0, 1e-8]]), 1e-6
)  # 1: the small singular value is below tol
W = np.array([[1, 2, 3], [4, 5, 6]])
r_wide = np.linalg.matrix_rank(W)  # rank of a non-square matrix: 2
