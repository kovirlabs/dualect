import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
At = A.T
Z = np.array([[1 + 2j, 3 - 1j], [2j, 4]])
Zh = Z.conj().T  # conjugate transpose
Zt = Z.T  # plain transpose
z = np.array([1 + 2j, 3 - 1j])
zh = z.conj()  # 1-D: nothing to flip, but the conjugate still applies
c = np.array([1, 2, 3])
c_total = c.sum()
