import numpy as np

A = np.array([[1, 2], [3, 4]])
P = np.linalg.matrix_power(A.astype(float), 2)
P3 = np.linalg.matrix_power(A.astype(float), 3)
Q = A**2.0  # a float exponent: integer squares wrap around above 3037000499
n = -(2**2)
h = 2 ** (-1)
x = np.array([1, 2, 3])
y = 2.0**x  # a float base: 2**x on integers wraps around for x >= 63
r = x ** (-1.0)  # a float exponent: integer arrays reject negative integer powers
big = np.array([3037000500, 2])
sq = big**2.0  # beyond 2^63: doubles in the source, no wrap-around
e2 = big[0] ** 2.0  # an element of an integer array is a NumPy integer
