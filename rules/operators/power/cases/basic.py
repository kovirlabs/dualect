import numpy as np

A = np.array([[1, 2], [3, 4]])
P = np.linalg.matrix_power(A, 2)
P3 = np.linalg.matrix_power(A, 3)
Q = A**2
n = -(2**2)
h = 2**-1
x = np.array([1, 2, 3])
y = 2**x
r = x**-1.0  # a float exponent: integer arrays reject negative integer powers
