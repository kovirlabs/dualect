import numpy as np

v = np.array([1, 2, 3])
c = np.array([4, 5, 6])  # a column is 1-D in NumPy too
A = np.array([[1, 2], [3, 4]])
e = np.array([])
neg = np.array([1, -2])  # two elements
one = 1 - 2  # one element: -1
w = np.concatenate((v, [4]))  # concatenation, not nesting
x = 2
row = np.array([x, 2 * x, x**2])
B = np.vstack((v, v))  # two rows stacked: 2x3
