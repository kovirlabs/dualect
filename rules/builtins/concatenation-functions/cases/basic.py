import numpy as np

# The function forms join like the bracket literals.
a = np.array([1, 2])
b = np.array([3, 4, 5])
r = np.concatenate((a, b))  # a longer row
c = np.array([1, 2])
d = np.array([3, 4])
C2 = np.column_stack((c, d))  # two columns side by side
col = np.concatenate((c, d))  # a longer column
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
V = np.vstack((A, B))  # 4x2
H = np.column_stack((A, B))  # 2x4
S = np.vstack((A, np.array([9, 9])))  # a row stacked under a matrix
e = np.concatenate((a, np.array([]), b))  # empty parts are skipped
