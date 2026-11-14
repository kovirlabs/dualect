import numpy as np

# The sample points come first in trapz and second in np.trapezoid.
y = np.array([1, 4, 9, 16])
unit = np.trapezoid(y)  # unit spacing: 22
x = np.array([0, 0.5, 2, 3])
area = np.trapezoid(y, x)
t = np.linspace(0, np.pi, 101)
s = np.trapezoid(np.sin(t), t)  # close to 2
Y = np.array([[1, 2], [3, 5], [4, 9]])
cols = np.trapezoid(Y, axis=0)  # down each column
cols_x = np.trapezoid(Y, [0, 1, 3], axis=0)
