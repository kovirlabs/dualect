import numpy as np

# The grids are numel(y) x numel(x): X varies along rows, Y down columns.
x = np.array([0, 1, 2])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y)  # 2x3 each
Z = X + Y
P, Q = np.meshgrid(np.arange(1, 5), np.arange(1, 5))  # meshgrid(x) is meshgrid(x, x)
corner = X[1, 2]  # x(3), from row 2
