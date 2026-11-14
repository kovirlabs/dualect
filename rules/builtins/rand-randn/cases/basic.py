import numpy as np

# Random values are compared by shape: rand(n) is square, 1xn or nx1 is 1-D.
U = np.random.default_rng().random((3, 2))
S = np.random.default_rng().random((3, 3))  # 3x3, not three numbers
z = np.random.default_rng().standard_normal(5)
c = np.random.default_rng().standard_normal(4)
u = np.random.default_rng().random()  # one number
in_range = U.min() >= 0 and U.max() < 1
m = z.mean()  # computed from random numbers
# dualect: shape m
