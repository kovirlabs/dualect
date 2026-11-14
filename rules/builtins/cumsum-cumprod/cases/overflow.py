import numpy as np

# A product of integer-valued data is computed in doubles: no wrap-around.
big = np.array([100000, 100000, 100000, 100000])
p = np.cumprod(big, dtype=float)  # 1e5 1e10 1e15 1e20
m = np.arange(1, 4) > 1
s = np.cumsum(m)  # a running count of true values: 0 1 2
