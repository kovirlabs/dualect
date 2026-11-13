import numpy as np


def sum_prod(x):
    return x.sum(), x.prod()


def bounds_of(x):
    return x.min(), x.max()


v = np.array([2, 3, 4])
s, p = sum_prod(v)
s_only = sum_prod(v)[0]
lo_only = bounds_of(v)[0]
lo, hi = bounds_of(v)
