import numpy as np


def sum_prod(x):
    s = x.sum()
    p = x.prod(dtype=float)
    return s, p


def spread(x):
    lo = x.min()
    mid = x.mean()
    hi = x.max()
    return lo, mid, hi


v = np.array([2, 3, 4])
s, p = sum_prod(v)
s_only = sum_prod(v)[0]  # the first output only
_, p_only = sum_prod(v)  # the first output discarded
a, b = spread(v)[:2]  # fewer outputs than the function has
