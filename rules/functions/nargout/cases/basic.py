import numpy as np


def bounds_of(x):
    return x.min(), x.max()  # Python computes every output


v = np.array([2, 3, 4])
lo_only = bounds_of(v)[0]
lo, hi = bounds_of(v)
