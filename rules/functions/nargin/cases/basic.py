import numpy as np


def scale(x, factor=10):
    return x * factor


def clip_to(x, lo=None, hi=None):
    if lo is None:
        lo = 0
    if hi is None:
        hi = x.max()  # the default depends on another argument
    return np.clip(x, lo, hi)


a = scale(2)
b = scale(2, 3)
x = np.array([-1, 5, 12])
c1 = clip_to(x)
c2 = clip_to(x, None, 4)
c3 = clip_to(x, 1, 6)
