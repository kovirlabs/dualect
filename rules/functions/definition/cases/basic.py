import numpy as np


def area(r):
    return np.pi * r**2


def zero_first(x):
    x = x.copy()  # Octave works on a copy; NumPy would change the caller's array
    x[0] = 0
    return x


a1 = area(2)
radii = np.array([1, 2, 3])
areas = area(radii)
v = np.array([5, 6, 7])
w = zero_first(v)
sq = lambda t: t**2
nine = sq(3)
