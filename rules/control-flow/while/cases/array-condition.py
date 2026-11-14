# An array condition holds only while every element is nonzero.
import numpy as np

v = np.array([3, 2, 1])
steps = 0
while v.size > 0 and v.all():
    v = v - 1
    steps += 1
e = np.array([])
empty_runs = 0
while e.size > 0 and e.all():  # e.all() alone is True for an empty array
    empty_runs += 1
