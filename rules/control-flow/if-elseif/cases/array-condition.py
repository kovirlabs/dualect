import numpy as np

v = np.array([1, 2, 0])
if v.size > 0 and v.all():
    all_nonzero = True
else:
    all_nonzero = False
e = np.array([])
if e.size > 0 and e.all():  # e.all() alone is True for an empty array
    empty_is_true = True
else:
    empty_is_true = False
