import numpy as np

# On empty data any is false and all is true.
e = np.array([])
a1 = e.any()
a2 = e.all()
x = np.array([3, 4, 5])
small = x[x < 0]  # no element: empty
all_small_neg = (small < 0).all()
any_small = small.any()
