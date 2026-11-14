import numpy as np

# NaN goes last ascending and first descending.
x = np.array([3, np.nan, 1, np.nan, 2])
up = np.sort(x)
down = np.flip(np.sort(x))
# dualect: index k
k = np.argsort(x, kind="stable")
up2 = x[k]  # the NaN positions stay in order: 2 4 last
# dualect: index k
