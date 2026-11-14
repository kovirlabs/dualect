import numpy as np

x = np.array([-2, 5, 0, 3, -1, 8])
mask = x > 2
big = x[mask]
y = x.copy()  # without copy, the next line would also change x
y[y < 0] = 0
z = x
z = z[~(z < 0)]  # deletion keeps the complement
n_big = np.count_nonzero(mask)
