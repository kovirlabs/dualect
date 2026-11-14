import numpy as np

# With dimensions, the constants fill arrays; one dimension means a square.
n = 3
mask = np.full((n, n), True)  # 3x3, not 1x3
none = np.full((2, 4), False)
gaps = np.full(n, np.nan)  # a row
lo = -np.full(2, np.inf)  # a column
cnt = mask.sum()
