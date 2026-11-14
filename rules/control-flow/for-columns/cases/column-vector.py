# A column vector is one column, so the loop body runs once.
import numpy as np

v = np.array([1, 2, 3])
iterations = 0
for x in [v]:  # not `for x in v`, which would run three times
    iterations += 1
