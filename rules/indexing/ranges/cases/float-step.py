# A fractional step: count the points instead of trusting float arithmetic.
import numpy as np

t = np.linspace(0, 1, 11)  # floor((1 - 0) / 0.1) + 1 points
n = t.size
t_last = t[-1]
