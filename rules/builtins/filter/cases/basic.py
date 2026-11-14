import numpy as np
from scipy.signal import lfilter

# Moving average (FIR) and a first-order low-pass (IIR) of a short signal.
x = np.array([1, 4, 2, 8, 5, 7, 3, 6])
b = np.ones(3) / 3
y_fir = lfilter(b, 1, x)
alpha = 0.25
y_iir = lfilter(alpha, [1, alpha - 1], x)
# a(1) need not be 1: both sides divide through by it.
y_scaled = lfilter([2, 2], [4, 1], x.astype(float))
