import numpy as np
from scipy.signal import lfilter

# A matrix is filtered column by column; a column vector stays a column.
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
Y = lfilter([1, -1], 1, X.astype(float), axis=0)  # first differences down each column
c = np.array([1, 0, 0, 0])
h = lfilter(1, [1, -0.5], c)  # impulse response: 1, 0.5, 0.25, 0.125
