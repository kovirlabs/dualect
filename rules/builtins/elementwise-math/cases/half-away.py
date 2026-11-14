# round() rounds halves away from zero.
import numpy as np

x = np.array([2.5, -2.5, 0.4])
r = np.sign(x) * np.floor(np.abs(x) + 0.5)  # np.round(x) would give 2, -2, 0
