import numpy as np

x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 100)  # NumPy's default would be 50
n = y.size
step = y[1] - y[0]
