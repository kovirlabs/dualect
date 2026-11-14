import numpy as np

x = np.array([1, 2, 3])
x = np.append(x, 4)
y = np.array([])
for k in range(1, 5):
    y = np.append(y, k**2)
z = np.array([1, 2])
z = np.pad(z, (0, 3))
z[4] = 9  # the gap is filled with zeros
