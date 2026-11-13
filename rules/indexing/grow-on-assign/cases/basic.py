import numpy as np

x = np.array([1, 2, 3])
x = np.append(x, 4)
squares = []
for k in range(1, 5):
    squares.append(k**2)
y = np.array(squares)
z = np.zeros(5)
z[:2] = [1, 2]
z[4] = 9
