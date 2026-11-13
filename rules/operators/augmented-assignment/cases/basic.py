import numpy as np

n = 0
n += 1
n += 4
k = 10
k -= 3
k *= 2
v = np.array([1, 2, 3])
v = v + 1  # += would change the array in place, and every alias of it
