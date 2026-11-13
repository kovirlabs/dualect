import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1, 0, 3, 0])
d1 = a != b
d2 = a != b
mask = a > 2
nm = ~mask
z = np.logical_not(b)  # ~b would be bitwise on integers
s = not 5
t = not 0
