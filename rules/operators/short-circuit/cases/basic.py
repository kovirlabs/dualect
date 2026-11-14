import numpy as np

x = np.array([5, 0, 3])
n = 0
guarded = n > 0 and x[n - 1] > 1  # x[-1] is never evaluated
n = 2
second_big = n > 0 and x[n - 1] > 1
flag = x.size == 0 or x[0] == 5
a = np.array([1, -1, 2, -2])
b = np.array([-1, -1, 2, 2])
both = (a > 0) & (b > 0)
either = (a > 0) | (b > 0)
num_and = np.logical_and([1, 2], [2, 2])  # & would be bitwise: [0, 2]
one = np.logical_xor(True, False)
