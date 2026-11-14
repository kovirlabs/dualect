import numpy as np

# The second output holds 1-based positions; equal values keep their order.
v = np.array([2, 1, 2, 1])
# dualect: index i
i = np.argsort(v, kind="stable")
s = v[i]  # i = 2 4 1 3
# dualect: index j
j = np.argsort(-v, kind="stable")
d = v[j]  # j = 1 3 2 4: ties in their original order
w = np.array([10, 20, 30, 40])
w_by_v = w[i]  # reorder another array the same way
# dualect: index i j
