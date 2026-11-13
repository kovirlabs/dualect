import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
total = np.zeros(2)
count = 0
for c in A.T:  # the rows of A.T are the columns of A
    total = total + c
    count += 1
s = 0
for k in range(1, 5):
    s += k
