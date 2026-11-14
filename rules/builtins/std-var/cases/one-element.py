import numpy as np

# One value: N - 1 would divide by zero, but the source returns 0.
single = np.std(7)
single_var = np.var(7)
pair = np.array([3, 5]).std(ddof=1)  # two elements: N - 1 = 1
