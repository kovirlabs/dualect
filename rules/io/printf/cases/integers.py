import numpy as np

# %d needs a whole number: a computed count is one, a quotient may not be.
k = np.floor(7 / 2)
total = np.array([1, 2, 3]).sum()
print(f"k = {int(k)}, total = {int(total)}, padded = {int(k):04d}")
print(f"big: {int(2**40)}")
# dualect: stdout
