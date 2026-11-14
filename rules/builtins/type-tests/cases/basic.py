import numpy as np

# Logicals are not numeric; characters are neither; an empty array keeps its class.
x = 3.5
mask = np.array([1, 5, 2]) > 2
name = "beam"
e = np.array([])
num_x = np.issubdtype(np.asarray(x).dtype, np.number)
num_mask = np.issubdtype(np.asarray(mask).dtype, np.number)  # false
num_name = np.issubdtype(np.asarray(name).dtype, np.number)  # false
num_e = np.issubdtype(np.asarray(e).dtype, np.number)  # true
log_mask = np.asarray(mask).dtype == bool
log_x = np.asarray(x).dtype == bool
log_e = np.asarray(e).dtype == bool  # false
chr_name = isinstance(name, str)
chr_x = isinstance(x, str)
