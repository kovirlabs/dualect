import numpy as np

w = np.array([1, np.nan, 3])
m = np.nanmax(w)  # w.max() would be nan
k = np.nanargmax(w)
m2 = w[k]
s = w.sum()  # NaN propagates
s_valid = w[~np.isnan(w)].sum()
