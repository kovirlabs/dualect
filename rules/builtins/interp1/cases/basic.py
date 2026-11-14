import numpy as np

# Query points come first in np.interp; outside the samples the result is NaN.
x = np.array([0, 1, 2, 4])
y = np.array([0, 10, 15, 35])
xi = np.array([0.5, 3, 4])
yi = np.interp(xi, x, y, left=np.nan, right=np.nan)
outside = np.interp([-1, 5], x, y, left=np.nan, right=np.nan)  # NaN NaN: no clamping
col = np.interp(
    np.array([1.5, 2.5]), x, y, left=np.nan, right=np.nan
)  # the shape of the query points
t = np.linspace(0, 2, 5)
v = np.interp(1.25, t, t**2, left=np.nan, right=np.nan)
