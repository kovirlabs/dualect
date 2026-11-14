import numpy as np

# A least-squares quadratic; coefficients are listed highest power first.
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1.1, 2.9, 9.2, 19.1, 32.8, 51.2])
p = np.polyfit(x, y, 2)
y_fit = np.polyval(p, x)
residual = np.nanmax(np.abs(y - y_fit))
line = np.polyfit(x, 3 * x - 2, 1)  # exact: 3 -2
at_half = np.polyval([2, 0, -1], 0.5)  # 2x^2 - 1 at 0.5: -0.5
