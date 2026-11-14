import numpy as np

x = np.array([-2.5, -1, 0, 1.5, 4])
a = np.abs(x)
r = np.sqrt(np.abs(x))
e = np.exp(-x)
f = np.floor(x)
c = np.ceil(x)
t = np.trunc(x)  # toward zero
m = np.mod(-7, 3)  # 2: the sign of the divisor
k = np.fmod(-7, 3)  # -1: the sign of the dividend
s = np.sin(np.pi * x)
nan_mask = np.isnan([1, np.nan, 3])
h = abs(-3)  # a scalar
ang = np.arctan2(1, -1)
