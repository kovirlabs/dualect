import numpy as np

# Whole numbers print every digit; otherwise give a precision or a format.
n = 42
label = "n = " + f"{n:.0f}"
neg = f"{-7:.0f}"
big = f"{123456789012:.0f}"
flag = f"{3 > 2:.0f}"  # '1', not 'True'
ratio = 1 / 3
r = f"{ratio:.4g}"  # 0.3333
p8 = f"{np.pi:.8g}"  # 3.1415927
t = f"{2.5:8.3f}".strip()  # trimmed: '2.500'
nz = f"{-0.0:.0f}"  # '-0': the sign of zero is printed
