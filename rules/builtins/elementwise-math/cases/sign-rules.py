import numpy as np

# Floor division, mod and rem with negative and fractional operands.
a = np.array([-7, 7, -7.5, 7.5])
q = a // 2  # floors toward -Inf: -4 3 -4 3
m = np.mod(a, 2)  # the sign of the divisor
r = np.fmod(a, 2)  # the sign of the dividend
m_neg = np.mod(7.5, -2)  # -0.5
r_neg = np.fmod(5, -3)  # 2
small = np.array([1e-10, -1e-10])
lp = np.log1p(small)  # accurate near zero
em = np.expm1(small)
z = np.array([3 + 4j, -1 - 2j])
re = np.real(z)
im = np.imag(z)
c = np.conj(z)
