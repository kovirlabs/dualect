% Floor division, mod and rem with negative and fractional operands.
a = [-7 7 -7.5 7.5];
q = floor(a / 2);           % floors toward -Inf: -4 3 -4 3
m = mod(a, 2);              % the sign of the divisor
r = rem(a, 2);              % the sign of the dividend
m_neg = mod(7.5, -2);       % -0.5
r_neg = rem(5, -3);         % 2
small = [1e-10 -1e-10];
lp = log1p(small);          % accurate near zero
em = expm1(small);
z = [3+4i, -1-2i];
re = real(z);
im = imag(z);
c = conj(z);
