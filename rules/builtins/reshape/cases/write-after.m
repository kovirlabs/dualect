% Writing into a reshaped array must not change the original.
v = 1:6;
M = reshape(v, 2, 3);
M(1, 1) = 100;
first = v(1);                 % still 1
