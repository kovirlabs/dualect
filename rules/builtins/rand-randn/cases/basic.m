% Random values are compared by shape: rand(n) is square, 1xn or nx1 is 1-D.
U = rand(3, 2);
S = rand(3);                  % 3x3, not three numbers
z = randn(1, 5);
c = randn(4, 1);
u = rand;                     % one number
in_range = min(U(:)) >= 0 && max(U(:)) < 1;
m = mean(z);                  % computed from random numbers
% dualect: shape U S z c u m
