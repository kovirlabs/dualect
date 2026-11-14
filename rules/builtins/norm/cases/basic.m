% Vector norms agree; the default matrix norm does not.
v = [3 -4 0];
n2 = norm(v);                 % 5
taxi = norm(v, 1);            % 7
peak = norm(v, Inf);          % 4
A = [1 2; 3 4];
s = norm(A);                  % spectral norm (largest singular value)
f = norm(A, 'fro');           % NumPy's default for a matrix
c1 = norm(A, 1);              % largest column sum
ri = norm(A, Inf);            % largest row sum
