% Square matrices: the same LAPACK computations under NumPy names.
A = [4 7; 2 6];
B = inv(A);
d = det(A);                   % 10
t = trace(A);                 % 10
S = [1 2; 2 4];               % singular: the second row is twice the first
r = rank(S);                  % 1
r_full = rank(A);             % 2
r_tol = rank([1 0; 0 1e-8], 1e-6);   % 1: the small singular value is below tol
W = [1 2 3; 4 5 6];
r_wide = rank(W);             % rank of a non-square matrix: 2
