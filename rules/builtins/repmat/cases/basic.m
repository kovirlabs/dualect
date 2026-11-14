% Matrices, rows and columns tile differently in NumPy.
A = [1 2; 3 4];
T = repmat(A, 2, 3);          % 4x6
S = repmat(A, 2);             % 2x2 copies: 4x4
v = [1 2 3];
w = repmat(v, 1, 3);          % a longer row
V = repmat(v, 2, 1);          % two rows: a matrix
c = [5; 6];
d = repmat(c, 3, 1);          % a longer column
W = repmat(c, [1 3]);         % three columns side by side
z = repmat(7, 1, 4);          % a scalar
E = repmat(A, 0, 2);          % zero copies: empty
