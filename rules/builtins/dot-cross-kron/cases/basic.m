% dot ignores orientation; kron needs it.
u = [1 2 3];
v = [4; 5; 6];
d = dot(u, v);                % 32: a row and a column are fine
w = cross([1 0 0], [0 1 0]);  % [0 0 1]
A = [1 2; 3 4];
K = kron(A, eye(2));          % 4x4 block matrix
r = [1 2];
c = [10; 20];
M = kron(c, r);               % 2x2: the outer product
L = kron(r, [1 1]);           % two rows: a longer row
P = kron(c, A);               % a column next to a matrix: 4x2
