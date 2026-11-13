x = [3 0 7 1 9];
idx = find(x > 2);
first = find(x > 2, 1);
none = find(x > 100);
vals = x(find(x > 2));
A = [0 2; 3 0; 0 4];
lin = find(A);          % column-major: 2, 4, 6
[r, c] = find(A);
% dualect: index idx first none lin r c
