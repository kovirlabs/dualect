x = [1 4 9 16 25];
d = diff(x);              % 3 5 7 9: one element shorter
d2 = diff(x, 2);          % 2 2 2
A = [1 2; 4 8; 9 9];
D = diff(A);              % down each column: 3 6; 5 1
E = diff(A, 1, 2);        % along each row: 1; 4; 0
m = [1 3 2 5] > 2;
steps = diff(m);          % logical in, numbers out: 1 -1 1
