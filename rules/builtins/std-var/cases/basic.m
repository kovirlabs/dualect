% Sample statistics divide by N - 1; a weight of 1 divides by N.
x = [2 4 4 4 5 5 7 9];
s = std(x);            % N - 1: 2.1381
s_pop = std(x, 1);     % N: 2
v = var(x);
A = [1 2; 3 5; 4 9];
col_std = std(A);      % down each column
row_var = var(A, 0, 2);  % the third argument is the dimension
row_pop = var(A, 1, 2);
