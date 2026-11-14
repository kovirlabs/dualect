% The middle value; an even count averages the two middle ones.
x = [7 1 3 10];
m = median(x);          % 5: (3 + 7) / 2
odd = median([5 2 9]);  % 5
A = [1 8; 4 2; 9 3];
col_m = median(A);      % down each column: 4 3
row_m = median(A, 2);   % 4.5 3 6
with_nan = median([1 NaN 3]);
