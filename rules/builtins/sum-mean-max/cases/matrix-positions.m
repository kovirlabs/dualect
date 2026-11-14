% Of a matrix, [m, i] = max(A) gives each column's maximum and its row.
A = [3 9 2; 7 1 8; 7 4 5];
[col_max, row_of_max] = max(A);   % ties: the first row (column 1: row 2)
[col_min, row_of_min] = min(A);
