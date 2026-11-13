A = [1 2 3; 4 5 6];
col_sums = sum(A);
row_sums = sum(A, 2);
total = sum(A(:));
col_means = mean(A);
row_means = mean(A, 2);
v = [4 9 2 9];
[mx, i] = max(v);     % first occurrence of the maximum
[mn, j] = min(v);
col_max = max(A);
row_max = max(A, [], 2);
pairwise = max([1 5 3], [4 2 6]);
% dualect: index i j
