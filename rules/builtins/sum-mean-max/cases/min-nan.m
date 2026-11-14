% min skips NaN like max: the value, its first position, rows and pairs.
w = [4 NaN 1 1];
m = min(w);               % 1: NaN is skipped
[m2, k] = min(w);         % k is the first of the tied minima: 3
A = [3 NaN; 2 5];
col_min = min(A);         % 2 5
row_min = min(A, [], 2);  % 3; 2
pair = min([1 NaN 3], [2 2 NaN]);   % 1 2 3: NaN only where both are NaN
% dualect: index k
