% NaN goes last ascending and first descending.
x = [3 NaN 1 NaN 2];
up = sort(x);
down = sort(x, 'descend');
[up2, k] = sort(x);           % the NaN positions stay in order: 2 4 last
% dualect: index k
