% The second output holds 1-based positions; equal values keep their order.
v = [2 1 2 1];
[s, i] = sort(v);             % i = 2 4 1 3
[d, j] = sort(v, 'descend');  % j = 1 3 2 4: ties in their original order
w = [10 20 30 40];
w_by_v = w(i);                % reorder another array the same way
% dualect: index i j
