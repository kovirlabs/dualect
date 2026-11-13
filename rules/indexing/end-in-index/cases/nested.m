x = [10 20 30 40 50 60];
idx = [2 4 6];
v = x(idx(end));    % end is numel(idx) here, so this is x(6)
n = 2;
w = x(end - n);     % end is numel(x): x(4)
% dualect: index idx
