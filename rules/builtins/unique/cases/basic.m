v = [3 1 3 2 1];
u = unique(v);            % 1 2 3, a row
c = unique([5; 4; 5]);    % 4; 5, a column
A = [2 1; 1 3];
w = unique(A);            % 1; 2; 3: a matrix gives a column
m = unique([1 0 1] > 0);  % logical values stay logical
e = unique([]);           % empty
n = numel(unique(v));     % how many distinct values
