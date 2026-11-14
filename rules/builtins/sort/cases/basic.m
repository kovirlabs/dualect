v = [3 1 2 1];
s = sort(v);
d = sort(v, 'descend');
A = [3 1; 2 5; 1 4];
S = sort(A);                  % each column
R = sort(A, 2, 'descend');    % each row, largest first
c = sort([2; 3; 1]);          % a column stays a column
