% The function forms join like the bracket literals.
a = [1 2];
b = [3 4 5];
r = horzcat(a, b);            % a longer row
c = [1; 2];
d = [3; 4];
C2 = horzcat(c, d);           % two columns side by side
col = vertcat(c, d);          % a longer column
A = [1 2; 3 4];
B = [5 6; 7 8];
V = vertcat(A, B);            % 4x2
H = cat(2, A, B);             % 2x4
S = cat(1, A, [9 9]);         % a row stacked under a matrix
e = horzcat(a, [], b);        % empty parts are skipped
