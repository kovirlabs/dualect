% Implicit expansion: singleton dimensions stretch to match.
r = [1 2 3];
c = [10; 20];
grid = c + r;           % 2x3: every pairing
A = [1 2 3; 4 5 6];
centered = A - mean(A); % subtract each column's mean
shifted = A + c;        % add c(i) to row i
