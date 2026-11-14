v = [1 2 3 4];
c = cumsum(v);            % 1 3 6 10
p = cumprod(v);           % 1 2 6 24
A = [1 2; 3 4; 5 6];
C = cumsum(A);            % down each column
R = cumprod(A, 2);        % along each row
col = cumsum([1; 2; 3]);  % a column stays a column
e = cumsum([]);           % empty in, empty out
