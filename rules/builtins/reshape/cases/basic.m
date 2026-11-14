% Elements are taken and placed column by column.
B = reshape(1:6, 2, 3);       % [1 3 5; 2 4 6]
A = [1 2 3; 4 5 6];
C = reshape(A, 3, 2);         % [1 5; 4 3; 2 6]
D = reshape(A, [], 2);        % the [] size is numel(A) / 2
E = reshape(A, [3 2]);        % the sizes as one vector
r = reshape(A, 1, []);        % 1 4 2 5 3 6, a row
c = reshape(B, [], 1);        % 1 2 3 4 5 6, a column
