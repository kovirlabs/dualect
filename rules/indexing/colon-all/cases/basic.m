A = [1 2 3; 4 5 6];
v = A(:);           % 1 4 2 5 3 6
total = sum(A(:));
B = A;
B(:) = 0;
C = A;
C(:) = 1:6;         % fills column by column
