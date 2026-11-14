% A matrix is filtered column by column; a column vector stays a column.
X = [1 2; 3 4; 5 6; 7 8];
Y = filter([1 -1], 1, X);   % first differences down each column
c = [1; 0; 0; 0];
h = filter(1, [1 -0.5], c); % impulse response: 1, 0.5, 0.25, 0.125
