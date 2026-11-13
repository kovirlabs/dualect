q = 7 / 2;          % 3.5: no integer division
v = [2 4 8];
half = v / 2;       % by a scalar
recip = 1 ./ v;     % element-wise
A = [4 1; 2 3];
b = [1; 2];
x = A \ b;          % solves A * x = b
B = [1 2; 3 4];
R = B / A;          % solves R * A = B
