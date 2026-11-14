x = [-2 5 0 3 -1 8];
mask = x > 2;
big = x(mask);
y = x;
y(y < 0) = 0;       % x is unchanged: y is a copy
z = x;
z(z < 0) = [];      % deletion
n_big = sum(mask);
