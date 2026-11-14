% The grids are numel(y) x numel(x): X varies along rows, Y down columns.
x = [0 1 2];
y = [10; 20];
[X, Y] = meshgrid(x, y);      % 2x3 each
Z = X + Y;
[P, Q] = meshgrid(1:4);       % meshgrid(x) is meshgrid(x, x)
corner = X(2, 3);             % x(3), from row 2
