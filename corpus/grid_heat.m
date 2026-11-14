% Steady temperature on a unit square: a hot top edge, cold elsewhere,
% solved by Jacobi iteration on a grid, then integrated with trapz.
nx = 21;
ny = 11;
x = linspace(0, 1, nx);
y = linspace(0, 1, ny);
[X, Y] = meshgrid(x, y);       % ny-by-nx: rows follow y, columns follow x

T = zeros(ny, nx);
T(end, :) = 100;               % the top edge (y = 1) is hot

for iter = 1:400
  T_old = T;
  T(2:end-1, 2:end-1) = 0.25 * (T_old(1:end-2, 2:end-1) + T_old(3:end, 2:end-1) ...
                              + T_old(2:end-1, 1:end-2) + T_old(2:end-1, 3:end));
end
change = max(max(abs(T - T_old)));

% Mean temperature over the square, by the trapezoidal rule in both directions.
inner = trapz(x, T, 2);        % along x, for each row
mean_T = trapz(y, inner);

% The centre line, its gradient, and the hottest interior point.
centre = T(:, (nx + 1) / 2);
dT = diff(centre) / (y(2) - y(1));
interior = T(2:end-1, 2:end-1);
[t_max, idx] = max(interior(:));
[row, col] = ind2sub(size(interior), idx);

% A smooth reference field on the same grid, and its distance from T.
F = 100 * sinh(pi * Y) .* sin(pi * X) / sinh(pi);
gap = norm(T - F, 'fro') / sqrt(numel(T));
profile = reshape(T(:, 1:2:end), 1, []);
