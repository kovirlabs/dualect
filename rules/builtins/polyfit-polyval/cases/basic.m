% A least-squares quadratic; coefficients are listed highest power first.
x = [0 1 2 3 4 5];
y = [1.1 2.9 9.2 19.1 32.8 51.2];
p = polyfit(x, y, 2);
y_fit = polyval(p, x);
residual = max(abs(y - y_fit));
line = polyfit(x, 3 * x - 2, 1);    % exact: 3 -2
at_half = polyval([2 0 -1], 0.5);   % 2x^2 - 1 at 0.5: -0.5
