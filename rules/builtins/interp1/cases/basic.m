% Query points come first in np.interp; outside the samples the result is NaN.
x = [0 1 2 4];
y = [0 10 15 35];
xi = [0.5 3 4];
yi = interp1(x, y, xi);
outside = interp1(x, y, [-1 5]);        % NaN NaN: no clamping
col = interp1(x, y, [1.5; 2.5], 'linear');  % the shape of the query points
t = 0:0.5:2;
v = interp1(t, t .^ 2, 1.25);
