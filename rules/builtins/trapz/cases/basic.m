% The sample points come first in trapz and second in np.trapezoid.
y = [1 4 9 16];
unit = trapz(y);              % unit spacing: 22
x = [0 0.5 2 3];
area = trapz(x, y);
t = linspace(0, pi, 101);
s = trapz(t, sin(t));         % close to 2
Y = [1 2; 3 5; 4 9];
cols = trapz(Y);              % down each column
cols_x = trapz([0 1 3], Y);
