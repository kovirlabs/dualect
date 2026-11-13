a = [1 2 3 4];
b = [1 0 3 0];
d1 = a ~= b;
d2 = a != b;        % Octave only
mask = a > 2;
nm = ~mask;
z = ~b;             % NOT of numbers: true where b is zero
s = ~5;
t = !0;             % Octave only
