% An anonymous function keeps the value k had when it was created.
k = 2;
f = @(t) k * t;
k = 10;
y = f(3);           % 6, not 30
