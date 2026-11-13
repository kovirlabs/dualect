% Newton's method for sqrt(2), then a do-until counter.
a = 2;
x = 1;
iterations = 0;
while abs(x^2 - a) > 1e-12
  x = x - (x^2 - a) / (2*x);
  iterations = iterations + 1;
end
n = 0;
do                  % Octave only
  n++;
until n >= 3
