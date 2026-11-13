% A fractional step: count the points instead of trusting float arithmetic.
t = 0:0.1:1;
n = numel(t);
t_last = t(end);
