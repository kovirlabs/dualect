% Moving average (FIR) and a first-order low-pass (IIR) of a short signal.
x = [1 4 2 8 5 7 3 6];
b = ones(1, 3) / 3;
y_fir = filter(b, 1, x);
alpha = 0.25;
y_iir = filter(alpha, [1, alpha - 1], x);
% a(1) need not be 1: both sides divide through by it.
y_scaled = filter([2 2], [4 1], x);
