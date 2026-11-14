% Moving-average (FIR) and first-order low-pass (IIR) filtering of a noisy
% sine, written out as loops and checked against the built-in filter().
fs = 200;                           % sample rate [Hz]
t = (0:1/fs:1-1/fs)';               % sample times, as a column
clean = sin(2*pi*5*t);
noise = 0.3*sin(2*pi*60*t) + 0.1*cos(2*pi*90*t);   % deterministic "noise"
x = clean + noise;

% Moving average over N samples.
N = 9;
b = ones(N, 1) / N;
y_fir = zeros(size(x));
for n = 1:numel(x)
  acc = 0;
  for k = 1:N
    if n - k + 1 >= 1
      acc = acc + b(k) * x(n - k + 1);
    end
  end
  y_fir(n) = acc;
end

% First-order low-pass: y(n) = alpha x(n) + (1 - alpha) y(n-1).
fc = 10;
alpha = 1 - exp(-2*pi*fc/fs);
y_iir = zeros(size(x));
y_iir(1) = alpha * x(1);
for n = 2:numel(x)
  y_iir(n) = alpha * x(n) + (1 - alpha) * y_iir(n-1);
end

% The same filters with the built-in.
y_fir_ref = filter(b, 1, x);
y_iir_ref = filter(alpha, [1, alpha - 1], x);
fir_err = max(abs(y_fir - y_fir_ref));
iir_err = max(abs(y_iir - y_iir_ref));

% Residual noise power once the moving average has filled up.
settled = t >= (N - 1) / fs;
p_fir = mean((y_fir(settled) - clean(settled)).^2);
p_iir = mean((y_iir(settled) - clean(settled)).^2);
