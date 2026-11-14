% Damped harmonic oscillator x'' + 2 zeta w0 x' + w0^2 x = 0, integrated with
% the classic fourth-order Runge-Kutta method and compared with the exact
% solution.
w0 = 2*pi;          % natural frequency [rad/s]
zeta = 0.1;         % damping ratio
f = @(t, y) [y(2); -2*zeta*w0*y(2) - w0^2*y(1)];

t_end = 3;
h = 0.01;
t = 0:h:t_end;
n = numel(t);
y = zeros(2, n);
y(:, 1) = [1; 0];   % x(0) = 1, x'(0) = 0

for k = 1:n-1
  k1 = f(t(k), y(:, k));
  k2 = f(t(k) + h/2, y(:, k) + h/2*k1);
  k3 = f(t(k) + h/2, y(:, k) + h/2*k2);
  k4 = f(t(k) + h, y(:, k) + h*k3);
  y(:, k+1) = y(:, k) + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

wd = w0*sqrt(1 - zeta^2);
x_exact = exp(-zeta*w0*t) .* (cos(wd*t) + zeta*w0/wd*sin(wd*t));
err = max(abs(y(1, :) - x_exact));
energy = 0.5*y(2, :).^2 + 0.5*w0^2*y(1, :).^2;
decay = energy(end) / energy(1);
