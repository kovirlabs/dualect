% Fit a quadratic to noisy measurements, three ways: polyfit, the normal
% equations, and a QR-free backslash solve; compare and summarise.
t = linspace(0, 2, 21);
noise = 0.05 * sin(37 * t) .* cos(11 * t);   % deterministic "noise"
y = 1.5 - 0.8 * t + 0.6 * t.^2 + noise;

% 1) polyfit / polyval
p = polyfit(t, y, 2);
y_fit = polyval(p, t);

% 2) the same model by least squares on a design matrix
A = [ones(numel(t), 1), t', (t.^2)'];
c = A \ y';
c_normal = (A' * A) \ (A' * y');

% Residuals and goodness of fit
r = y - y_fit;
ss_res = sum(r.^2);
ss_tot = sum((y - mean(y)).^2);
r_squared = 1 - ss_res / ss_tot;
rms_err = sqrt(mean(r.^2));
coef_diff = max(abs(fliplr(p) - c'));
cond_A = cond(A);

% Prediction at a few new points, and where the fitted curve is lowest
t_new = [2.5 3.0];
y_new = polyval(p, t_new);
[y_low, k_low] = min(y_fit);
t_low = t(k_low);

fprintf('fit: y = %.4f + %.4f t + %.4f t^2\n', p(3), p(2), p(1));
fprintf('R^2 = %.6f, rms = %.3e\n', r_squared, rms_err);
