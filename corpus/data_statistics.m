% Summary statistics of a data matrix: one column per sensor, one row per
% sample (deterministic data, so both sides see the same numbers).
n = 12;
k = (1:n)';
D = [20 + 2*sin(k), 15 + 0.5*k, 30 - cos(3*k).^2, mod(7*k, 5)];

col_mean = mean(D);
col_std = std(D);              % N-1 normalisation
col_var = var(D, 1);           % N normalisation
col_median = median(D);
[col_max, row_of_max] = max(D);
[col_min, row_of_min] = min(D);
spread = col_max - col_min;

% Standardise each column (implicit expansion), then find outliers.
Z = (D - mean(D)) ./ std(D);
outlier = abs(Z) > 1.5;
n_outliers = sum(outlier(:));
any_outlier = any(outlier);
all_positive = all(D > 0);

% Running totals and ranks of the second sensor.
running = cumsum(D(:, 2));
[sorted2, order2] = sort(D(:, 2), 'descend');
top3 = order2(1:3);

% Rounded values and how many distinct levels the last sensor takes.
levels = unique(D(:, 4));
n_levels = numel(levels);
D_round = round(D * 10) / 10;

for j = 1:size(D, 2)
  fprintf('sensor %d: mean %8.3f  std %7.3f  median %8.3f\n', ...
          j, col_mean(j), col_std(j), col_median(j));
end
