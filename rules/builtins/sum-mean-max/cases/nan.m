w = [1 NaN 3];
m = max(w);           % NaN is skipped: 3
[m2, k] = max(w);
s = sum(w);           % NaN propagates
s_valid = sum(w(~isnan(w)));
% dualect: index k
