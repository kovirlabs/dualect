1;
function [s, p] = sum_prod(x)
  s = sum(x);
  p = prod(x);
end
function [lo, mid, hi] = spread(x)
  lo = min(x);
  mid = mean(x);
  hi = max(x);
end
v = [2 3 4];
[s, p] = sum_prod(v);
s_only = sum_prod(v);       % the first output only
[~, p_only] = sum_prod(v);  % the first output discarded
[a, b] = spread(v);         % fewer outputs than the function has
