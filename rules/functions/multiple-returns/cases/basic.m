1;
function [s, p] = sum_prod(x)
  s = sum(x);
  p = prod(x);
end
function [lo, hi] = bounds_of(x)
  lo = min(x);
  if nargout > 1
    hi = max(x);
  end
end
v = [2 3 4];
[s, p] = sum_prod(v);
s_only = sum_prod(v);
lo_only = bounds_of(v);
[lo, hi] = bounds_of(v);
