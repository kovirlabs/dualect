1;
function [lo, hi] = bounds_of(x)
  lo = min(x);
  if nargout > 1
    hi = max(x);
  end
end
v = [2 3 4];
lo_only = bounds_of(v);
[lo, hi] = bounds_of(v);
