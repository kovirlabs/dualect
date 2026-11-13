1;
function y = scale(x, factor)
  if nargin < 2
    factor = 10;
  end
  y = x * factor;
end
function y = clip_to(x, lo, hi)
  if nargin < 2 || isempty(lo)
    lo = 0;
  end
  if nargin < 3
    hi = max(x);    % the default depends on another argument
  end
  y = min(max(x, lo), hi);
end
a = scale(2);
b = scale(2, 3);
x = [-1 5 12];
c1 = clip_to(x);
c2 = clip_to(x, [], 4);
c3 = clip_to(x, 1, 6);
