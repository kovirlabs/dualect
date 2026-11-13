1;
function y = area(r)
  y = pi * r.^2;
end
function x = zero_first(x)
  x(1) = 0;         % changes a copy: the caller's array is untouched
end
a1 = area(2);
radii = [1 2 3];
areas = area(radii);
v = [5 6 7];
w = zero_first(v);
sq = @(t) t.^2;
nine = sq(3);
