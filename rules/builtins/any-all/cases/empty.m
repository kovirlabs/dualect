% On empty data any is false and all is true.
e = [];
a1 = any(e);
a2 = all(e);
x = [3 4 5];
small = x(x < 0);         % no element: empty
all_small_neg = all(small < 0);
any_small = any(small);
