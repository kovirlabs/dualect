% A product of integer-valued data is computed in doubles: no wrap-around.
big = [100000 100000 100000 100000];
p = cumprod(big);         % 1e5 1e10 1e15 1e20
m = (1:3) > 1;
s = cumsum(m);            % a running count of true values: 0 1 2
