x = [5 0 3];
n = 0;
guarded = n > 0 && x(n) > 1;    % x(0) is never evaluated
n = 2;
second_big = n > 0 && x(n) > 1;
flag = isempty(x) || x(1) == 5;
a = [1 -1 2 -2];
b = [-1 -1 2 2];
both = a > 0 & b > 0;
either = a > 0 | b > 0;
num_and = [1 2] & [2 2];        % logical, not bitwise
one = xor(true, false);
