mask = [1 5 3] > 2;
counts = double(mask);        % 0 1 1, as numbers
n = double(true) + 1;
flags = logical([2 0 -1]);    % nonzero is true
x = [1 2 3];
y = double(x) / 2;            % already a double: no change
