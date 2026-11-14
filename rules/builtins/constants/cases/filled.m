% With dimensions, the constants fill arrays; one dimension means a square.
n = 3;
mask = true(n);           % 3x3, not 1x3
none = false(2, 4);
gaps = NaN(1, n);         % a row
lo = -Inf(2, 1);          % a column
cnt = sum(mask(:));
