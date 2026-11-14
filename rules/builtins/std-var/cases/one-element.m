% One value: N - 1 would divide by zero, but the source returns 0.
single = std(7);
single_var = var(7, 1);
pair = std([3 5]);      % two elements: N - 1 = 1
