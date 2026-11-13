% An array condition holds only while every element is nonzero.
v = [3 2 1];
steps = 0;
while v
  v = v - 1;
  steps = steps + 1;
end
e = [];
empty_runs = 0;
while e             % an empty condition is false
  empty_runs = empty_runs + 1;
end
