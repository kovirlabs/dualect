A = [1 2 3; 4 5 6];
total = zeros(2, 1);
count = 0;
for c = A
  total = total + c;
  count = count + 1;
end
s = 0;
for k = 1:4
  s = s + k;
end
