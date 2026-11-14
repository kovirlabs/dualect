x = [1 2 3];
x(end+1) = 4;
y = [];
for k = 1:4
  y(end+1) = k^2;
end
z = [1 2];
z(5) = 9;           % the gap is filled with zeros
