% Whole numbers print every digit; otherwise give a precision or a format.
n = 42;
label = ['n = ', num2str(n)];
neg = num2str(-7);
big = num2str(123456789012);
flag = num2str(3 > 2);               % '1', not 'True'
ratio = 1 / 3;
r = num2str(ratio, 4);               % 0.3333
p8 = num2str(pi, 8);                 % 3.1415927
t = num2str(2.5, '%8.3f');           % trimmed: '2.500'
nz = num2str(-0);                    % '-0': the sign of zero is printed
