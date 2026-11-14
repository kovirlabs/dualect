% The format repeats while values remain; an array recycles it per element.
v = [1 2 3];
fprintf('%d\n', v);
A = [1 2 3; 4 5 6];
fprintf('%d-%d\n', A);          % one pass per column: 1-4, 2-5, 3-6
fprintf('%d %d\n', 7, 8, 9);    % prints "7 8", then "9 " and stops
fprintf('\n');
csv = sprintf('%.1f,', [0.5 1.5]);
% dualect: stdout
