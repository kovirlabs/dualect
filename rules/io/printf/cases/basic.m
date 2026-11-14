% Each conversion becomes an f-string field; '\n' ends print's line.
x = 3.14159;
n = 12;
name = 'beam';
fprintf('x = %.2f, n = %d\n', x, n);
printf('[%8.3f|%-8.3f|%+.2e|%08.3f]\n', x, x, x, -x);
fprintf('%5s|%-6s|%s\n', name, name, 'ok');
fprintf(1, '100%% done\t(%g)\n', x);
fprintf('no newline: %d', n);
fprintf('\n');
label = sprintf('%s has %d parts', name, n);
disp(label)
% dualect: stdout
