n = 2;
switch n
  case 1
    label = 'one';
  case {2, 3}
    label = 'few';
  otherwise
    label = 'many';
end
