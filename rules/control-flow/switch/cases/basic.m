1;
function k = order_of(method)
  switch method
    case 'linear'
      k = 1;
    case {'cubic', 'spline'}
      k = 3;
    otherwise
      k = 0;
  end
end
k_lin = order_of('linear');
k_spl = order_of('spline');
k_other = order_of('nearest');
