% Deflection of a simply supported steel beam under a uniform load and a
% point load, by superposition of the Euler-Bernoulli closed-form solutions.
L = 4.0;            % span [m]
E = 200e9;          % Young's modulus [Pa]
I = 8.5e-6;         % second moment of area [m^4]
w = 5e3;            % uniform load [N/m]
P = 12e3;           % point load [N]
a = 1.5;            % position of the point load from the left support [m]
b = L - a;

n = 81;
x = linspace(0, L, n);

% Uniform load: v(x) = w x (L^3 - 2 L x^2 + x^3) / (24 E I)
v_uniform = w * x .* (L^3 - 2*L*x.^2 + x.^3) / (24*E*I);

% Point load at x = a: the solution is piecewise.
v_point = zeros(1, n);
for k = 1:n
  if x(k) <= a
    v_point(k) = P*b*x(k)*(L^2 - b^2 - x(k)^2) / (6*E*I*L);
  else
    v_point(k) = P*a*(L - x(k))*(2*L*x(k) - x(k)^2 - a^2) / (6*E*I*L);
  end
end

v = v_uniform + v_point;
[v_max, k_max] = max(v);
x_max = x(k_max);

% Support reactions and the bending moment along the beam.
R_left = w*L/2 + P*b/L;
R_right = w*L + P - R_left;
M = R_left*x - w*x.^2/2;
right_of_load = x > a;
M(right_of_load) = M(right_of_load) - P*(x(right_of_load) - a);
M_max = max(M);

fprintf('max deflection %.2f mm at x = %.2f m\n', v_max*1e3, x_max);
