# Newton's method for sqrt(2), then a do-until counter.
a = 2
x = 1.0
iterations = 0
while abs(x**2 - a) > 1e-12:
    x = x - (x**2 - a) / (2 * x)
    iterations += 1
n = 0
while True:  # do ... until: the body runs at least once
    n += 1
    if n >= 3:
        break
