# An anonymous function keeps the value k had when it was created.
k = 2
f = lambda t, k=k: k * t  # bind k now; a plain lambda would see k = 10
k = 10
y = f(3)  # 6, not 30
