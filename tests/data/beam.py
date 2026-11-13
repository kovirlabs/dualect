# Cantilever beam with an end load: tip deflection and the deflection curve.
# Hand-written twin of beam.m used to check the verify harness itself.
import numpy as np

E = 200e9  # Young's modulus [Pa]
Iz = 8.33e-6  # second moment of area [m^4]
L = 2.0  # length [m]
P = 1000  # end load [N]
x = np.linspace(0, L, 5)
w = P * x**2 * (3 * L - x) / (6 * E * Iz)
tip = w[-1]
imax = int(np.argmax(w))
wmax = w[imax]
