import numpy as np

# The format repeats while values remain; an array recycles it per element.
v = np.array([1, 2, 3])
for value in v:
    print(f"{int(value)}")
A = np.array([[1, 2, 3], [4, 5, 6]])
for column in A.T:
    print(f"{int(column[0])}-{int(column[1])}")  # one pass per column: 1-4, 2-5, 3-6
print("7 8\n9 ", end="")  # prints "7 8", then "9 " and stops
print()
csv = "".join([f"{value:.1f}," for value in np.array([0.5, 1.5])])
# dualect: stdout
