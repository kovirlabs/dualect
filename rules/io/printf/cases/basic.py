# Each conversion becomes an f-string field; '\n' ends print's line.
x = 3.14159
n = 12
name = "beam"
print(f"x = {x:.2f}, n = {int(n)}")
print(f"[{x:8.3f}|{x:<8.3f}|{x:+.2e}|{-x:08.3f}]")
print(f"{name:>5}|{name:<6}|ok")
print(f"100% done\t({x:g})")
print(f"no newline: {int(n)}", end="")
print()
label = f"{name} has {int(n)} parts"
print(label)
# dualect: stdout
