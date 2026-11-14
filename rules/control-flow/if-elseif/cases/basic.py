def sign_of(x):
    if x > 0:
        s = 1
    elif x < 0:
        s = -1
    else:
        s = 0
    return s


s_pos = sign_of(3)
s_neg = sign_of(-2)
s_zero = sign_of(0)
