def order_of(method):
    match method:
        case "linear":
            k = 1
        case "cubic" | "spline":
            k = 3
        case _:
            k = 0
    return k


k_lin = order_of("linear")
k_spl = order_of("spline")
k_other = order_of("nearest")
