n = 2
match n:
    case 1:
        label = "one"
    case 2 | 3:
        label = "few"
    case _:
        label = "many"
