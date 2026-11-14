# Command syntax: every word after the command name is passed as a string.
import numpy as np

a = 5
b = 7
c = 9
np.set_printoptions(precision=16)  # format long: display only
del a, b  # clear a b
words = "a b"  # what clear received: text, not values
n_words = len(words.split(" "))
np.set_printoptions(precision=8)  # format short: back to NumPy's default
