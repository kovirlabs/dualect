# The elapsed time differs on every run, so it is not compared.
import time

import numpy as np

t0 = time.perf_counter()
total = np.arange(1, 1001).sum()
elapsed = time.perf_counter() - t0
# dualect: ignore elapsed
