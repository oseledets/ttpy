"""Example of using tt.optimize module."""

import tt
from tt.optimize import tt_min
from scipy.optimize import rosen

def my_rosen(x):
    return rosen(x.T)

val, x_full = tt_min.min_func(my_rosen, -2, 2, d=4, n0=512, rmax=10, nswp=30)
