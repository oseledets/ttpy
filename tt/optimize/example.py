"""Example of using tt.optimize module."""

import tt
from tt.optimize import tt_min
from scipy.optimize import rosen


def my_rosen(x):
    return rosen(x.T)

print("Minimize 4-d Rosenbrock function on a 4-dimensional grid (512 points " +
      "along each dimension). The global minimum is 0 in the (1, 1, 1, 1) point.")
val, x_full = tt_min.min_func(my_rosen, -2, 2, d=4, n0=512, rmax=10, nswp=30)

tens = tt.rand([3, 4, 5, 4, 3], 5, 3)
min_element = min(tens.full().flatten())
print("Minimize random 5-dimensional TT tensor with ranks equal to 3. " +
      "The minimal element is %f" % min_element)
val, point = tt_min.min_tens(tens, rmax=10, nswp=30)
