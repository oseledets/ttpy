import sys
sys.path.append('../')
from tt import *
import numpy as np

x = xfun(2, 3)
e = ones(2, 2)
X = matrix(x, [2]*3, [1]*3) # [0, 1, 2, 3, 4, 5, 6, 7]^T
E = matrix(e, [1]*2, [2]*2) # [1, 1, 1, 1]
print (X * E).full()
assert np.all((X * E) * np.arange(4) == np.arange(8) * 6.)

A = matrix(xfun(2, 3), [1] * 3, [2] * 3)
u = np.arange(8)
assert A * u == 140.
