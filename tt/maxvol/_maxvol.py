from maxvol import dmaxvol
from numpy import arange, asanyarray

def maxvol(a, nswp = 20, tol = 5e-2):
    a = asanyarray(a)
    if a.shape[0] <= a.shape[1]:
        return arange(a.shape[0])
    return dmaxvol(a, nswp, tol) - 1
