from maxvol import dmaxvol, zmaxvol
from numpy import arange, asanyarray, iscomplexobj


def maxvol(a, nswp=20, tol=5e-2):
    a = asanyarray(a)
    if a.shape[0] <= a.shape[1]:
        return arange(a.shape[0])
    if iscomplexobj(a):
        return zmaxvol(a, nswp, tol) - 1
    else:
        return dmaxvol(a, nswp, tol) - 1
