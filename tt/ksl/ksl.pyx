""" This module implements the KSL splitting scheme 
    for the dynamical low-rank approximation; it is basically 
    a sweep + integration of the local stuff """

import numpy as np
from numpy.linalg import qr
cimport numpy as cnp
from tt import tensor
""" The method for the solution of non-stationary problems in the TT-format """
def ksl(A,y0, f = None, double tau = 1e-3):
    for i in range(d-1,-1,-1):
        #p1 = psy[i] - 1
        #p2 = psy[i+1]
        cr = cry[psy[i]-1:psy[i+1]]
        #cr = cr.reshape((ry[i] * n[i], ry[i+1]),order='F')
        #cr = cr.T
        #q,r = qr(cr)
        #print r
    y = y0
    return y
