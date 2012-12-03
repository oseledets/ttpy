""" This module implements the KSL splitting scheme 
    for the dynamical low-rank approximation; it is basically 
    a sweep + integration of the local stuff """

""" I am still not very familiar with Cython. I get an 
    input as a tt object with the common fields; I can try 
    to reorthogonalize the vector from right-to-left using 
    convenient numpy reshape stuff;
    that would be slow due to memory (reallocations) and all
    these, so I will have to stick to pointer arithmetics (bue)
    Maybe I will first just implement the stuff in numpy then
    go to memory views """
import numpy as np
from numpy.linalg import qr
cimport numpy as cnp
from tt import tensor
def ksl(A,y0, double tau):
    cdef int d = y0.d
    cdef int[:] ry = y0.r
    cdef int[:] ra = A.tt.r
    cdef int[:] n = y0.n
    cdef double [:] cry = y0.core
    cdef double [:] cra = A.tt.core
    cdef int[:] psa = A.tt.ps
    cdef int[:] psy = y0.ps
    cdef int p1,p2
    cdef i
    for i in range(d-1,-1,-1):
        p1 = psy[i] - 1
        p2 = psy[i+1]
        cr = cry[<int>p1]
        #cr = cr.reshape((ry[i] * n[i], ry[i+1]),order='F')
        #cr = cr.T
        #q,r = qr(cr)
        #print r
    y = y0
    return y
