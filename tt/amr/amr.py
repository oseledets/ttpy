#This module is about AMR-type algorithms for the TT
import numpy as np
import amr_f90
from tt import tensor
#from tt_tensor2 import tt_tensor
def mvk4(A,x,y0,eps,rmax=150,kickrank=5,nswp=20,verb=1):
    """ Approximate matrix-by-vector multiplication
            Y = MVK4(A,X,Y0,EPS) Multiply a TT-matrix A with a TT-vector x with accuracy eps
            using the AMR/DMRG algorithm
    """
    ry = y0.r.copy()
    amr_f90.tt_adapt_als.tt_mvk4(x.d,A.n,A.m,x.r,A.tt.r,A.tt.core, x.core, y0.core, ry, eps, rmax, kickrank, nswp, verb)
    y = tensor()
    y.d = x.d
    y.n = A.n.copy()
    y.r = ry 
    y.core = amr_f90.tt_adapt_als.result_core.copy()
    amr_f90.tt_adapt_als.deallocate_result()
    y.get_ps()
    return y
