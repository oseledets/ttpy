#This module is about AMR-type algorithms for the TT
import numpy as np
import tt_eigb
from tt import tensor
def eigb(A, y0, eps, rmax = 150, kickrank = 5, nswp = 20, max_full_size = 1000, verb = 1):
    """ Approximate matrix-by-vector multiplication
            Y = EIGB(A,X,Y0,EPS) Find several eigenvalues of the TT-matrix
    """
    ry = y0.r.copy()
    #lam = np.zeros(ry[y0.d])
    lam = tt_eigb.tt_block_eig.tt_eigb(y0.d, A.n, A.m, A.tt.r, A.tt.core, y0.core, ry, eps, \
                                       rmax, ry[y0.d], kickrank, nswp, max_full_size, verb)
    y = tensor()
    y.d = y0.d
    y.n = A.n.copy()
    y.r = ry 
    y.core = tt_eigb.tt_block_eig.result_core.copy()
    tt_eigb.tt_block_eig.deallocate_result()
    y.get_ps()
    return y,lam
