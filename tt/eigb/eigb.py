import numpy as np
import tt_eigb
from tt import tensor


def eigb(A, y0, eps, rmax=150, nswp=20, max_full_size=1000, verb=1):
    """ Approximate computation of minimal eigenvalues in tensor train format
    This function uses alternating least-squares algorithm for the computation of several
    minimal eigenvalues. If you want maximal eigenvalues, just send -A to the function.

    :Reference:


        S. V. Dolgov, B. N. Khoromskij, I. V. Oseledets, and D. V. Savostyanov.
        Computation of extreme eigenvalues in higher dimensions using block tensor train format. Computer Phys. Comm.,
        185(4):1207-1216, 2014. http://dx.doi.org/10.1016/j.cpc.2013.12.017


    :param A: Matrix in the TT-format
    :type A: matrix
    :param y0: Initial guess in the block TT-format, r(d+1) is the number of eigenvalues sought
    :type y0: tensor
    :param eps: Accuracy required
    :type eps: float
    :param rmax: Maximal rank
    :type rmax: int
    :param kickrank: Addition rank, the larger the more robus the method,
    :type kickrank: int
    :rtype: A tuple (ev, tensor), where ev is a list of eigenvalues, tensor is an approximation to eigenvectors.

    :Example:


        >>> import tt
        >>> import tt.eigb
        >>> d = 8; f = 3
        >>> r = [8] * (d * f + 1); r[d * f] = 8; r[0] = 1
        >>> x = tt.rand(n, d * f, r)
        >>> a = tt.qlaplace_dd([8, 8, 8])
        >>> sol, ev = tt.eigb.eigb(a, x, 1e-6, verb=0)
        Solving a block eigenvalue problem
        Looking for 8 eigenvalues with accuracy 1E-06
        swp: 1 er = 35.93 rmax:19
        swp: 2 er = 4.51015E-04 rmax:18
        swp: 3 er = 1.87584E-12 rmax:17
        Total number of matvecs: 0
        >>> print ev
        [ 0.00044828  0.00089654  0.00089654  0.00089654  0.0013448   0.0013448
                  0.0013448   0.00164356]



    """
    ry = y0.r.copy()
    lam = tt_eigb.tt_block_eig.tt_eigb(y0.d, A.n, A.m, A.tt.r, A.tt.core, y0.core, ry, eps,
                                       rmax, ry[y0.d], 0, nswp, max_full_size, verb)
    y = tensor()
    y.d = y0.d
    y.n = A.n.copy()
    y.r = ry
    y.core = tt_eigb.tt_block_eig.result_core.copy()
    tt_eigb.tt_block_eig.deallocate_result()
    y.get_ps()
    return y, lam
