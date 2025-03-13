""" Dynamical TT-approximation using interpolatory projection"""
from __future__ import print_function, absolute_import, division
from six.moves import xrange
import tt
import scipy
import numpy as np

def ksl_deim(A, Nf, y0, tau):
    """ Dynamical tensor-train approximation based on interpolatory projector splitting (1st order). 
        This function performs one step of dynamical tensor-train approximation
        for the equation

        .. math ::
            \\frac{dy}{dt} = A y + Nf(y), \\quad y(0) = y_0

        and outputs approximation for :math:`y(\\tau)`

    :References:

        1. Alec Dektor.
        Collocation methods for nonlinear differential equations on low-rank manifolds.
        Linear Algebra and its Applications, 2025, pp. 143-184.

        https://doi.org/10.1016/j.laa.2024.11.001

        2. Alec Dektor and Lukas Einkemmer. 
        Interpolatory dynamical low-rank approximation for the 3+3d Boltzmann-BGK equation.
        arXiv preprint 2411.15990, 2024.

        https://arxiv.org/abs/2411.15990

    :param A: Matrix in the TT-format
    :type A: matrix
    :param y0: Initial condition in the TT-format,
    :type y0: tensor
    :param tau: Timestep
    :type tau: float
    :param use_normest: Use matrix norm estimation instead of the true 1-norm in KSL procedure. 0 -use true norm, 1 - Higham norm estimator, 2 - fixed norm=1.0 (for testing purposes only)
    :type use_normest: int, default: 1
    :rtype: tensor

    :Example:
        >>> import tt
        >>> import tt.ksl_deim
        >>> import numpy as np
        >>> d = 8
        >>> a = tt.qlaplace_dd([d, d, d])
        >>> y0, ev = tt.eigb.eigb(a, tt.rand(2 , 24, 2), 1e-6, verb=0)
        Solving a block eigenvalue problem
        Looking for 1 eigenvalues with accuracy 1E-06
        swp: 1 er = 1.1408 rmax:2
        swp: 2 er = 190.01 rmax:2
        swp: 3 er = 2.72582E-08 rmax:2
        Total number of matvecs: 0
        >>> y1 = tt.ksl.ksl(a, y0, 1e-2)
        Solving a real-valued dynamical problem with tau=1E-02
        >>> print tt.dot(y1, y0) / (y1.norm() * y0.norm()) - 1 #Eigenvectors should not change
        0.0
    """
    r, n, d = y0.r, y0.n, y0.d
    y = tt.vector.to_list(y0)
    y, J = nested_J(y)
    I = d*[None]

    # ----------------- left-to-right sweep ----------------- #
    # first core forward step
    Ax_e = eval_rhs(A, Nf, y, I, J, -1, 1)
    R = right_eval(y, J, 1)
    y[0] = y[0] + tau*np.tensordot(Ax_e, np.linalg.pinv(R), axes=((2),(0)))
    
    # backward step
    cr = np.reshape(y[0], (r[0]*n[0],r[1]))
    cr, S = np.linalg.qr(cr)
    y[0] = np.reshape(cr, (r[0],n[0],r[1]))         # left-orth core 0
    cr2 = y[1]                                      # save for later
    y[1] = np.tensordot(S, y[1], axes=((1),(0)))    # temp. update to core 1
    I[0] = nested_Ik(y, n, r, I, 0)
    ax_e = eval_rhs(A, Nf, y, I, J, 0, 1)
    L = left_eval(y, I, 0)
    S = S - tau*(np.linalg.pinv(L)@ax_e@np.linalg.pinv(R))
    y[1] = np.tensordot(S, cr2, axes=((1),(0)))

    # interior cores
    for k in range(1,d-1,1):
        # forward step
        ax_e = eval_rhs(A, Nf, y, I, J, k-1, k+1)
        L = left_eval(y,I,k-1)
        R = right_eval(y,J,k+1)
        y[k] = y[k] + tau*np.einsum('ia,ajb,bk->ijk', np.linalg.pinv(L), ax_e, np.linalg.pinv(R))

        # backward step
        cr = np.reshape(y[k], (r[k]*n[k],r[k+1]))
        cr, S = np.linalg.qr(cr)
        y[k] = np.reshape(cr, (r[k],n[k],r[k+1]))
        cr2 = y[k+1]    # save for later
        y[k+1] = np.tensordot(S, y[k+1], axes=((1),(0)))
        I[k] = nested_Ik(y, n, r, I, k) 
        ax_e = eval_rhs(A, Nf, y, I, J, k, k+1)
        L = left_eval(y,I,k)
        S = S - tau*(np.linalg.pinv(L)@ax_e@np.linalg.pinv(R))
        y[k+1] = np.tensordot(S, cr2, axes=((1),(0)))

    # final core forward step
    ax_e = eval_rhs(A, Nf, y, I, J, d-2, d)
    L = left_eval(y, I, d-2)
    y[d-1] = y[d-1] + tau*np.tensordot(np.linalg.pinv(L), ax_e, axes=((1),(0)))

    return tt.vector.from_list(y)

def qdeim(M, r):
    row_inds = scipy.linalg.qr(M.transpose(), pivoting=True, mode='economic')[2]
    return row_inds[0:r]

def left_eval(x, I, k):
    ''' Evaluate the left k dimensions of TT x using multi-indices I
    :param x: vector in the TT-format
    :type x: TT vector
    :param I: Multi-indices
    :type I: list
    :param k: index for left evaluation
    :type k: integer 
    :rtype: matrix '''
    
    L = x[0][:,I[k][:,0],:]
    L = np.squeeze(L)
    
    for i in range(1,k+1,1):
        L = np.tensordot(L, x[i][:,I[k][:,i],:], axes=((1),(0)))
        L = np.diagonal(L).T
    return L

def nested_Ik(x, n, r, I, k):
    ''' Generate a single left nested index set I_k from I_{k-1} for interpolatory projection onto x
    :param x: vector in the TT-format
    :type x: TT vector
    :param n: mode sizes
    :type n: vector
    :param r: TT-ranks
    :type r: vector
    :param I: Multi-indices
    :type I: list
    :param k: index for left evaluation
    :type k: integer 
    :rtype: matrix '''

    if k == 0:
        cr = np.reshape(x[0], (r[0]*n[0],r[1]))
        Ik = qdeim(cr, r[1])
        Ik = Ik[:,np.newaxis]
    else:
        L = left_eval(x,I,k-1)    # evaluate left part of tt at preceding multi-indices
        T = np.tensordot(L, x[k], axes=((1),(0)))
        T = np.reshape(T, (r[k]*n[k],r[k+1]))

        p = qdeim(T, r[k+1])                            
        [p1, p2] = np.unravel_index(p, (r[k],n[k]))    # split indices
        p2 = p2[:,np.newaxis]
        Ik = np.hstack((I[k-1][p1,:], p2))   # nested multi-indices
    return Ik

def nested_I(x):
    ''' Generate left nested indices I for interpolatory projection onto x
    :param x: vector in the TT-format
    :type x: TT vector
    :rtype: TT vector, list '''
    d = len(x)
    n = np.zeros(d)
    r = np.ones(d + 1, dtype=np.int32)
    for i in xrange(d):
        [_, n[i], r[i + 1]] = x[i].shape

    I = d*[None]
    for k in range(d-1):
        # left-orth core k
        cr = np.reshape(x[k], (r[k]*n[k],r[k+1]))
        cr, R = np.linalg.qr(cr)
        x[k] = np.reshape(cr, (r[k],n[k],r[k+1]))
        x[k+1] = np.tensordot(R, x[k+1], axes=((1),(0)))

        I[k] = nested_Ik(x,n,r,I,k) # nested multi-index set
    return x, I

def right_eval(x, J, k):
    ''' Evaluate the right k dimensions of TT x using multi-indices I
    :param x: vector in the TT-format
    :type x: TT-vector
    :param J: Multi-indices
    :type J: list
    :param k: index for left evaluation
    :type k: integer 
    :rtype: matrix '''

    d = len(x)
    R = x[k][:,J[k][:,0],:]

    p = 1
    for i in range(k+1,d):
        R = np.tensordot(R, x[i][:,J[k][:,p],:], axes=((2),(0)))
        R = np.diagonal(R, axis1=1, axis2=2)
        R = np.transpose(R, (0,2,1))
        p += 1
    return np.squeeze(R)

def nested_Jk(x, n, r, J, k):
    ''' Generate a single nested right index set J_k from J_{k+1} for interpolatory projection onto x
    :param x: vector in the TT-format
    :type x: TT-vector
    :param n: mode sizes
    :type n: vector
    :param r: TT-ranks
    :type r: vector
    :param J: Right multi-indices
    :type J: list
    :param k: index for left evaluation
    :type k: integer 
    :rtype: matrix '''

    d = len(x)
    if k == d-1:
        cr = np.reshape(x[d-1], (r[d-1],n[d-1]*r[d]))
        Jk = qdeim(cr.T,r[d-1])
        Jk = Jk[:,np.newaxis]
    else:
        R = right_eval(x,J,k+1)    # evaluate right part of tt at preceding multi-indices
        T = np.tensordot(x[k], R, axes=((2),(0)))
        T = np.reshape(T, (r[k],n[k]*r[k+1]))
        #[T, _] = np.linalg.qr(T)
        p = qdeim(T.T, r[k])                            # qDEIM
        [p1, p2] = np.unravel_index(p, (n[k],r[k+1]))   # split indices
        p1 = p1[:,np.newaxis]
        Jk = np.hstack((p1,J[k+1][p2,:]))   # nested multi-index set
    return Jk

def nested_J(x):
    ''' Generate right nested indices J for interpolatory projection onto x
    :param x: vector in the TT-format
    :type x: TT vector
    :rtype: TT vector, list '''

    d = len(x)
    n = np.zeros(d, dtype=np.int32)
    r = np.ones(d + 1, dtype=np.int32)
    for i in xrange(d):
        [_, n[i], r[i + 1]] = x[i].shape

    J = d*[None]
    for k in range(d-1,0,-1):
        # right-orth core k
        cr = np.reshape(x[k], (r[k],n[k]*r[k+1]))
        cr, R = np.linalg.qr(cr.T)
        x[k] = np.reshape(cr.T, (r[k],n[k],r[k+1]))
        x[k-1] = np.tensordot(x[k-1], R.T, axes=((2),(0)))
        
        cr = np.reshape(x[k], (r[k],n[k]*r[k+1]))
        J[k] = nested_Jk(x,n,r,J,k) # nested multi-index set
    return x, J

def eval_rhs(A, Nf, y, I, J, k1, k2):
    ''' Evaluate Ay + Nf(y) at multi-indices defined by I,J,k1,k2
    :param A: Matrix in the TT-format
    :type A: matrix
    :param Nf: Nonlinear function 
    :type Nf: function handle
    :param y: tensor in the TT-format
    :type y: tensor
    :param I: left nested indices
    :type I: list
    :param J: right nested indices
    :type J: list
    :param k1: left-nested index set used for evaluation
    :type k1: integer
    :param k1: right-nested index set used for evaluation
    :type k1: integer
    '''

    d = len(y)
    Ay = tt.matvec(A, tt.vector.from_list(y))
    Ay = tt.vector.to_list(Ay)

    if k2-k1 == 2: # 3d tensor forward step
        k = k1+1
        if k == 0:
            RAy = right_eval(Ay, J, k+1)
            Ay_e = np.tensordot(Ay[k], RAy, axes=((2),(0)))

            Ry = right_eval(y, J, k+1)
            y_e = np.tensordot(y[k], Ry, axes=((2),(0)))
        elif k == d-1:
            LAy = left_eval(Ay, I, k-1)
            Ay_e = np.tensordot(LAy, Ay[k], axes=((1),(0)))

            Ly = left_eval(y, I, k-1)
            y_e = np.tensordot(Ly, y[k], axes=((1),(0)))
        else:
            LAy = left_eval(Ay, I, k-1)
            RAy = right_eval(Ay, J, k+1)
            Ay_e = np.einsum('ia,ajb,bk->ijk', LAy, Ay[k], RAy)

            Ly = left_eval(y, I, k-1)
            Ry = right_eval(y, J, k+1)
            y_e = np.einsum('ia,ajb,bk->ijk', Ly, y[k], Ry)
            
    elif k2-k1 == 1: # 2d tensor backward step
        LAy = left_eval(Ay, I, k1)
        RAy = right_eval(Ay, J, k2)
        Ay_e = LAy@RAy

        Ly = left_eval(y, I, k1)
        Ry = right_eval(y, J, k2)
        y_e = Ly@Ry
    else: 
        raise Exception('Invalid evaluation indices k_1,k_2')

    return Ay_e + Nf(y_e)