from __future__ import absolute_import, division, print_function

from copy import copy, deepcopy

import numpy as np
from six import integer_types
from six.moves import xrange

from .matrix import matrix
from .utils import gcd, ind2sub, my_chop2
from .vector import vector

# Some binary operations (put aside to wrap something in future)
# TT-matrix by a TT-vector product

# Available functions:
# matvec, col, kron, dot, diag, mkron, concatenate, sum, ones, rand, eye,
# Toeplitz, qlaplace_dd, xfun, linspace, sin, cos, delta, stepfun, qshift, unit,
# IpaS, reshape

def matvec(a, b, compression=False):
    """Matrix-vector product in TT format."""
    if not isinstance(a, matrix):
        raise ValueError('The first operand expected to be TT-matrix but '
                         f'given {type(a)}.')
    if not isinstance(b, vector):
        raise ValueError('The second operand expected to be TT-vector but '
                         f'given {type(a)}.')

    acrs = vector.to_list(a.tt)
    bcrs = vector.to_list(b)
    ccrs = []
    d = b.d

    def get_core(i):
        acr = np.reshape(
            acrs[i],
            (a.tt.r[i],
             a.n[i],
             a.m[i],
             a.tt.r[
                 i + 1]),
            order='F')
        acr = acr.transpose([3, 0, 1, 2])  # a(R_{i+1}, R_i, n_i, m_i)
        bcr = bcrs[i].transpose([1, 0, 2])  # b(m_i, r_i, r_{i+1})
        # c(R_{i+1}, R_i, n_i, r_i, r_{i+1})
        ccr = np.tensordot(acr, bcr, axes=(3, 0))
        ccr = ccr.transpose([1, 3, 2, 0, 4]).reshape(
            (a.tt.r[i] * b.r[i], a.n[i], a.tt.r[i + 1] * b.r[i + 1]), order='F')
        return ccr

    if compression:  # the compression is laaaaazy and one-directioned
        # calculate norm of resulting _vector first
        nrm = np.array([[1.0]])  # 1 x 1
        v = np.array([[1.0]])
        for i in xrange(d):
            ccr = get_core(i)
            # print(str(ccr.shape) + " -> "),
            # minimal loss compression
            ccr = np.tensordot(v, ccr, (1, 0))
            rl, n, rr = ccr.shape
            if i < d - 1:
                u, s, v = np.linalg.svd(
                    ccr.reshape(
                        (rl * n, rr), order='F'), full_matrices=False)
                newr = min(rl * n, rr)
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = np.dot(np.diag(s[:newr]), v[:newr, :])
            # print(ccr.shape)
            # r x r . r x n x R -> r x n x R
            nrm = np.tensordot(nrm, ccr, (0, 0))
            # r x n x R . r x n x R -> n x R x n x R
            nrm = np.tensordot(nrm, np.conj(ccr), (0, 0))
            nrm = nrm.diagonal(axis1=0, axis2=2)  # n x R x n x R -> R x R x n
            nrm = nrm.sum(axis=2)  # R x R x n -> R x R
        if nrm.size > 1:
            raise Exception('too many numbers in norm')
        # print("Norm calculated:", nrm)
        nrm = np.sqrt(np.linalg.norm(nrm))
        # print("Norm predicted:", nrm)
        compression = compression * nrm / np.sqrt(d - 1)
        v = np.array([[1.0]])

    for i in xrange(d):
        ccr = get_core(i)
        rl, n, rr = ccr.shape
        if compression:
            ccr = np.tensordot(v, ccr, (1, 0))  # c(s_i, n_i, r_i, r_{i+1})
            if i < d - 1:
                rl = v.shape[0]
                u, s, v = np.linalg.svd(
                    ccr.reshape(
                        (rl * n, rr), order='F'), full_matrices=False)
                ss = np.cumsum(s[::-1])[::-1]
                newr = max(min([r for r in range(ss.size) if ss[
                    r] <= compression] + [min(rl * n, rr)]), 1)
                # print("Rank % 4d replaced by % 4d" % (rr, newr))
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = np.dot(np.diag(s[:newr]), v[:newr, :])
        ccrs.append(ccr)
    result = vector.from_list(ccrs)
    if compression:
        # print(result)
        print("Norm actual:", result.norm(), " mean rank:", result.rmean())
        # print("Norm very actual:", matvec(a,b).norm())
    return result


# TT-by-a-full _matrix product (wrapped in Fortran 90, inspired by
# MATLAB prototype)
# def tt_full_mv(a,b):
#    mv = _matrix_f90.matrix.tt_mv_full
#    if b.ndim is 1:
#        rb = 1
#    else:
#        rb = b.shape[1]
#    x1 = b.reshape(b.shape[0],rb)
#    y = np.zeros(a.n.prod(),dtype=np.float64)
#    y = mv(a.n,a.m,a.tt.r,a.tt.ps,a.tt.core,x1,a.n.prod())
#    return y

def col(a, k):
    """Get the column of the block TT-vector"""
    if hasattr(a, '__col__'):
        return a.__col__(k)
    else:
        raise ValueError('col is waiting for a TT-vector or a TT-matrix')


def kron(a, b):
    """Kronecker product of two TT-matrices or two TT-vectors"""
    if hasattr(a, '__kron__'):
        return a.__kron__(b)
    if a is None:
        return b
    else:
        raise ValueError(
            'Kron is waiting for two TT-vectors or two TT-matrices')


def dot(a, b):
    """Dot product of two TT-matrices or two TT-vectors"""
    if hasattr(a, '__dot__'):
        return a.__dot__(b)
    if a is None:
        return b
    else:
        raise ValueError(
            'Dot is waiting for two TT-vectors or two TT-    matrices')


def diag(a):
    """ Diagonal of a TT-matrix OR diagonal _matrix from a TT-vector."""
    if hasattr(a, '__diag__'):
        return a.__diag__()
    else:
        raise ValueError('Can be called only on TT-vector or a TT-matrix')


def mkron(a, *args):
    """Kronecker product of all the arguments"""
    if not isinstance(a, list):
        a = [a]
    a = list(a)  # copy list
    for i in args:
        if isinstance(i, list):
            a.extend(i)
        else:
            a.append(i)

    c = vector()
    c.d = 0
    c.n = np.array([], dtype=np.int32)
    c.r = np.array([], dtype=np.int32)
    c.core = []

    for t in a:
        thetensor = t.tt if isinstance(t, matrix) else t
        c.d += thetensor.d
        c.n = np.concatenate((c.n, thetensor.n))
        c.r = np.concatenate((c.r[:-1], thetensor.r))
        c.core = np.concatenate((c.core, thetensor.core))

    c.get_ps()
    return c


def zkron(ttA, ttB):
    """
    Do kronecker product between cores of two matrices ttA and ttB.
    Look about kronecker at: https://en.wikipedia.org/wiki/Kronecker_product
    For details about operation refer: https://arxiv.org/abs/1802.02839
    :param ttA: first TT-matrix;
    :param ttB: second TT-matrix;
    :return: TT-matrix in z-order
    """
    Al = matrix.to_list(ttA)
    Bl = matrix.to_list(ttB)
    Hl = [np.kron(B, A) for (A, B) in zip(Al, Bl)]
    return matrix.from_list(Hl)


def zkronv(ttA, ttB):
    """
    Do kronecker product between vectors ttA and ttB.
    Look about kronecker at: https://en.wikipedia.org/wiki/Kronecker_product
    For details about operation refer: https://arxiv.org/abs/1802.02839
    :param ttA: first TT-vector;
    :param ttB: second TT-vector;
    :return: operation result in z-order
    """
    Al = vector.to_list(ttA)
    Bl = vector.to_list(ttB)
    Hl = [np.kron(B, A) for (A, B) in zip(Al, Bl)]
    return vector.from_list(Hl)


def zmeshgrid(d):
    """
    Returns a meshgrid like np.meshgrid but in z-order
    :param d: you'll get 4**d nodes in meshgrid
    :return: xx, yy in z-order
    """
    lin = xfun(2, d)
    one = ones(2, d)

    xx = zkronv(lin, one)
    yy = zkronv(one, lin)

    return xx, yy


def zaffine(c0, c1, c2, d):
    """
    Generate linear function c0 + c1 ex + c2 ey in z ordering with d cores in QTT
    :param c0:
    :param c1:
    :param c2:
    :param d:
    :return:
    """

    xx, yy = zmeshgrid(d)
    Hx, Hy = vector.to_list(xx), vector.to_list(yy)

    Hs = deepcopy(Hx)
    Hs[0][:, :, 0] = c1 * Hx[0][:, :, 0] + c2 * Hy[0][:, :, 0]
    Hs[-1][1, :, :] = c1 * Hx[-1][1, :, :] + (c0 + c2 * Hy[-1][1, :, :])

    d = len(Hs)
    for k in range(1, d - 1):
        Hs[k][1, :, 0] = c1 * Hx[k][1, :, 0] + c2 * Hy[k][1, :, 0]

    return vector.from_list(Hs)


def concatenate(*args):
    """Concatenates given TT-vectors.

    For two tensors :math:`X(i_1,\\ldots,i_d),Y(i_1,\\ldots,i_d)` returns :math:`(d+1)`-dimensional
    tensor :math:`Z(i_0,i_1,\\ldots,i_d)`, :math:`i_0=\\overline{0,1}`, such that

    .. math::
       Z(0, i_1, \\ldots, i_d) = X(i_1, \\ldots, i_d),

       Z(1, i_1, \\ldots, i_d) = Y(i_1, \\ldots, i_d).

    """
    tmp = np.array([[1] + [0] * (len(args) - 1)])
    result = kron(vector(tmp), args[0])
    for i in range(1, len(args)):
        result += kron(vector(np.array([[0] * i +
                                                 [1] + [0] * (len(args) - i - 1)])), args[i])
    return result


def sum(a, axis=-1):
    """Sum TT-vector over specified axes"""
    d = a.d
    crs = vector.to_list(a.tt if isinstance(a, matrix) else a)
    if axis < 0:
        axis = range(a.d)
    elif isinstance(axis, int):
        axis = [axis]
    axis = list(axis)[::-1]
    for ax in axis:
        crs[ax] = np.sum(crs[ax], axis=1)
        rleft, rright = crs[ax].shape
        if (rleft >= rright or rleft < rright and ax + 1 >= d) and ax > 0:
            crs[ax - 1] = np.tensordot(crs[ax - 1], crs[ax], axes=(2, 0))
        elif ax + 1 < d:
            crs[ax + 1] = np.tensordot(crs[ax], crs[ax + 1], axes=(1, 0))
        else:
            return np.sum(crs[ax])
        crs.pop(ax)
        d -= 1
    return vector.from_list(crs)


# Basic functions for the arrays creation


def ones(n, d=None):
    """ Creates a TT-vector of all ones"""
    c = vector()
    if d is None:
        c.n = np.array(n, dtype=np.int32)
        c.d = c.n.size
    else:
        c.n = np.array([n] * d, dtype=np.int32)
        c.d = d
    c.r = np.ones((c.d + 1,), dtype=np.int32)
    c.get_ps()
    c.core = np.ones(c.ps[c.d] - 1)
    return c


def rand(n, d=None, r=2, samplefunc=np.random.randn):
    """Generate a random d-dimensional TT-vector with ranks ``r``.
    Distribution to sample cores is provided by the samplefunc.
    Default is to sample from normal distribution.
    """
    n0 = np.asanyarray(n, dtype=np.int32)
    r0 = np.asanyarray(r, dtype=np.int32)
    if d is None:
        d = n.size
    if n0.size == 1:
        n0 = np.ones((d,), dtype=np.int32) * n0
    if r0.size == 1:
        r0 = np.ones((d + 1,), dtype=np.int32) * r0
        r0[0] = 1
        r0[d] = 1
    c = vector()
    c.d = d
    c.n = n0
    c.r = r0
    c.get_ps()
    c.core = samplefunc(c.ps[d] - 1)
    return c


# Identity _matrix
def eye(n, d=None):
    """ Creates an identity TT-matrix"""
    c = matrix()
    c.tt = vector()
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
        c.tt.d = n0.size
    else:
        n0 = np.asanyarray([n] * d, dtype=np.int32)
        c.tt.d = d
    c.n = n0.copy()
    c.m = n0.copy()
    c.tt.n = (c.n) * (c.m)
    c.tt.r = np.ones((c.tt.d + 1,), dtype=np.int32)
    c.tt.get_ps()
    c.tt.alloc_core()
    for i in xrange(c.tt.d):
        c.tt.core[
        c.tt.ps[i] -
        1:c.tt.ps[
              i +
              1] -
          1] = np.eye(
            c.n[i]).flatten()
    return c


# Arbitrary multilevel Toeplitz _matrix
def Toeplitz(x, d=None, D=None, kind='F'):
    """ Creates multilevel Toeplitz TT-matrix with ``D`` levels.

        Possible _matrix types:

        * 'F' - full Toeplitz _matrix,             size(x) = 2^{d+1}
        * 'C' - circulant _matrix,                 size(x) = 2^d
        * 'L' - lower triangular Toeplitz _matrix, size(x) = 2^d
        * 'U' - upper triangular Toeplitz _matrix, size(x) = 2^d

        Sample calls:

        >>> # one-level Toeplitz _matrix:
        >>> T = tt.Toeplitz(x)
        >>> # one-level circulant _matrix:
        >>> T = tt.Toeplitz(x, kind='C')
        >>> # three-level upper-triangular Toeplitz _matrix:
        >>> T = tt.Toeplitz(x, D=3, kind='U')
        >>> # two-level mixed-type Toeplitz _matrix:
        >>> T = tt.Toeplitz(x, kind=['L', 'U'])
        >>> # two-level mixed-size Toeplitz _matrix:
        >>> T = tt.Toeplitz(x, [3, 4], kind='C')

    """

    # checking for arguments consistency
    def check_kinds(D, kind):
        if D % len(kind) == 0:
            kind.extend(kind * (D // len(kind) - 1))
        if len(kind) != D:
            raise ValueError(
                "Must give proper amount of _matrix kinds (one or D, for example)")

    kind = list(kind)
    if not set(kind).issubset(['F', 'C', 'L', 'U']):
        raise ValueError("Toeplitz _matrix kind must be one of F, C, L, U.")
    if d is None:
        if D is None:
            D = len(kind)
        if x.d % D:
            raise ValueError(
                "x.d must be divisible by D when d is not specified!")
        if len(kind) == 1:
            d = np.array([x.d // D - (1 if kind[0] == 'F' else 0)]
                          * D, dtype=np.int32)
            kind = kind * D
        else:
            check_kinds(D, kind)
            if set(kind).issubset(['F']):
                d = np.array([x.d // D - 1] * D, dtype=np.int32)
            elif set(kind).issubset(['C', 'L', 'U']):
                d = np.array([x.d // D] * D, dtype=np.int32)
            else:
                raise ValueError(
                    "Only similar _matrix kinds (only F or only C, L and U) are accepted when d is not specified!")
    elif d is not None:
        d = np.asarray(d, dtype=np.int32).flatten()
        if D is None:
            D = d.size
        elif d.size == 1:
            d = np.array([d[0]] * D, dtype=np.int32)
        if D != d.size:
            raise ValueError("D must be equal to len(d)")
        check_kinds(D, kind)
        if np.sum(d) + np.sum([(1 if knd == 'F' else 0)
                                 for knd in kind]) != x.d:
            raise ValueError(
                "Dimensions inconsistency: x.d != d_1 + d_2 + ... + d_D")

    # predefined matrices and tensors:
    I = [[1, 0], [0, 1]]
    J = [[0, 1], [0, 0]]
    JT = [[0, 0], [1, 0]]
    H = [[0, 1], [1, 0]]
    S = np.array([[[0], [1]], [[1], [0]]]).transpose()  # 2 x 2 x 1
    P = np.zeros((2, 2, 2, 2))
    P[:, :, 0, 0] = I
    P[:, :, 1, 0] = H
    P[:, :, 0, 1] = H
    P[:, :, 1, 1] = I
    P = np.transpose(P)  # 2 x 2! x 2 x 2 x '1'
    Q = np.zeros((2, 2, 2, 2))
    Q[:, :, 0, 0] = I
    Q[:, :, 1, 0] = JT
    Q[:, :, 0, 1] = JT
    Q = np.transpose(Q)  # 2 x 2! x 2 x 2 x '1'
    R = np.zeros((2, 2, 2, 2))
    R[:, :, 1, 0] = J
    R[:, :, 0, 1] = J
    R[:, :, 1, 1] = I
    R = np.transpose(R)  # 2 x 2! x 2 x 2 x '1'
    W = np.zeros([2] * 5)  # 2 x 2! x 2 x 2 x 2
    W[0, :, :, 0, 0] = W[1, :, :, 1, 1] = I
    W[0, :, :, 1, 0] = W[0, :, :, 0, 1] = JT
    W[1, :, :, 1, 0] = W[1, :, :, 0, 1] = J
    W = np.transpose(W)  # 2 x 2! x 2 x 2 x 2
    V = np.zeros((2, 2, 2, 2))
    V[0, :, :, 0] = I
    V[0, :, :, 1] = JT
    V[1, :, :, 1] = J
    V = np.transpose(V)  # '1' x 2! x 2 x 2 x 2

    crs = []
    xcrs = vector.to_list(x)
    dp = 0  # dimensions passed
    for j in xrange(D):
        currd = d[j]
        xcr = xcrs[dp]
        cr = np.tensordot(V, xcr, (0, 1))
        cr = cr.transpose(3, 0, 1, 2, 4)  # <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
        cr = cr.reshape((x.r[dp], 2, 2, 2 * x.r[dp + 1]),
                        order='F')  # <r_dp| x 2 x 2 x |2r_{dp+1}>
        dp += 1
        crs.append(cr)
        for i in xrange(1, currd - 1):
            xcr = xcrs[dp]
            # (<2| x 2 x 2 x |2>) x <r_dp| x |r_{dp+1}>
            cr = np.tensordot(W, xcr, (1, 1))
            # <2| x <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
            cr = cr.transpose([0, 4, 1, 2, 3, 5])
            # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp + 1]), order='F')
            dp += 1
            crs.append(cr)
        if kind[j] == 'F':
            xcr = xcrs[dp]  # r_dp x 2 x r_{dp+1}
            cr = np.tensordot(W, xcr, (1, 1)).transpose([0, 4, 1, 2, 3, 5])
            # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp + 1]), order='F')
            dp += 1
            xcr = xcrs[dp]  # r_dp x 2 x r_{dp+1}
            # <2| x |1> x <r_dp| x |r_{dp+1}>
            tmp = np.tensordot(S, xcr, (1, 1))
            # tmp = tmp.transpose([0, 2, 1, 3]) # TODO: figure out WHY THE HELL
            # this spoils everything
            # <2r_dp| x |r_{dp+1}>
            tmp = tmp.reshape((2 * x.r[dp], x.r[dp + 1]), order='F')
            # <2r_{dp-1}| x 2 x 2 x |r_{dp+1}>
            cr = np.tensordot(cr, tmp, (3, 0))
            dp += 1
            crs.append(cr)
        else:
            dotcore = None
            if kind[j] == 'C':
                dotcore = P
            elif kind[j] == 'L':
                dotcore = Q
            elif kind[j] == 'U':
                dotcore = R
            xcr = xcrs[dp]  # r_dp x 2 x r_{dp+1}
            # <2| x 2 x 2 x |'1'> x <r_dp| x |r_{dp+1}>
            cr = np.tensordot(dotcore, xcr, (1, 1))
            # <2| x <r_dp| x 2 x 2 x |r_{dp+1}>
            cr = cr.transpose([0, 3, 1, 2, 4])
            cr = cr.reshape((2 * x.r[dp], 2, 2, x.r[dp + 1]), order='F')
            dp += 1
            crs.append(cr)
    return matrix.from_list(crs)


# Laplace operator
def qlaplace_dd(d):
    """Creates a QTT representation of the Laplace operator"""
    res = matrix()
    d0 = d[::-1]
    D = len(d0)
    I = np.eye(2)
    J = np.array([[0, 1], [0, 0]])
    cr = []
    if D == 1:
        for k in xrange(1, d0[0] + 1):
            if k == 1:
                cur_core = np.zeros((1, 2, 2, 3))
                cur_core[:, :, :, 0] = 2 * I - J - J.T
                cur_core[:, :, :, 1] = -J
                cur_core[:, :, :, 2] = -J.T
            elif k == d0[0]:
                cur_core = np.zeros((3, 2, 2, 1))
                cur_core[0, :, :, 0] = I
                cur_core[1, :, :, 0] = J.T
                cur_core[2, :, :, 0] = J
            else:
                cur_core = np.zeros((3, 2, 2, 3))
                cur_core[0, :, :, 0] = I
                cur_core[1, :, :, 1] = J
                cur_core[2, :, :, 2] = J.T
                cur_core[1, :, :, 0] = J.T
                cur_core[2, :, :, 0] = J
            cr.append(cur_core)
    else:
        for k in xrange(D):
            for kappa in xrange(1, d0[k] + 1):
                if kappa == 1:
                    if k == 0:
                        cur_core = np.zeros((1, 2, 2, 4))
                        cur_core[:, :, :, 0] = 2 * I - J - J.T
                        cur_core[:, :, :, 1] = -J
                        cur_core[:, :, :, 2] = -J.T
                        cur_core[:, :, :, 3] = I
                    elif k == D - 1:
                        cur_core = np.zeros((2, 2, 2, 3))
                        cur_core[0, :, :, 0] = 2 * I - J - J.T
                        cur_core[0, :, :, 1] = -J
                        cur_core[0, :, :, 2] = -J.T
                        cur_core[1, :, :, 0] = I
                    else:
                        cur_core = np.zeros((2, 2, 2, 4))
                        cur_core[0, :, :, 0] = 2 * I - J - J.T
                        cur_core[0, :, :, 1] = -J
                        cur_core[0, :, :, 2] = -J.T
                        cur_core[0, :, :, 3] = I
                        cur_core[1, :, :, 0] = I
                elif kappa is d0[k]:
                    if k is D - 1:
                        cur_core = np.zeros((3, 2, 2, 1))
                        cur_core[0, :, :, 0] = I
                        cur_core[1, :, :, 0] = J.T
                        cur_core[2, :, :, 0] = J
                    else:
                        cur_core = np.zeros((4, 2, 2, 2))
                        cur_core[3, :, :, 0] = I
                        cur_core[0, :, :, 1] = I
                        cur_core[1, :, :, 1] = J.T
                        cur_core[2, :, :, 1] = J
                else:
                    if k is D - 1:
                        cur_core = np.zeros((3, 2, 2, 3))
                        cur_core[0, :, :, 0] = I
                        cur_core[1, :, :, 1] = J
                        cur_core[2, :, :, 2] = J.T
                        cur_core[1, :, :, 0] = J.T
                        cur_core[2, :, :, 0] = J
                    else:
                        cur_core = np.zeros((4, 2, 2, 4))
                        cur_core[0, :, :, 0] = I
                        cur_core[1, :, :, 1] = J
                        cur_core[2, :, :, 2] = J.T
                        cur_core[1, :, :, 0] = J.T
                        cur_core[2, :, :, 0] = J
                        cur_core[3, :, :, 3] = I
                cr.append(cur_core)
    return matrix.from_list(cr)


def xfun(n, d=None):
    """ Create a QTT-representation of 0:prod(n) _vector
        call examples:
        tt.xfun(2, 5)         # create 2 x 2 x 2 x 2 x 2 TT-vector
        tt.xfun(3)            # create [0, 1, 2] one-dimensional TT-vector
        tt.xfun([3, 5, 7], 2) # create 3 x 5 x 7 x 3 x 5 x 7 TT-vector
    """
    if isinstance(n, integer_types):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    if d == 1:
        return vector.from_list(
            [np.reshape(np.arange(n0[0]), (1, n0[0], 1))])
    cr = []
    cur_core = np.ones((1, n0[0], 2))
    cur_core[0, :, 0] = np.arange(n0[0])
    cr.append(cur_core)
    ni = float(n0[0])
    for i in xrange(1, d - 1):
        cur_core = np.zeros((2, n0[i], 2))
        for j in xrange(n0[i]):
            cur_core[:, j, :] = np.eye(2)
        cur_core[1, :, 0] = ni * np.arange(n0[i])
        ni *= n0[i]
        cr.append(cur_core)
    cur_core = np.ones((2, n0[d - 1], 1))
    cur_core[1, :, 0] = ni * np.arange(n0[d - 1])
    cr.append(cur_core)
    return vector.from_list(cr)


def linspace(n, d=None, a=0.0, b=1.0, right=True, left=True):
    """ Create a QTT-representation of a uniform grid on an interval [a, b] """
    if isinstance(n, integer_types):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    t = xfun(n0)
    e = ones(n0)
    N = np.prod(n0)  # Size
    if left and right:
        h = (b - a) * 1.0 / (N - 1)
        res = a * e + t * h
    elif left and not right:
        h = (b - a) * 1.0 / N
        res = a * e + t * h
    elif right and not left:
        h = (b - a) * 1.0 / N
        res = a * e + (t + e) * h
    else:
        h = (b - a) * 1.0 / (N - 1)
        res = a * e + (t + e) * h
    return res.round(1e-13)


def sin(d, alpha=1.0, phase=0.0):
    """ Create TT-vector for :math:`\\sin(\\alpha n + \\varphi)`."""
    cr = []
    cur_core = np.zeros([1, 2, 2], dtype=np.float64)
    cur_core[0, 0, :] = [np.cos(phase), np.sin(phase)]
    cur_core[0, 1, :] = [np.cos(alpha + phase), np.sin(alpha + phase)]
    cr.append(cur_core)
    for i in xrange(1, d - 1):
        cur_core = np.zeros([2, 2, 2], dtype=np.float64)
        cur_core[0, 0, :] = [1.0, 0.0]
        cur_core[1, 0, :] = [0.0, 1.0]
        cur_core[
        0,
        1,
        :] = [
            np.cos(
                alpha *
                2 ** i),
            np.sin(
                alpha *
                2 ** i)]
        cur_core[1,
        1,
        :] = [-np.sin(alpha * 2 ** i),
              np.cos(alpha * 2 ** i)]
        cr.append(cur_core)
    cur_core = np.zeros([2, 2, 1], dtype=np.float64)
    cur_core[0, :, 0] = [0.0, np.sin(alpha * 2 ** (d - 1))]
    cur_core[1, :, 0] = [1.0, np.cos(alpha * 2 ** (d - 1))]
    cr.append(cur_core)
    return vector.from_list(cr)


def cos(d, alpha=1.0, phase=0.0):
    """ Create TT-vector for :math:`\\cos(\\alpha n + \\varphi)`."""
    return sin(d, alpha, phase + np.pi * 0.5)


def delta(n, d=None, center=0):
    """ Create TT-vector for delta-function :math:`\\delta(x - x_0)`. """
    if isinstance(n, integer_types):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size

    if center < 0:
        cind = [0] * d
    else:
        cind = []
        for i in xrange(d):
            cind.append(center % n0[i])
            center //= n0[i]
        if center > 0:
            cind = [0] * d
    cr = []
    for i in xrange(d):
        cur_core = np.zeros((1, n0[i], 1))
        cur_core[0, cind[i], 0] = 1
        cr.append(cur_core)
    return vector.from_list(cr)


def stepfun(n, d=None, center=1, direction=1):
    """ Create TT-vector for Heaviside step function :math:`\chi(x - x_0)`.

    Heaviside step function is defined as

    .. math::

        \chi(x) = \\left\{ \\begin{array}{l} 1 \mbox{ when } x \ge 0, \\\\ 0 \mbox{ when } x < 0. \\end{array} \\right.

    For negative value of ``direction`` :math:`\chi(x_0 - x)` is approximated. """
    if isinstance(n, integer_types):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    N = np.prod(n0)

    if center >= N and direction < 0 or center <= 0 and direction > 0:
        return ones(n0)

    if center <= 0 and direction < 0 or center >= N and direction > 0:
        raise ValueError(
            "Heaviside function with specified center and direction gives zero tensor!")
    if direction > 0:
        center = N - center
    cind = []
    for i in xrange(d):
        cind.append(center % n0[i])
        center //= n0[i]

    def gen_notx(currcind, currn):
        return [0.0] * (currn - currcind) + [1.0] * currcind

    def gen_notx_rev(currcind, currn):
        return [1.0] * currcind + [0.0] * (currn - currcind)

    def gen_x(currcind, currn):
        result = [0.0] * currn
        result[currn - currcind - 1] = 1.0
        return result

    def gen_x_rev(currcind, currn):
        result = [0.0] * currn
        result[currcind] = 1.0
        return result

    if direction > 0:
        x = gen_x
        notx = gen_notx
    else:
        x = gen_x_rev
        notx = gen_notx_rev

    crs = []
    prevrank = 1
    for i in range(d)[::-1]:
        break_further = max([0] + cind[:i])
        nextrank = 2 if break_further else 1
        one = [1] * n0[i]
        cr = np.zeros([nextrank, n0[i], prevrank], dtype=np.float64)
        tempx = x(cind[i], n0[i])
        tempnotx = notx(cind[i], n0[i])
        # high-conditional magic
        if not break_further:
            if cind[i]:
                if prevrank > 1:
                    cr[0, :, 0] = one
                    cr[0, :, 1] = tempnotx
                else:
                    cr[0, :, 0] = tempnotx
            else:
                cr[0, :, 0] = one
        else:
            if prevrank > 1:
                cr[0, :, 0] = one
                if cind[i]:
                    cr[0, :, 1] = tempnotx
                    cr[1, :, 1] = tempx
                else:
                    cr[1, :, 1] = tempx
            else:
                if cind[i]:
                    cr[0, :, 0] = tempnotx
                    cr[1, :, 0] = tempx
                else:
                    nextrank = 1
                    cr = cr[:1, :, :]
                    cr[0, :, 0] = tempx
        prevrank = nextrank
        crs.append(cr)
    return vector.from_list(crs[::-1])


def qshift(d):
    x = []
    x.append(np.array([0.0, 1.0]).reshape((1, 2, 1)))
    for _ in xrange(1, d):
        x.append(np.array([1.0, 0.0]).reshape((1, 2, 1)))
    return Toeplitz(vector.from_list(x), kind='L')


####### Recent update #######
def unit(n, d=None, j=None, tt_instance=True):
    ''' Generates e_j _vector in tt.vector format
    ---------
    Parameters:
        n - modes (either integer or array)
        d - dimensionality (integer)
        j - position of 1 in full-format e_j (integer)
        tt_instance - if True, returns tt.vector;
                      if False, returns tt cores as a list
    '''
    if isinstance(n, int):
        if d is None:
            d = 1
        n = n * np.ones(d, dtype=np.int32)
    else:
        d = len(n)
    if j is None:
        j = 0
    rv = []

    j = ind2sub(n, j)

    for k in xrange(d):
        rv.append(np.zeros((1, n[k], 1)))
        rv[-1][0, j[k], 0] = 1
    if tt_instance:
        rv = vector.from_list(rv)
    return rv


def IpaS(d, a, tt_instance=True):
    '''A special bidiagonal _matrix in the QTT-format
    M = IPAS(D, A)
    Generates I+a*S_{-1} _matrix in the QTT-format:
    1 0 0 0
    a 1 0 0
    0 a 1 0
    0 0 a 1
    Convenient for Crank-Nicolson and time gradient matrices
    '''

    if d == 1:
        M = np.array([[1, 0], [a, 1]]).reshape((1, 2, 2, 1), order='F')
    else:
        M = [None] * d
        M[0] = np.zeros((1, 2, 2, 2))
        M[0][0, :, :, 0] = np.array([[1, 0], [a, 1]])
        M[0][0, :, :, 1] = np.array([[0, a], [0, 0]])
        for i in xrange(1, d - 1):
            M[i] = np.zeros((2, 2, 2, 2))
            M[i][:, :, 0, 0] = np.eye(2)
            M[i][:, :, 1, 0] = np.array([[0, 0], [1, 0]])
            M[i][:, :, 1, 1] = np.array([[0, 1], [0, 0]])
        M[d - 1] = np.zeros((2, 2, 2, 1))
        M[d - 1][:, :, 0, 0] = np.eye(2)
        M[d - 1][:, :, 1, 0] = np.array([[0, 0], [1, 0]])
    if tt_instance:
        M = matrix.from_list(M)
    return M


def reshape(tt_array, shape, eps=1e-14, rl=1, rr=1):
    ''' Reshape of the TT-vector
       [TT1]=TT_RESHAPE(TT,SZ) reshapes TT-vector or TT-matrix into another
       with mode sizes SZ, accuracy 1e-14

       [TT1]=TT_RESHAPE(TT,SZ,EPS) reshapes TT-vector/matrix into another with
       mode sizes SZ and accuracy EPS

       [TT1]=TT_RESHAPE(TT,SZ,EPS, RL) reshapes TT-vector/matrix into another
       with mode size SZ and left tail rank RL

       [TT1]=TT_RESHAPE(TT,SZ,EPS, RL, RR) reshapes TT-vector/matrix into
       another with mode size SZ and tail ranks RL*RR
       Reshapes TT-vector/matrix into a new one, with dimensions specified by SZ.

       If the i_nput is TT-matrix, SZ must have the sizes for both modes,
       so it is a _matrix if sizes d2-by-2.
       If the i_nput is TT-vector, SZ may be either a column or a row _vector.
    '''

    tt1 = deepcopy(tt_array)
    sz = deepcopy(shape)
    ismatrix = False
    if isinstance(tt1, matrix):
        d1 = tt1.tt.d
        d2 = sz.shape[0]
        ismatrix = True
        # The size should be [n,m] in R^{d x 2}
        restn2_n = sz[:, 0]
        restn2_m = sz[:, 1]
        sz_n = copy(sz[:, 0])
        sz_m = copy(sz[:, 1])
        n1_n = tt1.n
        n1_m = tt1.m
        # We will split/convolve using the _vector form anyway
        sz = np.prod(sz, axis=1)
        tt1 = tt1.tt
    else:
        d1 = tt1.d
        d2 = len(sz)

    # Recompute sz to include r0,rd,
    # and the items of tt1

    sz[0] = sz[0] * rl
    sz[d2 - 1] = sz[d2 - 1] * rr
    tt1.n[0] = tt1.n[0] * tt1.r[0]
    tt1.n[d1 - 1] = tt1.n[d1 - 1] * tt1.r[d1]
    if ismatrix:  # in _matrix: 1st tail rank goes to the n-mode, last to the m-mode
        restn2_n[0] = restn2_n[0] * rl
        restn2_m[d2 - 1] = restn2_m[d2 - 1] * rr
        n1_n[0] = n1_n[0] * tt1.r[0]
        n1_m[d1 - 1] = n1_m[d1 - 1] * tt1.r[d1]

    tt1.r[0] = 1
    tt1.r[d1] = 1

    n1 = tt1.n

    assert np.prod(n1) == np.prod(sz), 'Reshape: incorrect sizes'

    needQRs = False
    if d2 > d1:
        needQRs = True

    if d2 <= d1:
        i2 = 0
        n2 = deepcopy(sz)
        for i1 in range(d1):
            if n2[i2] == 1:
                i2 = i2 + 1
                if i2 > d2:
                    break
            if n2[i2] % n1[i1] == 0:
                n2[i2] = n2[i2] // n1[i1]
            else:
                needQRs = True
                break

    r1 = tt1.r
    tt1 = tt1.to_list(tt1)

    if needQRs:  # We have to split some cores -> perform QRs
        for i in range(d1 - 1, 0, -1):
            cr = tt1[i]
            cr = np.reshape(cr, (r1[i], n1[i] * r1[i + 1]), order='F')
            [cr, rv] = np.linalg.qr(cr.T)  # Size n*r2, r1new - r1nwe,r1
            cr0 = tt1[i - 1]
            cr0 = np.reshape(cr0, (r1[i - 1] * n1[i - 1], r1[i]), order='F')
            cr0 = np.dot(cr0, rv.T)  # r0*n0, r1new
            r1[i] = cr.shape[1]
            cr0 = np.reshape(cr0, (r1[i - 1], n1[i - 1], r1[i]), order='F')
            cr = np.reshape(cr.T, (r1[i], n1[i], r1[i + 1]), order='F')
            tt1[i] = cr
            tt1[i - 1] = cr0

    r2 = np.ones(d2 + 1, dtype=np.int32)

    i1 = 0  # Working index in tt1
    i2 = 0  # Working index in tt2
    core2 = np.zeros((0))
    curcr2 = 1
    restn2 = sz
    n2 = np.ones(d2, dtype=np.int32)
    if ismatrix:
        n2_n = np.ones(d2, dtype=np.int32)
        n2_m = np.ones(d2, dtype=np.int32)

    while i1 < d1:
        curcr1 = tt1[i1]
        if gcd(restn2[i2], n1[i1]) == n1[i1]:
            # The whole core1 fits to core2. Convolve it
            if (i1 < d1 - 1) and (needQRs):  # QR to the next core - for safety
                curcr1 = np.reshape(
                    curcr1, (r1[i1] * n1[i1], r1[i1 + 1]), order='F')
                [curcr1, rv] = np.linalg.qr(curcr1)
                curcr12 = tt1[i1 + 1]
                curcr12 = np.reshape(
                    curcr12, (r1[i1 + 1], n1[i1 + 1] * r1[i1 + 2]), order='F')
                curcr12 = np.dot(rv, curcr12)
                r1[i1 + 1] = curcr12.shape[0]
                tt1[i1 + 1] = np.reshape(curcr12,
                                          (r1[i1 + 1],
                                           n1[i1 + 1],
                                           r1[i1 + 2]),
                                          order='F')
            # Actually merge is here
            curcr1 = np.reshape(
                curcr1, (r1[i1], n1[i1] * r1[i1 + 1]), order='F')
            curcr2 = np.dot(curcr2, curcr1)  # size r21*nold, dn*r22
            if ismatrix:  # Permute if we are working with tt_matrix
                curcr2 = np.reshape(curcr2, (r2[i2], n2_n[i2], n2_m[i2], n1_n[
                    i1], n1_m[i1], r1[i1 + 1]), order='F')
                curcr2 = np.transpose(curcr2, [0, 1, 3, 2, 4, 5])
                # Update the "matrix" sizes
                n2_n[i2] = n2_n[i2] * n1_n[i1]
                n2_m[i2] = n2_m[i2] * n1_m[i1]
                restn2_n[i2] = restn2_n[i2] // n1_n[i1]
                restn2_m[i2] = restn2_m[i2] // n1_m[i1]
            r2[i2 + 1] = r1[i1 + 1]
            # Update the sizes of tt2
            n2[i2] = n2[i2] * n1[i1]
            restn2[i2] = restn2[i2] // n1[i1]
            curcr2 = np.reshape(
                curcr2, (r2[i2] * n2[i2], r2[i2 + 1]), order='F')
            i1 = i1 + 1  # current core1 is over
        else:
            if (gcd(restn2[i2], n1[i1]) != 1) or (restn2[i2] == 1):
                # There exists a nontrivial divisor, or a singleton requested
                # Split it and convolve
                n12 = gcd(restn2[i2], n1[i1])
                if ismatrix:  # Permute before the truncation
                    # _matrix sizes we are able to split
                    n12_n = gcd(restn2_n[i2], n1_n[i1])
                    n12_m = gcd(restn2_m[i2], n1_m[i1])
                    curcr1 = np.reshape(curcr1,
                                         (r1[i1],
                                          n12_n,
                                          n1_n[i1] // n12_n,
                                          n12_m,
                                          n1_m[i1] // n12_m,
                                          r1[i1 + 1]),
                                         order='F')
                    curcr1 = np.transpose(curcr1, [0, 1, 3, 2, 4, 5])
                    # Update the _matrix sizes of tt2 and tt1
                    n2_n[i2] = n2_n[i2] * n12_n
                    n2_m[i2] = n2_m[i2] * n12_m
                    restn2_n[i2] = restn2_n[i2] // n12_n
                    restn2_m[i2] = restn2_m[i2] // n12_m
                    n1_n[i1] = n1_n[i1] // n12_n
                    n1_m[i1] = n1_m[i1] // n12_m

                curcr1 = np.reshape(
                    curcr1, (r1[i1] * n12, (n1[i1] // n12) * r1[i1 + 1]), order='F')
                [u, s, v] = np.linalg.svd(curcr1, full_matrices=False)
                r = my_chop2(s, eps * np.linalg.norm(s) / (d2 - 1) ** 0.5)
                u = u[:, :r]
                v = v.T
                v = v[:, :r] * s[:r]
                u = np.reshape(u, (r1[i1], n12 * r), order='F')
                # u is our admissible chunk, merge it to core2
                curcr2 = np.dot(curcr2, u)  # size r21*nold, dn*r22
                r2[i2 + 1] = r
                # Update the sizes of tt2
                n2[i2] = n2[i2] * n12
                restn2[i2] = restn2[i2] // n12
                curcr2 = np.reshape(
                    curcr2, (r2[i2] * n2[i2], r2[i2 + 1]), order='F')
                r1[i1] = r
                # and tt1
                n1[i1] = n1[i1] // n12
                # keep v in tt1 for next operations
                curcr1 = np.reshape(
                    v.T, (r1[i1], n1[i1], r1[i1 + 1]), order='F')
                tt1[i1] = curcr1
            else:
                # Bad case. We have to merge cores of tt1 until a common
                # divisor appears
                i1new = i1 + 1
                curcr1 = np.reshape(
                    curcr1, (r1[i1] * n1[i1], r1[i1 + 1]), order='F')
                while (gcd(restn2[i2], n1[i1]) == 1) and (i1new < d1):
                    cr1new = tt1[i1new]
                    cr1new = np.reshape(
                        cr1new, (r1[i1new], n1[i1new] * r1[i1new + 1]), order='F')
                    # size r1(i1)*n1(i1), n1new*r1new
                    curcr1 = np.dot(curcr1, cr1new)
                    if ismatrix:  # Permutes and _matrix size updates
                        curcr1 = np.reshape(curcr1, (r1[i1], n1_n[i1], n1_m[i1], n1_n[
                            i1new], n1_m[i1new], r1[i1new + 1]), order='F')
                        curcr1 = np.transpose(curcr1, [0, 1, 3, 2, 4, 5])
                        n1_n[i1] = n1_n[i1] * n1_n[i1new]
                        n1_m[i1] = n1_m[i1] * n1_m[i1new]
                    n1[i1] = n1[i1] * n1[i1new]
                    curcr1 = np.reshape(
                        curcr1, (r1[i1] * n1[i1], r1[i1new + 1]), order='F')
                    i1new = i1new + 1
                # Inner cores merged => squeeze tt1 data
                n1 = np.concatenate((n1[:i1], n1[i1new:]))
                r1 = np.concatenate((r1[:i1], r1[i1new:]))
                tt1[i] = np.reshape(
                    curcr1, (r1[i1], n1[i1], r1[i1new]), order='F')
                tt1 = tt1[:i1] + tt1[i1new:]
                d1 = len(n1)

        if (restn2[i2] == 1) and ((i1 >= d1) or ((i1 < d1) and (n1[i1] != 1))):
            # The core of tt2 is finished
            # The second condition prevents core2 from finishing until we
            # squeeze all tailing singletons in tt1.
            curcr2 = curcr2.flatten(order='F')
            core2 = np.concatenate((core2, curcr2))
            i2 = i2 + 1
            # Start new core2
            curcr2 = 1

    # If we have been asked for singletons - just add them
    while (i2 < d2):
        core2 = np.concatenate((core2, np.ones(1)))
        r2[i2] = 1
        i2 = i2 + 1

    tt2 = ones(2, 1)  # dummy tensor
    tt2.d = d2
    tt2.n = n2
    tt2.r = r2
    tt2.core = core2
    tt2.ps = np.int32(np.cumsum(np.concatenate((np.ones(1), r2[:-1] * n2 * r2[1:]))))

    tt2.n[0] = tt2.n[0] // rl
    tt2.n[d2 - 1] = tt2.n[d2 - 1] // rr
    tt2.r[0] = rl
    tt2.r[d2] = rr

    if ismatrix:
        ttt = eye(1, 1)  # dummy tt _matrix
        ttt.n = sz_n
        ttt.m = sz_m
        ttt.tt = tt2
        return ttt
    else:
        return tt2


def permute(x, order, eps=None, return_cores=False):
    '''
    Permute dimensions (python translation of original matlab code)
       Y = permute(X, ORDER, EPS) permutes the dimensions of the TT-tensor X
       according to ORDER, delivering a result at relative accuracy EPS. This
       function is equivalent to
          Y = tt_tensor(permute(reshape(full(X), X.n'),ORDER), EPS)
       but avoids the conversion to the full format.


     Simon Etter, Summer 2015
     Seminar of Applied Mathematics, ETH Zurich


     TT-Toolbox 2.2, 2009-2012

    This is TT Toolbox, written by Ivan Oseledets et al. Institute of
    Numerical Mathematics, Moscow, Russia webpage:
    http://spring.inm.ras.ru/osel

    For all questions, bugs and suggestions please mail
    ivan.oseledets@gmail.com
    ---------------------------

     This code basically performs insertion sort on the TT dimensions:
      for k = 2:d
         Bubble the kth dimension to the right (according to ORDER) position in the first 1:k dimensions.

     The current code could be optimised at the following places:
      - Instead of initially orthogonalising with respect to the first two vertices,
        orthogonalise directly with respect to the first inversion.
      - When performing the SVD, check on which side of the current position the
        next swap will occur and directly establish the appropriate orthogonality
        (current implementation always assumes we move left).
     Both changes reduce the number of QRs by at most O(d) and are therefore likely
     to produce negligible speedup while rendering the code more complicated.
     '''

    def _reshape(tensor, shape):
        return np.reshape(tensor, shape, order='F')

    # Parse input
    if eps is None:
        eps = np.spacing(1)
    cores = vector.to_list(x)
    d = deepcopy(x.d)
    n = deepcopy(x.n)
    r = deepcopy(x.r)

    idx = np.empty(len(order))
    idx[order] = np.arange(len(order))
    eps /= d ** 1.5
    # ^Numerical evidence suggests that eps = eps/d may be sufficient, however I can only prove correctness
    #  for this choice of global-to-local conversion factor.
    assert len(order) > d - 1, 'ORDER must have at least D elements for a D-dimensional tensor'

    # RL-orthogonalise x
    for kk in xrange(d - 1, 1, -1):  ##########################################
        new_shape = [r[kk], n[kk] * r[kk + 1]]
        Q, R = np.linalg.qr(_reshape(cores[kk], new_shape).T)
        tr = min(new_shape)
        cores[kk] = _reshape(Q.T, [tr, n[kk], r[kk + 1]])
        tmp = _reshape(cores[kk - 1], [r[kk - 1] * n[kk - 1], r[kk]])
        tmp = np.dot(tmp, R.T)
        cores[kk - 1] = _reshape(tmp, [r[kk - 1], n[kk - 1], tr])
        r[kk] = tr
    k = 0
    while (True):
        # Find next inversion
        nk = k
        while (nk < d - 1) and (idx[nk] < idx[nk + 1]):
            nk += 1
        if (nk == d - 1):
            break

        # Move orthogonal centre there
        for kk in xrange(k, nk - 1):  #############
            new_shape = [r[kk] * n[kk], r[kk + 1]]
            Q, R = np.linalg.qr(_reshape(cores[kk], new_shape))
            tr = min(new_shape)
            new_shape = [r[kk], n[kk], tr]
            cores[kk] = _reshape(Q, new_shape)
            tmp = _reshape(cores[kk + 1], [r[kk + 1], n[kk + 1] * r[kk + 2]])
            tmp = np.dot(R, tmp)
            cores[kk + 1] = _reshape(tmp, [tr, n[kk + 1], r[kk + 2]])
            r[kk + 1] = tr
        k = nk

        # Swap dimensions
        tmp = _reshape(cores[k], [r[k] * n[k], r[k + 1]])
        tmp = np.dot(tmp, _reshape(cores[k + 1], [r[k + 1], n[k + 1] * r[k + 2]]))
        c = _reshape(tmp, [r[k], n[k], n[k + 1], r[k + 2]])
        c = np.transpose(c, [0, 2, 1, 3])
        tmp = _reshape(c, [r[k] * n[k + 1], n[k] * r[k + 2]])
        U, S, Vt = np.linalg.svd(tmp, full_matrices=False)
        r[k + 1] = max(my_chop2(S, np.linalg.norm(S) * eps), 1)
        lenS = len(S)
        tmp = U[:, :lenS] * S  # multiplication by diagonal matrix
        cores[k] = _reshape(tmp[:, :r[k + 1]], [r[k], n[k + 1], r[k + 1]])
        cores[k + 1] = _reshape(Vt[:r[k + 1], :], [r[k + 1], n[k], r[k + 2]])
        idx[[k, k + 1]] = idx[[k + 1, k]]
        n[[k, k + 1]] = n[[k + 1, k]]
        k = max(k - 1, 0)  ##################
    # Parse output
    if return_cores:
        return cores
    return vector.from_list(cores)
