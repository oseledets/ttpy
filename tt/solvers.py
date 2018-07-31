from __future__ import print_function, absolute_import, division
import numpy as np
import tt

import scipy.linalg as la

# TT-GMRES


def GMRES(A, u_0, b, eps=1e-6, maxit=100, m=20, _iteration=0, callback=None, verbose=0):
    """
    Flexible TT GMRES
    :param A: matvec(x[, eps])
    :param u_0: initial vector
    :param b: answer
    :param maxit: max number of iterations
    :param eps: required accuracy
    :param m: number of iteration without restart
    :param _iteration: iteration counter
    :param callback:
    :param verbose: to print debug info or not
    :return: answer, residual

    >>> from tt import GMRES
    >>> def matvec(x, eps):
    >>>     return tt.matvec(S, x).round(eps)
    >>> answer, res = GMRES(matvec, u_0, b, eps=1e-8)
    """
    maxitexceeded = False
    converged = False

    if verbose:
        print('GMRES(m=%d, _iteration=%d, maxit=%d)' % (m, _iteration, maxit))
    v = np.ones((m + 1), dtype=object) * np.nan
    R = np.ones((m, m)) * np.nan
    g = np.zeros(m)
    s = np.ones(m) * np.nan
    c = np.ones(m) * np.nan
    v[0] = b - A(u_0, eps=eps)
    v[0] = v[0].round(eps)
    resnorm = v[0].norm()
    curr_beta = resnorm
    bnorm = b.norm()
    wlen = resnorm
    q = m
    for j in range(m):
        _iteration += 1

        delta = eps / (curr_beta / resnorm)

        if verbose:
            print("it = %d delta = " % _iteration, delta)

        v[j] *= 1.0 / wlen
        v[j + 1] = A(v[j], eps=delta)
        for i in range(j + 1):
            R[i, j] = tt.dot(v[j + 1], v[i])
            v[j + 1] = v[j + 1] - R[i, j] * v[i]
        v[j + 1] = v[j + 1].round(delta)

        wlen = v[j + 1].norm()
        for i in range(j):
            r1 = R[i, j]
            r2 = R[i + 1, j]
            R[i, j] = c[i] * r1 - s[i] * r2
            R[i + 1, j] = c[i] * r2 + s[i] * r1
        denom = np.hypot(wlen, R[j, j])
        s[j] = wlen / denom
        c[j] = -R[j, j] / denom
        R[j, j] = -denom

        g[j] = c[j] * curr_beta
        curr_beta *= s[j]

        if verbose:
            print("it = {}, ||r|| = {}".format(_iteration, curr_beta / bnorm))

        converged = (curr_beta / bnorm) < eps or (curr_beta / resnorm) < eps
        maxitexceeded = _iteration >= maxit
        if converged or maxitexceeded:
            q = j + 1
            break

    y = la.solve_triangular(R[:q, :q], g[:q], check_finite=False)
    for idx in range(q):
        u_0 += v[idx] * y[idx]

    u_0 = u_0.round(eps)

    if callback is not None:
        callback(u_0)

    if converged or maxitexceeded:
        return u_0, resnorm / bnorm
    return GMRES(A, u_0, b, eps, maxit, m, _iteration, callback=callback, verbose=verbose)
