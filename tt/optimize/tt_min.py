"""This module contains a prototype implementation of the
TT-cross-based minimization procedure
"""
import numpy as np
import math
import tt
from ..maxvol import maxvol
from ..utils.rect_maxvol import rect_maxvol


def reshape(a, sz):
    return np.reshape(a, sz, order='F')


def mkron(a, b):
    return np.kron(a, b)


def mysvd(a, full_matrices=False):
    try:
        return np.linalg.svd(a, full_matrices)
    except:
        return np.linalg.svd(a + np.max(np.abs(a).flatten()) * 1e-14 *
                             np.random.randn(a.shape[0], a.shape[1]), full_matrices)


def min_func(fun, bounds_min, bounds_max, d=None, rmax=10,
             n0=64, nswp=10, verb=True, smooth_fun=None):
    """Find (approximate) minimal value of the function on a d-dimensional grid."""
    if d is None:
        d = len(bounds_min)
        a = np.asanyarray(bounds_min).copy()
        b = np.asanyarray(bounds_max).copy()
    else:
        a = np.ones(d) * bounds_min
        b = np.ones(d) * bounds_max

    if smooth_fun is None:
        smooth_fun = lambda p, lam: (math.pi / 2 - np.arctan(p - lam))
    #smooth_fun = lambda p, lam: np.exp(-10*(p - lam))

        # We do not need to store the cores, only the interfaces!
    Rx = [[]] * (d + 1)  # Python list for the interfaces
    Rx[0] = np.ones((1, 1))
    Rx[d] = np.ones((1, 1))
    Jy = [np.empty(0)] * (d + 1)
    ry = rmax * np.ones(d + 1, dtype=np.int)
    ry[0] = 1
    ry[d] = 1
    n = n0 * np.ones(d, dtype=np.int)
    fun_evals = 0

    grid = [np.reshape(np.linspace(a[i], b[i], n[i]), (n[i], 1))
            for i in xrange(d)]
    for i in xrange(d - 1):
        #cr1 = y[i]
        ry[i + 1] = min(ry[i + 1], n[i] * ry[i])
        cr1 = np.random.randn(ry[i], n[i], ry[i + 1])
        cr1 = reshape(cr1, (ry[i] * n[i], ry[i + 1]))
        q, r = np.linalg.qr(cr1)
        ind = maxvol(q)
        w1 = mkron(np.ones((n[i], 1)), Jy[i])
        w2 = mkron(grid[i], np.ones((ry[i], 1)))
        Jy[i + 1] = np.hstack((w1, w2))
        Jy[i + 1] = reshape(Jy[i + 1], (ry[i] * n[i], -1))
        Jy[i + 1] = Jy[i + 1][ind, :]

        # Jy{i+1} = [kron(ones(n(i),1), Jy{i}), kron((1:n(i))', ones(ry(i),1))];
        # Jy{i+1} = Jy{i+1}(ind,:);
    swp = 0
    dirn = -1
    i = d - 1
    lm = 999999999999
    while swp < nswp:
        # Right-to-left sweep
        # The idea: compute the current core; compute the function of it;
        # Shift locally or globally? Local shift would be the first try
        # Compute the current core

        if np.size(Jy[i]) == 0:
            w1 = np.zeros((ry[i] * n[i] * ry[i + 1], 0))
        else:
            w1 = mkron(np.ones((n[i] * ry[i + 1], 1)), Jy[i])
        w2 = mkron(mkron(np.ones((ry[i + 1], 1)),
                         grid[i]), np.ones((ry[i], 1)))
        if np.size(Jy[i + 1]) == 0:
            w3 = np.zeros((ry[i] * n[i] * ry[i + 1], 0))
        else:
            w3 = mkron(Jy[i + 1], np.ones((ry[i] * n[i], 1)))

        J = np.hstack((w1, w2, w3))
        # Just add some random indices to J, which is rnr x d, need to make rn (r + r0) x add,
        # i.e., just generate random r, random n and random multiindex

        cry = fun(J)
        fun_evals += cry.size
        cry = reshape(cry, (ry[i], n[i], ry[i + 1]))
        min_cur = np.min(cry.flatten("F"))
        ind_cur = np.argmin(cry.flatten("F"))
        if lm > min_cur:
            lm = min_cur
            x_full = J[ind_cur, :]
            val = fun(x_full)
            if verb:
                print 'New record:', val, 'Point:', x_full, 'fevals:', fun_evals
        cry = smooth_fun(cry, lm)
        if (dirn < 0 and i > 0):
            cry = reshape(cry, (ry[i], n[i] * ry[i + 1]))
            cry = cry.T
            #q, r = np.linalg.qr(cry)
            u, s, v = mysvd(cry, full_matrices=False)
            ry[i] = min(ry[i], rmax)
            q = u[:, :ry[i]]
            ind = rect_maxvol(q)[0]  # maxvol(q)
            ry[i] = ind.size
            w1 = mkron(np.ones((ry[i + 1], 1)), grid[i])
            if np.size(Jy[i + 1]) == 0:
                w2 = np.zeros((n[i] * ry[i + 1], 0))
            else:
                w2 = mkron(Jy[i + 1], np.ones((n[i], 1)))
            Jy[i] = np.hstack((w1, w2))
            Jy[i] = reshape(Jy[i], (n[i] * ry[i + 1], -1))
            Jy[i] = Jy[i][ind, :]

        if (dirn > 0 and i < d - 1):
            cry = reshape(cry, (ry[i] * n[i], ry[i + 1]))
            q, r = np.linalg.qr(cry)
            #ind = maxvol(q)
            ind = rect_maxvol(q)[0]
            ry[i + 1] = ind.size
            w1 = mkron(np.ones((n[i], 1)), Jy[i])
            w2 = mkron(grid[i], np.ones((ry[i], 1)))
            Jy[i + 1] = np.hstack((w1, w2))
            Jy[i + 1] = reshape(Jy[i + 1], (ry[i] * n[i], -1))
            Jy[i + 1] = Jy[i + 1][ind, :]

        i += dirn
        if i == d or i == -1:
            dirn = -dirn
            i += dirn
            swp = swp + 1
    return val, x_full


def min_tens(tens, rmax=10, nswp=10, verb=True, smooth_fun=None):
    """Find (approximate) minimal element in a TT-tensor."""
    if smooth_fun is None:
        smooth_fun = lambda p, lam: (math.pi / 2 - np.arctan(p - lam))
    d = tens.d
    Rx = [[]] * (d + 1)  # Python list for the interfaces
    Rx[0] = np.ones((1, 1))
    Rx[d] = np.ones((1, 1))
    Jy = [np.empty(0)] * (d + 1)
    ry = rmax * np.ones(d + 1, dtype=np.int)
    ry[0] = 1
    ry[d] = 1
    n = tens.n
    elements_seen = 0
    phi_left = [np.empty(0)] * (d + 1)
    phi_left[0] = np.array([1])
    phi_right = [np.empty(0)] * (d + 1)
    phi_right[d] = np.array([1])
    cores = tt.tensor.to_list(tens)

    # Fill initial multiindex J randomly.
    grid = [np.reshape(range(n[i]), (n[i], 1)) for i in xrange(d)]
    for i in xrange(d - 1):
        ry[i + 1] = min(ry[i + 1], n[i] * ry[i])
        ind = sorted(np.random.permutation(ry[i] * n[i])[0:ry[i + 1]])
        w1 = mkron(np.ones((n[i], 1)), Jy[i])
        w2 = mkron(grid[i], np.ones((ry[i], 1)))
        Jy[i + 1] = np.hstack((w1, w2))
        Jy[i + 1] = reshape(Jy[i + 1], (ry[i] * n[i], -1))
        Jy[i + 1] = Jy[i + 1][ind, :]
        phi_left[i + 1] = np.tensordot(phi_left[i], cores[i], 1)
        phi_left[i + 1] = reshape(phi_left[i + 1], (ry[i] * n[i], -1))
        phi_left[i + 1] = phi_left[i + 1][ind, :]
    swp = 0
    dirn = -1
    i = d - 1
    lm = 999999999999
    while swp < nswp:
        # Right-to-left sweep
        # The idea: compute the current core; compute the function of it;
        # Shift locally or globally? Local shift would be the first try
        # Compute the current core

        if np.size(Jy[i]) == 0:
            w1 = np.zeros((ry[i] * n[i] * ry[i + 1], 0))
        else:
            w1 = mkron(np.ones((n[i] * ry[i + 1], 1)), Jy[i])
        w2 = mkron(mkron(np.ones((ry[i + 1], 1)),
                         grid[i]), np.ones((ry[i], 1)))
        if np.size(Jy[i + 1]) == 0:
            w3 = np.zeros((ry[i] * n[i] * ry[i + 1], 0))
        else:
            w3 = mkron(Jy[i + 1], np.ones((ry[i] * n[i], 1)))
        J = np.hstack((w1, w2, w3))

        phi_right[i] = np.tensordot(cores[i], phi_right[i + 1], 1)
        phi_right[i] = reshape(phi_right[i], (-1, n[i] * ry[i + 1]))

        cry = np.tensordot(
            phi_left[i], np.tensordot(
                cores[i], phi_right[
                    i + 1], 1), 1)
        elements_seen += cry.size
        cry = reshape(cry, (ry[i], n[i], ry[i + 1]))
        min_cur = np.min(cry.flatten("F"))
        ind_cur = np.argmin(cry.flatten("F"))
        if lm > min_cur:
            lm = min_cur
            x_full = J[ind_cur, :]
            val = tens[x_full]
            if verb:
                print 'New record:', val, 'Point:', x_full, 'elements seen:', elements_seen
        cry = smooth_fun(cry, lm)
        if dirn < 0 and i > 0:
            cry = reshape(cry, (ry[i], n[i] * ry[i + 1]))
            cry = cry.T
            #q, r = np.linalg.qr(cry)
            u, s, v = mysvd(cry, full_matrices=False)
            ry[i] = min(ry[i], rmax)
            q = u[:, :ry[i]]
            ind = rect_maxvol(q)[0]  # maxvol(q)
            ry[i] = ind.size
            w1 = mkron(np.ones((ry[i + 1], 1)), grid[i])
            if np.size(Jy[i + 1]) == 0:
                w2 = np.zeros((n[i] * ry[i + 1], 0))
            else:
                w2 = mkron(Jy[i + 1], np.ones((n[i], 1)))
            Jy[i] = np.hstack((w1, w2))
            Jy[i] = reshape(Jy[i], (n[i] * ry[i + 1], -1))
            Jy[i] = Jy[i][ind, :]
            phi_right[i] = np.tensordot(cores[i], phi_right[i + 1], 1)
            phi_right[i] = reshape(phi_right[i], (-1, n[i] * ry[i + 1]))
            phi_right[i] = phi_right[i][:, ind]

        if dirn > 0 and i < d - 1:
            cry = reshape(cry, (ry[i] * n[i], ry[i + 1]))
            q, r = np.linalg.qr(cry)
            #ind = maxvol(q)
            ind = rect_maxvol(q)[0]
            ry[i + 1] = ind.size
            phi_left[i + 1] = np.tensordot(phi_left[i], cores[i], 1)
            phi_left[i + 1] = reshape(phi_left[i + 1], (ry[i] * n[i], -1))
            phi_left[i + 1] = phi_left[i + 1][ind, :]
            w1 = mkron(np.ones((n[i], 1)), Jy[i])
            w2 = mkron(grid[i], np.ones((ry[i], 1)))
            Jy[i + 1] = np.hstack((w1, w2))
            Jy[i + 1] = reshape(Jy[i + 1], (ry[i] * n[i], -1))
            Jy[i + 1] = Jy[i + 1][ind, :]

        i += dirn
        if i == d or i == -1:
            dirn = -dirn
            i += dirn
            swp = swp + 1
    return val, x_full
