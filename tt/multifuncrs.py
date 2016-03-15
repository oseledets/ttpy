import numpy as np
from numpy import prod, nonzero, size
import math
import tt
from tt.maxvol import maxvol
import copy


def reshape(a, size):
    return np.reshape(a, size, order='F')


def my_chop2(sv, eps):
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return np.amin(ff)

# Cross approximation of a (vector-)function of several TT-tensors.
# [Y]=MULTIFUNCRS(X,FUNS,EPS,VARARGIN)
# Computes approximation to the functions FUNS(X{1},...,X{N}) with accuracy EPS
# X should be a cell array of nx TT-tensors of equal sizes.
# The function FUNS should receive a 2d array V of sizes I x N, where the
# first dimension stays for the reduced set of spatial indices, and the
# second is the enumerator of X.
# The returned sizes should be I x D2, where D2 is the number of
# components in FUNS. D2 should be either provided as the last (d+1)-th
# TT-rank of the initial guess, or given explicitly as an option (see
# below).
# For example, a linear combination reads FUNS=@(x)(x*W), W is a N x D2
# matrix.
#
# Options are provided in form
# 'PropertyName1',PropertyValue1,'PropertyName2',PropertyValue2 and so
# on. The parameters are set to default (in brackets in the following)
# The list of option names and default values are:
#   o y0 - initial approximation [random rank-2 tensor]
#   o nswp - maximal number of DMRG sweeps [10]
#   o rmax - maximal TT rank [Inf]
#   o verb - verbosity level, 0-silent, 1-sweep info, 2-block info [1]
#   o kickrank - the rank-increasing parameter [5]
#   o d2 - the last rank of y, that is dim(FUNS) [1]
#   o qr - do (or not) qr before maxvol [false]
# o pcatype - How to compute the enrichment of the basis, 'uchol' -
# Incomplete Cholesky, 'svd' - SVD [svd]


def multifuncrs(X, funs, eps=1E-6,
                nswp=10,
                kickrank=5,
                y0=None,
                rmax=999999,  # TODO:infinity \
                kicktype='amr-two',        \
                pcatype='svd',             \
                trunctype='fro',           \
                d2=1,                      \
                do_qr=False,               \
                verb=1):
    """Cross approximation of a (vector-)function of several TT-tensors.

    :param X: tuple of TT-tensors
    :param funs: multivariate function
    :param eps: accuracy
    """

    dtype = np.float64
    if len(filter(lambda x: x.is_complex, X)) > 0:
        dtype = np.complex128

    y = y0
    wasrand = False

    nx = len(X)
    d = X[0].d
    n = X[0].n
    rx = np.transpose(np.array([ttx.r for ttx in X]))
    #crx = [tt.tensor.to_list(ttx) for x in X]
    #crx = zip(*crx)
    crx = np.transpose(np.array([tt.tensor.to_list(ttx)
                                 for ttx in X], dtype=np.object))
    crx = np.empty((nx, d), dtype=np.object)
    i = 0
    for ttx in X:
        v = tt.tensor.to_list(ttx)
        j = 0
        for w in v:
            crx[i, j] = w
            j = j + 1
        i = i + 1
    crx = crx.T
    if y is None:
        ry = d2 * np.ones((d + 1,), dtype=np.int32)
        ry[0] = 1
        y = tt.rand(n, d, ry)
        wasrand = True

    ry = y.r
    cry = tt.tensor.to_list(y)

    Ry = np.zeros((d + 1, ), dtype=np.object)
    Ry[0] = np.array([[1.0]], dtype=dtype)
    Ry[d] = np.array([[1.0]], dtype=dtype)
    Rx = np.zeros((d + 1, nx), dtype=np.object)
    Rx[0, :] = np.ones(nx, dtype=dtype)
    Rx[d, :] = np.ones(nx, dtype=dtype)

    block_order = [+d, -d]

    # orth
    for i in range(0, d - 1):
        cr = cry[i]
        cr = reshape(cr, (ry[i] * n[i], ry[i + 1]))
        cr, rv = np.linalg.qr(cr)
        cr2 = cry[i + 1]
        cr2 = reshape(cr2, (ry[i + 1], n[i + 1] * ry[i + 2]))
        cr2 = np.dot(rv, cr2)  # matrix multiplication
        ry[i + 1] = cr.shape[1]
        cr = reshape(cr, (ry[i], n[i], ry[i + 1]))
        cry[i + 1] = reshape(cr2, (ry[i + 1], n[i + 1], ry[i + 2]))
        cry[i] = cr

        Ry[i + 1] = np.dot(Ry[i], reshape(cr, (ry[i], n[i] * ry[i + 1])))
        Ry[i + 1] = reshape(Ry[i + 1], (ry[i] * n[i], ry[i + 1]))
        curind = []
        if wasrand:
            # EVERY DAY I'M SHUFFLIN'
            curind = np.random.permutation(n[i] * ry[i])[:ry[i + 1]]
        else:
            curind = maxvol(Ry[i + 1])
        Ry[i + 1] = Ry[i + 1][curind, :]
        for j in range(0, nx):
            try:
                Rx[i + 1, j] = reshape(crx[i, j],
                                       (rx[i, j], n[i] * rx[i + 1, j]))
            except:
                pass
            Rx[i + 1, j] = np.dot(Rx[i, j], Rx[i + 1, j])
            Rx[i + 1, j] = reshape(Rx[i + 1, j], (ry[i] * n[i], rx[i + 1, j]))
            Rx[i + 1, j] = Rx[i + 1, j][curind, :]

    d2 = ry[d]
    ry[d] = 1
    cry[d - 1] = np.transpose(cry[d - 1], [2, 0, 1])  # permute

    last_sweep = False
    swp = 1

    dy = np.zeros((d, ))
    max_dy = 0

    cur_order = copy.copy(block_order)
    order_index = 1
    i = d - 1
    # can't use 'dir' identifier in python
    dirn = int(math.copysign(1, cur_order[order_index]))

    # DMRG sweeps
    while swp <= nswp or dirn > 0:

        oldy = reshape(cry[i], (d2 * ry[i] * n[i] * ry[i + 1],))

        if not last_sweep:
            # compute the X superblocks
            curbl = np.zeros((ry[i] * n[i] * ry[i + 1], nx), dtype=dtype)
            for j in range(0, nx):
                cr = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                cr = np.dot(Rx[i, j], cr)
                cr = reshape(cr, (ry[i] * n[i], rx[i + 1, j]))
                cr = np.dot(cr, Rx[i + 1, j])
                curbl[:, j] = cr.flatten('F')
            # call the function
            newy = funs(curbl)
            # multiply with inverted Ry
            newy = reshape(newy, (ry[i], n[i] * ry[i + 1] * d2))
            newy = np.linalg.solve(Ry[i], newy)  # y = R \ y
            newy = reshape(newy, (ry[i] * n[i] * ry[i + 1], d2))
            newy = reshape(np.transpose(newy), (d2 * ry[i] * n[i], ry[i + 1]))
            newy = np.transpose(np.linalg.solve(
                np.transpose(Ry[i + 1]), np.transpose(newy)))  # y=y/R
            newy = reshape(newy, (d2 * ry[i] * n[i] * ry[i + 1],))
        else:
            newy = oldy

        dy[i] = np.linalg.norm(newy - oldy) / np.linalg.norm(newy)
        max_dy = max(max_dy, dy[i])

        # truncation
        if dirn > 0:  # left-to-right
            newy = reshape(newy, (d2, ry[i] * n[i] * ry[i + 1]))
            newy = reshape(np.transpose(newy), (ry[i] * n[i], ry[i + 1] * d2))
        else:
            newy = reshape(newy, (d2 * ry[i], n[i] * ry[i + 1]))

        r = 0  # defines a variable in global scope

        if kickrank >= 0:
            u, s, v = np.linalg.svd(newy, full_matrices=False)
            v = np.conj(np.transpose(v))
            if trunctype == "fro" or last_sweep:
                r = my_chop2(s, eps / math.sqrt(d) * np.linalg.norm(s))
            else:
                # truncate taking into account the (r+1) overhead in the cross
                # (T.S.: what?)
                cums = abs(s * np.arange(2, len(s) + 2)) ** 2
                cums = np.cumsum(cums[::-1])[::-1]
                cums = cums / cums[0]
                ff = [i for i in range(len(cums)) if cums[i] < eps ** 2 / d]
                if len(ff) == 0:
                    r = len(s)
                else:
                    r = np.amin(ff)
            r = min(r, rmax, len(s))
        else:
            if dirn > 0:
                u, v = np.linalg.qr(newy)
                v = np.conj(np.transpose(v))
                r = u.shape[1]
                s = np.ones((r, ))
            else:
                v, u = np.linalg.qr(np.transpose(newy))
                v = np.conj(v)
                u = np.transpose(u)
                r = u.shape[1]
                s = np.ones((r, ))

        if verb > 1:
            print '=multifuncrs=   block %d{%d}, dy: %3.3e, r: %d' % (i, dirn, dy[i], r)

        # kicks and interfaces
        if dirn > 0 and i < d - 1:
            u = u[:, :r]
            v = np.dot(v[:, :r], np.diag(s[:r]))

            # kick
            radd = 0
            rv = 1
            if not last_sweep and kickrank > 0:
                uk = None
                if kicktype == 'amr-two':
                    # AMR(two)-like kick.

                    # compute the X superblocks
                    ind2 = np.unique(np.random.randint(
                        0, ry[i + 2] * n[i + 1], ry[i + 1]))
                    #ind2 = np.unique(np.floor(np.random.rand(ry[i + 1]) * (ry[i + 2] * n[i + 1])))
                    rkick = len(ind2)
                    curbl = np.zeros((ry[i] * n[i] * rkick, nx), dtype=dtype)
                    for j in range(nx):
                        cr1 = reshape(
                            crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                        cr1 = np.dot(Rx[i, j], cr1)
                        cr1 = reshape(cr1, (ry[i] * n[i], rx[i + 1, j]))
                        cr2 = reshape(
                            crx[i + 1, j], (rx[i + 1, j] * n[i + 1], rx[i + 2, j]))
                        cr2 = np.dot(cr2, Rx[i + 2, j])
                        cr2 = reshape(
                            cr2, (rx[i + 1, j], n[i + 1] * ry[i + 2]))
                        cr2 = cr2[:, ind2]
                        curbl[:, j] = reshape(
                            np.dot(cr1, cr2), (ry[i] * n[i] * rkick,))
                    # call the function
                    uk = funs(curbl)
                    uk = reshape(uk, (ry[i], n[i] * rkick * d2))
                    uk = np.linalg.solve(Ry[i], uk)
                    uk = reshape(uk, (ry[i] * n[i], rkick * d2))
                    if pcatype == 'svd':
                        uk, sk, vk = np.linalg.svd(uk, full_matrices=False)
                        vk = np.conj(np.transpose(vk))
                        uk = uk[:, :min(kickrank, uk.shape[1])]
                    else:
                        # uk = uchol(np.transpose(uk), kickrank + 1) # TODO
                        uk = uk[:, :max(uk.shape[1] - kickrank + 1, 1):-1]
                else:
                    uk = np.random.rand(ry[i] * n[i], kickrank)
                u, rv = np.linalg.qr(np.concatenate((u, uk), axis=1))
                radd = uk.shape[1]
            v = np.concatenate(
                (v, np.zeros((ry[i + 1] * d2, radd), dtype=dtype)), axis=1)
            v = np.dot(rv, np.conj(np.transpose(v)))
            r = u.shape[1]

            cr2 = cry[i + 1]
            cr2 = reshape(cr2, (ry[i + 1], n[i + 1] * ry[i + 2]))
            v = reshape(v, (r * ry[i + 1], d2))
            v = reshape(np.transpose(v), (d2 * r, ry[i + 1]))
            v = np.dot(v, cr2)

            ry[i + 1] = r

            u = reshape(u, (ry[i], n[i], r))
            v = reshape(v, (d2, r, n[i + 1], ry[i + 2]))

            cry[i] = u
            cry[i + 1] = v

            Ry[i + 1] = np.dot(Ry[i], reshape(u, (ry[i], n[i] * ry[i + 1])))
            Ry[i + 1] = reshape(Ry[i + 1], (ry[i] * n[i], ry[i + 1]))
            curind = maxvol(Ry[i + 1])
            Ry[i + 1] = Ry[i + 1][curind, :]
            for j in range(nx):
                Rx[i + 1, j] = reshape(crx[i, j],
                                       (rx[i, j], n[i] * rx[i + 1, j]))
                Rx[i + 1, j] = np.dot(Rx[i, j], Rx[i + 1, j])
                Rx[i + 1, j] = reshape(Rx[i + 1, j],
                                       (ry[i] * n[i], rx[i + 1, j]))
                Rx[i + 1, j] = Rx[i + 1, j][curind, :]
        elif dirn < 0 and i > 0:
            u = np.dot(u[:, :r], np.diag(s[:r]))
            v = np.conj(v[:, :r])

            radd = 0
            rv = 1
            if not last_sweep and kickrank > 0:
                if kicktype == 'amr-two':
                    # compute the X superblocks
                    ind2 = np.unique(np.random.randint(
                        0, ry[i - 1] * n[i - 1], ry[i]))
                    rkick = len(ind2)
                    curbl = np.zeros(
                        (rkick * n[i] * ry[i + 1], nx), dtype=dtype)
                    for j in range(nx):
                        cr1 = reshape(
                            crx[i, j], (rx[i, j] * n[i], rx[i + 1, j]))
                        cr1 = np.dot(cr1, Rx[i + 1, j])
                        cr1 = reshape(cr1, (rx[i, j], n[i] * ry[i + 1]))
                        cr2 = reshape(
                            crx[i - 1, j], (rx[i - 1, j], n[i - 1] * rx[i, j]))
                        cr2 = np.dot(Rx[i - 1, j], cr2)
                        cr2 = reshape(cr2, (ry[i - 1] * n[i - 1], rx[i, j]))
                        cr2 = cr2[ind2, :]
                        curbl[:, j] = reshape(
                            np.dot(cr2, cr1), (rkick * n[i] * ry[i + 1],))
                    # calling the function
                    uk = funs(curbl)
                    uk = reshape(uk, (rkick * n[i] * ry[i + 1], d2))
                    uk = reshape(np.transpose(
                        uk), (d2 * rkick * n[i], ry[i + 1]))
                    uk = np.transpose(np.linalg.solve(
                        np.transpose(Ry[i + 1]), np.transpose(uk)))
                    uk = reshape(uk, (d2 * rkick, n[i] * ry[i + 1]))
                    if pcatype == 'svd':
                        vk, sk, uk = np.linalg.svd(uk, full_matrices=False)
                        uk = np.conj(np.transpose(uk))
                        # TODO: refactor
                        uk = uk[:, :min(kickrank, uk.shape[1])]
                    else:
                        # uk = uchol(uk, kickrank + 1) # TODO
                        uk = uk[:, :max(uk.shape[1] - kickrank + 1, 1):-1]
                else:
                    uk = np.random.rand(n[i] * ry[i + 1], kickrank)
                v, rv = np.linalg.qr(np.concatenate((v, uk), axis=1))
                radd = uk.shape[1]
            u = np.concatenate(
                (u, np.zeros((d2 * ry[i], radd), dtype=dtype)), axis=1)
            u = np.dot(u, np.transpose(rv))
            r = v.shape[1]
            cr2 = cry[i - 1]
            cr2 = reshape(cr2, (ry[i - 1] * n[i - 1], ry[i]))
            u = reshape(u, (d2, ry[i] * r))
            u = reshape(np.transpose(u), (ry[i], r * d2))
            u = np.dot(cr2, u)

            u = reshape(u, (ry[i - 1] * n[i - 1] * r, d2))
            u = reshape(np.transpose(u), (d2, ry[i - 1], n[i - 1], r))
            v = reshape(np.transpose(v), (r, n[i], ry[i + 1]))

            ry[i] = r
            cry[i - 1] = u
            cry[i] = v

            Ry[i] = np.dot(reshape(v, (ry[i] * n[i], ry[i + 1])), Ry[i + 1])
            Ry[i] = reshape(Ry[i], (ry[i], n[i] * ry[i + 1]))
            curind = maxvol(np.transpose(Ry[i]))
            Ry[i] = Ry[i][:, curind]
            for j in range(nx):
                Rx[i, j] = reshape(crx[i, j], (rx[i, j] * n[i], rx[i + 1, j]))
                Rx[i, j] = np.dot(Rx[i, j], Rx[i + 1, j])
                Rx[i, j] = reshape(Rx[i, j], (rx[i, j], n[i] * ry[i + 1]))
                Rx[i, j] = Rx[i, j][:, curind]
        elif dirn > 0 and i == d - 1:
            newy = np.dot(np.dot(u[:, :r], np.diag(s[:r])),
                          np.conj(np.transpose(v[:, :r])))
            newy = reshape(newy, (ry[i] * n[i] * ry[i + 1], d2))
            cry[i] = reshape(np.transpose(newy), (d2, ry[i], n[i], ry[i + 1]))
        elif dirn < 0 and i == 0:
            newy = np.dot(np.dot(u[:, :r], np.diag(s[:r])),
                          np.conj(np.transpose(v[:, :r])))
            newy = reshape(newy, (d2, ry[i], n[i], ry[i + 1]))
            cry[i] = newy

        i = i + dirn
        cur_order[order_index] = cur_order[order_index] - dirn
        if cur_order[order_index] == 0:
            order_index = order_index + 1
            if verb > 0:
                print '=multifuncrs= sweep %d{%d}, max_dy: %3.3e, erank: %g' % (swp, order_index, max_dy,
                                                                                math.sqrt(np.dot(ry[:d], n * ry[1:]) / np.sum(n)))

            if last_sweep:
                break
            if max_dy < eps and dirn < 0:
                last_sweep = True
                kickrank = 0

            if order_index >= len(cur_order):
                cur_order = copy.copy(block_order)
                order_index = 0
                if last_sweep:
                    cur_order = [d - 1]

                max_dy = 0
                swp = swp + 1

            dirn = int(math.copysign(1, cur_order[order_index]))
            i = i + dirn

    cry[d - 1] = np.transpose(cry[d - 1][:, :, :, 0], [1, 2, 0])
    y = tt.tensor.from_list(cry)
    return y
