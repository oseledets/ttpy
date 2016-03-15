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


def multifuncrs2(X, funs, eps=1e-6,
                 nswp=10,
                 rmax=9999999,
                 verb=1,
                 kickrank=5,
                 kickrank2=0,
                 d2=1,
                 eps_exit=None,
                 y0=None,
                 do_qr=False,
                 restart_it=0):

    dtype = np.float64
    if len(filter(lambda x: x.is_complex, X)) > 0:
        dtype = np.complex128

    if eps_exit is None:
        eps_exit = eps
    nx = len(X)
    d = X[0].d
    n = X[0].n
    rx = np.transpose(np.array([ttx.r for ttx in X]))
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
    wasrand = False
    if y0 is None:
        ry = d2 * np.ones((d + 1,), dtype=np.int32)
        ry[0] = 1
        y = tt.rand(n, d, ry)
        wasrand = True
    else:  # Initial guess available
        y = y0.copy()
        ry = y.r.copy()
    # Error vector
    z = tt.rand(n, d, kickrank)
    rz = z.r
    z = tt.tensor.to_list(z)
    ry = y.r
    cry = tt.tensor.to_list(y)
    # Interface matrices - for solution
    one_arr = np.ones((1, 1), dtype=dtype)
    Ry = np.zeros((d + 1, ), dtype=np.object)
    Ry[0] = one_arr
    Ry[d] = one_arr
    Rx = np.zeros((d + 1, nx), dtype=np.object)
    Rx[0, :] = np.ones(nx, dtype=dtype)
    Rx[d, :] = np.ones(nx, dtype=dtype)
    Ryz = np.zeros((d + 1, ), dtype=np.object)
    Ryz[0] = one_arr
    Ryz[d] = one_arr
    Rz = np.zeros((d + 1, ), dtype=np.object)
    Rz[0] = one_arr
    Rz[d] = one_arr
    Rxz = np.zeros((d + 1, nx), dtype=np.object)
    Rxz[0, :] = np.ones(nx, dtype=dtype)
    Rxz[d, :] = np.ones(nx, dtype=dtype)
    block_order = [+d, -d]
    # orth
    for i in range(0, d - 1):
        cr = cry[i].copy()
        cr = reshape(cr, (ry[i] * n[i], ry[i + 1]))
        cr, rv = np.linalg.qr(cr)
        cr2 = cry[i + 1].copy()
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
        # Interface matrices for X
        for j in range(0, nx):
            Rx[i + 1, j] = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
            Rx[i + 1, j] = np.dot(Rx[i, j], Rx[i + 1, j])
            Rx[i + 1, j] = reshape(Rx[i + 1, j], (ry[i] * n[i], rx[i + 1, j]))
            Rx[i + 1, j] = Rx[i + 1, j][curind, :]
        # Error for kick
        if kickrank > 0:
            crz = z[i]
            crz = reshape(crz, (rz[i] * n[i], rz[i + 1]))
            crz, rv = np.linalg.qr(crz)
            cr2 = z[i + 1]
            cr2 = reshape(cr2, (rz[i + 1], n[i + 1] * rz[i + 2]))
            cr2 = np.dot(rv, cr2)
            rz[i + 1] = crz.shape[1]
            crz = reshape(crz, (rz[i], n[i], rz[i + 1]))
            z[i + 1] = reshape(cr2, (rz[i + 1], n[i + 1], rz[i + 2]))
            z[i] = crz
            # Interfaces for error
            Rz[i + 1] = np.dot(Rz[i], reshape(crz, [rz[i], n[i] * rz[i + 1]]))
            Rz[i + 1] = reshape(Rz[i + 1], [rz[i] * n[i], rz[i + 1]])
            Ryz[i + 1] = np.dot(Ryz[i], reshape(cr, [ry[i], n[i] * ry[i + 1]]))
            Ryz[i + 1] = reshape(Ryz[i + 1], [rz[i] * n[i], ry[i + 1]])
            # Pick random initial indices
            curind = np.random.permutation(n[i] * rz[i])[:rz[i + 1]]
            Ryz[i + 1] = Ryz[i + 1][curind, :]
            Rz[i + 1] = Rz[i + 1][curind, :]
            # Interface matrices for X
            for j in range(0, nx):
                Rxz[i + 1, j] = reshape(crx[i, j],
                                        (rx[i, j], n[i] * rx[i + 1, j]))
                Rxz[i + 1, j] = np.dot(Rxz[i, j], Rxz[i + 1, j])
                Rxz[i + 1, j] = reshape(Rxz[i + 1, j],
                                        (rz[i] * n[i], rx[i + 1, j]))
                Rxz[i + 1, j] = Rxz[i + 1, j][curind, :]
    d2 = ry[d]
    ry[d] = 1
    cry[d - 1] = np.transpose(cry[d - 1], [2, 0, 1])  # permute
    swp = 1
    max_dy = 0.0
    cur_order = copy.copy(block_order)
    order_index = 1
    i = d - 1
    # can't use 'dir' identifier in python
    dirn = int(math.copysign(1, cur_order[order_index]))
    # DMRG sweeps
    while swp <= nswp or dirn > 0:
        oldy = reshape(cry[i].copy(), (d2 * ry[i] * n[i] * ry[i + 1], ))
        # Compute X superblocks
        curbl = np.zeros((ry[i] * n[i] * ry[i + 1], nx), dtype)
        for j in range(0, nx):
            cr = reshape(crx[i, j].copy(), (rx[i, j], n[i] * rx[i + 1, j]))
            cr = np.dot(Rx[i, j], cr)
            cr = reshape(cr, (ry[i] * n[i], rx[i + 1, j]))
            cr = np.dot(cr, Rx[i + 1, j])
            curbl[:, j] = cr.flatten('F')
        newy = funs(curbl)
        # multiply with inverted Ry
        newy = reshape(newy, (ry[i], n[i] * ry[i + 1] * d2))
        newy = np.linalg.solve(Ry[i], newy)  # y = R \ y
        newy = reshape(newy, (ry[i] * n[i] * ry[i + 1], d2))
        newy = reshape(np.transpose(newy), (d2 * ry[i] * n[i], ry[i + 1]))
        newy = np.transpose(np.linalg.solve(
            np.transpose(Ry[i + 1]), np.transpose(newy)))  # y=y/R
        newy = reshape(newy, (d2 * ry[i] * n[i] * ry[i + 1],))
        try:
            dy = np.linalg.norm(newy - oldy) / np.linalg.norm(newy)
        except ZeroDivisionError:
            print 'Bad initial indices, the solution is exactly zero. Restarting'
            return
        max_dy = max(max_dy, dy)
        # truncation
        if dirn > 0:  # left-to-right
            newy = reshape(newy, (d2, ry[i] * n[i] * ry[i + 1]))
            newy = reshape(np.transpose(newy), (ry[i] * n[i], ry[i + 1] * d2))
        else:
            newy = reshape(newy, (d2 * ry[i], n[i] * ry[i + 1]))
        if kickrank >= 0:
            try:
                u, s, v = np.linalg.svd(newy, full_matrices=False)
            except:
                tmp = np.array(
                    np.random.randn(
                        newy.shape[1],
                        newy.shape[1]),
                    dtype=dtype)
                tmp, ru_tmp = np.linalg.qr(tmp)
                u, s, v = np.linalg.svd(np.dot(newy, tmp))
                # u * s * v = A * tmp
                v = np.dot(v, np.conj(tmp).T)
            v = np.conj(np.transpose(v))
            r = my_chop2(s, eps / math.sqrt(d) * np.linalg.norm(s))
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
            print '=multifuncrs2=   block %d{%d}, dy: %3.3e, r: %d' % (i, dirn, dy, r)
        #Kicks and interfaces
        if dirn > 0 and i < d - 1:
            u = u[:, :r]
            v = np.dot(v[:, :r], np.diag(s[:r]))
            # kick
            radd = 0
            rv = 1
            if kickrank > 0:
                # Compute the function at residual indices
                curbl_y = np.zeros((ry[i] * n[i] * rz[i + 1], nx), dtype=dtype)
                curbl_z = np.zeros((rz[i] * n[i] * rz[i + 1], nx), dtype=dtype)
                for j in xrange(nx):
                    # For kick
                    cr = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                    cr = np.dot(Rx[i, j], cr)
                    cr = reshape(cr, (ry[i] * n[i], rx[i + 1, j]))
                    cr = np.dot(cr, Rxz[i + 1, j])
                    curbl_y[:, j] = cr.flatten('F')
                    # For z update
                    cr = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                    cr = np.dot(Rxz[i, j], cr)
                    cr = reshape(cr, (rz[i] * n[i], rx[i + 1, j]))
                    cr = np.dot(cr, Rxz[i + 1, j])
                    curbl_z[:, j] = cr.flatten('F')
                # Call the function
                zy = reshape(funs(curbl_y), (-1, d2))
                zz = reshape(funs(curbl_z), (-1, d2))
                # Assemble y at z indices (sic!) and subtract
                dzy = reshape(np.dot(u, v.T), (ry[i] * n[i] * ry[i + 1], d2))
                dzy = reshape(dzy.T, (d2 * ry[i] * n[i], ry[i + 1]))
                # Cast dzy from core items to samples at right indices
                dzy = np.dot(dzy, Ryz[i + 1])
                dzy = reshape(dzy, (d2, ry[i] * n[i] * rz[i + 1]))
                dzy = dzy.T
                # zy still requires casting from samples to core entities
                zy = reshape(zy, (ry[i], n[i] * rz[i + 1] * d2))
                zy = np.linalg.solve(Ry[i], zy)
                zy = reshape(zy, (ry[i] * n[i] * rz[i + 1], d2))
                zy = zy - dzy
                dzy = reshape(dzy, (ry[i], n[i] * rz[i + 1] * d2))
                dzy = np.dot(Ryz[i], dzy)
                # Sample from both sizes
                dzy = reshape(dzy, (rz[i] * n[i] * rz[i + 1], d2))
                zz = zz - dzy
                # Interpolate all remaining samples into core elements
                zy = reshape(zy.T, (d2 * ry[i] * n[i], rz[i + 1]))
                zy = np.linalg.solve(Rz[i + 1].T, zy.T).T
                zy = reshape(zy, (d2, ry[i] * n[i] * rz[i + 1]))
                zy = reshape(zy.T, (ry[i] * n[i], rz[i + 1] * d2))
                # SVD to eliminate d2 and possibly overestimated rz
                zy, sz, vz = np.linalg.svd(zy, full_matrices=False)
                zy = zy[:, :min(kickrank, zy.shape[1])]
                # For z update
                zz = reshape(zz, (rz[i], n[i] * rz[i + 1] * d2))
                zz = np.linalg.solve(Rz[i], zz)
                zz = reshape(zz, (rz[i] * n[i] * rz[i + 1], d2))
                zz = reshape(zz.T, (d2 * rz[i] * n[i], rz[i + 1]))
                zz = np.linalg.solve(Rz[i + 1].T, zz.T).T
                zz = reshape(zz, (d2, rz[i] * n[i] * rz[i + 1]))
                zz = reshape(zz.T, (rz[i] * n[i], rz[i + 1] * d2))
                zz, sz, vz = np.linalg.svd(zz, full_matrices=False)
                zz = zz[:, :min(kickrank, zz.shape[1])]
                # Second random kick rank
                zz = np.hstack((zz, np.random.randn(rz[i] * n[i], kickrank2)))
                u, rv = np.linalg.qr(np.hstack((u, zy)))
                radd = zy.shape[1]
            v = np.hstack((v, np.zeros((ry[i + 1] * d2, radd), dtype=dtype)))
            v = np.dot(rv, v.T)
            r = u.shape[1]
            cr2 = cry[i + 1].copy()
            cr2 = reshape(cr2, (ry[i + 1], n[i + 1] * ry[i + 2]))
            v = reshape(v, (r * ry[i + 1], d2))
            v = reshape(v.T, (d2 * r, ry[i + 1]))
            v = np.dot(v, cr2)
            ry[i + 1] = r
            u = reshape(u, (ry[i], n[i], r))
            v = reshape(v, (d2, r, n[i + 1], ry[i + 2]))
            # Stuff back
            cry[i] = u
            cry[i + 1] = v
            # Update kick
            if kickrank > 0:
                zz, rv = np.linalg.qr(zz)
                rz[i + 1] = zz.shape[1]
                z[i] = reshape(zz, (rz[i], n[i], rz[i + 1]))
            # z[i+1] is recomputed from scratch we do not need it now
            # Compute left interface matrices
            # Interface matrix for Y
            Ry[i + 1] = np.dot(Ry[i], reshape(u, (ry[i], n[i] * ry[i + 1])))
            Ry[i + 1] = reshape(Ry[i + 1], (ry[i] * n[i], ry[i + 1]))
            curind = maxvol(Ry[i + 1])
            Ry[i + 1] = Ry[i + 1][curind, :]
            # Interface matrices for X
            for j in xrange(nx):
                Rx[i + 1, j] = reshape(crx[i, j],
                                       (rx[i, j], n[i] * rx[i + 1, j]))
                Rx[i + 1, j] = np.dot(Rx[i, j], Rx[i + 1, j])
                Rx[i + 1, j] = reshape(Rx[i + 1, j],
                                       (ry[i] * n[i], rx[i + 1, j]))
                Rx[i + 1, j] = Rx[i + 1, j][curind, :]
            # for kick
            if kickrank > 0:
                Ryz[i + 1] = np.dot(Ryz[i], reshape(u,
                                                    (ry[i], n[i] * ry[i + 1])))
                Ryz[i + 1] = reshape(Ryz[i + 1], (rz[i] * n[i], ry[i + 1]))
                Rz[i + 1] = np.dot(Rz[i], reshape(zz,
                                                  (rz[i], n[i] * rz[i + 1])))
                Rz[i + 1] = reshape(Rz[i + 1], (rz[i] * n[i], rz[i + 1]))
                curind = maxvol(Rz[i + 1])
                Ryz[i + 1] = Ryz[i + 1][curind, :]
                Rz[i + 1] = Rz[i + 1][curind, :]
                # Interface matrices for X
                for j in xrange(nx):
                    Rxz[i + 1, j] = reshape(crx[i, j],
                                            (rx[i, j], n[i] * rx[i + 1, j]))
                    Rxz[i + 1, j] = np.dot(Rxz[i, j], Rxz[i + 1, j])
                    Rxz[i + 1, j] = reshape(Rxz[i + 1, j],
                                            (rz[i] * n[i], rx[i + 1, j]))
                    Rxz[i + 1, j] = Rxz[i + 1, j][curind, :]
        elif dirn < 0 and i > 0:  # Right to left
            u = np.dot(u[:, :r], np.diag(s[:r]))
            v = np.conj(v[:, :r])
            # kick
            radd = 0
            rv = 0
            if kickrank > 0:
                # AMEN kick
                # Compute the function at residual indices
                curbl_y = np.zeros((rz[i] * n[i] * ry[i + 1], nx), dtype=dtype)
                curbl_z = np.zeros((rz[i] * n[i] * rz[i + 1], nx), dtype=dtype)
                for j in xrange(nx):
                    cr = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                    cr = np.dot(Rxz[i, j], cr)
                    cr = reshape(cr, (rz[i] * n[i], rx[i + 1, j]))
                    cr = np.dot(cr, Rx[i + 1, j])
                    curbl_y[:, j] = cr.flatten('F')
                    # for z update
                    cr = reshape(crx[i, j], (rx[i, j], n[i] * rx[i + 1, j]))
                    cr = np.dot(Rxz[i, j], cr)
                    cr = reshape(cr, (rz[i] * n[i], rx[i + 1, j]))
                    cr = np.dot(cr, Rxz[i + 1, j])
                    curbl_z[:, j] = cr.flatten('F')
                # Call the function
                zy = reshape(funs(curbl_y), (-1, d2))
                zz = reshape(funs(curbl_z), (-1, d2))
                # Assemble y at z indices (sic!) and subtract
                dzy = reshape(np.dot(u, v.T), (ry[i], n[i] * ry[i + 1] * d2))
                dzy = np.dot(Ryz[i], dzy)
                dzy = reshape(dzy, (rz[i] * n[i] * ry[i + 1], d2))
                # zy still requires casting from samples to core entries
                zy = zy.T
                zy = reshape(zy, (d2 * rz[i] * n[i], ry[i + 1]))
                zy = np.linalg.solve(Ry[i + 1].T, zy.T).T
                zy = reshape(zy, (d2, rz[i] * n[i] * ry[i + 1]))
                zy = zy.T
                zy = zy - dzy
                dzy = reshape(dzy.T, (d2 * rz[i] * n[i], ry[i + 1]))
                dzy = np.dot(dzy, Ryz[i + 1])
                dzy = reshape(dzy, (d2, rz[i] * n[i] * rz[i + 1]))
                zz = zz - dzy.T
                # Cast sample indices to core elements
                # ...for kick
                zy = reshape(zy, (rz[i], n[i] * ry[i + 1] * d2))
                zy = np.linalg.solve(Rz[i], zy)
                zy = reshape(zy, (rz[i] * n[i] * ry[i + 1], d2))
                zy = zy.T
                zy = reshape(zy, (d2 * rz[i], n[i] * ry[i + 1]))
                zu, zs, zy = np.linalg.svd(zy, full_matrices=False)
                zy = zy[:min(kickrank, zy.shape[0]), :]
                zy = zy.T
                # ...for z update
                zz = reshape(zz, (rz[i], n[i] * rz[i + 1] * d2))
                zz = np.linalg.solve(Rz[i], zz)
                zz = reshape(zz, (rz[i] * n[i] * rz[i + 1], d2))
                zz = reshape(zz.T, (d2 * rz[i] * n[i], rz[i + 1]))
                zz = np.linalg.solve(Rz[i + 1].T, zz.T).T
                zz = reshape(zz, (d2 * rz[i], n[i] * rz[i + 1]))
                zu, zs, zz = np.linalg.svd(zz, full_matrices=False)
                zz = zz[:min(kickrank, zz.shape[0]), :]
                zz = zz.T
                zz = np.hstack(
                    (zz, np.random.randn(n[i] * rz[i + 1], kickrank2)))
                v, rv = np.linalg.qr(np.hstack((v, zy)))
                radd = zy.shape[1]
                u = np.hstack((u, np.zeros((d2 * ry[i], radd), dtype=dtype)))
                u = np.dot(u, rv.T)
            r = v.shape[1]
            cr2 = cry[i - 1].copy()
            cr2 = reshape(cr2, (ry[i - 1] * n[i - 1], ry[i]))
            u = reshape(u, (d2, ry[i] * r))
            u = reshape(u.T, (ry[i], r * d2))
            u = np.dot(cr2, u)
            u = reshape(u, (ry[i - 1] * n[i - 1] * r, d2))
            u = reshape(u.T, (d2, ry[i - 1], n[i - 1], r))
            v = reshape(v.T, (r, n[i], ry[i + 1]))
            # Stuff back
            ry[i] = r
            cry[i - 1] = u
            cry[i] = v
            # kick
            if kickrank > 0:
                zz, rv = np.linalg.qr(zz)
                rz[i] = zz.shape[1]
                zz = reshape(zz.T, (rz[i], n[i], rz[i + 1]))
                z[i] = zz
            # z[i-1] is recomputed from scratch we do not need it
            # Recompute left interface matrices
            # Interface matrix for Y
            Ry[i] = np.dot(reshape(v, (ry[i] * n[i], ry[i + 1])), Ry[i + 1])
            Ry[i] = reshape(Ry[i], (ry[i], n[i] * ry[i + 1]))
            curind = maxvol(Ry[i].T)
            Ry[i] = Ry[i][:, curind]
            # Interface matrices for X
            for j in xrange(nx):
                Rx[i, j] = reshape(crx[i, j], (rx[i, j] * n[i], rx[i + 1, j]))
                Rx[i, j] = np.dot(Rx[i, j], Rx[i + 1, j])
                Rx[i, j] = reshape(Rx[i, j], (rx[i, j], n[i] * ry[i + 1]))
                Rx[i, j] = Rx[i, j][:, curind]
            # for kick
            if kickrank > 0:
                Rz[i] = np.dot(
                    reshape(zz, (rz[i] * n[i], rz[i + 1])), Rz[i + 1])
                Rz[i] = reshape(Rz[i], (rz[i], n[i] * rz[i + 1]))
                Ryz[i] = np.dot(
                    reshape(v, (ry[i] * n[i], ry[i + 1])), Ryz[i + 1])
                Ryz[i] = reshape(Ryz[i], (ry[i], n[i] * rz[i + 1]))
                curind = maxvol(Rz[i].T)
                Ryz[i] = Ryz[i][:, curind]
                Rz[i] = Rz[i][:, curind]
                # Interface matrices for X
                for j in xrange(nx):
                    Rxz[i, j] = reshape(
                        crx[i, j], (rx[i, j] * n[i], rx[i + 1, j]))
                    Rxz[i, j] = np.dot(Rxz[i, j], Rxz[i + 1, j])
                    Rxz[i, j] = reshape(
                        Rxz[i, j], (rx[i, j], n[i] * rz[i + 1]))
                    Rxz[i, j] = Rxz[i, j][:, curind]
        elif dirn > 0 and i == d - 1:
            # Just stuff back the last core
            newy = np.dot(u[:, :r], np.dot(
                np.diag(s[:r]), np.conj(v[:, :r].T)))
            newy = reshape(newy, (ry[i] * n[i] * ry[i + 1], d2))
            cry[i] = reshape(newy.T, (d2, ry[i], n[i], ry[i + 1]))
        elif dirn < 0 and i == 0:
            newy = np.dot(u[:, :r], np.dot(
                np.diag(s[:r]), np.conj(v[:, :r].T)))
            newy = reshape(newy, (d2, ry[i], n[i], ry[i + 1]))
            cry[i] = newy
        i += dirn
        # Reversing, residue check, etc
        cur_order[order_index] = cur_order[order_index] - dirn
        # New direction
        if cur_order[order_index] == 0:
            order_index = order_index + 1
            if verb > 0:
                print '=multifuncrs= sweep %d{%d}, max_dy: %3.3e, erank: %g' % (swp, order_index, max_dy,
                                                                                math.sqrt(np.dot(ry[:d], n * ry[1:]) / np.sum(n)))
            if max_dy < eps_exit and dirn > 0:
                break
            if order_index >= len(cur_order):  # New global sweep
                cur_order = copy.copy(block_order)
                order_index = 0
                max_dy = 0
                swp = swp + 1
            dirn = int(math.copysign(1, cur_order[order_index]))
            i = i + dirn
    cry[d - 1] = np.transpose(cry[d - 1][:, :, :, 0], [1, 2, 0])
    y = tt.tensor.from_list(cry)
    return y
