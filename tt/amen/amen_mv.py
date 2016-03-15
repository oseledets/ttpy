import tt as _tt
import numpy as _np
from scipy.sparse import spdiags as _spdiags
from scipy.linalg import cholesky as _cholesky


def _reshape(a, shape):
    return _np.reshape(a, shape, order='F')


def _tconj(a):
    return a.T.conjugate()


def _my_chop2(sv, eps):  # from ttpy/multifuncr.py
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = _np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return _np.amin(ff)


def _svdgram(A, tol=None, tol2=1e-7):
    ''' Highly experimental acceleration of SVD/QR using Gram matrix.
        Use with caution for m>>n only!
        function [u,s,r]=_svdgram(A,[tol])
        u is the left singular factor of A,
        s is the singular values (vector!),
        r has the meaning of diag(s)*v'.
        if tol is given, performs the truncation with Fro-threshold.
    '''

    R2 = _np.dot(_tconj(A), A)
    [u, s, vt] = _np.linalg.svd(R2, full_matrices=False)
    u = _np.dot(A, _tconj(vt))
    s = (u**2).sum(axis=0)
    s = s ** 0.5
    if tol is not None:
        p = _my_chop2(s, _np.linalg.norm(s) * tol)
        u = u[:, :p]
        s = s[:p]
        vt = vt[:p, :]

    tmp = _spdiags(1. / s, 0, len(s), len(s)).todense()
    tmp = _np.array(tmp)
    u = _np.dot(u, tmp)
    r = _np.dot(_np.diag(s), vt)

    # Run chol for reortogonalization.
    # It will stop if the matrix will be singular.
    # Fortunately, it means rank truncation with eps in our business.

    if (s[0] / s[-1] > 1. / tol2):
        p = 1
        while (p > 0):
            R2 = _np.dot(_tconj(u), u)
            #[u_r2, s_r2, vt_r2] = _np.linalg.svd(R2) # in matlab [R, p] = chol(a) - here is a *dirty* patch
            #p = s_r2[s_r2 > eps].size
            #R2 = R2[:p, :p]
            R = _cholesky(R2, lower=True)
            if (p > 0):
                u = u[:, :p]
                s = s[:p]
                r = r[:p, :]
            iR = _np.linalg.inv(R)
            u = _np.dot(u, iR)
            r = _np.dot(R, r)
    return u, s, r


def amen_mv(A, x, tol, y=None, z=None, nswp=20, kickrank=4,
            kickrank2=0, verb=True, init_qr=True, renorm='direct', fkick=False):
    '''
       Approximate the matrix-by-vector via the AMEn iteration
       [y,z]=amen_mv(A, x, tol, varargin)
       Attempts to approximate the y = A*x
       with accuracy TOL using the AMEn+ALS iteration.
       Matrix A has to be given in the TT-format, right-hand side x should be
       given in the TT-format also.

       Options are provided in form
       'PropertyName1',PropertyValue1,'PropertyName2',PropertyValue2 and so
       on. The parameters are set to default (in brackets in the following)
       The list of option names and default values are:
           o y0 - initial approximation to Ax [rand rank-2]
           o nswp - maximal number of sweeps [20]
           o verb - verbosity level, 0-silent, 1-sweep info, 2-block info [1]
           o kickrank - compression rank of the error,
             i.e. enrichment size [3]
           o init_qr - perform QR of the input (save some time in ts, etc) [true]
           o renorm - Orthog. and truncation methods: direct (svd,qr) or gram
             (apply svd to the gram matrix, faster for m>>n) [direct]
           o fkick - Perform solution enrichment during forward sweeps [false]
             (rather questionable yet; false makes error higher, but "better
             structured": it does not explode in e.g. subsequent matvecs)
           o z0 - initial approximation to the error Ax-y [rand rank-kickrank]


    ********
       For description of adaptive ALS please see
       Sergey V. Dolgov, Dmitry V. Savostyanov,
       Alternating minimal energy methods for linear systems in higher dimensions.
       Part I: SPD systems, http://arxiv.org/abs/1301.6068,
       Part II: Faster algorithm and application to nonsymmetric systems, http://arxiv.org/abs/1304.1222

       Use {sergey.v.dolgov, dmitry.savostyanov}@gmail.com for feedback
    ********
    '''

    if renorm is 'gram':
        print "Not implemented yet. Renorm is switched to 'direct'"
        renorm = 'direct'

    if isinstance(x, _tt.vector):
        d = x.d
        m = x.n
        rx = x.r
        x = _tt.vector.to_list(x)
        vectype = 1  # tt_tensor
    elif isinstance(x, list):
        d = len(x)
        m = _np.zeros(d)
        rx = _np.ones(d + 1)
        for i in xrange(d):
            [_, m[i], rx[i + 1]] = x[i].shape
        vectype = 0  # cell
    else:
        raise Exception('x: use tt.tensor or list of cores as numpy.arrays')

    if isinstance(A, _tt.matrix):
        n = A.n
        ra = A.tt.r
        A = _tt.matrix.to_list(A)
        # prepare A for fast ALS-mv
        for i in xrange(d):
            A[i] = _reshape(A[i], (ra[i] * n[i], m[i] * ra[i + 1]))
        atype = 1  # tt_matrix
    # Alternative: A is a cell of cell: sparse canonical format
    elif isinstance(A, list):
        n = _np.zeros(d)
        for i in xrange(d):
            n[i] = A[i][0].shape[0]
        ra = len(A[0])
        atype = 0  # cell
    else:
        raise Exception('A: use tt.matrix or list of cores as numpy.arrays')

    if y is None:
        y = _tt.rand(n, d, 2)
        y = _tt.vector.to_list(y)
    else:
        if isinstance(y, _tt.vector):
            y = _tt.vector.to_list(y)

    ry = _np.ones(d + 1)
    for i in range(d):
        ry[i + 1] = y[i].shape[2]

    if (kickrank + kickrank2 > 0):
        if z is None:
            z = _tt.rand(n, d, kickrank + kickrank2)
            rz = z.r
            z = _tt.vector.to_list(z)
        else:
            if isinstance(z, _tt.vector):
                z = _tt.vector.to_list(z)
            rz = _np.ones(d + 1)
            for i in range(d):
                rz[i + 1] = z[i].shape[2]

        phizax = [None] * (d + 1)  # cell(d+1,1);
        if (atype == 1):
            phizax[0] = _np.ones((1, 1, 1))  # 1
            phizax[d] = _np.ones((1, 1, 1))  # 1
        else:
            phizax[0] = _np.ones((1, ra))  # 33
            phizax[d] = _np.ones((1, ra))
        phizy = [None] * (d + 1)
        phizy[0] = _np.ones((1))  # , 1))
        phizy[d] = _np.ones((1))  # , 1))

    phiyax = [None] * (d + 1)
    if (atype == 1):
        phiyax[0] = _np.ones((1, 1, 1))  # 1
        phiyax[d] = _np.ones((1, 1, 1))  # 1
    else:
        phiyax[0] = _np.ones((1, ra))  # 3
        phiyax[d] = _np.ones((1, ra))

    nrms = _np.ones(d)

    # Initial ort
    for i in range(d - 1):
        if init_qr:
            cr = _reshape(y[i], (ry[i] * n[i], ry[i + 1]))
            if (renorm is 'gram') and (ry[i] * n[i] > 5 * ry[i + 1]):
                [cr, s, R] = _svdgram(cr)
            else:
                [cr, R] = _np.linalg.qr(cr)
            nrmr = _np.linalg.norm(R)  # , 'fro')
            if (nrmr > 0):
                R = R / nrmr
            cr2 = _reshape(y[i + 1], (ry[i + 1], n[i + 1] * ry[i + 2]))
            cr2 = _np.dot(R, cr2)
            ry[i + 1] = cr.shape[1]
            y[i] = _reshape(cr, (ry[i], n[i], ry[i + 1]))
            y[i + 1] = _reshape(cr2, (ry[i + 1], n[i + 1], ry[i + 2]))

        [phiyax[i + 1], nrms[i]
         ] = _compute_next_Phi(phiyax[i], y[i], x[i], 'lr', A[i])

        if (kickrank + kickrank2 > 0):
            cr = _reshape(z[i], (rz[i] * n[i], rz[i + 1]))
            if (renorm == 'gram') and (rz[i] * n[i] > 5 * rz[i + 1]):
                [cr, s, R] = _svdgram(cr)
            else:
                [cr, R] = _np.linalg.qr(cr)
            nrmr = _np.linalg.norm(R)  # , 'fro')
            if (nrmr > 0):
                R = R / nrmr
            cr2 = _reshape(z[i + 1], (rz[i + 1], n[i + 1] * rz[i + 2]))
            cr2 = _np.dot(R, cr2)
            rz[i + 1] = cr.shape[1]
            z[i] = _reshape(cr, (rz[i], n[i], rz[i + 1]))
            z[i + 1] = _reshape(cr2, (rz[i + 1], n[i + 1], rz[i + 2]))
            phizax[
                i +
                1] = _compute_next_Phi(
                phizax[i],
                z[i],
                x[i],
                'lr',
                A[i],
                nrms[i],
                return_norm=False)
            phizy[
                i +
                1] = _compute_next_Phi(
                phizy[i],
                z[i],
                y[i],
                'lr',
                return_norm=False)

    i = d - 1
    direct = -1
    swp = 1
    max_dx = 0

    while swp <= nswp:
        # Project the MatVec generating vector
        crx = _reshape(x[i], (rx[i] * m[i] * rx[i + 1], 1))
        cry = _bfun3(phiyax[i], A[i], phiyax[i + 1], crx)
        nrms[i] = _np.linalg.norm(cry)  # , 'fro')
        # The main goal is to keep y[i] of norm 1
        if (nrms[i] > 0):
            cry = cry / nrms[i]
        else:
            nrms[i] = 1
        y[i] = _reshape(y[i], (ry[i] * n[i] * ry[i + 1], 1))
        dx = _np.linalg.norm(cry - y[i])
        max_dx = max(max_dx, dx)

        # Truncation and enrichment
        if ((direct > 0) and (i < d - 1)):  # ?? i<d
            cry = _reshape(cry, (ry[i] * n[i], ry[i + 1]))
            if (renorm == 'gram'):
                [u, s, v] = _svdgram(cry, tol / d**0.5)
                v = v.T
                r = u.shape[1]
            else:
                [u, s, vt] = _np.linalg.svd(cry, full_matrices=False)
                #s = diag(s)
                r = _my_chop2(s, tol * _np.linalg.norm(s) / d**0.5)
                u = u[:, :r]
                # ????? s - matrix or vector
                v = _np.dot(_tconj(vt[:r, :]), _np.diag(s[:r]))

            # Prepare enrichment, if needed
            if (kickrank + kickrank2 > 0):
                cry = _np.dot(u, v.T)
                cry = _reshape(cry, (ry[i] * n[i], ry[i + 1]))
                # For updating z
                crz = _bfun3(phizax[i], A[i], phizax[i + 1], crx)
                crz = _reshape(crz, (rz[i] * n[i], rz[i + 1]))
                ys = _np.dot(cry, phizy[i + 1])
                yz = _reshape(ys, (ry[i], n[i] * rz[i + 1]))
                yz = _np.dot(phizy[i], yz)
                yz = _reshape(yz, (rz[i] * n[i], rz[i + 1]))
                crz = crz / nrms[i] - yz
                nrmz = _np.linalg.norm(crz)  # , 'fro')
                if (kickrank2 > 0):
                    [crz, _, _] = _np.linalg.svd(crz, full_matrices=False)
                    crz = crz[:, : min(crz.shape[1], kickrank)]
                    crz = _np.hstack(
                        (crz, _np.random.randn(
                            rz[i] * n[i], kickrank2)))
                # For adding into solution
                if fkick:
                    crs = _bfun3(phiyax[i], A[i], phizax[i + 1], crx)
                    crs = _reshape(crs, (ry[i] * n[i], rz[i + 1]))
                    crs = crs / nrms[i] - ys
                    u = _np.hstack((u, crs))
                    if (renorm == 'gram') and (
                            ry[i] * n[i] > 5 * (ry[i + 1] + rz[i + 1])):
                        [u, s, R] = _svdgram(u)
                    else:
                        [u, R] = _np.linalg.qr(u)
                    v = _np.hstack((v, _np.zeros((ry[i + 1], rz[i + 1]))))
                    v = _np.dot(v, R.T)
                    r = u.shape[1]
            y[i] = _reshape(u, (ry[i], n[i], r))

            cr2 = _reshape(y[i + 1], (ry[i + 1], n[i + 1] * ry[i + 2]))
            v = _reshape(v, (ry[i + 1], r))
            cr2 = _np.dot(v.T, cr2)
            y[i + 1] = _reshape(cr2, (r, n[i + 1], ry[i + 2]))

            ry[i + 1] = r

            [phiyax[i + 1], nrms[i]
             ] = _compute_next_Phi(phiyax[i], y[i], x[i], 'lr', A[i])

            if (kickrank + kickrank2 > 0):
                if (renorm == 'gram') and (rz[i] * n[i] > 5 * rz[i + 1]):
                    [crz, s, R] = _svdgram(crz)
                else:
                    [crz, R] = _np.linalg.qr(crz)
                rz[i + 1] = crz.shape[1]
                z[i] = _reshape(crz, (rz[i], n[i], rz[i + 1]))
                # z[i+1] will be recomputed from scratch in the next step

                phizax[
                    i +
                    1] = _compute_next_Phi(
                    phizax[i],
                    z[i],
                    x[i],
                    'lr',
                    A[i],
                    nrms[i],
                    return_norm=False)
                phizy[
                    i +
                    1] = _compute_next_Phi(
                    phizy[i],
                    z[i],
                    y[i],
                    'lr',
                    return_norm=False)

        elif ((direct < 0) and (i > 0)):
            cry = _reshape(cry, (ry[i], n[i] * ry[i + 1]))
            if (renorm == 'gram'):
                [v, s, u] = _svdgram(cry.T, tol / d**0.5)
                u = u.T
                r = v.shape[1]
            else:
                #[v, s, u] = _np.linalg.svd(cry.T, full_matrices=False)
                [u, s, vt] = _np.linalg.svd(cry, full_matrices=False)
                #s = diag(s);
                r = _my_chop2(s, tol * _np.linalg.norm(s) / d**0.5)
                v = _tconj(vt[:r, :])

                #v = vt[:r, :]
                #v = _np.dot(v[:, :r], _np.diag(s[:r]))
                u = _np.dot(u[:, :r], _np.diag(s[:r]))  # ??????????????????

            # Prepare enrichment, if needed
            if (kickrank + kickrank2 > 0):
                cry = _np.dot(u, v.T)  # .T)
                cry = _reshape(cry, (ry[i], n[i] * ry[i + 1]))
                # For updating z
                crz = _bfun3(phizax[i], A[i], phizax[i + 1], crx)
                crz = _reshape(crz, (rz[i], n[i] * rz[i + 1]))
                ys = _np.dot(phizy[i], cry)
                yz = _reshape(ys, (rz[i] * n[i], ry[i + 1]))
                yz = _np.dot(yz, phizy[i + 1])
                yz = _reshape(yz, (rz[i], n[i] * rz[i + 1]))
                crz = crz / nrms[i] - yz
                nrmz = _np.linalg.norm(crz)  # , 'fro')
                if (kickrank2 > 0):
                    [_, _, crz] = _np.linalg.svd(crz, full_matrices=False)
                    crz = crz[:, : min(crz.shape[1], kickrank)]
                    crz = _tconj(crz)
                    crz = _np.vstack(
                        (crz, _np.random.randn(kickrank2, n[i] * rz[i + 1])))
                # For adding into solution
                crs = _bfun3(phizax[i], A[i], phiyax[i + 1], crx)
                crs = _reshape(crs, (rz[i], n[i] * ry[i + 1]))
                crs = crs / nrms[i] - ys
                v = _np.hstack((v, crs.T))  # .T
                #v = v.T
                if (renorm == 'gram') and (
                        n[i] * ry[i + 1] > 5 * (ry[i] + rz[i])):
                    [v, s, R] = _svdgram(v)
                else:
                    [v, R] = _np.linalg.qr(v)
                u = _np.hstack((u, _np.zeros((ry[i], rz[i]))))
                u = _np.dot(u, R.T)
                r = v.shape[1]

            cr2 = _reshape(y[i - 1], (ry[i - 1] * n[i - 1], ry[i]))
            cr2 = _np.dot(cr2, u)
            y[i - 1] = _reshape(cr2, (ry[i - 1], n[i - 1], r))
            y[i] = _reshape(v.T, (r, n[i], ry[i + 1]))

            ry[i] = r

            [phiyax[i], nrms[i]] = _compute_next_Phi(
                phiyax[i + 1], y[i], x[i], 'rl', A[i])

            if (kickrank + kickrank2 > 0):
                if (renorm == 'gram') and (n[i] * rz[i + 1] > 5 * rz[i]):
                    [crz, s, R] = _svdgram(crz.T)
                else:
                    [crz, R] = _np.linalg.qr(crz.T)
                rz[i] = crz.shape[1]
                z[i] = _reshape(crz.T, (rz[i], n[i], rz[i + 1]))
                # don't update z[i-1], it will be recomputed from scratch

                phizax[i] = _compute_next_Phi(
                    phizax[
                        i + 1],
                    z[i],
                    x[i],
                    'rl',
                    A[i],
                    nrms[i],
                    return_norm=False)
                phizy[i] = _compute_next_Phi(
                    phizy[i + 1], z[i], y[i], 'rl', return_norm=False)

        if (verb > 1):
            print 'amen-mv: swp=[%d,%d], dx=%.3e, r=%d, |y|=%.3e, |z|=%.3e' % (swp, i, dx, r, _np.linalg.norm(cry), nrmz)

        # Stopping or reversing
        if ((direct > 0) and (i == d - 1)) or ((direct < 0) and (i == 0)):
            if (verb > 0):
                print 'amen-mv: swp=%d{%d}, max_dx=%.3e, max_r=%d' % (swp, (1 - direct) / 2, max_dx, max(ry))
            if ((max_dx < tol) or (swp == nswp)) and (direct > 0):
                break
            else:
                # We are at the terminal block
                y[i] = _reshape(cry, (ry[i], n[i], ry[i + 1]))
                if (direct > 0):
                    swp = swp + 1
            max_dx = 0
            direct = -direct
        else:
            i = i + direct
    # if (direct>0)
    y[d - 1] = _reshape(cry, (ry[d - 1], n[d - 1], ry[d]))
    # else
    #     y{1} = reshape(cry, ry(1), n(1), ry(2));
    # end;

    # Distribute norms equally...
    nrms = _np.exp(sum(_np.log(nrms)) / d)
    # ... and plug them into y
    for i in xrange(d):
        y[i] = _np.dot(y[i], nrms)

    if (vectype == 1):
        y = _tt.vector.from_list(y)
        if kickrank == 0:
            z = None
        else:
            z = _tt.vector.from_list(z)

    return y, z


def _compute_next_Phi(Phi_prev, x, y, direction, A=None,
                      extnrm=None, return_norm=True):
    '''
    Performs the recurrent Phi (or Psi) matrix computation
    Phi = Phi_prev * (x'Ay).
    If direction is 'lr', computes Psi
    if direction is 'rl', computes Phi
    A can be empty, then only x'y is computed.

        Phi1: rx1, ry1, ra1, or {rx1, ry1}_ra, or rx1, ry1
        Phi2: ry2, ra2, rx2, or {ry2, rx2}_ra, or ry2, rx2
    '''

    [rx1, n, rx2] = x.shape
    [ry1, m, ry2] = y.shape

    if A is not None:
        if isinstance(A, list):  # ?????????????????????????????????
            # A is a canonical block
            ra = len(A)
        else:
            # Just full format
            [ra1, ra2] = A.shape
            ra1 = ra1 / n
            ra2 = ra2 / m
    # ?????????????????????????????????????
    else:
        [ra1, ra2] = [1, 1]

    if isinstance(Phi_prev, list):
        Phi = [None] * ra
        if return_norm:
            nrm = 0
        if (direction == 'lr'):
            # lr: Phi1
            x = _reshape(x, (rx1, n * rx2))
            y = _reshape(y, (ry1 * m, ry2))
            for i in xrange(ra):
                Phi[i] = _np.dot(_tconj(x), Phi_prev[i])
                Phi[i] = _reshape(Phi[i], (n, rx2 * ry1))
                Phi[i] = Phi[i].T
                Phi[i] = _np.dot(Phi[i], A[i])
                Phi[i] = _reshape(Phi[i], (rx2, ry1 * m))
                Phi[i] = _np.dot(Phi[i], y)
                if return_norm:
                    nrm = max(nrm, _np.linalg.norm(Phi[i]))  # , 'fro'))
        else:
            # rl: Phi2
            y = _reshape(y, (ry1, m * ry2))
            x = _reshape(x, (rx1 * n, rx2))
            for i in xrange(ra):
                Phi[i] = _np.dot(Phi_prev[i], x.T)
                Phi[i] = _reshape(Phi[i], (ry2 * rx1, n))
                Phi[i] = _np.dot(Phi[i], A[i])
                Phi[i] = Phi[i].T
                Phi[i] = _reshape(Phi[i], (m * ry2, rx1))
                Phi[i] = _np.dot(y, Phi[i])
                if return_norm:
                    nrm = max(nrm, _np.linalg.norm(Phi[i]))  # , 'fro'))
        if return_norm:
            # Extract the scale to prevent overload
            if (nrm > 0):
                for i in xrange(ra):
                    Phi[i] = Phi[i] / nrm
            else:
                nrm = 1
        elif extnrm is not None:
            # Override the normalization
            for i in xrange(ra):
                Phi[i] = Phi[i] / extnrm
    else:
        if (direction == 'lr'):
            # lr: Phi1
            x = _reshape(x, (rx1, n * rx2))
            Phi = _reshape(Phi_prev, (rx1, ry1 * ra1))
            Phi = _np.dot(_tconj(x), Phi)
            if A is not None:
                Phi = _reshape(Phi, (n * rx2 * ry1, ra1))
                Phi = Phi.T
                Phi = _reshape(Phi, (ra1 * n, rx2 * ry1))
                Phi = _np.dot(A.T, Phi)
                Phi = _reshape(Phi, (m, ra2 * rx2 * ry1))
            else:
                Phi = _reshape(Phi, (n, rx2 * ry1))
            Phi = Phi.T
            Phi = _reshape(Phi, (ra2 * rx2, ry1 * m))
            y = _reshape(y, (ry1 * m, ry2))
            Phi = _np.dot(Phi, y)
            if A is not None:
                Phi = _reshape(Phi, (ra2, rx2 * ry2))
                Phi = Phi.T
                Phi = _reshape(Phi, (rx2, ry2, ra2))
            else:
                Phi = _reshape(Phi, (rx2, ry2))
        else:
            # rl: Phi2
            y = _reshape(y, (ry1 * m, ry2))
            Phi = _reshape(Phi_prev, (ry2, ra2 * rx2))
            Phi = _np.dot(y, Phi)
            if A is not None:
                Phi = _reshape(Phi, (ry1, m * ra2 * rx2))
                Phi = Phi.T
                Phi = _reshape(Phi, (m * ra2, rx2 * ry1))
                Phi = _np.dot(A, Phi)
                Phi = _reshape(Phi, (ra1 * n * rx2, ry1))
                Phi = Phi.T
            Phi = _reshape(Phi, (ry1 * ra1, n * rx2))
            x = _reshape(x, (rx1, n * rx2))
            Phi = _np.dot(Phi, _tconj(x))
            if A is not None:
                Phi = _reshape(Phi, (ry1, ra1, rx1))
            else:
                Phi = _reshape(Phi, (ry1, rx1))

        if return_norm:
            # Extract the scale to prevent overload
            nrm = _np.linalg.norm(Phi)  # , 'fro')
            if (nrm > 0):
                Phi = Phi / nrm
            else:
                nrm = 1
        elif extnrm is not None:
            # Override the normalization by the external one
            Phi = Phi / extnrm

    if return_norm:
        return Phi, nrm
    else:
        return Phi


# new
def _bfun3(Phi1, A, Phi2, x):
    b = x.shape[1]

    if isinstance(A, list):
        ra = len(A)
        [ry1, rx1] = Phi1[0].shape  # [:3]
        [rx2, ry2] = Phi2[0].shape  # [:3]
        [n, m] = A.shape

        y = _np.zeros((ry1 * n * ry2, b))
        for i in xrange(ra):
            cy = _reshape(x.T, (b * rx1 * m, rx2))
            cy = _np.dot(cy, Phi2[i])
            cy = _reshape(cy, (b * rx1, m * ry2))
            cy = cy.T
            cy = _reshape(cy, (m, ry2 * b * rx1))
            cy = _np.dot(A[i], cy)
            cy = _reshape(cy, (n * ry2 * b, rx1))
            cy = _np.dot(cy, Phi1[i].T)
            cy = cy.T
            cy = _reshape(cy, (ry1 * n * ry2, b))
            y = y + cy
    else:
        # Phi1: ry1, rx1, ra1
        [ry1, rx1, ra1] = Phi1.shape
        # Phi2: rx2, ra2, ry2
        [rx2, ra2, ry2] = Phi2.shape
        [n, m] = A.shape
        n = n / ra1
        m = m / ra2

        y = _reshape(x.T, (b * rx1 * m, rx2))
        Phi2 = _reshape(Phi2, (rx2, ra2 * ry2))
        y = _np.dot(y, Phi2)
        y = _reshape(y, (b * rx1, m * ra2 * ry2))
        y = y.T
        y = _reshape(y, (m * ra2, ry2 * b * rx1))
        y = _np.dot(A, y)
        y = _reshape(y, (ra1 * n * ry2 * b, rx1))
        y = y.T
        y = _reshape(y, (rx1 * ra1, n * ry2 * b))
        Phi1 = _reshape(Phi1, (ry1, rx1 * ra1))
        y = _np.dot(Phi1, y)
        y = _reshape(y, (ry1 * n * ry2, b))

    return y


if __name__ == '__main__':

    d = 12
    n = 15
    m = 15
    ra = 30
    rb = 10
    eps = 1e-6

    a = 0 * _tt.rand(n * m, d, r=ra)
    a = a + _tt.ones(n * m, d)
    #a = a.round(1e-12)
    a = _tt.vector.to_list(a)
    for i in xrange(d):
        sa = a[i].shape
        a[i] = _reshape(a[i], (sa[0], m, n, sa[-1]))
    A = _tt.matrix.from_list(a)

    b = _tt.rand(n, d, r=rb)

    c = amen_mv(A, b, eps, y=None, z=None, nswp=20, kickrank=4,
                kickrank2=0, verb=True, init_qr=True, renorm='gram', fkick=False)
    d = _tt.matvec(A, b).round(eps)

    print (c[0] - d).norm() / d.norm()
