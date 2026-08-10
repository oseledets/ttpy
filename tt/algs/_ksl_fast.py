"""Compiled KSL sweeps (optional, needs numba).

A KSL step at the manifold ranks this integrator is used with is ~44 local
exponentials of matrices of 4..40 numbers plus as many QRs and interface
updates: ~260k flops that the interpreted path spends almost entirely on call
dispatch (measured; ``docs/PERFORMANCE.md`` 3b).  This module runs the whole
sweep -- initial orthogonalization, K- and S-steps, interfaces, the stiffness
guard -- as jitted code, for the case that actually occurs: real float64 on the
numpy backend with every local size within ``DENSE_EXPM_LIMIT`` (the exact-
exponential regime, where there is no Krylov machinery to reproduce).
Everything else falls back to the interpreted path, which computes the same
thing.

The local matrix exponential is scaling-and-squaring with the degree-13 Pade
approximant (Higham 2005), the same algorithm scipy's ``expm`` uses for
well-scaled input; parity with scipy on these block sizes is pinned by a test.
The stiffness guard is evaluated inside the kernel and reported out as data;
the Python wrapper raises the same error, with the same message, as the
interpreted path.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    from numba.typed import List as TypedList
    HAVE_NUMBA = True
except ImportError:                        # pragma: no cover - [fast] absent
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @njit(cache=True)
    def _qr(m):
        """Householder QR, reduced: ``m = Q R`` with ``Q (rows, k)``.

        numba's ``np.linalg.qr`` goes through a LAPACK envelope that costs
        tens of microseconds per call regardless of size; at the 8x4 blocks of
        a KSL sweep that envelope IS the sweep.  Householder by hand is the
        same algorithm LAPACK runs, backward stable, and nanoseconds here.
        """
        rows, cols = m.shape
        k = min(rows, cols)
        r = m.copy()
        vs = np.zeros((k, rows))
        for j in range(k):
            normx = 0.0
            for i in range(j, rows):
                normx += r[i, j] * r[i, j]
            normx = np.sqrt(normx)
            if normx == 0.0:
                continue
            alpha = -normx if r[j, j] >= 0.0 else normx
            v0 = r[j, j] - alpha
            vs[j, j] = v0
            vnorm2 = v0 * v0
            for i in range(j + 1, rows):
                vs[j, i] = r[i, j]
                vnorm2 += r[i, j] * r[i, j]
            if vnorm2 == 0.0:
                continue
            for c in range(j, cols):
                dot = 0.0
                for i in range(j, rows):
                    dot += vs[j, i] * r[i, c]
                f = 2.0 * dot / vnorm2
                for i in range(j, rows):
                    r[i, c] -= f * vs[j, i]
        q = np.zeros((rows, k))
        for j in range(k):
            q[j, j] = 1.0
        for j in range(k - 1, -1, -1):
            vnorm2 = 0.0
            for i in range(j, rows):
                vnorm2 += vs[j, i] * vs[j, i]
            if vnorm2 == 0.0:
                continue
            for c in range(k):
                dot = 0.0
                for i in range(j, rows):
                    dot += vs[j, i] * q[i, c]
                f = 2.0 * dot / vnorm2
                for i in range(j, rows):
                    q[i, c] -= f * vs[j, i]
        return q, r[:k, :]

    @njit(cache=True)
    def _solve(a, b):
        """``a^{-1} b`` by LU with partial pivoting, row ops as slices."""
        n = a.shape[0]
        lu = np.ascontiguousarray(a).copy()
        x = np.ascontiguousarray(b).copy()
        for col in range(n):
            piv = col
            best = abs(lu[col, col])
            for i in range(col + 1, n):
                if abs(lu[i, col]) > best:
                    best = abs(lu[i, col])
                    piv = i
            if piv != col:
                for c in range(n):
                    t = lu[col, c]; lu[col, c] = lu[piv, c]; lu[piv, c] = t
                    t = x[col, c]; x[col, c] = x[piv, c]; x[piv, c] = t
            d = lu[col, col]
            for i in range(col + 1, n):
                f = lu[i, col] / d
                lu[i, col] = f
                # scalar loops, not slices: a slice op allocates a temporary,
                # and a thousand 30-element temporaries cost more than the LU
                for c in range(col + 1, n):
                    lu[i, c] -= f * lu[col, c]
                for c in range(n):
                    x[i, c] -= f * x[col, c]
        for col in range(n - 1, -1, -1):
            for i in range(col + 1, n):
                f = lu[col, i]
                for c in range(n):
                    x[col, c] -= f * x[i, c]
            d = lu[col, col]
            for c in range(n):
                x[col, c] /= d
        return x

    @njit(cache=True)
    def _expm(a):
        """Scaling-and-squaring Pade-13 exponential (Higham 2005).

        scipy's ``expm`` picks lower Pade degrees for small norms as a speed
        optimisation; always-13 is at least as accurate, and at 40x40 the
        cost difference is nothing.
        """
        a = np.ascontiguousarray(a)   # layout 'A' inputs would push every @
        n = a.shape[0]                # below off the BLAS path
        norm = 0.0
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += abs(a[i, j])
            if s > norm:
                norm = s
        eye = np.eye(n)
        # scipy's degree ladder: the cheapest Pade approximant whose backward
        # error at this norm is below eps -- a KSL step usually lands at
        # degree 3 or 5 (tau ||B|| is small), which is two matmuls, not eight
        if norm <= 1.495585217958292e-2:      # theta_3
            w2 = a @ a
            u = a @ (w2 + 60.0 * eye)
            v = 12.0 * w2 + 120.0 * eye
            return _solve(v - u, v + u)
        if norm <= 2.539398330063230e-1:      # theta_5
            w2 = a @ a
            w4 = w2 @ w2
            u = a @ (w4 + 420.0 * w2 + 15120.0 * eye)
            v = 30.0 * w4 + 3360.0 * w2 + 30240.0 * eye
            return _solve(v - u, v + u)
        if norm <= 9.504178996162932e-1:      # theta_7
            w2 = a @ a
            w4 = w2 @ w2
            w6 = w2 @ w4
            u = a @ (w6 + 1512.0 * w4 + 277200.0 * w2 + 8648640.0 * eye)
            v = (56.0 * w6 + 25200.0 * w4 + 1995840.0 * w2
                 + 17297280.0 * eye)
            return _solve(v - u, v + u)
        b = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
             1187353796428800.0, 129060195264000.0, 10559470521600.0,
             670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
             960960.0, 16380.0, 182.0, 1.0)
        theta13 = 5.371920351148152
        sq = 0
        if norm > theta13:
            sq = int(np.ceil(np.log2(norm / theta13)))
        w = a / (2.0 ** sq)
        w2 = w @ w
        w4 = w2 @ w2
        w6 = w2 @ w4
        u = w @ (w6 @ (b[13] * w6 + b[11] * w4 + b[9] * w2)
                 + b[7] * w6 + b[5] * w4 + b[3] * w2 + b[1] * eye)
        v = (w6 @ (b[12] * w6 + b[10] * w4 + b[8] * w2)
             + b[6] * w6 + b[4] * w4 + b[2] * w2 + b[0] * eye)
        r = _solve(v - u, v + u)
        for _ in range(sq):
            r = r @ r
        return r

    @njit(cache=True)
    def _local_matrix(left, acore, right):
        """``(p i P, q j Q)`` dense local operator; loops, sizes are tiny."""
        p, ra, q = left.shape
        _, ni, nj, ra2 = acore.shape
        P, _, Q = right.shape
        rows = p * ni * P
        cols = q * nj * Q
        m = np.zeros((rows, cols))
        for ip in range(p):
            for ii in range(ni):
                for iP in range(P):
                    r0 = (ip * ni + ii) * P + iP
                    for iq in range(q):
                        for ij in range(nj):
                            for iQ in range(Q):
                                c0 = (iq * nj + ij) * Q + iQ
                                s = 0.0
                                for a in range(ra):
                                    for A in range(ra2):
                                        s += (left[ip, a, iq]
                                              * acore[a, ii, ij, A]
                                              * right[iP, A, iQ])
                                m[r0, c0] = s
        return m

    @njit(cache=True)
    def _interface_matrix(left, right):
        p, ra, q = left.shape
        P, _, Q = right.shape
        m = np.zeros((p * P, q * Q))
        for ip in range(p):
            for iP in range(P):
                for iq in range(q):
                    for iQ in range(Q):
                        s = 0.0
                        for a in range(ra):
                            s += left[ip, a, iq] * right[iP, a, iQ]
                        m[ip * P + iP, iq * Q + iQ] = s
        return m

    @njit(cache=True)
    def _phi_left(phi, acore, frame):
        """L' = sum phi[p,a,q] frame[p,i,P] acore[a,i,j,A] frame[q,j,Q]."""
        p, ra, q = phi.shape
        _, ni, nj, ra2 = acore.shape
        P = frame.shape[2]
        t1 = np.zeros((ra, q, P, ni))                 # phi * conj(bra)
        for a in range(ra):
            for iq in range(q):
                for iP in range(P):
                    for ii in range(ni):
                        s = 0.0
                        for ip in range(p):
                            s += phi[ip, a, iq] * frame[ip, ii, iP]
                        t1[a, iq, iP, ii] = s
        t2 = np.zeros((q, P, ra2, nj))                # * acore
        for iq in range(q):
            for iP in range(P):
                for A in range(ra2):
                    for ij in range(nj):
                        s = 0.0
                        for a in range(ra):
                            for ii in range(ni):
                                s += t1[a, iq, iP, ii] * acore[a, ii, ij, A]
                        t2[iq, iP, A, ij] = s
        out = np.zeros((P, ra2, P))                   # * ket
        for iP in range(P):
            for A in range(ra2):
                for iQ in range(P):
                    s = 0.0
                    for iq in range(q):
                        for ij in range(nj):
                            s += t2[iq, iP, A, ij] * frame[iq, ij, iQ]
                    out[iP, A, iQ] = s
        return out

    @njit(cache=True)
    def _phi_right(phi, acore, frame):
        """R = sum phi[P,A,Q] frame[p,i,P] acore[a,i,j,A] frame[q,j,Q]."""
        P, ra2, Q = phi.shape
        ra, ni, nj, _ = acore.shape
        p = frame.shape[0]
        t1 = np.zeros((ra2, Q, p, ni))                # phi * conj(bra)
        for A in range(ra2):
            for iQ in range(Q):
                for ip in range(p):
                    for ii in range(ni):
                        s = 0.0
                        for iP in range(P):
                            s += phi[iP, A, iQ] * frame[ip, ii, iP]
                        t1[A, iQ, ip, ii] = s
        t2 = np.zeros((Q, p, ra, nj))                 # * acore
        for iQ in range(Q):
            for ip in range(p):
                for a in range(ra):
                    for ij in range(nj):
                        s = 0.0
                        for A in range(ra2):
                            for ii in range(ni):
                                s += t1[A, iQ, ip, ii] * acore[a, ii, ij, A]
                        t2[iQ, ip, a, ij] = s
        out = np.zeros((p, ra, p))                    # * ket
        for ip in range(p):
            for a in range(ra):
                for iq in range(p):
                    s = 0.0
                    for iQ in range(Q):
                        for ij in range(nj):
                            s += t2[iQ, ip, a, ij] * frame[iq, ij, iQ]
                    out[ip, a, iq] = s
        return out

    @njit(cache=True)
    def _left_orth(core):
        r0, n, r1 = core.shape
        q, s = _qr(np.ascontiguousarray(core).reshape(r0 * n, r1))
        return np.ascontiguousarray(q).reshape(r0, n, q.shape[1]), s

    @njit(cache=True)
    def _right_orth(core):
        r0, n, r1 = core.shape
        # LQ through QR of the transpose: core = S Q with Q Q^T = I
        m = np.zeros((n * r1, r0))
        for a in range(r0):
            for i in range(n):
                for c in range(r1):
                    m[i * r1 + c, a] = core[a, i, c]
        q, s = _qr(m)
        rnew = q.shape[1]
        qq = np.zeros((rnew, n, r1))
        for c in range(rnew):
            for i in range(n):
                for b in range(r1):
                    qq[c, i, b] = q[i * r1 + b, c]
        st = np.zeros((r0, rnew))
        for a in range(r0):
            for c in range(rnew):
                st[a, c] = s[c, a]
        return st, qq

    @njit(cache=True)
    def _fro(a):
        b = np.ascontiguousarray(a).reshape(a.size)
        s = 0.0
        for i in range(b.size):
            s += b[i] * b[i]
        return np.sqrt(s)

    @njit(cache=True)
    def _exp_apply(mat, x, t, growth_exponent):
        """``expm(t mat) @ x`` with the growth factor; status 1 = refused."""
        size = x.size
        w = _expm(t * mat) @ np.ascontiguousarray(x).reshape(size)
        nx = _fro(x)
        nw = 0.0
        for v in w:
            nw += v * v
        nw = np.sqrt(nw)
        growth = nw / nx if nx > 0.0 else 1.0
        floor = 2.220446049250313e-16 * growth ** growth_exponent
        status = 1 if floor > 1.0 else 0
        return w, growth, status

    @njit(cache=True)
    def run_sweeps(cores, acores, tau0, symm, growth_exponent):
        """Init orthogonalization + backward (+ forward) sweep, compiled.

        Returns ``(cores, rec, nrec, status, bad_site, bad_kind)``; ``rec``
        rows are ``(sweep, site, kind(0=K,1=S), size, growth)``, and a nonzero
        status means the stiffness guard fired at ``bad_site`` -- the wrapper
        raises the same error the interpreted path does.
        """
        d = len(cores)
        maxrec = 2 * (2 * d - 1) + 2
        rec = np.zeros((maxrec, 5))
        nrec = 0

        left = TypedList()
        right = TypedList()
        one = np.ones((1, 1, 1))
        for _k in range(d + 1):
            left.append(one)
            right.append(one)

        # init: left-orthogonalize all but the last core, build left interfaces
        for k in range(d - 1):
            q, s = _left_orth(cores[k])
            cores[k] = q
            r0, nk, r1 = cores[k + 1].shape
            cores[k + 1] = np.ascontiguousarray(
                s @ np.ascontiguousarray(cores[k + 1]).reshape(r0, nk * r1)
            ).reshape(s.shape[0], nk, r1)
            left[k + 1] = _phi_left(left[k], acores[k], q)

        nsweep = 2 if symm else 1
        for sweep in range(nsweep):
            backward = sweep == 0
            for step in range(d):
                i = d - 1 - step if backward else step
                mat = _local_matrix(left[i], acores[i], right[i + 1])
                w, growth, status = _exp_apply(mat, cores[i], tau0,
                                               growth_exponent)
                rec[nrec, 0] = sweep; rec[nrec, 1] = i; rec[nrec, 2] = 0.0
                rec[nrec, 3] = mat.shape[0]; rec[nrec, 4] = growth
                nrec += 1
                if status != 0:
                    return cores, rec, nrec, 1, i, 0
                k = np.ascontiguousarray(w).reshape(cores[i].shape)
                last = (i == 0) if backward else (i == d - 1)
                if last:
                    cores[i] = k.copy()
                    continue
                if backward:
                    s, q = _right_orth(k)
                    cores[i] = q
                    right[i] = _phi_right(right[i + 1], acores[i], q)
                    smat = _interface_matrix(left[i], right[i])
                else:
                    q, s = _left_orth(k)
                    cores[i] = q
                    left[i + 1] = _phi_left(left[i], acores[i], q)
                    smat = _interface_matrix(left[i + 1], right[i + 1])
                ws, growth, status = _exp_apply(smat, s, -tau0,
                                                growth_exponent)
                rec[nrec, 0] = sweep; rec[nrec, 1] = i; rec[nrec, 2] = 1.0
                rec[nrec, 3] = smat.shape[0]; rec[nrec, 4] = growth
                nrec += 1
                if status != 0:
                    return cores, rec, nrec, 1, i, 1
                snew = np.ascontiguousarray(ws).reshape(s.shape)
                if backward:
                    r0, nk, _ = cores[i - 1].shape
                    cores[i - 1] = np.ascontiguousarray(
                        np.ascontiguousarray(cores[i - 1]).reshape(r0 * nk, -1)
                        @ snew).reshape(r0, nk, snew.shape[1])
                else:
                    _, nk, r2 = cores[i + 1].shape
                    cores[i + 1] = np.ascontiguousarray(
                        snew @ np.ascontiguousarray(cores[i + 1]).reshape(
                            snew.shape[1], nk * r2)).reshape(snew.shape[0], nk, r2)
        return cores, rec, nrec, 0, -1, -1

    @njit(cache=True)
    def tangent_defect_kernel(ycores, acores):
        """``(defect_gap^2 terms)``: compiled twin of ``ksl.tangent_defect``.

        Returns ``(proj2, znorm)`` with ``znorm`` computed exactly as
        ``_ops.norm`` computes it -- through an orthogonalization sweep of the
        matvec cores, not through the cancelling contraction.
        """
        d = len(ycores)
        # right-orthogonalize y (centre 0)
        yc = TypedList()
        for k in range(d):
            yc.append(ycores[k].copy())
        for k in range(d - 1, 0, -1):
            s, q = _right_orth(yc[k])
            yc[k] = q
            r0, nk, _ = yc[k - 1].shape
            yc[k - 1] = np.ascontiguousarray(
                np.ascontiguousarray(yc[k - 1]).reshape(r0 * nk, -1)
                @ s).reshape(r0, nk, s.shape[1])

        # z = A y, core by core: z_k[(a p), i, (A q)] = sum_j acore[a,i,j,A] y[p,j,q]
        zc = TypedList()
        for k in range(d):
            ak = acores[k]
            yk = yc[k]
            ra, ni, nj, ra2 = ak.shape
            p, _, q = yk.shape
            z = np.zeros((ra * p, ni, ra2 * q))
            for a in range(ra):
                for ip in range(p):
                    for ii in range(ni):
                        for A in range(ra2):
                            for iq in range(q):
                                s = 0.0
                                for ij in range(nj):
                                    s += ak[a, ii, ij, A] * yk[ip, ij, iq]
                                z[a * p + ip, ii, A * q + iq] = s
            zc.append(z)

        # right interfaces of <y|z>
        mr = TypedList()
        for _k in range(d + 1):
            mr.append(np.ones((1, 1)))
        for k in range(d - 1, -1, -1):
            yk = yc[k]
            zk_ = zc[k]
            mrk = mr[k + 1]
            ry0, nk, ry1 = yk.shape
            rz0 = zk_.shape[0]
            rz1 = zk_.shape[2]
            # two binary contractions, not one 5-deep loop
            t = np.zeros((ry0, nk, rz1))
            for a in range(ry0):
                for i in range(nk):
                    for e in range(rz1):
                        s = 0.0
                        for b in range(ry1):
                            s += yk[a, i, b] * mrk[b, e]
                        t[a, i, e] = s
            out = np.zeros((ry0, rz0))
            for a in range(ry0):
                for c in range(rz0):
                    s = 0.0
                    for i in range(nk):
                        for e in range(rz1):
                            s += t[a, i, e] * zk_[c, i, e]
                    out[a, c] = s
            mr[k] = out

        ml = np.ones((1, 1))
        proj2 = 0.0
        for k in range(d):
            zck = zc[k]
            rz0, nk, rz1 = zck.shape
            ry0 = ml.shape[0]
            zk = np.zeros((ry0, nk, rz1))
            for a in range(ry0):
                for i in range(nk):
                    for e in range(rz1):
                        s = 0.0
                        for c in range(rz0):
                            s += ml[a, c] * zck[c, i, e]
                        zk[a, i, e] = s
            ry1 = mr[k + 1].shape[0]
            zkr = np.zeros((ry0, nk, ry1))
            for a in range(ry0):
                for i in range(nk):
                    for b in range(ry1):
                        s = 0.0
                        for e in range(rz1):
                            s += zk[a, i, e] * mr[k + 1][b, e]
                        zkr[a, i, b] = s
            if k < d - 1:
                q, s = _left_orth(yc[k])
                yc[k] = q
                r0, nk1, r1 = yc[k + 1].shape
                yc[k + 1] = np.ascontiguousarray(
                    s @ np.ascontiguousarray(yc[k + 1]).reshape(
                        r0, nk1 * r1)).reshape(s.shape[0], nk1, r1)
                qm = np.ascontiguousarray(q).reshape(-1, q.shape[2])
                zm = np.ascontiguousarray(zkr).reshape(-1, zkr.shape[2])
                res = zm - qm @ (qm.T @ zm)
                proj2 += _fro(res) ** 2
                # advance ml with the fresh frame (zk already is ml * z)
                tmp = zk
                mlnew = np.zeros((q.shape[2], zc[k].shape[2]))
                for b in range(q.shape[2]):
                    for e in range(zc[k].shape[2]):
                        s2 = 0.0
                        for a in range(ry0):
                            for i in range(nk):
                                s2 += q[a, i, b] * tmp[a, i, e]
                        mlnew[b, e] = s2
                ml = mlnew
            else:
                proj2 += _fro(zkr) ** 2

        # ||z|| through an orthogonalization sweep, as _ops.norm does
        for k in range(d - 1, 0, -1):
            s, q = _right_orth(zc[k])
            zc[k] = q
            r0, nk, _ = zc[k - 1].shape
            zc[k - 1] = np.ascontiguousarray(
                np.ascontiguousarray(zc[k - 1]).reshape(r0 * nk, -1)
                @ s).reshape(r0, nk, s.shape[1])
        znorm = _fro(zc[0])
        return proj2, znorm
