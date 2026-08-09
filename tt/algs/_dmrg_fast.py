"""Compiled inner loop of the greedy DMRG cross (optional, needs numba).

One bond visit of :func:`tt.algs.dmrg_cross.dmrg_cross` -- lottery seed, rook
alternation, residual evaluation, fiber bookkeeping -- as a single jitted
kernel that calls the user's ``fun`` directly.  It engages only when ``fun``
itself is a numba dispatcher: then the whole visit runs without touching the
interpreter, which is where the pure-numpy path spends most of its time (a
visit is ~50 small numpy calls on ~1300-point fibers; the arithmetic is
microseconds, the call overhead is not).

The linear algebra is a hand-written LAPACK ``getrs`` -- row swaps, unit-lower
forward substitution, upper back substitution -- on the ``getrf`` factors
computed outside.  It must stay a *solve*: the numpy path's docstring records
what happened when an explicit inverse was tried.

Everything here mirrors the numpy path decision for decision; the two paths
consume identical lottery draws (drawn outside, passed in), so they walk the
same pivots up to floating-point tie-breaking between LAPACK's blocked
triangular solves and the loops below.  Parity is pinned by
``tests/test_dmrg_cross.py``.

Status codes returned by :func:`bond_kernel`: 0 = pivot found, 1 = no
admissible lottery position, 2 = ``fun`` returned a non-finite value
(``badval``/``badidx`` name it), 3 = ``fun`` returned the wrong batch length.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit
    from numba.extending import is_jitted
    HAVE_NUMBA = True
except ImportError:                       # pragma: no cover - [fast] not installed
    HAVE_NUMBA = False

    def is_jitted(fun):                   # noqa: ARG001
        return False


if HAVE_NUMBA:

    @njit(cache=True)
    def getrs(lu, piv0, b):
        """Solve ``M x = b`` from the ``getrf`` factors, 0-based pivots."""
        r = lu.shape[0]
        x = b.copy()
        for i in range(r):
            pi = piv0[i]
            if pi != i:
                t = x[i]; x[i] = x[pi]; x[pi] = t
        for i in range(r):                        # L, unit lower
            s = x[i]
            for k in range(i):
                s -= lu[i, k] * x[k]
            x[i] = s
        for i in range(r - 1, -1, -1):            # U, upper
            s = x[i]
            for k in range(i + 1, r):
                s -= lu[i, k] * x[k]
            x[i] = s / lu[i, i]
        return x

    @njit(cache=True)
    def getrs_t(lu, piv0, b):
        """Solve ``M^T x = b`` from the same factors."""
        r = lu.shape[0]
        x = b.copy()
        for i in range(r):                        # U^T, lower with diagonal
            s = x[i]
            for k in range(i):
                s -= lu[k, i] * x[k]
            x[i] = s / lu[i, i]
        for i in range(r - 1, -1, -1):            # L^T, unit upper
            s = x[i]
            for k in range(i + 1, r):
                s -= lu[k, i] * x[k]
            x[i] = s
        for i in range(r - 1, -1, -1):            # swaps, reversed
            pi = piv0[i]
            if pi != i:
                t = x[i]; x[i] = x[pi]; x[pi] = t
        return x

    @njit(cache=True)
    def _draw(w, u):
        """Inverse-CDF draws over weights ``w`` -- the numpy path's lottery."""
        c = np.cumsum(w)
        total = c[-1]
        out = np.empty(len(u), dtype=np.int64)
        for l in range(len(u)):
            pos = np.searchsorted(c, u[l] * total, side="right")
            if pos > len(w) - 1:
                pos = len(w) - 1
            out[l] = pos
        return out

    @njit
    def _fill_col_idx(idx, lpref, rsuf, n1, kk, qq):
        """Column-fiber multi-indices, (i outer, j inner) as in the numpy path."""
        r1, npl = lpref.shape
        nsr = rsuf.shape[1]
        d = npl + 2 + nsr
        row = 0
        for a in range(r1):
            for j in range(n1):
                for s in range(npl):
                    idx[row, s] = lpref[a, s]
                idx[row, npl] = j
                idx[row, npl + 1] = kk
                for s in range(nsr):
                    idx[row, npl + 2 + s] = rsuf[qq, s]
                row += 1

    @njit
    def _fill_row_idx(idx, lpref, rsuf, ii, jj, n2):
        """Row-fiber multi-indices, (k outer, q inner)."""
        r2, nsr = rsuf.shape
        npl = lpref.shape[1]
        row = 0
        for k in range(n2):
            for q in range(r2):
                for s in range(npl):
                    idx[row, s] = lpref[ii, s]
                idx[row, npl] = jj
                idx[row, npl + 1] = k
                for s in range(nsr):
                    idx[row, npl + 2 + s] = rsuf[q, s]
                row += 1

    @njit
    def bond_kernel(fun, Cp, Cq, lu, piv0, lpref, rsuf, u1, u2, wcol, wrow,
                    rook, start_with_row, amax):
        """One bond visit: the ``piv >= 0`` branch of ``_bond_pivot``, compiled.

        Returns ``(status, pivot, ii, jj, kk, qq, acol, arow, neval, amax,
        badval, badidx)``; see the module docstring for the status codes.
        """
        r1, n1, rp = Cp.shape
        rp2, n2, r2 = Cq.shape
        npl = lpref.shape[1]
        nsr = rsuf.shape[1]
        d = npl + 2 + nsr
        neval = 0
        acol = np.empty(r1 * n1)
        arow = np.empty(n2 * r2)
        badidx = np.empty(d, dtype=np.int64)
        empty = np.empty(0)

        scol = wcol.sum()
        srow = wrow.sum()
        if scol == 0.0 or srow == 0.0:
            return 1, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx

        # -- lottery seed -----------------------------------------------------
        nlot = len(u1)
        ijpos = _draw(wcol, u1)
        kqpos = _draw(wrow, u2)
        idx = np.empty((nlot, d), dtype=np.int64)
        for l in range(nlot):
            i = ijpos[l] // n1
            j = ijpos[l] % n1
            k = kqpos[l] // r2
            q = kqpos[l] % r2
            for s in range(npl):
                idx[l, s] = lpref[i, s]
            idx[l, npl] = j
            idx[l, npl + 1] = k
            for s in range(nsr):
                idx[l, npl + 2 + s] = rsuf[q, s]
        b = fun(idx)
        neval += nlot
        if b.shape[0] != nlot:
            return 3, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx
        for l in range(nlot):
            if not np.isfinite(b[l]):
                badidx[:] = idx[l]
                return 2, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, b[l], badidx
            a = abs(b[l])
            if a > amax:
                amax = a

        rhs = np.empty(rp)
        best = 0
        pivot = 0.0
        bestabs = -1.0
        for l in range(nlot):
            k = kqpos[l] // r2
            q = kqpos[l] % r2
            for s in range(rp):
                rhs[s] = Cq[s, k, q]
            x = getrs(lu, piv0, rhs)
            i = ijpos[l] // n1
            j = ijpos[l] % n1
            e = b[l]
            for s in range(rp):
                e -= Cp[i, j, s] * x[s]
            if abs(e) > bestabs:
                bestabs = abs(e)
                best = l
                pivot = e
        ii = ijpos[best] // n1
        jj = ijpos[best] % n1
        kk = kqpos[best] // r2
        qq = kqpos[best] % r2

        # -- fibers / rook alternation ---------------------------------------
        colidx = np.empty((r1 * n1, d), dtype=np.int64)
        rowidx = np.empty((n2 * r2, d), dtype=np.int64)
        havecol = False
        haverow = False
        crs = 0
        done = rook == 0
        skipcol = start_with_row

        if done:                                  # piv == 0: fibers, no search
            _fill_col_idx(colidx, lpref, rsuf, n1, kk, qq)
            v = fun(colidx)
            neval += r1 * n1
            if v.shape[0] != r1 * n1:
                return 3, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx
            for l in range(r1 * n1):
                if not np.isfinite(v[l]):
                    badidx[:] = colidx[l]
                    return 2, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, v[l], badidx
                acol[l] = v[l]
                if abs(v[l]) > amax:
                    amax = abs(v[l])
            _fill_row_idx(rowidx, lpref, rsuf, ii, jj, n2)
            v = fun(rowidx)
            neval += n2 * r2
            if v.shape[0] != n2 * r2:
                return 3, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx
            for l in range(n2 * r2):
                if not np.isfinite(v[l]):
                    badidx[:] = rowidx[l]
                    return 2, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, v[l], badidx
                arow[l] = v[l]
                if abs(v[l]) > amax:
                    amax = abs(v[l])
            return 0, pivot, ii, jj, kk, qq, acol, arow, neval, amax, 0.0, badidx

        while not done:
            if not skipcol:
                _fill_col_idx(colidx, lpref, rsuf, n1, kk, qq)
                v = fun(colidx)
                neval += r1 * n1
                if v.shape[0] != r1 * n1:
                    return 3, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx
                for l in range(r1 * n1):
                    if not np.isfinite(v[l]):
                        badidx[:] = colidx[l]
                        return (2, 0.0, 0, 0, 0, 0, empty, empty, neval, amax,
                                v[l], badidx)
                    acol[l] = v[l]
                    if abs(v[l]) > amax:
                        amax = abs(v[l])
                crs += 1
                done = haverow and crs >= 2 * rook
                havecol = True
                if not done:
                    for s in range(rp):
                        rhs[s] = Cq[s, kk, qq]
                    x = getrs(lu, piv0, rhs)
                    bestabs = -1.0
                    pos = 0
                    for l in range(r1 * n1):
                        i = l // n1
                        j = l % n1
                        e = acol[l]
                        for s in range(rp):
                            e -= Cp[i, j, s] * x[s]
                        if abs(e) > bestabs:
                            bestabs = abs(e)
                            pos = l
                            best_e = e
                    i = pos // n1
                    j = pos % n1
                    done = haverow and i == ii and j == jj
                    ii = i
                    jj = j
                    pivot = best_e
            skipcol = False
            if not done:
                _fill_row_idx(rowidx, lpref, rsuf, ii, jj, n2)
                v = fun(rowidx)
                neval += n2 * r2
                if v.shape[0] != n2 * r2:
                    return 3, 0.0, 0, 0, 0, 0, empty, empty, neval, amax, 0.0, badidx
                for l in range(n2 * r2):
                    if not np.isfinite(v[l]):
                        badidx[:] = rowidx[l]
                        return (2, 0.0, 0, 0, 0, 0, empty, empty, neval, amax,
                                v[l], badidx)
                    arow[l] = v[l]
                    if abs(v[l]) > amax:
                        amax = abs(v[l])
                crs += 1
                done = havecol and crs >= 2 * rook
                haverow = True
                if not done:
                    for s in range(rp):
                        rhs[s] = Cp[ii, jj, s]
                    x = getrs_t(lu, piv0, rhs)
                    bestabs = -1.0
                    pos = 0
                    for l in range(n2 * r2):
                        k = l // r2
                        q = l % r2
                        e = arow[l]
                        for s in range(rp):
                            e -= x[s] * Cq[s, k, q]
                        if abs(e) > bestabs:
                            bestabs = abs(e)
                            pos = l
                            best_e = e
                    k = pos // r2
                    q = pos % r2
                    done = havecol and k == kk and q == qq
                    kk = k
                    qq = q
                    pivot = best_e
        return 0, pivot, ii, jj, kk, qq, acol, arow, neval, amax, 0.0, badidx

    @njit
    def probe(fun, idx):
        """Call the user's jitted ``fun`` from compiled code once, as a check."""
        return fun(idx)
