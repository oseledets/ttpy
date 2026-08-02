"""Compiled kernels for the innermost AMEn loops (optional, numba).

Why this file exists, in numbers.  On the 2D QTT Laplacian (1024x1024) the local
GMRES runs ~4600 inner iterations.  Each one costs 65 us with numpy against 45 us
for the Fortran it replaces, and the difference is not arithmetic: the matvec is
26 us on both, while orthogonalisation (18 us), the preconditioner (10 us) and
the rotation bookkeeping (11 us) are dominated by numpy call overhead on arrays
of a few thousand elements.  Those three are pure loops, so a compiled kernel
removes the overhead without changing a single formula.

Nothing here is required: if numba is absent, :data:`HAVE_NUMBA` is False and the
callers use their numpy paths.  ``pip install ttpy[fast]`` provides it as a
wheel, so the package still installs without a compiler.

Every kernel is checked against its numpy twin in tests/test_fast.py.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):                       # noqa: D103
        def wrap(fn):
            return fn
        return wrap if args and callable(args[0]) is False else (
            args[0] if args else wrap)


_JIT = dict(cache=True, nogil=True, fastmath=False)


@njit(**_JIT)
def orth_step(V, j, w, h):
    """One Arnoldi step: classical Gram-Schmidt against ``V[:j+1]``, then a
    second pass only if the vector lost orthogonality (Daniel-Gragg-Kaufman-
    Stewart: repeat when the norm drops by more than 1/sqrt(2)).

    ``h`` is filled with the Hessenberg column; the new norm is returned.
    ``V`` is (restart+1, size) and ``w`` is overwritten in place.
    """
    # np.dot, not hand-written loops: numba maps it to BLAS, and a scalar loop
    # over the basis measured 50 us against 18 us for the gemv it replaces.
    Vj = V[:j + 1]
    nrm0 = np.sqrt(np.dot(w, w))
    hh = Vj @ w
    w -= hh @ Vj
    for i in range(j + 1):
        h[i] = hh[i]
    nrm = np.sqrt(np.dot(w, w))
    if nrm < 0.707 * nrm0:                    # reorthogonalise
        h2 = Vj @ w
        w -= h2 @ Vj
        for i in range(j + 1):
            h[i] += h2[i]
        nrm = np.sqrt(np.dot(w, w))
    return nrm


@njit(**_JIT)
def givens_step(col, cs, sn, g, j):
    """Apply the stored rotations to a new Hessenberg column, then eliminate.

    ``col`` holds ``h[0..j+1]`` and is overwritten; ``cs``/``sn``/``g`` carry the
    factorisation.  Returns the new residual estimate ``|g[j+1]|``.
    """
    for i in range(j):
        t = cs[i] * col[i] + sn[i] * col[i + 1]
        col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1]
        col[i] = t
    denom = np.sqrt(col[j] * col[j] + col[j + 1] * col[j + 1])
    if denom == 0.0:
        cs[j] = 1.0
        sn[j] = 0.0
    else:
        cs[j] = col[j] / denom
        sn[j] = col[j + 1] / denom
    col[j] = cs[j] * col[j] + sn[j] * col[j + 1]
    col[j + 1] = 0.0
    g[j + 1] = -sn[j] * g[j]
    g[j] = cs[j] * g[j]
    return abs(g[j + 1])


@njit(**_JIT)
def jacobi_c_apply(invT, w, out):
    """Central block-Jacobi: an ``n x n`` solve for every rank pair.

    ``invT`` is (n, n, b, f) laid out once; ``w`` and ``out`` are (b, n, f).
    """
    nn = invT.shape[0]
    bb = w.shape[0]
    ff = w.shape[2]
    for i in range(nn):
        for b in range(bb):
            for f in range(ff):
                acc = 0.0
                for jj in range(nn):
                    acc += invT[i, jj, b, f] * w[b, jj, f]
                out[b, i, f] = acc
    return out


@njit(**_JIT)
def axpy_basis(V, y, used, out):
    """``out = sum_i y[i] * V[i]`` -- the GMRES correction from the basis."""
    out[:] = y[:used] @ V[:used]
    return out


def warmup():
    """Compile the kernels once (they are cached on disk afterwards)."""
    if not HAVE_NUMBA:
        return False
    V = np.zeros((3, 4)); w = np.zeros(4); h = np.zeros(3)
    orth_step(V, 0, w, h)
    givens_step(np.zeros(3), np.zeros(2), np.zeros(2), np.zeros(3), 0)
    jacobi_c_apply(np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
    axpy_basis(V, np.zeros(3), 1, w)
    return True


@njit(**_JIT)
def _mv(phi1, amat, phi2, w, i_, m_, j_, p_, n_, c_, b_, a_):
    """The local operator, as three BLAS products (tt-fort's d2d_mv)."""
    t1 = w.reshape(i_ * m_, j_) @ phi2                       # (i m, c b)
    t2 = np.ascontiguousarray(t1.reshape(i_, m_ * c_ * b_).T)
    t3 = amat @ t2.reshape(m_ * c_, b_ * i_)                 # (p n, b i)
    t4 = np.ascontiguousarray(t3.reshape(p_ * n_ * b_, i_).T)
    return phi1 @ t4.reshape(i_ * p_, n_ * b_)               # (a, n b)


@njit(**_JIT)
def gmres_local(phi1, amat, phi2, invT, use_prec, rhs, tol, restart, maxit,
                i_, m_, j_, p_, n_, c_, b_, a_):
    """One complete local GMRES solve, compiled.

    Everything the inner loop touches is here -- the three products of the local
    operator, the block-Jacobi apply, the Arnoldi step and the Givens rotations
    -- so the loop never returns to python.  The heavy products still go to BLAS
    through ``np.dot``; what is removed is the ~20 us of interpreter and
    temporary-allocation cost per iteration.

    Right preconditioning: the iteration runs on ``B M`` and the correction is
    ``M y``.  Returns ``(sol, relres, nmatvec, converged)`` with ``relres``
    recomputed from the returned iterate, never the Arnoldi estimate.
    """
    size = i_ * m_ * j_
    sol = np.zeros(size)
    bnorm = np.sqrt(np.sum(rhs * rhs))
    if bnorm == 0.0:
        return sol, 0.0, 0, True

    mrest = restart
    V = np.zeros((mrest + 1, size))
    hcol = np.zeros(mrest + 2)
    cs = np.zeros(mrest + 1)
    sn = np.zeros(mrest + 1)
    g = np.zeros(mrest + 2)
    R = np.zeros((mrest + 1, mrest + 1))
    tmp = np.zeros(size)

    r = rhs.copy()
    nmv = 0
    relres = 1.0
    for _cycle in range(maxit):
        beta = np.sqrt(np.sum(r * r))
        relres = beta / bnorm
        if relres <= tol:
            return sol, relres, nmv, True
        for t in range(size):
            V[0, t] = r[t] / beta
        for t in range(mrest + 2):
            g[t] = 0.0
        g[0] = beta

        used = 0
        for jj in range(mrest):
            if use_prec:
                jacobi_c_apply(invT, V[jj].reshape(i_, m_, j_),
                               tmp.reshape(i_, m_, j_))
                w = _mv(phi1, amat, phi2, tmp, i_, m_, j_, p_, n_, c_, b_, a_)
            else:
                w = _mv(phi1, amat, phi2, V[jj].copy(), i_, m_, j_,
                        p_, n_, c_, b_, a_)
            nmv += 1
            wf = w.reshape(size).copy()
            hnext = orth_step(V, jj, wf, hcol)

            for t in range(jj + 1):
                R[t, jj] = hcol[t]
            for t in range(jj + 2):
                hcol[t] = R[t, jj] if t <= jj else hnext
            hcol[jj + 1] = hnext
            est = givens_step(hcol, cs, sn, g, jj)
            for t in range(jj + 1):
                R[t, jj] = hcol[t]
            used = jj + 1
            if est <= tol * bnorm or hnext <= 1e-15 * beta:
                break
            for t in range(size):
                V[jj + 1, t] = wf[t] / hnext

        y = np.zeros(used)
        for t in range(used - 1, -1, -1):
            acc = g[t]
            for u in range(t + 1, used):
                acc -= R[t, u] * y[u]
            y[t] = acc / R[t, t]
        axpy_basis(V, y, used, tmp)
        if use_prec:
            step = np.zeros(size)
            jacobi_c_apply(invT, tmp.reshape(i_, m_, j_), step.reshape(i_, m_, j_))
        else:
            step = tmp.copy()
        for t in range(size):
            sol[t] += step[t]

        w = _mv(phi1, amat, phi2, sol.copy(), i_, m_, j_, p_, n_, c_, b_, a_)
        nmv += 1
        wf = w.reshape(size)
        for t in range(size):
            r[t] = rhs[t] - wf[t]
        relres = np.sqrt(np.sum(r * r)) / bnorm
        if relres <= tol:
            return sol, relres, nmv, True
    return sol, relres, nmv, relres <= tol
