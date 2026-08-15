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



@njit(**_JIT)
def recycled_pcg_local(phi1, amat, phi2, invT, use_prec, rhs, coarse,
                       use_coarse, tol, maxit,
                       i_, m_, j_, p_, n_, c_, b_, a_):
    """Augmented PCG with one exactly treated recycled direction.

    The local correction is minimized first over ``span(coarse)``.  Every
    subsequently preconditioned residual is projected to the
    ``B``-orthogonal complement of that direction.  The projection is kept
    inside the compiled loop: for one coarse vector it is only a dot and an
    axpy, while returning to Python once per local iteration costs more than
    the operation itself at QTT core sizes.

    ``coarse`` is assumed Euclidean-normalized when ``use_coarse`` is true.
    The returned ``last_search`` is the last fresh PCG direction; callers keep
    it as the one-vector memory for the next visit of this TT core.
    """
    size = rhs.size
    correction = np.zeros(size)
    residual = rhs.copy()
    last_search = np.zeros(size)
    zvec = np.zeros(size)
    hp = np.zeros(size)
    denominator = 0.0
    matvecs = 0
    coarse_ok = False

    right_norm = np.sqrt(np.dot(rhs, rhs))
    if right_norm == 0.0:
        return (correction, residual, last_search, matvecs, 0, False,
                coarse_ok)

    if use_coarse:
        hp[:] = _mv(phi1, amat, phi2, coarse, i_, m_, j_, p_, n_, c_, b_, a_).reshape(size)
        matvecs += 1
        denominator = np.dot(coarse, hp)
        scale = np.sqrt(np.dot(coarse, coarse) * np.dot(hp, hp))
        if denominator > 100.0 * np.finfo(np.float64).eps * scale:
            alpha = np.dot(coarse, residual) / denominator
            correction += alpha * coarse
            residual -= alpha * hp
            coarse_ok = True

    if use_prec:
        jacobi_c_apply(invT, residual.reshape(i_, m_, j_),
                       zvec.reshape(i_, m_, j_))
    else:
        zvec[:] = residual
    if coarse_ok:
        zvec -= coarse * (np.dot(hp, zvec) / denominator)

    rho = np.dot(residual, zvec)
    search = zvec.copy()
    used = 0
    breakdown = False
    for _iteration in range(maxit):
        if np.sqrt(np.dot(residual, residual)) <= tol * right_norm:
            break
        image = _mv(phi1, amat, phi2, search, i_, m_, j_, p_, n_, c_, b_, a_).reshape(size)
        matvecs += 1
        curvature = np.dot(search, image)
        scale = np.sqrt(np.dot(search, search) * np.dot(image, image))
        if (curvature <= 100.0 * np.finfo(np.float64).eps * scale
                or rho <= 0.0):
            breakdown = True
            break
        last_search[:] = search
        used += 1
        alpha = rho / curvature
        correction += alpha * search
        residual -= alpha * image
        if np.sqrt(np.dot(residual, residual)) <= tol * right_norm:
            break

        if use_prec:
            jacobi_c_apply(invT, residual.reshape(i_, m_, j_),
                           zvec.reshape(i_, m_, j_))
        else:
            zvec[:] = residual
        if coarse_ok:
            # Strict augmented PCG: do not let preconditioning leak back into
            # the recycled coarse space.
            zvec -= coarse * (np.dot(hp, zvec) / denominator)
        rho_new = np.dot(residual, zvec)
        if rho_new <= 0.0:
            breakdown = True
            break
        search = zvec + (rho_new / rho) * search
        rho = rho_new

    return (correction, residual, last_search, matvecs, used, breakdown,
            coarse_ok)




@njit(**_JIT)
def invert_2x2_blocks(blocks, out):
    """Invert every 2x2 diagonal block of the central Jacobi preconditioner.

    ``blocks`` is (b, f, 2, 2), ``out`` is (2, 2, b, f) -- the layout
    :func:`jacobi_c_apply` wants.  numpy's batched ``inv`` over ~1156 two-by-two
    matrices spends its time dispatching, not inverting.  Returns False if any
    block is singular; the caller raises rather than silently substituting.
    """
    bb = blocks.shape[0]
    ff = blocks.shape[1]
    for b in range(bb):
        for f in range(ff):
            a11 = blocks[b, f, 0, 0]
            a12 = blocks[b, f, 0, 1]
            a21 = blocks[b, f, 1, 0]
            a22 = blocks[b, f, 1, 1]
            det = a11 * a22 - a12 * a21
            if det == 0.0 or not np.isfinite(det):
                return False
            out[0, 0, b, f] = a22 / det
            out[0, 1, b, f] = -a12 / det
            out[1, 0, b, f] = -a21 / det
            out[1, 1, b, f] = a11 / det
    return True


@njit(**_JIT)
def project_lr(phi, acore, xcore):
    """``phi[a,i,p], A[p,n,m,c], x[i,m,j] -> w[a,n,j,c]``.

    Two BLAS products with the permutations done explicitly, in one call: the
    two-einsum version spends about two thirds of its 77 us on dispatch and
    temporaries rather than on the 0.84 Mflop it computes.
    """
    a, i, p = phi.shape
    _, n, m, c = acore.shape
    _, _, j = xcore.shape
    # t[a,p,m,j] = sum_i phi[a,i,p] x[i,m,j]
    phi_t = np.ascontiguousarray(phi.transpose(0, 2, 1)).reshape(a * p, i)
    t = (phi_t @ np.ascontiguousarray(xcore).reshape(i, m * j)).reshape(a, p, m, j)
    # out[a,n,j,c] = sum_{p,m} t[a,p,m,j] A[p,n,m,c]
    t2 = np.ascontiguousarray(t.transpose(0, 3, 1, 2)).reshape(a * j, p * m)
    a2 = np.ascontiguousarray(acore.transpose(0, 2, 1, 3)).reshape(p * m, n * c)
    out = (t2 @ a2).reshape(a, j, n, c)
    return np.ascontiguousarray(out.transpose(0, 2, 1, 3))


@njit(**_JIT)
def project_rl(phi, acore, xcore):
    """``phi[b,j,c], A[p,n,m,c], x[i,m,j] -> w[b,n,i,p]`` (the mirror image)."""
    b, j, c = phi.shape
    p, n, m, _ = acore.shape
    i, _, _ = xcore.shape
    # t[b,c,i,m] = sum_j phi[b,j,c] x[i,m,j]
    phi_t = np.ascontiguousarray(phi.transpose(0, 2, 1)).reshape(b * c, j)
    x_t = np.ascontiguousarray(xcore.transpose(2, 0, 1)).reshape(j, i * m)
    t = (phi_t @ x_t).reshape(b, c, i, m)
    # out[b,n,i,p] = sum_{c,m} t[b,c,i,m] A[p,n,m,c]
    t2 = np.ascontiguousarray(t.transpose(0, 2, 3, 1)).reshape(b * i, m * c)
    a2 = np.ascontiguousarray(acore.transpose(2, 3, 1, 0)).reshape(m * c, n * p)
    out = (t2 @ a2).reshape(b, i, n, p)
    return np.ascontiguousarray(out.transpose(0, 2, 1, 3))


@njit(**_JIT)
def apply_lr(w, phi):
    """``w[a,n,j,c], phi[b,j,c] -> y[a,n,b]``."""
    a, n, j, c = w.shape
    b = phi.shape[0]
    wc = np.ascontiguousarray(w).reshape(a * n, j * c)
    pc = np.ascontiguousarray(np.ascontiguousarray(phi).reshape(b, j * c).T)
    return (wc @ pc).reshape(a, n, b)


@njit(**_JIT)
def apply_rl(w, phi):
    """``w[b,n,i,p], phi[a,i,p] -> y[a,n,b]``."""
    b, n, i, p = w.shape
    a = phi.shape[0]
    wt = np.ascontiguousarray(w.transpose(2, 3, 0, 1)).reshape(i * p, b * n)
    t = (np.ascontiguousarray(phi).reshape(a, i * p) @ wt).reshape(a, b, n)
    return np.ascontiguousarray(t.transpose(0, 2, 1))


@njit(**_JIT)
def phi_next_lr(w, ycore):
    """``w[a,n,j,c], conj(y)[a,n,b] -> phi[b,j,c]``."""
    a, n, j, c = w.shape
    b = ycore.shape[2]
    yt = np.ascontiguousarray(np.ascontiguousarray(ycore).reshape(a * n, b).T)
    return (yt @ np.ascontiguousarray(w).reshape(a * n, j * c)).reshape(b, j, c)


@njit(**_JIT)
def phi_next_rl(w, ycore):
    """``w[b,n,i,p], conj(y)[a,n,b] -> phi[a,i,p]``."""
    b, n, i, p = w.shape
    a = ycore.shape[0]
    y2 = np.ascontiguousarray(ycore.transpose(0, 2, 1)).reshape(a, b * n)
    w2 = np.ascontiguousarray(w).reshape(b * n, i * p)
    return (y2 @ w2).reshape(a, i, p)


@njit(**_JIT)
def jacobi_c_blocks(phiL, acore, phiR, out):
    """Diagonal blocks of the central Jacobi preconditioner.

    ``out[b, f, i, j] = sum_{p,q} phiL[b,b,p] A[p,i,j,q] phiR[f,f,q]`` -- the
    n x n diagonal block for every pair of rank indices.  Written as loops
    because the whole thing is ~80k multiply-adds: two einsums plus a diagonal
    extraction spend their time on dispatch, not on that.
    """
    r1 = phiL.shape[0]
    r2 = phiR.shape[0]
    R1 = acore.shape[0]
    n = acore.shape[1]
    m = acore.shape[2]
    R2 = acore.shape[3]
    for b in range(r1):
        for f in range(r2):
            for i in range(n):
                for j in range(m):
                    acc = 0.0
                    for p in range(R1):
                        lp = phiL[b, b, p]
                        if lp == 0.0:
                            continue
                        for q in range(R2):
                            acc += lp * acore[p, i, j, q] * phiR[f, f, q]
                    out[b, f, i, j] = acc
    return out
