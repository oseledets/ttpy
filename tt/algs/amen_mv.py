"""AMEn matrix-by-vector: ``y ~ A x`` with adaptive TT ranks.

The exact product ``A x`` has TT ranks ``r(A) * r(x)``; forming it and rounding
it costs ``O(d n (r_A r_x)^3)``.  AMEn never forms it: it runs an ALS sweep for
``y`` (each local step is a projection of ``A x`` onto the current TT frames,
i.e. a *linear* operation, not a linear solve -- this is why the matvec version
has no local system to solve), and it fixes the well-known rank stagnation of
plain ALS by enriching the frames with a cheap TT approximation ``z`` of the
residual ``A x - y``.

Algorithm
---------
One-site ALS over ``y`` with left-to-right and right-to-left half-sweeps.  With
the interfaces (partial contractions) of the current frames

    Phi^L_k[a, i, p] = (Y_L^H (AX)_L)[a, (i,p)],
    Phi^R_k[b, j, c] = (Y_R^H (AX)_R)[b, (j,c)],

the local block is the projection

    y_k[a, n, b] = sum Phi^L_k[a,i,p] A_k[p,n,m,c] x_k[i,m,j] Phi^R_{k+1}[b,j,c],

which is exactly what :func:`_project` + :func:`_apply` compute.  After the
block is truncated with an SVD at ``tol/sqrt(d)`` it is enriched (backward
sweep) by the corresponding block of ``z``, projected onto the ``y`` frames.
``z`` itself is carried along as a rank-``kickrank`` ALS approximation of
``A x - y``, updated with the same projections.

Scaling invariant
-----------------
Every interface is normalized to unit Frobenius norm and the extracted scalar is
kept in ``nrms[k]``.  The invariant maintained everywhere in the loop is

    y_exact_projection = (product of the stored cores) * prod(nrms),

so the stored cores never overflow for large ``d`` (interface norms grow
exponentially in ``d`` for e.g. QTT Laplacians) and the local truncation
threshold ``tol/sqrt(d) * ||cry||`` is a *relative* one, because the stored
block always has unit norm and all other stored cores are orthonormal.  The
accumulated scale is put back into the cores at the very end, spread as the
geometric mean over all ``d`` cores.

References
----------
* S. V. Dolgov, D. V. Savostyanov, "Alternating minimal energy methods for
  linear systems in higher dimensions. Part I: SPD systems",
  arXiv:1301.6068, SIAM J. Sci. Comput. 36(5):A2248-A2271, 2014.
* S. V. Dolgov, D. V. Savostyanov, "... Part II: Faster algorithm and
  application to nonsymmetric systems", arXiv:1304.1222.
* Ported from ``tt/amen/amen_mv.py`` of legacy ttpy (Dolgov, Savostyanov),
  which is itself a port of the MATLAB TT-Toolbox ``amen_mv.m``.

Notes
-----
Backend agnostic: all arithmetic goes through :mod:`tt.backend` and ``einops``,
the only numpy in here is the integer bookkeeping of ranks (``ry``, ``rz``) and
the singular-value ``chop``, which :mod:`tt.core._ops` owns.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from einops import einsum, rearrange

from .. import backend as bk
from ..core import _ops
from ..core.matrix import matrix
from ..core.vector import vector

__all__ = ["amen_mv", "AmenMvHistory"]


# --- history -----------------------------------------------------------------

@dataclass
class AmenMvHistory:
    """What the run knows about itself; recorded even with ``verb=0``.

    Attributes:
        tol: Requested relative accuracy.
        sweeps: One dict per half-sweep, with keys ``sweep, direction, max_dx,
            max_rank, res_est, time``.
        converged: ``max_dx < tol`` was reached before ``nswp`` was exhausted.
        max_dx: Largest relative change of a block during the last forward
            half-sweep -- the stopping indicator.  It is an *estimate* of the
            error of ``y``, not a bound.
        res_est: Norm of the last residual block ``z``, in units of ``||y||``.
            ``z`` lives in a rank-``kickrank`` subspace, so this is a lower
            estimate of the true relative residual, never an upper bound.
        ranks: TT ranks of the returned ``y``.
        nswp_done: Number of full sweeps performed.
        time: Wall-clock seconds.
    """

    tol: float
    sweeps: list = field(default_factory=list)
    converged: bool = False
    max_dx: float = float("nan")
    res_est: float = float("nan")
    ranks: list = field(default_factory=list)
    nswp_done: int = 0
    time: float = 0.0

    def __repr__(self):
        return (f"AmenMvHistory(sweeps={self.nswp_done}, "
                f"converged={self.converged}, max_dx={self.max_dx:.2e}, "
                f"res_est={self.res_est:.2e}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


# --- orthogonalization / truncation, with the two ``renorm`` paths -----------

def _gram_svd(a):
    """SVD of a tall matrix through its Gram matrix ``a^H a``.

    Returns ``(u, s, vh)`` with ``a ~ u diag(s) vh`` and orthonormal ``u``.
    Costs one ``(m, k) x (m, k)`` GEMM plus a ``k x k`` eigendecomposition
    instead of an ``(m, k)`` SVD, which is the win for ``m >> k``.

    The price is that the singular values are recovered as square roots of the
    eigenvalues of ``a^H a``: everything below ``sqrt(eps_machine) * s[0]`` is
    noise and is dropped here, and the orthogonality of ``u`` degrades like
    ``eps_machine * cond(a)^2`` rather than ``eps_machine``, because the Gram
    matrix squares the condition number.  Measured (float64, dense ``a`` of size
    4096x64 with a logarithmically graded spectrum, seed 90 -- reproduced by
    ``test_gram_orthogonality_degrades_with_the_condition_number``):

        cond(a)   gram ||u^H u - I||   direct QR ||q^H q - I||
        1e0       9.6e-15              2.9e-15
        1e3       6.0e-11              2.5e-15
        1e7       2.3e-03              2.8e-15

    The direct path is flat in ``cond(a)``; the Gram path is not, and at
    ``cond(a) = 1e7`` it has no orthogonality left to speak of.  The
    *reconstruction* ``u diag(s) vh ~ a`` stays at ``~1e-15`` throughout -- it is
    the orthogonality of the frame, not the factorization, that is lost, which
    is exactly what an ALS sweep depends on.  Never use ``renorm='gram'`` when
    the blocks may be ill-conditioned; see the ``amen_mv`` docstring for the
    end-to-end number.
    """
    ah = rearrange(a.conj(), "i j -> j i")
    w, v = bk.eigh(ah @ a)                    # ascending eigenvalues
    # descending order without negative strides (torch has none): fancy indexing
    order = np.arange(w.shape[0] - 1, -1, -1)
    s_np = np.sqrt(np.clip(np.asarray(bk.to_numpy(w), dtype=np.float64), 0.0,
                           None))[order]
    keep = max(1, int(np.sum(s_np > s_np[0] * np.sqrt(bk.eps_of(bk.dtype_of(a))))))
    v = v[:, order[:keep]]
    s = bk.asarray(s_np[:keep], bk.real_dtype(bk.dtype_of(a)),
                   backend=bk.backend_of(a))
    u = (a @ v) / s.reshape((1, keep))
    return u, s, rearrange(v.conj(), "i j -> j i")


def _svd(a, renorm):
    """``a = u diag(s) vh``; ``renorm='gram'`` takes the Gram short cut."""
    if renorm == "gram" and a.shape[0] > 5 * a.shape[1]:
        return _gram_svd(a)
    return bk.svd(a)


def _qr(a, renorm):
    """``a = q r`` with orthonormal columns of ``q``."""
    if renorm == "gram" and a.shape[0] > 5 * a.shape[1]:
        u, s, vh = _gram_svd(a)
        return u, s.reshape((-1, 1)) * vh
    return bk.qr(a)


def _lq(a, renorm):
    """``a = l q`` with orthonormal *rows* of ``q``."""
    q, r = _qr(rearrange(a.conj(), "i j -> j i"), renorm)
    return rearrange(r.conj(), "i j -> j i"), rearrange(q.conj(), "i j -> j i")


# --- the three contractions the whole method is made of ----------------------
#
# Index letters, used consistently below:
#   a, b : y (or z) ranks, left and right      n : row mode of A (mode of y)
#   i, j : x ranks, left and right             m : column mode of A (mode of x)
#   p, c : A ranks, left and right

def _project(phi, acore, xcore, direction):
    """Contract ``A`` and ``x`` into an interface, leaving the ``y`` legs free.

    ``lr``: ``phi[a,i,p] -> w[a,n,j,c]``, ``rl``: ``phi[b,j,c] -> w[b,n,i,p]``.
    This is the expensive step; both ``_apply`` and ``_phi_next`` reuse its
    result, which is why it is a separate function.
    """
    if direction == "lr":
        t = einsum(phi, xcore, "a i p, i m j -> a p m j")
        return einsum(t, acore, "a p m j, p n m c -> a n j c")
    t = einsum(phi, xcore, "b j c, i m j -> b c i m")
    return einsum(t, acore, "b c i m, p n m c -> b n i p")


def _apply(w, phi_other, direction):
    """Close ``w`` with the opposite interface: the local block of ``A x``."""
    if direction == "lr":
        return einsum(w, phi_other, "a n j c, b j c -> a n b")
    return einsum(w, phi_other, "b n i p, a i p -> a n b")


def _phi_next(w, ycore, direction):
    """Close ``w`` with a (conjugated) frame core: the next interface."""
    if direction == "lr":
        return einsum(w, ycore.conj(), "a n j c, a n b -> b j c")
    return einsum(w, ycore.conj(), "b n i p, a n b -> a i p")


def _phi_yy_next(phi, zcore, ycore, direction):
    """Interface between two TT vectors: ``Z^H Y`` over the contracted modes."""
    if direction == "lr":
        t = einsum(phi, ycore, "a e, e n f -> a n f")
        return einsum(t, zcore.conj(), "a n f, a n b -> b f")
    t = einsum(phi, ycore, "b f, e n f -> b n e")
    return einsum(t, zcore.conj(), "b n e, a n b -> a e")


def _normalized(phi, extnrm=None):
    """Scale an interface to unit norm; return ``(phi, scale)``.

    With ``extnrm`` the scale is imposed from outside instead (the ``z``
    interfaces must carry exactly the same scaling as the ``y`` ones, otherwise
    the residual ``crz - yz`` subtracts two differently scaled quantities).
    """
    if extnrm is not None:
        return phi / extnrm, extnrm
    nrm = float(bk.norm(phi))
    if nrm > 0:
        return phi / nrm, nrm
    return phi, 1.0


# --- argument normalization --------------------------------------------------

def _canonical_to_matrix(a):
    """Canonical (CP) format ``a[k] = [A_k^(1), ..., A_k^(ra)]`` -> TT-matrix.

    The TT cores are diagonal in the rank index, so the TT-matrix has rank
    ``ra`` and represents ``sum_s kron_k A_k^(s)`` exactly.  Note that the local
    products then cost a factor ``ra`` more than a dedicated canonical code path
    would (a dense ``(ra, n, m, ra)`` core instead of ``ra`` separate
    ``(n, m)`` blocks); the format is supported for compatibility, not for
    speed.
    """
    d = len(a)
    ra = len(a[0])
    cores = []
    for k, terms in enumerate(a):
        if len(terms) != ra:
            raise ValueError(
                f"canonical A: mode {k} has {len(terms)} terms, mode 0 has {ra}")
        blocks = [bk.asarray(t) for t in terms]
        n, m = blocks[0].shape
        left = 1 if k == 0 else ra
        right = 1 if k == d - 1 else ra
        core = bk.zeros((left, n, m, right), dtype=bk.dtype_of(blocks[0]),
                        like=blocks[0])
        for s, blk in enumerate(blocks):
            if blk.shape != (n, m):
                raise ValueError(f"canonical A: term {s} of mode {k} has shape "
                                 f"{tuple(blk.shape)}, expected {(n, m)}")
            core[0 if left == 1 else s, :, :, 0 if right == 1 else s] = blk
        cores.append(core)
    return matrix.from_list(cores)


def _matrix_cores(A, d):
    """Bring the ``A`` argument to a single list of ``(ra, n, m, ra)`` cores."""
    if isinstance(A, matrix):
        cores = matrix.to_list(A)
    elif isinstance(A, (list, tuple)) and A and isinstance(A[0], matrix):
        # A list of TT-matrices means their sum.  Summing them up front is not
        # a shortcut: the block-diagonal sum has rank sum(ra_s), and both the
        # interfaces and the local products then cost exactly what carrying the
        # terms separately would cost.  One code path, same flops.
        total = A[0]
        for term in A[1:]:
            total = total + term
        cores = matrix.to_list(total)
    elif isinstance(A, (list, tuple)) and A and isinstance(A[0], (list, tuple)):
        cores = matrix.to_list(_canonical_to_matrix(A))
    elif isinstance(A, (list, tuple)) and A and hasattr(A[0], "ndim"):
        if A[0].ndim != 4:
            raise ValueError(
                f"a list of raw A cores must have ndim 4 (r,n,m,r), got {A[0].ndim}")
        cores = [bk.asarray(c) for c in A]
    else:
        raise TypeError(
            "A: expected a tt.matrix, a list of tt.matrix (summed), a list of "
            "(r,n,m,r) cores, or a canonical list of lists of dense blocks; "
            f"got {type(A).__name__}")
    if len(cores) != d:
        raise ValueError(f"A has {len(cores)} cores, x has {d}")
    return cores


def _vector_cores(x):
    """Cores of ``x`` plus a flag telling how to give the answer back."""
    if isinstance(x, vector):
        return list(x.cores), True
    if isinstance(x, (list, tuple)) and x and hasattr(x[0], "ndim"):
        return [bk.asarray(c) for c in x], False
    raise TypeError("x: expected a tt.vector or a list of (r,n,r) cores, "
                    f"got {type(x).__name__}")


# --- the method --------------------------------------------------------------

def amen_mv(A, x, tol, y=None, z=None, nswp=20, kickrank=4, kickrank2=0,
            verb=1, init_qr=True, renorm='direct', fkick=False,
            seed=None, return_history=False):
    """Approximate ``y = A x`` in the TT format by the AMEn iteration.

    Args:
        A: The matrix.  A :class:`tt.matrix`; or a list of :class:`tt.matrix`,
            which is interpreted as their **sum**; or a canonical (CP) list
            ``A[k] = [A_k^(1), ..., A_k^(ra)]`` of dense ``(n_k, m_k)`` blocks,
            meaning ``sum_s kron_k A_k^(s)``; or a raw list of ``(r,n,m,r)``
            cores.
        x: The vector, a :class:`tt.vector` or a list of ``(r,n,r)`` cores.
            The return type follows this one.
        tol: Relative Frobenius accuracy.  Blocks are truncated at
            ``tol / sqrt(d)`` and the iteration stops when the largest relative
            change of a block over a forward half-sweep drops below ``tol``.
        y: Initial guess (default: a random rank-2 TT).
        z: Initial guess for the residual ``A x - y`` (default: random of rank
            ``kickrank + kickrank2``).
        nswp: Maximal number of sweeps.  A sweep raises each TT rank of ``y`` by
            at most ``kickrank + kickrank2``, so reaching a target rank ``r``
            from the default rank-2 guess needs at least ``(r - 2)/kickrank``
            sweeps.  This is the usual reason for a non-convergence warning, and
            it is cheaper to fix with ``kickrank`` than with ``nswp``: measured
            on ``d=16``, ``n=m=8``, ``r_A=r_x=12``, ``tol=1e-8`` (exact ranks
            144), ``kickrank=4`` needs 36 sweeps / 27 s, ``kickrank=16`` needs 9
            sweeps / 6.8 s and ``kickrank=40`` needs 4 sweeps / 6.1 s, all three
            landing on rank 144 at ``1.5e-14``.  With ``nswp=20, kickrank=4``
            the same problem stops at rank 82 with a relative error of ``0.86``
            -- and warns, loudly, rather than returning it as an answer.
        kickrank: Rank of the residual enrichment (0 switches AMEn off and
            leaves plain one-site ALS, which cannot increase ranks).  It bounds
            the per-sweep rank growth; see ``nswp``.
        kickrank2: Extra *random* enrichment on top of the residual one; with
            ``kickrank2 > 0`` the residual block is first compressed to
            ``kickrank`` columns and ``kickrank2`` random ones are appended.
        verb: 0 silent, 1 one line per half-sweep, 2 one line per block.
            The history is recorded at every level, including 0.
        init_qr: Orthogonalize the initial guess.  ``False`` is a promise that
            ``y`` is already left-orthogonal; the promise is **checked** (once,
            at a cost comparable to the QR it skips) and a violation raises.
        renorm: ``'direct'`` (QR/SVD, default) or ``'gram'`` (orthogonalize
            through the Gram matrix when a block has more than 5x more rows
            than columns).  They are **not** equivalent: ``'gram'`` squares the
            condition number of the block.  Measured end-to-end (float64,
            reproduced by the two ``test_gram_*`` cases): on a well-conditioned
            random operator (``d=6``, ``n=m=8``, ``r_A=r_x=3``, ``tol=1e-10``)
            both reach ``~3e-15`` relative error, ``direct`` 3.3e-15 vs
            ``gram`` 3.7e-15.  On an ill-conditioned one (``A = eye([8]*6)``,
            ``x`` with a block spectrum spanning ``1e-10``, ``tol=1e-12``)
            ``direct`` reaches ``6e-16`` while ``gram`` stalls at ``1e-8`` --
            seven orders of magnitude worse, and it does not even report
            convergence (``max_dx`` plateaus at ``~2e-10``).  Use ``'gram'``
            only for tall thin blocks of moderate condition number.
        fkick: Also enrich during the forward half-sweep.  It lowers the error
            of a single call but leaves ``y`` with a less compact structure
            (the extra directions are not truncated afterwards), which shows up
            in subsequent matvecs; hence the ``False`` default of the original.
        seed: Seed for the random initial guesses and for ``kickrank2``, so a
            run can be reproduced.
        return_history: Also return the :class:`AmenMvHistory`.

    Returns:
        ``(y, z)``, or ``(y, z, history)`` with ``return_history=True``.
        ``z`` is the last approximation of the residual ``A x - y`` (scaled so
        that ``||y|| = 1``); pass it back as ``z0`` to warm-start a related
        matvec.  It is ``None`` when ``kickrank + kickrank2 == 0``.

    Raises:
        ValueError: on inconsistent shapes, or when ``init_qr=False`` is a
            false promise.

    Warns:
        UserWarning: when ``nswp`` sweeps were spent without reaching ``tol``.
            The returned ``y`` is then whatever the last sweep produced -- the
            history says so, and ``history.max_dx`` is the value actually
            reached.
    """
    t_start = time.time()
    if renorm not in ("direct", "gram"):
        raise ValueError(f"renorm must be 'direct' or 'gram', got {renorm!r}")

    xc, x_was_vector = _vector_cores(x)
    d = len(xc)
    ac = _matrix_cores(A, d)

    dtype = bk.result_dtype(bk.dtype_of(ac[0]), bk.dtype_of(xc[0]))
    ac = [bk.asarray(c, dtype, backend=bk.backend_of(xc[0])) for c in ac]
    xc = _ops.to_dtype(xc, dtype)

    n = [int(c.shape[1]) for c in ac]
    m = [int(c.shape[2]) for c in ac]
    ra = [int(ac[0].shape[0])] + [int(c.shape[3]) for c in ac]
    rx = _ops.ranks(xc)
    if [int(c.shape[1]) for c in xc] != m:
        raise ValueError(f"mode mismatch: A has column modes {m}, x has "
                         f"{[int(c.shape[1]) for c in xc]}")
    if ra[0] != 1 or ra[-1] != 1 or rx[0] != 1 or rx[-1] != 1:
        raise ValueError("amen_mv needs boundary ranks 1 for A and x, got "
                         f"A: {ra[0]},{ra[-1]}  x: {rx[0]},{rx[-1]}")

    rng = np.random.default_rng(seed)
    if y is None:
        yc = _ops.random_tt(n, [1] + [2] * (d - 1) + [1], dtype=dtype,
                            like=xc[0], seed=rng.integers(2 ** 31))
    else:
        yc, _ = _vector_cores(y)
        yc = _ops.to_dtype(yc, dtype)
        if [int(c.shape[1]) for c in yc] != n:
            raise ValueError(f"y has modes {[int(c.shape[1]) for c in yc]}, "
                             f"A has row modes {n}")
    ry = _ops.ranks(yc)

    kick = kickrank + kickrank2
    if kick > 0:
        if z is None:
            zc = _ops.random_tt(n, [1] + [kick] * (d - 1) + [1], dtype=dtype,
                                like=xc[0], seed=rng.integers(2 ** 31))
        else:
            zc, _ = _vector_cores(z)
            zc = _ops.to_dtype(zc, dtype)
        rz = _ops.ranks(zc)
    else:
        zc, rz = None, None

    one = bk.zeros((1, 1, 1), dtype=dtype, like=xc[0]) + 1.0
    phiyax = [None] * (d + 1)
    phiyax[0] = one
    phiyax[d] = one
    if kick > 0:
        phizax = [None] * (d + 1)
        phizax[0] = one
        phizax[d] = one
        flat_one = bk.zeros((1, 1), dtype=dtype, like=xc[0]) + 1.0
        phizy = [None] * (d + 1)
        phizy[0] = flat_one
        phizy[d] = flat_one

    nrms = np.ones(d)
    hist = AmenMvHistory(tol=float(tol))

    # --- initial left-to-right orthogonalization of y (and z) ----------------
    for i in range(d - 1):
        if init_qr:
            q, r = _qr(rearrange(yc[i], "a n b -> (a n) b"), renorm)
            nrmr = float(bk.norm(r))
            if nrmr > 0:
                r = r / nrmr
            ry[i + 1] = q.shape[1]
            yc[i] = rearrange(q, "(a n) b -> a n b", n=n[i])
            yc[i + 1] = einsum(r, yc[i + 1], "b e, e nn f -> b nn f")
        else:
            _check_left_orthogonal(yc[i], i)
        w = _project(phiyax[i], ac[i], xc[i], "lr")
        phiyax[i + 1], nrms[i] = _normalized(_phi_next(w, yc[i], "lr"))

        if kick > 0:
            q, r = _qr(rearrange(zc[i], "a n b -> (a n) b"), renorm)
            nrmr = float(bk.norm(r))
            if nrmr > 0:
                r = r / nrmr
            rz[i + 1] = q.shape[1]
            zc[i] = rearrange(q, "(a n) b -> a n b", n=n[i])
            zc[i + 1] = einsum(r, zc[i + 1], "b e, e nn f -> b nn f")
            wz = _project(phizax[i], ac[i], xc[i], "lr")
            phizax[i + 1], _ = _normalized(_phi_next(wz, zc[i], "lr"),
                                           extnrm=nrms[i])
            phizy[i + 1] = _phi_yy_next(phizy[i], zc[i], yc[i], "lr")

    # --- sweeps --------------------------------------------------------------
    i = d - 1
    direct = -1
    swp = 1
    max_dx = 0.0
    nrmz = float("nan")
    tol_local = float(tol) / np.sqrt(d)
    cry = None

    while True:
        # One projection per block and per tensor (y, z): everything else in
        # this iteration reuses w / wz, which is where the flops are.
        if direct > 0:
            w = _project(phiyax[i], ac[i], xc[i], "lr")
            cry = _apply(w, phiyax[i + 1], "lr")
        else:
            w = _project(phiyax[i + 1], ac[i], xc[i], "rl")
            cry = _apply(w, phiyax[i], "rl")

        nrm = float(bk.norm(cry))
        if nrm > 0:
            cry = cry / nrm
            nrms[i] = nrm
        else:
            nrms[i] = 1.0
        dx = float(bk.norm(cry - yc[i]))
        max_dx = max(max_dx, dx)
        r_new = ry[i + 1] if direct > 0 else ry[i]

        if direct > 0 and i < d - 1:
            u, s, vh = _svd(rearrange(cry, "a n b -> (a n) b"), renorm)
            r_new = max(1, min(_ops.chop(s, tol_local * float(bk.norm(s))),
                               u.shape[1]))
            u = u[:, :r_new]
            sv = s[:r_new].reshape((r_new, 1)) * vh[:r_new, :]   # (r, ry[i+1])

            if kick > 0:
                cry_t = rearrange(u @ sv, "(a n) b -> a n b", n=n[i])
                wz = _project(phizax[i], ac[i], xc[i], "lr")
                crz = _apply(wz, phizax[i + 1], "lr")
                ys = einsum(cry_t, phizy[i + 1], "a n b, c b -> a n c")
                yz = einsum(phizy[i], ys, "e a, a n c -> e n c")
                crz = crz / nrms[i] - yz
                nrmz = float(bk.norm(crz))
                crz = rearrange(crz, "e n c -> (e n) c")
                if kickrank2 > 0:
                    uz, _, _ = bk.svd(crz)
                    crz = bk.concatenate(
                        [uz[:, :min(uz.shape[1], kickrank)],
                         bk.asarray(rng.standard_normal(
                             (rz[i] * n[i], kickrank2)), dtype,
                             backend=bk.backend_of(xc[0]))], axis=1)
                if fkick:
                    crs = _apply(w, phizax[i + 1], "lr") / nrms[i] - ys
                    u = bk.concatenate(
                        [u, rearrange(crs, "a n c -> (a n) c")], axis=1)
                    u, rr = _qr(u, renorm)
                    pad = bk.zeros((crs.shape[2], sv.shape[1]), dtype=dtype,
                                   like=sv)
                    sv = rr @ bk.concatenate([sv, pad], axis=0)
                    r_new = u.shape[1]

            yc[i] = rearrange(u, "(a n) b -> a n b", n=n[i])
            yc[i + 1] = einsum(sv, yc[i + 1], "b e, e nn f -> b nn f")
            ry[i + 1] = r_new
            phiyax[i + 1], nrms[i] = _normalized(_phi_next(w, yc[i], "lr"))

            if kick > 0:
                qz, _ = _qr(crz, renorm)
                rz[i + 1] = qz.shape[1]
                zc[i] = rearrange(qz, "(a n) b -> a n b", n=n[i])
                phizax[i + 1], _ = _normalized(_phi_next(wz, zc[i], "lr"),
                                               extnrm=nrms[i])
                phizy[i + 1] = _phi_yy_next(phizy[i], zc[i], yc[i], "lr")

        elif direct < 0 and i > 0:
            u, s, vh = _svd(rearrange(cry, "a n b -> a (n b)"), renorm)
            r_new = max(1, min(_ops.chop(s, tol_local * float(bk.norm(s))),
                               u.shape[1]))
            v = vh[:r_new, :]                                    # (r, n*ry[i+1])
            us = u[:, :r_new] * s[:r_new].reshape((1, r_new))    # (ry[i], r)

            if kick > 0:
                cry_t = rearrange(us @ v, "a (n b) -> a n b", n=n[i])
                wz = _project(phizax[i + 1], ac[i], xc[i], "rl")
                crz = _apply(wz, phizax[i], "rl")
                ys = einsum(phizy[i], cry_t, "e a, a n b -> e n b")
                yz = einsum(ys, phizy[i + 1], "e n b, c b -> e n c")
                crz = crz / nrms[i] - yz
                nrmz = float(bk.norm(crz))
                crz = rearrange(crz, "e n c -> e (n c)")
                if kickrank2 > 0:
                    _, _, vz = bk.svd(crz)
                    crz = bk.concatenate(
                        [vz[:min(vz.shape[0], kickrank), :],
                         bk.asarray(rng.standard_normal(
                             (kickrank2, n[i] * rz[i + 1])), dtype,
                             backend=bk.backend_of(xc[0]))], axis=0)
                # The backward enrichment of y itself: this is the AMEn step.
                # z on the left, y on the right, so it closes the *y* interface
                # w with the *z* left interface.
                crs = _apply(w, phizax[i], "rl") / nrms[i] - ys
                v = bk.concatenate(
                    [v, rearrange(crs, "e n b -> e (n b)")], axis=0)
                ll, v = _lq(v, renorm)
                pad = bk.zeros((us.shape[0], crs.shape[0]), dtype=dtype, like=us)
                us = bk.concatenate([us, pad], axis=1) @ ll
                r_new = v.shape[0]

            yc[i - 1] = einsum(yc[i - 1], us, "e nn a, a b -> e nn b")
            yc[i] = rearrange(v, "b (n f) -> b n f", n=n[i])
            ry[i] = r_new
            phiyax[i], nrms[i] = _normalized(_phi_next(w, yc[i], "rl"))

            if kick > 0:
                _, qz = _lq(crz, renorm)
                rz[i] = qz.shape[0]
                zc[i] = rearrange(qz, "b (n f) -> b n f", n=n[i])
                phizax[i], _ = _normalized(_phi_next(wz, zc[i], "rl"),
                                           extnrm=nrms[i])
                phizy[i] = _phi_yy_next(phizy[i + 1], zc[i], yc[i], "rl")

        if verb > 1:
            print(f"amen_mv: swp=[{swp},{i}], dx={dx:.3e}, r={r_new}, "
                  f"|z|={nrmz:.3e}")

        at_end = (direct > 0 and i == d - 1) or (direct < 0 and i == 0)
        if at_end:
            hist.sweeps.append(dict(sweep=swp, direction="lr" if direct > 0
                                    else "rl", max_dx=max_dx,
                                    max_rank=int(max(ry)), res_est=nrmz,
                                    time=time.time() - t_start))
            if verb > 0:
                print(f"amen_mv: swp={swp}{{{(1 - direct) // 2}}}, "
                      f"max_dx={max_dx:.3e}, max_r={max(ry)}")
            if direct > 0:
                hist.max_dx = max_dx
                hist.nswp_done = swp
                if max_dx < tol:
                    hist.converged = True
                    break
                if swp == nswp:
                    break
            yc[i] = cry
            if direct > 0:
                swp += 1
            max_dx = 0.0
            direct = -direct
        else:
            i += direct

    yc[d - 1] = cry
    hist.res_est = nrmz

    if kick > 0:
        # The legacy code leaves the last z core stale (its left rank belongs to
        # the previous sweep), i.e. it returns a rank-inconsistent z.  Recompute
        # it here so that z is a valid TT vector and really is the residual
        # block: ||z|| is then the residual estimate reported in the history.
        wz = _project(phizax[d - 1], ac[d - 1], xc[d - 1], "lr")
        crz = _apply(wz, phizax[d], "lr")
        ys = einsum(cry, phizy[d], "a n b, c b -> a n c")
        yz = einsum(phizy[d - 1], ys, "e a, a n c -> e n c")
        zc[d - 1] = crz / nrms[d - 1] - yz
        hist.res_est = float(_ops.norm(zc))

    # Put the accumulated scale back, spread evenly over the cores: the stored
    # cores are all O(1), so this cannot overflow the way prod(nrms) could.
    scale = np.exp(np.sum(np.log(nrms)) / d)
    yc = [c * scale for c in yc]

    hist.ranks = _ops.ranks(yc)
    hist.time = time.time() - t_start
    if not hist.converged:
        warnings.warn(
            f"amen_mv did not converge: {hist.nswp_done} sweeps, "
            f"max_dx={hist.max_dx:.3e} > tol={tol:.3e}. The returned y is the "
            "last iterate; raise nswp or loosen tol.", UserWarning, stacklevel=2)

    if x_was_vector:
        yc = vector.from_list(yc)
        zc = vector.from_list(zc) if kick > 0 else None
    if return_history:
        return yc, zc, hist
    return yc, zc


def _check_left_orthogonal(core, k, tol=1e-8):
    """Fail loudly if ``init_qr=False`` was a false promise."""
    q = rearrange(core, "a n b -> (a n) b")
    qh = rearrange(q.conj(), "i j -> j i")
    err = float(bk.norm(qh @ q - bk.eye(q.shape[1], dtype=bk.dtype_of(q),
                                        like=q)))
    if err > tol * max(1.0, np.sqrt(q.shape[1])):
        raise ValueError(
            f"init_qr=False promises a left-orthogonal y0, but core {k} has "
            f"||Q^H Q - I|| = {err:.2e}. Pass init_qr=True (the QR costs one "
            "pass over y) or orthogonalize y0 yourself.")
