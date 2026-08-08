"""AMEn linear solver: ``A x = f`` in the TT format.

The Alternating Minimal Energy method of Dolgov and Savostyanov.  Plain
one-site ALS for a linear system minimizes the energy (or the residual) over
one TT block at a time; it converges but it cannot *increase* the TT ranks, so
it stalls at whatever rank the initial guess had.  AMEn fixes that by enriching
the left basis of every block with a cheap rank-``kickrank`` approximation
``z`` of the global residual ``A x - f``, which is the steepest-descent
direction of the energy functional: one ALS sweep plus the enrichment inherits
the (geometric) convergence of steepest descent while keeping the ALS cost.

Structure of one sweep (exactly the structure of the Fortran original,
``tt/tt-fort/ttamen.f90`` of legacy ttpy):

1. right-to-left orthogonalization of ``x`` and ``z``, building the right
   interfaces ``Phi^R``;
2. left-to-right optimization: at block ``k`` solve the *local* system

       B_k w = rhs_k,
       B_k[c,i,e; b,j,f]   = sum_{p,q} PhiA^L_k[c,b,p] A_k[p,i,j,q] PhiA^R_{k+1}[e,f,q],
       rhs_k[c,i,e]        = sum_{g,h}  Phif^L_k[c,g] f_k[g,i,h] Phif^R_{k+1}[e,h],

   densely (size < ``max_full_size``) or by matrix-free GMRES, truncate the
   block, and enrich it with the residual block of ``z``.

Interfaces
----------
``PhiA^L_k[c,b,p]`` contracts cores ``0..k-1`` of ``X^H A X`` and carries
(test rank, trial rank, A rank) in that order; ``PhiA^R_k`` is the same for
cores ``k..d-1``.  ``Phif`` interfaces carry (test rank, f rank).  The
contractions themselves are the ALS interface algebra shared with
:mod:`tt.algs.amen_mv`, which owns them (``_project``, ``_apply``,
``_phi_next``, ``_phi_yy_next``); if a third algorithm ever needs them they
should move to a dedicated ``tt/algs/_als.py``.  The local operator here is
*square* (a linear solve): ``n_k == m_k`` is required and checked.

What is reported
----------------
Every sweep records ``max_dx`` (largest relative block update), ``max_res``
(largest *local* relative residual seen before the local solves), ``max_rank``,
and the wall time.  ``max_res`` is the default stopping criterion: it is the
residual of the incoming iterate at every block, so it costs nothing beyond the
sweep that was happening anyway.

The exact global residual ``||A x - f|| / ||f||`` is available behind
``check_true_res=True`` and is **off by default**: forming ``A x`` multiplies
the ranks, and on a preconditioned operator that is ruinous for a number that is
only reported (``docs/NUMERICS.md``).  The history is on the returned vector as
``x.amen_info`` and, with ``return_info=True``, returned alongside it.  If the
requested accuracy is not reached within ``nswp`` sweeps the solver **warns
with the residual it actually achieved** and marks ``info.converged = False``;
it never reports success it did not verify.

References
----------
* S. V. Dolgov, D. V. Savostyanov, "Alternating minimal energy methods for
  linear systems in higher dimensions. Part I: SPD systems", arXiv:1301.6068,
  SIAM J. Sci. Comput. 36(5):A2248-A2271, 2014.
* S. V. Dolgov, D. V. Savostyanov, "... Part II: Faster algorithm and
  application to nonsymmetric systems", arXiv:1304.1222.
* Legacy implementation: ``tt/tt-fort/ttamen.f90``, ``ttals.f90``,
  ``ttnodeop.f90``, ``ttlocsolve.f90`` of ttpy (Dolgov, Savostyanov).

Known limits
------------
* The attainable relative residual is bounded from below by
  ``eps_machine * ||A|| ||x|| / ||f||``, so a tight ``eps`` on an
  ill-conditioned system cannot succeed for anyone.  The solver reports the
  failure; it does not pretend.
* The residual the solver reports about itself is computed in TT arithmetic and
  is not optimistic.
* The local solver is restarted GMRES, which is known to stagnate on strongly
  non-normal operators whose spectrum surrounds the origin, no matter how well
  conditioned they are.  When that happens the failure message names the local
  solver and the remedies (``max_full_size``, ``local_iters``,
  ``local_restart``, ``local_prec``).

Notes
-----
Backend agnostic through :mod:`tt.backend` and ``einops`` with one deliberate
exception: the ``(m+1) x m`` Hessenberg matrix of the local GMRES and its
least-squares problem live in numpy (see :func:`_gmres`).  It is a tiny matrix
(``m = local_restart <= 40``), but it means one host-device sync per GMRES
inner iteration on a GPU backend.

The measurements behind the limits and the defaults -- accuracy floors, the cost
of ``check_true_res``, where the dense local solve stops paying -- are in
``docs/NUMERICS.md``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from einops import rearrange
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from ..core import _ops
from ..core.vector import vector
from . import _fast
from .amen_mv import (_apply, _matrix_cores, _phi_next, _phi_yy_next, _project,
                      _vector_cores)

__all__ = ["amen_solve", "AmenSolveHistory"]

#: The local tolerance is this much tighter than the global one, so that the
#: local solves are not the bottleneck of the outer iteration (``resid_damp``
#: in ttamen.f90).
RESID_DAMP = 2.0

_PREC_ALIASES = {"": "n", "n": "n", "none": "n",
                 "c": "c", "cjacobi": "c", "center": "c",
                 "l": "l", "ljacobi": "l", "left": "l",
                 "r": "r", "rjacobi": "r", "right": "r"}


# --- history -----------------------------------------------------------------

@dataclass
class AmenSolveHistory:
    """What the run knows about itself; recorded in full even with ``verb=0``.

    Attributes:
        tol: Requested relative accuracy.
        sweeps: One dict per sweep with keys ``sweep, max_dx, max_res,
            max_rank, true_res, time, local_matvecs, local_direct,
            local_solves, local_failed, local_failed_direct,
            local_failed_gmres``.
        converged: Whether the stopping criterion was met (see
            :func:`amen_solve` for which criterion that is).
        max_dx: Largest relative block update of the sweep that produced the
            returned iterate (``best_sweep``).
        max_res: Largest local relative residual of that same sweep.
        true_res: ``||A x - f|| / ||f||`` of the returned ``x``, computed
            exactly in the TT format.  ``nan`` iff ``check_true_res=False``.
        ranks: TT ranks of the returned ``x``.
        nswp_done: Number of sweeps performed.
        best_sweep: Sweep whose iterate is actually returned (the one with the
            smallest true residual).  0 iff ``check_true_res=False``, in which
            case the last iterate is returned.
        time: Wall-clock seconds.
        message: Human-readable outcome, identical to the text of the warning
            raised on failure.
    """

    tol: float
    sweeps: list = field(default_factory=list)
    converged: bool = False
    max_dx: float = float("nan")
    max_res: float = float("nan")
    true_res: float = float("nan")
    ranks: list = field(default_factory=list)
    nswp_done: int = 0
    best_sweep: int = 0
    time: float = 0.0
    message: str = ""

    def __repr__(self):
        return (f"AmenSolveHistory(sweeps={self.nswp_done}, "
                f"converged={self.converged}, max_dx={self.max_dx:.2e}, "
                f"max_res={self.max_res:.2e}, true_res={self.true_res:.2e}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


# --- small linear-algebra helpers -------------------------------------------

def _vdot(a, b):
    """``<a, b> = sum(conj(a) * b)`` over all axes, as a python scalar.

    On numpy this goes straight to BLAS ``dot``/``zdotc``: the readable version
    (``(a.conj() * b).sum()``) allocates two temporaries per call, and the
    Gram-Schmidt loop calls this 23569 times in one AMEn solve.
    """
    if type(a) is np.ndarray and type(b) is np.ndarray:
        return np.vdot(a, b)
    val = (a.conj() * b).sum()
    return complex(val) if bk.is_complex(a) else float(val)


def _qr_left(core):
    """``core = q r`` with ``q`` left-orthogonal; returns ``(q, r)``."""
    r1, n, r2 = core.shape
    q, rmat = bk.qr(core.reshape((r1 * n, r2)))
    return q.reshape((r1, n, q.shape[1])), rmat


def _rq_right(core):
    """``core = l q`` with ``q`` right-orthogonal; returns ``(l, q)``."""
    r1, n, r2 = core.shape
    lmat, q = _ops._lq(core.reshape((r1, n * r2)))   # single owner of the LQ
    return lmat, q.reshape((q.shape[0], n, r2))


def _push_left(core, rmat):
    """``rmat`` (from a QR at ``k``) absorbed into core ``k+1``."""
    r0, n, r1 = core.shape
    return (rmat @ core.reshape((r0, n * r1))).reshape((rmat.shape[0], n, r1))


def _push_right(core, lmat):
    """``lmat`` (from an LQ at ``k``) absorbed into core ``k-1``."""
    r0, n, r1 = core.shape
    return (core.reshape((r0 * n, r1)) @ lmat).reshape((r0, n, lmat.shape[1]))


# --- the local system --------------------------------------------------------
#
# Index letters: c, e -- test (row) ranks of x; b, f -- trial (column) ranks;
# p, q -- ranks of A; i -- row mode, j -- column mode.

def _local_matvec(phiL, acore, phiR, w):
    """``B_k w``: the local operator applied matrix-free."""
    return _apply(_project(phiL, acore, w, "lr"), phiR, "lr")


def _blas_layout(phiL, acore, phiR):
    """The three operands of the local matvec, laid out for BLAS.

    ``phi1`` is ``(a, i p)``, ``A`` is ``(p n, m c)``, ``phi2`` is ``(j, c b)``.
    """
    a, i, p = phiL.shape
    b, j, c = phiR.shape
    _, n, m, _ = acore.shape
    return (np.ascontiguousarray(phiL.reshape(a, i * p)),
            np.ascontiguousarray(acore.reshape(p * n, m * c)),
            np.ascontiguousarray(phiR.transpose(1, 2, 0).reshape(j, c * b)))


def _jacobi_c_inverse(phiL, acore, phiR):
    """The central block-Jacobi inverse, laid out as ``(n, n, b, f)``."""
    prec = _jacobi("c", phiL, acore, phiR)
    return getattr(prec, "invT", None)


def _residual_bytes(acores, xcores):
    """Bytes of the largest core of ``A x``, which is what forming it costs.

    Core ``k`` of the product has shape ``(rA_k rx_k, n_k, rA_{k+1} rx_{k+1})``:
    the ranks *multiply*.  For a well conditioned QTT problem that is harmless;
    for a preconditioned one it is not, by orders of magnitude
    (``docs/NUMERICS.md``).  Nothing here needs the full-rank object; it exists
    only long enough to be orthogonalized away.
    """
    worst = 0
    for a, x in zip(acores, xcores):
        worst = max(worst, (a.shape[0] * x.shape[0]) * a.shape[1]
                    * (a.shape[3] * x.shape[2]))
    return worst * 8


def _local_operator(phiL, acore, phiR):
    """Build ``w -> B_k w`` once, with the interfaces already laid out for BLAS.

    The generic contraction path re-arranges ``phiL`` and ``phiR`` on *every*
    application, and a local GMRES applies the operator tens of times with the
    same interfaces.  Hoisting that out is worth a factor of two: the same three
    products, with the operands pre-arranged, run at 52 GFLOPS against 26.5 for
    the generic path (53 us -> 27 us at r=34, n=2, R=4 -- the Fortran's whole
    inner iteration is 45 us, so this is the difference between parity and
    twice the cost).

    The sequence is the one in ``tt-fort/ttlocsolve.f90::d2d_mv``:
    ``x (rx1 m, rx2) @ phi2 (rx2, ra2 ry2)``, transpose, ``A (ra1 n, m ra2) @``,
    transpose, ``phi1 (ry1, rx1 ra1) @``.

    Returns:
        ``(apply, shape)`` -- the closure and the shape it maps to and from.
    """
    a, i, p = phiL.shape                    # y-rank, x-rank, A-rank (left)
    b, j, c = phiR.shape                    # y-rank, x-rank, A-rank (right)
    _, n, m, _ = acore.shape
    if type(phiL) is not np.ndarray:        # torch and friends: generic path
        return (lambda w: _local_matvec(phiL, acore, phiR, w)), (a, n, b)

    phi1, amat, phi2 = _blas_layout(phiL, acore, phiR)

    def apply(w):
        t = w.reshape(i * m, j) @ phi2                    # (i m, c b)
        t = t.reshape(i, m * c * b).T                     # (m c b, i)
        t = amat @ t.reshape(m * c, b * i)                # (p n, b i)
        t = t.reshape(p * n * b, i).T                     # (i, p n b)
        return (phi1 @ t.reshape(i * p, n * b)).reshape(a, n, b)

    return apply, (a, n, b)


def _dense_solve(mat, rhs, symmetric=False):
    """Solve the local system.

    A Cholesky route for the symmetric case was tried and dropped -- no benefit
    at the sizes these blocks actually have (``docs/NUMERICS.md``).  The
    ``symmetric`` flag is kept in the signature because callers know the answer
    cheaply and a future backend may use it.
    """
    return bk.solve(mat, rhs)


def _local_matrix(phiL, acore, phiR):
    """The local operator as a dense ``(ry1*n*ry2, rx1*m*rx2)`` matrix.

    The flattening matches ``w.reshape(-1)`` of a ``(r1, n, r2)`` block.
    """
    t = einsum(phiL, acore, "c b p, p i j q -> c b i j q")
    mat = einsum(t, phiR, "c b i j q, e f q -> c i e b j f")
    return rearrange(mat, "c i e b j f -> (c i e) (b j f)")


def _local_rhs(phiL, fcore, phiR):
    """``rhs_k``: the right-hand side projected on the current frames."""
    t = einsum(phiL, fcore, "c g, g i h -> c i h")
    return einsum(t, phiR, "c i h, e h -> c i e")


def _diag_of(phi):
    """``phi[b, b, p]`` -- the interface restricted to its diagonal frames."""
    ident = bk.eye(phi.shape[0], phi.shape[1], dtype=bk.dtype_of(phi), like=phi)
    return einsum(phi, ident, "a b p, a b -> a p")


def _invert_blocks(blocks, kind):
    """Invert the diagonal blocks, refusing loudly if one is singular."""
    try:
        return bk.inv(blocks)
    except Exception as exc:                 # LinAlgError differs per backend
        raise RuntimeError(
            f"local Jacobi preconditioner {kind!r}: a diagonal block of the "
            "local matrix is singular, the preconditioner does not exist; "
            "run with local_prec='n'") from exc


def _jacobi(kind, phiL, acore, phiR):
    """Block-Jacobi preconditioner for the local operator.

    The three variants of the legacy code are *the same object* seen with three
    different block groupings of the local matrix ``B_k``:

    * ``'c'`` (central): one ``n x n`` block per rank pair ``(b1, b2)``,
      i.e. ``B_k[b1,:,b2; b1,:,b2]``;
    * ``'l'`` (left): one ``(r1 n) x (r1 n)`` block per ``b2``;
    * ``'r'`` (right): one ``(n r2) x (n r2)`` block per ``b1``.

    In every case the block is exactly the corresponding diagonal block of the
    dense ``B_k``, so the preconditioner is the inverse of the block-diagonal
    part of the local matrix -- that identity is what the unit test checks.
    Applied on the right: GMRES iterates on ``B M`` and the correction is
    ``M y``.

    Raises:
        RuntimeError: if a diagonal block is singular.  A silent fallback to
            the identity would hide a broken preconditioner behind slow
            convergence; pass ``local_prec='n'`` if the blocks are singular by
            construction.
    """
    r1, n, m, r2 = phiL.shape[1], acore.shape[1], acore.shape[2], phiR.shape[1]
    if kind == "c":
        # Which way to build the blocks is a *cost* decision, not a "is numba
        # available" one.  The compiled kernel fuses all six loops, so it costs
        # r1 r2 n m R1 R2; the two-step contraction below costs
        # r1 n m R1 R2 + r1 r2 n m R2.  The fused form is therefore worse by a
        # factor ~min(r2, R1), and only wins when that factor is small enough
        # for numpy's dispatch overhead to dominate.
        #
        # It matters: at the shapes the kernel was tuned for (r=34, n=2, R_A=4)
        # the ratio is 3.6 and fusing wins.  On a BPX-preconditioned 2D system
        # (r=175, n=4, R_A=161) it is 84, the fused loop does 1.3e10 multiply-
        # adds where the staged one does 1.5e8, and building this preconditioner
        # took 48% of the whole solve -- for a local system that BPX has already
        # made well conditioned (measured: 6.7 GMRES iterations per block with
        # it, 7.1 without).
        big1, big2 = acore.shape[0], acore.shape[3]
        fused = r1 * r2 * n * m * big1 * big2
        staged = r1 * n * m * big1 * big2 + r1 * r2 * n * m * big2
        if (_fast.HAVE_NUMBA and type(phiL) is np.ndarray
                and phiL.dtype == np.float64 and acore.dtype == np.float64
                and fused <= 4 * staged):
            blocks = np.empty((phiL.shape[0], phiR.shape[0],
                               acore.shape[1], acore.shape[2]))
            _fast.jacobi_c_blocks(phiL, acore, phiR, blocks)
        else:
            t = einsum(_diag_of(phiL), acore, "b p, p i j q -> b i j q")
            blocks = einsum(t, _diag_of(phiR), "b i j q, f q -> b f i j")
    elif kind == "l":
        t = einsum(phiL, acore, "c b p, p i j q -> c b i j q")
        blocks = rearrange(
            einsum(t, _diag_of(phiR), "c b i j q, f q -> f c i b j"),
            "f c i b j -> f (c i) (b j)")
    elif kind == "r":
        t = einsum(_diag_of(phiL), acore, "b p, p i j q -> b i j q")
        blocks = rearrange(
            einsum(t, phiR, "b i j q, e f q -> b i e j f"),
            "b i e j f -> b (i e) (j f)")
    else:                                    # pragma: no cover - guarded above
        raise NotImplementedError(kind)

    if kind == "c":
        # The n = 2 path inverts the blocks itself, so the batched numpy inverse
        # below must not run first: it was doing the same 1156 two-by-two
        # inversions a second time (16 ms of a 200 ms solve).
        if _fast.HAVE_NUMBA and type(blocks) is np.ndarray and n == 2:
            invT = np.empty((2, 2, blocks.shape[0], blocks.shape[1]))
            ok = _fast.invert_2x2_blocks(blocks, invT)
            if not ok:
                raise RuntimeError(
                    "local Jacobi preconditioner 'c': a 2x2 diagonal block is "
                    "singular, the preconditioner does not exist; "
                    "run with local_prec='n'")

            def apply_c2(w):
                out = np.empty_like(w)
                return _fast.jacobi_c_apply(invT, w, out)

            apply_c2.invT = invT
            return apply_c2
        inv = _invert_blocks(blocks, kind)
        if type(inv) is np.ndarray and n * n <= 16:
            # n^2 elementwise products on (b, f) arrays instead of a batched
            # matmul over r1*r2 tiny n x n blocks: numpy spends 34 us
            # dispatching 1156 two-by-two products that take 10 us to do.
            invT = np.ascontiguousarray(np.transpose(inv, (2, 3, 0, 1)))

            def apply_c(w):   # noqa: D401
                out = np.empty_like(w)
                for i in range(n):
                    acc = invT[i, 0] * w[:, 0, :]
                    for jj in range(1, n):
                        acc = acc + invT[i, jj] * w[:, jj, :]
                    out[:, i, :] = acc
                return out

            apply_c.invT = invT       # the compiled path reuses this layout
            return apply_c
        inv = _invert_blocks(blocks, kind)
        return lambda w: einsum(inv, w, "b f i j, b j f -> b i f")
    inv = _invert_blocks(blocks, kind)
    if kind == "l":
        def apply_l(w):
            flat = rearrange(w, "b j f -> f (b j)")
            return rearrange(einsum(inv, flat, "f u v, f v -> f u"),
                             "f (c i) -> c i f", i=n)
        return apply_l

    def apply_r(w):
        flat = rearrange(w, "b j f -> b (j f)")
        return rearrange(einsum(inv, flat, "b u v, b v -> b u"),
                         "b (i e) -> b i e", i=n)
    return apply_r


def _gmres(matvec, b, tol, restart, maxit, prec=None):
    """Restarted, optionally right-preconditioned GMRES on TT blocks.

    Args:
        matvec: the local operator, block -> block.
        b: right-hand side block.
        tol: relative (to ``||b||``) residual to stop at.
        restart: Krylov dimension of one cycle (``local_restart``).
        maxit: number of cycles (``local_iters``).
        prec: right preconditioner or ``None``.

    Returns:
        ``(x, relres, nmatvec, converged)``.  ``relres`` is the *recomputed*
        residual of the returned iterate, not the Arnoldi estimate, so a
        non-converged run cannot report a tolerance it did not reach.

    Two things here are about cost, not mathematics.  The Krylov basis is one
    contiguous ``(restart+1, size)`` array, so orthogonalisation is two matrix
    products (classical Gram-Schmidt applied twice, which is as stable as the
    modified variant) instead of a python loop of ``j`` inner products -- that
    loop called ``_vdot`` 23569 times in one 2D solve.  And the least-squares
    problem is carried by Givens rotations, O(j) per step, instead of a fresh
    ``lstsq`` at every step.
    """
    bnorm = float(bk.norm(b))
    x = bk.zeros(b.shape, dtype=bk.dtype_of(b), like=b)
    if bnorm == 0.0:
        return x, 0.0, 0, True
    shape = b.shape
    size = int(np.prod(shape))
    dt = bk.dtype_of(b)
    cplx = bk.is_complex(b)
    eps_mach = bk.eps_of(dt)
    m = int(restart)

    fast = type(b) is np.ndarray          # in-place updates are safe on numpy
    r = b
    nmv = 0
    relres = 1.0
    for _cycle in range(max(int(maxit), 1)):
        beta = float(bk.norm(r))
        relres = beta / bnorm
        if relres <= tol:
            return x, relres, nmv, True

        V = bk.zeros((m + 1, size), dtype=dt, like=b)
        V[0] = r.reshape((size,)) / beta
        hess = bk.zeros((m + 1, m), dtype=dt, like=b)
        cs_l = [0.0] * m
        sn_l = [0.0] * m
        g = np.zeros(m + 1, dtype=np.complex128 if cplx else np.float64)
        g[0] = beta
        used = 0
        for j in range(m):
            vj = V[j].reshape(shape)
            w = matvec(prec(vj) if prec is not None else vj)
            nmv += 1
            w = w.reshape((size,))
            if fast:
                w = np.array(w, copy=True).reshape((size,))
            # Classical Gram-Schmidt, reorthogonalised only when the vector
            # actually lost orthogonality (Daniel-Gragg-Kaufman-Stewart: repeat
            # when the norm drops by more than 1/sqrt(2)).  Doing it twice
            # unconditionally cost 38 us per inner step against 19 us here, on a
            # 54 us matvec.
            nrm0 = float(bk.norm(w))
            # No .conj() on real data: numpy's conj allocates a copy of the whole
            # basis block (j x size doubles) on every inner step, which for
            # j = 30 and size = 2312 is half a megabyte of pure memory traffic.
            Vj = V[:j + 1]
            Vc = Vj.conj() if cplx else Vj
            # In place: every temporary here is a size-2312 allocation inside a
            # loop that runs thousands of times per solve.
            h = Vc @ w
            if fast:
                w -= h @ Vj
            else:
                w = w - h @ Vj
            hess[:j + 1, j] = h
            hnext = float(bk.norm(w))
            if hnext < 0.707 * nrm0:
                h2 = Vc @ w
                if fast:
                    w -= h2 @ Vj
                else:
                    w = w - h2 @ Vj
                hess[:j + 1, j] = hess[:j + 1, j] + h2
                hnext = float(bk.norm(w))
            hess[j + 1, j] = hnext

            # Python lists, not numpy scalars: this loop runs j times per inner
            # step, and indexing a numpy array to get a scalar costs about as
            # much as the arithmetic (15 us per step against 5 us here).
            col = [complex(v) if cplx else float(v)
                   for v in bk.to_numpy(hess[:j + 2, j])]
            for i in range(j):                     # apply the earlier rotations
                ci, si = cs_l[i], sn_l[i]
                t = ci * col[i] + si * col[i + 1]
                col[i + 1] = -(si.conjugate() if cplx else si) * col[i] \
                    + (ci.conjugate() if cplx else ci) * col[i + 1]
                col[i] = t
            denom = np.hypot(abs(col[j]), abs(col[j + 1]))
            if denom == 0.0:
                cs_l[j], sn_l[j] = 1.0, 0.0
            else:
                cs_l[j] = (col[j].conjugate() if cplx else col[j]) / denom
                sn_l[j] = (col[j + 1].conjugate() if cplx else col[j + 1]) / denom
            col[j] = cs_l[j] * col[j] + sn_l[j] * col[j + 1]
            col[j + 1] = 0.0
            hess[:j + 2, j] = bk.asarray(np.asarray(col), dt,
                                         backend=bk.backend_of(b))
            g[j + 1] = -(sn_l[j].conjugate() if cplx else sn_l[j]) * g[j]
            g[j] = cs_l[j] * g[j]
            used = j + 1
            if abs(g[j + 1]) <= tol * bnorm or hnext <= eps_mach * beta:
                break
            V[j + 1] = w / hnext

        rmat = np.asarray(bk.to_numpy(hess[:used, :used]))
        y = np.linalg.solve(np.triu(rmat), g[:used]) if used else np.zeros(0)
        step = (bk.asarray(y, dt, backend=bk.backend_of(b)) @ V[:used]).reshape(shape)
        x = x + (prec(step) if prec is not None else step)
        r = b - matvec(x)
        nmv += 1
        relres = float(bk.norm(r)) / bnorm
        if relres <= tol:
            return x, relres, nmv, True
    return x, relres, nmv, relres <= tol


def _solve_local(phiL, acore, phiR, rhs, tol, max_full_size, prec_kind,
                 local_iters, local_restart, symmetric=False):
    """Solve ``B_k sol = rhs``; returns ``(sol, info_dict)``."""
    size = int(rhs.shape[0] * rhs.shape[1] * rhs.shape[2])
    rhs_norm = float(bk.norm(rhs))
    if size < max_full_size:
        mat = _local_matrix(phiL, acore, phiR)
        sol = _dense_solve(mat, rhs.reshape((-1,)), symmetric).reshape(rhs.shape)
        res = _local_matvec(phiL, acore, phiR, sol) - rhs
        relres = float(bk.norm(res)) / rhs_norm if rhs_norm > 0 else 0.0
        # A dense solve is backward stable, not exact: on an ill-conditioned
        # local system its residual is ~cond*eps_machine and can sit above the
        # requested tolerance.  Reporting "converged" unconditionally would
        # hide that and misdirect the diagnosis of a stalled outer iteration.
        return sol, {"kind": "direct", "matvecs": 0, "relres": relres,
                     "converged": bool(relres <= tol), "size": size}
    fast = (_fast.HAVE_NUMBA and type(rhs) is np.ndarray
            and rhs.dtype == np.float64 and prec_kind in ("n", "c"))
    if fast:
        # One compiled call for the whole local solve: the inner loop never
        # returns to python.  Measured at r=34: 34 us per iteration against 63
        # in numpy (the Fortran it replaces is at 45).
        a, i, p = phiL.shape
        b_, j, c = phiR.shape
        _, n, m, _ = acore.shape
        phi1, amat, phi2 = _blas_layout(phiL, acore, phiR)
        invT = (_jacobi_c_inverse(phiL, acore, phiR) if prec_kind == "c"
                else np.zeros((n, n, 1, 1)))
        if invT is not None:
            sol_flat, relres, nmv, ok = _fast.gmres_local(
                phi1, amat, phi2, invT, prec_kind == "c",
                np.ascontiguousarray(rhs.reshape(-1)), float(tol),
                int(local_restart), max(int(local_iters), 1),
                i, m, j, p, n, c, b_, a)
            sol = sol_flat.reshape(rhs.shape)
            return sol, {"kind": "gmres", "matvecs": nmv, "relres": relres,
                         "converged": ok, "size": size}
    prec = None if prec_kind == "n" else _jacobi(prec_kind, phiL, acore, phiR)
    op, _ = _local_operator(phiL, acore, phiR)
    sol, relres, nmv, ok = _gmres(op, rhs, tol, local_restart, local_iters, prec)
    return sol, {"kind": "gmres", "matvecs": nmv, "relres": relres,
                 "converged": ok, "size": size}


# --- block truncation --------------------------------------------------------

def _truncate(core, next_core, tol, residual_ctx, rmax):
    """Split ``core`` by an SVD, choosing the rank, and pass the tail on.

    Args:
        core: block ``k``, shape ``(r1, n, r2)``, holding the norm.
        next_core: block ``k+1``, gets the singular tail.
        tol: relative tolerance (Frobenius or residual, see ``residual_ctx``).
        residual_ctx: ``None`` for Frobenius truncation; otherwise
            ``(phiL, acore, phiR, rhs, rhs_norm)`` and the rank is the smallest
            one whose *local residual* stays below ``tol`` -- the truncation of
            ``trunc_norm='residual'``, which is what makes AMEn insensitive to
            the (possibly huge) condition number when picking ranks.
        rmax: hard cap on the new rank, or ``None``.

    Returns:
        ``(core_new, next_new, core_old_basis)``.  The third one is the
        truncated block written back in the *old* right basis, which is what
        the enrichment needs (the ``z`` interfaces still refer to it).
    """
    r1, n, r2 = core.shape
    u, s, vh = bk.svd(core.reshape((r1 * n, r2)))
    rmin = int(s.shape[0])
    tail = s.reshape((rmin, 1)) * vh                      # (rmin, r2)

    if residual_ctx is None:
        rnew = max(1, _ops.chop(s, tol * float(bk.norm(s))))
    else:
        phiL, acore, phiR, rhs, rhs_norm = residual_ctx
        op, _ = _local_operator(phiL, acore, phiR)
        # Walked down from the top, not bisected.  The local residual is only
        # *nearly* monotone in the rank, and bisection on it picked ranks that
        # cost convergence (measured: a d=14 QTT Laplacian stopped at 1.3e-08
        # instead of reaching 1e-10).  The scan exits after a few steps in
        # practice, which is why it costs 30 ms of a 200 ms solve and not more.
        rnew = 1
        for r in range(rmin - 1, 0, -1):
            trial = (u[:, :r] @ tail[:r, :]).reshape((r1, n, r2))
            if float(bk.norm(op(trial) - rhs)) / rhs_norm > tol:
                rnew = r + 1
                break
    rnew = min(rnew, rmin)
    if rmax is not None:
        rnew = min(rnew, int(rmax))

    core_new = u[:, :rnew].reshape((r1, n, rnew))
    next_new = _push_left(next_core, tail[:rnew, :])
    core_old = (u[:, :rnew] @ tail[:rnew, :]).reshape((r1, n, r2))
    return core_new, next_new, core_old


# --- argument normalization --------------------------------------------------

def _canon_prec(local_prec):
    key = (local_prec or "n")
    key = key.lower() if isinstance(key, str) else key
    if key not in _PREC_ALIASES:
        raise NotImplementedError(
            f"local_prec={local_prec!r} is not implemented; supported values "
            "are 'n' (none), 'c'/'cjacobi' (central Jacobi), 'l'/'ljacobi' "
            "(left Jacobi), 'r'/'rjacobi' (right Jacobi)")
    return _PREC_ALIASES[key]


def _check_positive(**kwargs):
    """Reject nonsense integer arguments instead of quietly reinterpreting them.

    In the Fortran-era interface ``kickrank=-1`` meant "no enrichment" and
    ``rmax=0`` meant "rank 0, then whatever the enrichment adds": both are
    answers to a question the caller did not ask.
    """
    for name, (value, lo) in kwargs.items():
        if value is None:
            continue
        if int(value) < lo:
            raise ValueError(f"{name}={value!r}: expected an integer >= {lo}")


def _canon_trunc(trunc_norm):
    if trunc_norm in (1, "residual", "resid"):
        return 1
    if trunc_norm in (0, "fro", "frobenius"):
        return 0
    raise ValueError(
        f"trunc_norm={trunc_norm!r}: expected 1 / 'residual' or 0 / 'fro'")


# --- the method --------------------------------------------------------------

def amen_solve(A, f, x0, eps, kickrank=4, nswp=20, local_prec='c',
               local_iters=2, local_restart=40, trunc_norm=1, max_full_size=200,
               verb=1, *, rmax=None, seed=None, check_true_res=False,
               true_res_budget=2 * 1024 ** 3, return_info=False):
    """Solve ``A x = f`` in the TT format by the AMEn iteration.

    Args:
        A: The matrix, a :class:`tt.matrix` (or anything
            :func:`tt.algs.amen_mv._matrix_cores` accepts: a list of cores, a
            list of matrices meaning their sum, a canonical list).  Must be
            square mode-wise (``n_k == m_k``).
        f: Right-hand side, a :class:`tt.vector` or a list of cores.
        x0: Initial guess.  ``None`` gives a random rank-2 TT (use ``seed`` to
            make that reproducible).
        eps: Relative accuracy.
        kickrank: TT rank of the residual approximation ``z`` used to enrich
            the bases.  ``kickrank=0`` switches the enrichment off and leaves
            plain one-site ALS, which cannot increase the ranks of ``x0``.
        nswp: Maximal number of sweeps.
        local_prec: Local preconditioner for the GMRES path: ``'n'`` none,
            ``'c'`` central Jacobi, ``'l'`` left Jacobi, ``'r'`` right Jacobi
            (the legacy long names ``'cjacobi'`` etc. are accepted).  Anything
            else raises :class:`NotImplementedError`.
        local_iters: Number of GMRES restart cycles.
        local_restart: Krylov dimension of one GMRES cycle.
        trunc_norm: ``1`` / ``'residual'`` truncates blocks in the residual
            norm, ``0`` / ``'fro'`` in the Frobenius norm.
        max_full_size: Local systems strictly smaller than this are solved
            densely, larger ones by matrix-free GMRES.  The default is 200,
            not the 50 of ttpy 1.x: there the local solver was compiled
            Fortran, here it is interpreted, so the size at which a dense
            LAPACK solve stops being worth it is much larger (the crossover
            is measured in ``docs/NUMERICS.md``).  Raise it further if the
            local blocks are still small; lower it if a single dense solve of
            this size does not fit your time budget (cost grows as size^3).
        verb: 0 silent (the history is still recorded), 1 one line per sweep,
            2 one line per block.
        rmax: Hard cap on the TT rank chosen by the truncation.  The
            enrichment that follows it adds up to ``kickrank`` more, so the
            ranks of the returned ``x`` are bounded by ``rmax + kickrank``.
        seed: Integer seed (or ``None``).  The random ``z`` and the random
            ``x0`` are drawn from two *independent* streams spawned from it,
            so they never coincide.
        check_true_res: Compute ``||A x - f|| / ||f||`` exactly in the TT
            format after a sweep and use it as the stopping criterion.  **Off
            by default**, because forming it multiplies the ranks: core ``k``
            of ``A x`` has shape ``(rA_k rx_k, n_k, rA_{k+1} rx_{k+1})``.  On a
            preconditioned operator that is tens of gigabytes in a single core,
            for a number that is only reported and is thrown away immediately
            afterwards (``docs/NUMERICS.md``).

            With ``False`` the criterion is ``max_res`` (``max_dx`` for
            ``trunc_norm=0``): the local residual ``||B_k x_k - rhs_k||`` of
            every block *before* it is solved, i.e. how wrong the incoming
            iterate is.  It is a local quantity and can be optimistic, so
            ``info.true_res`` stays ``nan`` -- never a guess.  Turn this on for
            a certificate when the ranks are small enough to afford it; the run
            warns if the product would exceed ``true_res_budget``.
        true_res_budget: Bytes above which ``check_true_res=True`` warns before
            forming the product.  Reporting only -- it never silently skips the
            computation the caller asked for.
        return_info: Return ``(x, info)`` instead of ``x``.

    Returns:
        The solution as a :class:`tt.vector`, carrying the
        :class:`AmenSolveHistory` in ``x.amen_info``; with ``return_info`` the
        pair ``(x, info)``.

    Raises:
        ValueError: on mode/dimension mismatch, a non-square ``A``, a zero
            right-hand side (``A x = 0`` has the trivial solution and no
            relative residual), a negative ``eps``, or a non-positive
            ``nswp`` / ``rmax`` / ``local_iters`` / ``local_restart`` /
            negative ``kickrank``.
        NotImplementedError: on an unknown ``local_prec``.
        numpy.linalg.LinAlgError: if a local system is exactly singular (a
            singular ``A``); nothing is substituted for the missing solution.

    Warns:
        UserWarning: if the accuracy was not reached in ``nswp`` sweeps; the
        message carries the residual that *was* reached.

    Example:
        >>> import tt
        >>> from tt.algs.amen import amen_solve
        >>> A = tt.qlaplace_dd([10])
        >>> rhs = tt.ones(2, 10)
        >>> x, info = amen_solve(A, rhs, None, 1e-10, verb=0, seed=0,
        ...                      check_true_res=True, return_info=True)
        >>> bool(info.true_res < 1e-10)
        True
    """
    t0 = time.time()
    prec_kind = _canon_prec(local_prec)
    trunc_norm = _canon_trunc(trunc_norm)
    _check_positive(kickrank=(kickrank, 0), nswp=(nswp, 1), rmax=(rmax, 1),
                    local_iters=(local_iters, 1), local_restart=(local_restart, 1))
    kickrank = int(kickrank)
    warned_budget = False
    tol = float(eps)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(f"eps={eps!r}: expected a finite non-negative accuracy")

    fcores, f_is_vector = _vector_cores(f)
    d = len(fcores)
    acores = _matrix_cores(A, d)
    n = [c.shape[1] for c in acores]
    m = [c.shape[2] for c in acores]
    if n != m:
        raise ValueError(
            f"amen_solve needs a square operator, got row modes {n} and "
            f"column modes {m}; solve the normal equations for a rectangular A")
    if [c.shape[1] for c in fcores] != n:
        raise ValueError(f"f has modes {[c.shape[1] for c in fcores]}, "
                         f"A has row modes {n}")

    # ``x`` and ``z`` must be drawn from *independent* streams: seeding both
    # with ``seed`` made them the same tensor whenever kickrank == 2, i.e. the
    # enrichment started out inside the trial subspace and added nothing.
    seed_x, seed_z = np.random.SeedSequence(seed).spawn(2)

    dt = bk.result_dtype(bk.dtype_of(acores[0]), bk.dtype_of(fcores[0]))
    if x0 is None:
        xcores = _ops.random_tt(m, [1] + [2] * (d - 1) + [1], dtype=dt,
                                like=fcores[0], seed=seed_x)
    else:
        xcores, _ = _vector_cores(x0)
        if [c.shape[1] for c in xcores] != m:
            raise ValueError(f"x0 has modes {[c.shape[1] for c in xcores]}, "
                             f"A has column modes {m}")
        dt = bk.result_dtype(dt, bk.dtype_of(xcores[0]))
    acores = _ops.to_dtype(acores, dt)
    a_symmetric = False
    fcores = _ops.to_dtype(fcores, dt)
    xcores = _ops.to_dtype(list(xcores), dt)

    like = fcores[0]
    fnorm = float(_ops.norm(fcores))
    if fnorm == 0.0:
        # Same contract as tt.algs.solvers.GMRES: without a scale there is no
        # relative residual to converge, and every stopping test would divide
        # by zero.  (The answer is x = 0; say so rather than iterate on nan.)
        raise ValueError(
            "amen_solve: the right-hand side is zero, so a relative residual "
            "is undefined; the solution of A x = 0 is x = 0 (tt.zeros)")
    # Both the local and the global tolerance are split over the d-1 splittings
    # of the TT chain, exactly as in ttamen.f90.
    real_tol = tol / np.sqrt(max(d - 1, 1)) / RESID_DAMP

    one = bk.eye(1, 1, dtype=dt, like=like).reshape((1, 1, 1))
    one2 = bk.eye(1, 1, dtype=dt, like=like)

    def interfaces(scalar):
        """``[None] * (d+1)`` with the two trivial boundary interfaces set."""
        phi = [None] * (d + 1)
        phi[0] = phi[d] = scalar
        return phi

    phiax_l, phiax_r = interfaces(one), interfaces(one)
    phif_l, phif_r = interfaces(one2), interfaces(one2)

    if kickrank > 0:
        zcores = _ops.random_tt(n, [1] + [kickrank] * (d - 1) + [1], dtype=dt,
                                like=like, seed=seed_z)
        phizax_l, phizax_r = interfaces(one), interfaces(one)
        phizf_l, phizf_r = interfaces(one2), interfaces(one2)

    info = AmenSolveHistory(tol=tol)
    best_cores, best_res = None, float("inf")

    for swp in range(int(nswp)):
        # --- right-to-left orthogonalization, building the right interfaces --
        for k in range(d - 1, 0, -1):
            if kickrank > 0 and swp > 0:
                zcores[k] = (
                    _local_matvec(phizax_l[k], acores[k], phizax_r[k + 1], xcores[k])
                    - _local_rhs(phizf_l[k], fcores[k], phizf_r[k + 1]))
            lmat, xcores[k] = _rq_right(xcores[k])
            xcores[k - 1] = _push_right(xcores[k - 1], lmat)
            phiax_r[k] = _phi_next(
                _project(phiax_r[k + 1], acores[k], xcores[k], "rl"),
                xcores[k], "rl")
            phif_r[k] = _phi_yy_next(phif_r[k + 1], xcores[k], fcores[k], "rl")
            if kickrank > 0:
                lmat, zcores[k] = _rq_right(zcores[k])
                zcores[k - 1] = _push_right(zcores[k - 1], lmat)
                phizax_r[k] = _phi_next(
                    _project(phizax_r[k + 1], acores[k], xcores[k], "rl"),
                    zcores[k], "rl")
                phizf_r[k] = _phi_yy_next(phizf_r[k + 1], zcores[k],
                                          fcores[k], "rl")

        # --- left-to-right optimization --------------------------------------
        max_dx = 0.0
        max_res = 0.0
        n_matvecs = 0
        n_direct = 0
        n_local = 0
        n_failed_direct = 0
        n_failed_gmres = 0
        for k in range(d):
            rhs = _local_rhs(phif_l[k], fcores[k], phif_r[k + 1])
            rhs_norm = float(bk.norm(rhs))
            res = _local_matvec(phiax_l[k], acores[k], phiax_r[k + 1],
                                xcores[k]) - rhs
            # A zero local rhs makes the relative residual meaningless: fall
            # back to the absolute one rather than divide by zero.
            scale = rhs_norm if rhs_norm > 0 else 1.0
            err = float(bk.norm(res)) / scale
            max_res = max(max_res, err)

            if err > real_tol:
                # Inexact local solve, as in the AMEn papers: the target is an
                # ABSOLUTE local residual of real_tol/err * ||rhs||, so a block
                # that is already close is not solved any harder than it needs.
                # We solve the correction equation B d = res, whose right-hand
                # side has norm err*||rhs||, so the relative tolerance carries a
                # second factor of err.  Passing real_tol/err here (the same
                # number the full-system formulation uses) over-solves by 1/err
                # -- measured on the 2D QTT Laplacian: 7482 local iterations
                # against the Fortran's 4185 for the same sweeps and ranks.
                #
                # The cap is not cosmetic.  The legacy restarts its local GMRES
                # from zero, so a loose tolerance still moves the block; we start
                # from the current iterate, so the same tolerance can mean "do
                # nothing" and the sweep stops converging (measured: residual
                # 2.6e-04 instead of 5.7e-07 with no cap).  Demanding a tenfold
                # improvement of every block solved keeps that from happening;
                # 0.01 costs 36% more iterations, 0.3 costs three more sweeps.
                tol_local = min(0.1, real_tol / (err * err))
                sol, linfo = _solve_local(
                    phiax_l[k], acores[k], phiax_r[k + 1], res,
                    tol_local, max_full_size, prec_kind,
                    local_iters, local_restart, a_symmetric)
                n_matvecs += linfo["matvecs"]
                n_direct += int(linfo["kind"] == "direct")
                n_local += 1
                if not linfo["converged"]:
                    if linfo["kind"] == "direct":
                        n_failed_direct += 1
                    else:
                        n_failed_gmres += 1
                # ``sol`` solves B sol = A x - f, so it is the *negative*
                # correction.
                xcores[k] = xcores[k] - sol
                dx_abs = float(bk.norm(sol))
                x_abs = float(bk.norm(xcores[k]))
                if not np.isfinite(dx_abs) or not np.isfinite(x_abs):
                    raise FloatingPointError(
                        f"amen_solve: block {k} of sweep {swp + 1} is not "
                        "finite; the local system or the interfaces overflowed")
                max_dx = max(max_dx, dx_abs / x_abs if x_abs > 0 else dx_abs)
                res = _local_matvec(phiax_l[k], acores[k], phiax_r[k + 1],
                                    xcores[k]) - rhs
                err = float(bk.norm(res)) / scale
                if verb > 1:
                    print(f"amen_solve: swp={swp + 1}, block={k}, "
                          f"{linfo['kind']}(size={linfo['size']}), "
                          f"local_res={linfo['relres']:.3E}, "
                          f"block_res={err:.3E}")

            if k == d - 1:
                break

            if kickrank > 0:
                # Truncate, then enrich with the residual directions.  The
                # truncated block is needed in the *old* right basis because
                # phizax_r[k+1] is expressed in it.
                if trunc_norm == 1:
                    ctx = (phiax_l[k], acores[k], phiax_r[k + 1], rhs,
                           scale)
                    trunc_tol = max(err, real_tol * RESID_DAMP)
                else:
                    ctx = None
                    trunc_tol = real_tol * RESID_DAMP
                xcores[k], xcores[k + 1], xold = _truncate(
                    xcores[k], xcores[k + 1], trunc_tol, ctx, rmax)

                enrich = (_apply(_project(phiax_l[k], acores[k], xold, "lr"),
                                 phizax_r[k + 1], "lr")
                          - _local_rhs(phif_l[k], fcores[k], phizf_r[k + 1]))
                zcores[k] = (
                    _apply(_project(phizax_l[k], acores[k], xold, "lr"),
                           phizax_r[k + 1], "lr")
                    - _local_rhs(phizf_l[k], fcores[k], phizf_r[k + 1]))

                rz = enrich.shape[2]
                xcores[k] = bk.concatenate((xcores[k], enrich), axis=2)
                pad = bk.zeros((rz,) + tuple(xcores[k + 1].shape[1:]),
                               dtype=dt, like=like)
                xcores[k + 1] = bk.concatenate((xcores[k + 1], pad), axis=0)

                xcores[k], rmat = _qr_left(xcores[k])
                xcores[k + 1] = _push_left(xcores[k + 1], rmat)
                zcores[k], rmat = _qr_left(zcores[k])
                zcores[k + 1] = _push_left(zcores[k + 1], rmat)

                phizax_l[k + 1] = _phi_next(
                    _project(phizax_l[k], acores[k], xcores[k], "lr"),
                    zcores[k], "lr")
                phizf_l[k + 1] = _phi_yy_next(phizf_l[k], zcores[k],
                                              fcores[k], "lr")
            else:
                xcores[k], rmat = _qr_left(xcores[k])
                xcores[k + 1] = _push_left(xcores[k + 1], rmat)

            phiax_l[k + 1] = _phi_next(
                _project(phiax_l[k], acores[k], xcores[k], "lr"),
                xcores[k], "lr")
            phif_l[k + 1] = _phi_yy_next(phif_l[k], xcores[k], fcores[k], "lr")

        # --- report and stop --------------------------------------------------
        ranks = _ops.ranks(xcores)
        true_res = float("nan")
        # Forming ||A x - f|| in TT and orthogonalising it is the only way to
        # certify a residual below sqrt(eps_machine) -- the expansion
        # <Ax,Ax> - 2<Ax,f> + <f,f> cancels -- but it costs a full sweep over a
        # tensor of rank r_A r_x + r_f, measured at 5.6 ms against a 38 ms
        # sweep.  It is therefore computed when it can change the decision: on
        # the last sweep, and whenever the cheap local criteria say the run is
        # done.  A run never reports convergence on the cheap criteria alone.
        # ... and a sweep whose local residuals have not reached the target
        # cannot be the one that converges.  Note max_res, not max_dx: the step
        # size collapses long before the residual does, and keying on it had the
        # residual formed in 6 sweeps of 12 (57 ms of a 250 ms solve) to decide
        # nothing.  Keying on max_res costs two extra sweeps and saves five
        # residuals: 250 -> 206 ms.
        # A sweep whose cheap indicators have not reached the target
        # cannot be the one that converges, so nothing is decided by measuring
        # it.  nan records "not measured"; such a sweep is never selected as the
        # best iterate.  Measured: 61 ms of a 237 ms solve went into residuals
        # that changed nothing.
        worth_measuring = max_res <= 10 * tol
        last_sweep = swp + 1 >= int(nswp)
        if check_true_res and (worth_measuring or last_sweep):
            nbytes = _residual_bytes(acores, xcores)
            if nbytes > true_res_budget and not warned_budget:
                warned_budget = True
                warnings.warn(
                    f"check_true_res=True is about to form A x, whose largest "
                    f"core is {nbytes / 1024 ** 3:.1f} GiB (the ranks of A and "
                    f"x multiply). The stopping criterion does not need it -- "
                    f"max_res, the local residual before each block solve, is "
                    f"free. Pass check_true_res=False, or raise "
                    f"true_res_budget to silence this.",
                    RuntimeWarning, stacklevel=2)
            true_res = float(_ops.norm(_ops.sub(
                _ops.matvec_cores(acores, xcores), fcores)) / fnorm)
        entry = {"sweep": swp + 1, "max_dx": max_dx, "max_res": max_res,
                 "max_rank": int(max(ranks)), "true_res": true_res,
                 "time": time.time() - t0, "local_matvecs": n_matvecs,
                 "local_direct": n_direct, "local_solves": n_local,
                 "local_failed": n_failed_direct + n_failed_gmres,
                 "local_failed_direct": n_failed_direct,
                 "local_failed_gmres": n_failed_gmres}
        info.sweeps.append(entry)
        if verb > 0:
            line = (f"amen_solve: swp={swp + 1}, max_dx={max_dx:9.3E}, "
                    f"max_res={max_res:9.3E}, max_rank={max(ranks)}")
            if check_true_res:
                line += f", true_res={true_res:9.3E}"
            print(line)

        if check_true_res:
            converged = true_res <= tol   # nan when not measured -> False
        else:
            converged = (max_res if trunc_norm == 1 else max_dx) < tol
        info.max_dx, info.max_res, info.true_res = max_dx, max_res, true_res
        info.ranks = [int(r) for r in ranks]
        info.nswp_done = swp + 1
        # Never hand back an iterate that is worse than one already computed:
        # with a stalling local solver the sweeps can oscillate, and returning
        # the last one instead of the best one would throw away a better answer
        # for no reason.  Only possible when the true residual is known.
        score = true_res if check_true_res else max_res
        if score < best_res:
            # A shallow copy of the *list* is enough: every step of the sweep
            # rebinds ``xcores[k]`` to a freshly allocated array, nothing is
            # ever written into a core in place.
            best_cores, best_res = list(xcores), score
            info.best_sweep = swp + 1
        if converged:
            info.converged = True
            break

    current = info.true_res if check_true_res else info.max_res
    if best_cores is not None and best_res < current:
        # The whole summary must describe the vector that is handed back, not
        # the last one computed: a history that mixes two iterates is a lie.
        xcores = best_cores
        if check_true_res:
            info.true_res = best_res
        best = info.sweeps[info.best_sweep - 1]
        info.max_dx, info.max_res = best["max_dx"], best["max_res"]
    info.ranks = [int(r) for r in _ops.ranks(xcores)]

    info.time = time.time() - t0
    if info.converged:
        info.message = (f"converged in {info.nswp_done} sweeps, "
                        f"max_dx={info.max_dx:.3E}, max_res={info.max_res:.3E}")
    else:
        reached = (f"true residual {info.true_res:.3E}" if check_true_res
                   else f"max_res {info.max_res:.3E}, max_dx {info.max_dx:.3E}")
        last = info.sweeps[-1] if info.sweeps else {}
        if last.get("local_failed_gmres", 0):
            blame = (f" The local GMRES missed its tolerance in "
                     f"{last['local_failed_gmres']} of {last['local_solves']} "
                     "blocks of the last sweep, so it -- not the outer "
                     "iteration -- may be the bottleneck: raise max_full_size "
                     "(dense local solves), local_iters/local_restart, or set "
                     "local_prec.")
        elif last.get("local_failed_direct", 0):
            blame = (f" The *dense* local solves missed their tolerance in "
                     f"{last['local_failed_direct']} of "
                     f"{last['local_solves']} blocks of the last sweep: the "
                     "local systems are too ill-conditioned for the working "
                     "precision, so neither more sweeps nor a different local "
                     "solver will help.")
        else:
            blame = (" All local systems were solved to their tolerance, so "
                     "the outer iteration is what stalled: the accuracy may be "
                     "at the float64 floor of this problem "
                     "(eps_machine * ||A|| ||x|| / ||f||).")
        which = ("the last iterate" if info.best_sweep in (0, info.nswp_done)
                 else f"the best iterate (sweep {info.best_sweep})")
        info.message = (
            f"amen_solve did NOT reach eps={tol:.3E} in {info.nswp_done} "
            f"sweeps: reached {reached} (max rank "
            f"{max(info.ranks) if info.ranks else 0}). The returned vector is "
            f"{which}, not a solution to the requested accuracy." + blame)
        warnings.warn(info.message, UserWarning, stacklevel=2)

    if f_is_vector:
        x = vector.from_list(xcores)
        x.amen_info = info       # a raw core list cannot carry the history
    else:
        x = xcores
    return (x, info) if return_info else x
