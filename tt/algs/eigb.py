"""Block eigenvalue solver: the ``B`` smallest eigenpairs of a symmetric TT-matrix.

The unknown is a *block* TT-vector ``y``: a TT tensor whose last rank is
``B = y.r[-1]``, so that ``y[i_1, ..., i_d, b]`` is the ``b``-th eigenvector.
All ``B`` eigenvectors share one set of TT cores, which is what makes the method
cheap -- and what makes it converge, because the block index acts as the rank
enrichment that plain one-site ALS is missing: when the block index is pushed
through an interface by an SVD, the interface rank may grow up to
``min(B r, n r')``.

Algorithm (one-site block ALS / DMRG-1)
--------------------------------------
The block index travels with the current site.  At site ``k`` the frames
``Y_{<k}``, ``Y_{>k}`` are orthonormal, the projected operator

    B_k = (Y_{<k} (x) I (x) Y_{>k})^H A (Y_{<k} (x) I (x) Y_{>k})

of size ``(r_k n_k r_{k+1})^2`` is symmetric, and its ``B`` smallest eigenpairs
are the local update: the eigenvalues are Ritz values of ``A`` and decrease
monotonically along the sweep.  The eigenvector block ``(r_k n_k r_{k+1}, B)``
is then truncated by an SVD at ``eps / sqrt(d)`` (relative), the orthogonal
factor stays behind as the new core, and the rest -- carrying the block index --
is multiplied into the neighbouring core.

Sweeps run ``d -> 1`` and ``1 -> d``; the stopping indicator is the largest
relative drop of ``sum(lambda)`` observed during a full sweep, exactly as in the
Fortran original.

That indicator says how much the iteration still *moves*, which is not the same
as being right, and one-site ALS has a standard way of not being right: it
cannot grow a rank.  The block index is the only enrichment, so with ``B = 1``
and a rank-1 initial guess the iterate is trapped on the rank-1 manifold, the
Ritz value stops moving to 1e-14 and the run looks converged while being wrong
by a factor of 50.  For that reason the returned block always comes with its
measured eigenresidual (:func:`block_residuals`, ``history.res``), and a large
one warns -- the answer of an eigensolver is a pair, and the residual is the
only evidence that it is one.

Local eigensolver
-----------------
``r_k n_k r_{k+1} <= max_full_size``: the local matrix is built densely and sent
to ``bk.eigh``.  Above that it is never formed; ``scipy.sparse.linalg.lobpcg``
runs on a ``LinearOperator`` with the previous block as the initial guess (this
replaces PRIMME, which the Fortran version linked).  The iterative path is
numpy-only -- the small interface tensors are pulled to the host once per local
solve -- and its true residuals ``||B_k v_i - lam_i v_i||`` are measured and
recorded, never assumed.

References
----------
* S. V. Dolgov, B. N. Khoromskij, I. V. Oseledets, D. V. Savostyanov,
  "Computation of extreme eigenvalues in higher dimensions using block tensor
  train format", Computer Phys. Comm. 185(4):1207-1216, 2014,
  arXiv:1306.2269.
* Replaces ``tt/tt-fort/tt_eigb.f90`` of legacy ttpy (same public signature).
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
from ..core.matrix import matrix
from ..core.vector import vector
from . import _localops as lo

__all__ = ["eigb", "EigbHistory", "spectral_norm_estimate"]


@dataclass
class EigbHistory:
    """What the run knows about itself; recorded even with ``verb=0``.

    Attributes:
        eps: Requested accuracy.
        steps: One dict per local solve, with keys ``sweep, site, direction,
            loc_size, solver, lam, res, rank_new, time``.  ``res`` is the true
            local eigenresidual (0 for the dense solver, measured for lobpcg).
        sweeps: One dict per full sweep, with keys ``sweep, ermax, max_rank,
            lam, time``.
        lam: Eigenvalues of the returned block.
        converged: ``ermax < eps`` was reached before ``nswp`` was exhausted.
        ermax: The stopping indicator of the last sweep -- the largest relative
            drop of ``sum(lambda)`` over the sweep.  It measures how much the
            iteration still moves, not the distance to the true eigenvalues.
        res: ``||A y_i - lam_i y_i||`` for each returned eigenvector, or None if
            ``check_residual=False``.  This -- not ``ermax`` -- is the evidence
            that the answer is an eigenpair: a stagnating alternating iteration
            reports ``ermax = 0`` at a point that is not one.
        res_rel: ``res_i / ||A y_i||`` -- the *relative* accuracy of eigenvalue
            ``i``.  Reported, but not what the warning fires on: near the bottom
            of the spectrum ``||A y_i||`` is tiny and this is large even when
            the answer is right to machine precision.
        res_back: ``res_i / ||A||_2`` (see :func:`spectral_norm_estimate`) --
            the backward error, and the quantity the warning uses: the returned
            pair is exact for ``A + E`` with ``||E||/||A|| = res_back``.
        anorm: The estimate of ``||A||_2`` used for ``res_back``.
        max_local_res: Largest local eigenresidual seen (iterative solver only).
        ranks: TT ranks of the returned block vector.
        nswp_done: Number of full sweeps performed.
        time: Wall-clock seconds.
    """

    eps: float
    steps: list = field(default_factory=list)
    sweeps: list = field(default_factory=list)
    lam: np.ndarray = None
    converged: bool = False
    ermax: float = float("nan")
    res: np.ndarray = None
    res_rel: np.ndarray = None
    res_back: np.ndarray = None
    anorm: float = float("nan")
    max_local_res: float = 0.0
    ranks: list = field(default_factory=list)
    nswp_done: int = 0
    time: float = 0.0

    def __repr__(self):
        res = "n/a" if self.res_back is None else f"{float(np.max(self.res_back)):.2e}"
        return (f"EigbHistory(sweeps={self.nswp_done}, converged={self.converged}, "
                f"ermax={self.ermax:.2e}, max_rank={max(self.ranks) if self.ranks else 0}, "
                f"max_back_err={res}, "
                f"max_local_res={self.max_local_res:.2e}, time={self.time:.2f}s)")


def spectral_norm_estimate(A, its=12, eps=1e-3, seed=0):
    """A cheap lower estimate of ``||A||_2``, by power iteration in TT.

    This exists to give the eigenresidual a denominator that means something.
    The obvious candidates both fail:

    * ``||A y_i||`` (that is, ``lam_i``) asks every eigenvalue to be accurate
      *relatively*.  No eigensolver working at truncation accuracy ``eps`` can
      deliver that near the bottom of the spectrum: on ``qlaplace_dd([10])``
      with ``B = 4`` the returned pairs are right to 1e-9 absolute, yet
      ``res/||A y||`` is 3.0e-04 because ``lam_1 = 9.4e-06``.
    * ``||A||_F`` overestimates badly in exactly the regime we care about --
      measured 78.4 against ``||A||_2 = 4.0`` for the same operator, and 41.6
      against 4.26 for a 10-site Heisenberg chain.  A denominator 10-20x too
      large desensitizes the warning by the same factor.

    ``||r||/||A||_2`` is the backward error: the returned pair is exact for
    ``A + E`` with ``||E|| = ||r||``, so this is the perturbation of the
    operator that the answer corresponds to, and for a symmetric ``A`` it also
    bounds ``|lam - lam_exact|`` directly.

    The estimate is a *lower* bound, so it can only make the warning more
    eager, never quieter.  Measured against a dense ``||A||_2`` at ``its=12``:
    0.94-0.97 of the truth on ``qlaplace_dd``, 0.97 on Heisenberg, in 25-40 ms
    -- both operators have a clustered top of the spectrum, which is the slow
    case for power iteration, so this is close to the worst it does.  One digit
    is all a threshold needs.

    Deterministic: the starting vector comes from a fixed seed, so two runs on
    the same operator warn identically.

    Args:
        A: A :class:`tt.matrix`.
        its: Power iterations.
        eps: Rounding accuracy of the iterate (loose on purpose -- this is a
            threshold, not an answer).
        seed: Seed of the starting vector.

    Returns:
        ``float``, an estimate of ``||A||_2`` from below.  ``0.0`` if the
        iterate collapses, which is what a zero operator does.
    """
    from ..core import tools as _tools

    d = int(A.tt.d)
    n = [int(v) for v in A.n]
    # the starting vector must live where A lives, not on the default backend:
    # this is an *operation* on a user's operator, and operations follow their
    # input.  Going through tt.rand would build it on whatever set_backend last
    # selected and then die in the first contraction on a mixture.
    ab = bk.backend_of(A.tt.cores[0])
    dt = A.dtype
    rng = np.random.default_rng(seed)
    ranks = [1] + [3] * (d - 1) + [1]
    x = vector.from_list([ab.randn((ranks[k], n[k], ranks[k + 1]), dt, rng)
                          for k in range(d)])
    nrm = x.norm()
    if nrm == 0.0:
        return 0.0
    x = x * (1.0 / nrm)
    lam = 0.0
    for _ in range(int(its)):
        y = _tools.matvec(A, x).round(eps)
        lam = float(y.norm())
        if lam == 0.0:
            return 0.0
        x = y * (1.0 / lam)
    return lam


def block_residuals(A, y, lam):
    """``(||A y_i - lam_i y_i||, ||A y_i||)`` for every column of a block vector.

    The only evidence that the returned pairs are eigenpairs.  ``ermax``, the
    stopping indicator of the sweep, says how much the iteration still *moves*;
    an alternating iteration that stalls at a non-stationary starting frame (a
    rank-deficient or zero initial guess is the standard way to produce one)
    reports ``ermax = 0`` while sitting on a vector with residual O(1).

    Everything is done in the TT format: one matvec (ranks ``r_A r_y``), one
    addition (ranks ``r_A r_y + r_y``) and two block dots -- about one sweep's
    worth of work, never a dense vector.  The residual vector is *formed* and
    then normed instead of expanding ``||z||^2 - 2 lam <z, y> + lam^2 ||y||^2``:
    that expansion cancels down to ``sqrt(eps) ||A y||`` and would report 1e-9
    where the truth is 1e-16.

    Args:
        A: The TT-matrix.
        y: The block TT-vector, ``y.r[-1] == B``.
        lam: The ``B`` Ritz values.

    Returns:
        ``(res, znorm)``, two numpy arrays of length ``B``: the residual norms
        and ``||A y_i||``.
    """
    nblock = int(y.r[-1])
    dt = bk.result_dtype(A.dtype, y.dtype)
    ycores = _ops.to_dtype(list(y.cores), dt)
    acores = lo.operator_cores(A, ycores[0], dt)
    z = _ops.matvec_cores(acores, ycores)

    # -lam on the block index of the last core: (A y - y diag(lam))_i
    minus_lam = bk.asarray(-np.asarray(lam, dtype=np.float64), dt,
                           backend=bk.backend_of(ycores[0]))
    scaled = list(ycores)
    scaled[-1] = scaled[-1] * minus_lam
    return _block_norms(_ops.add(z, scaled)), _block_norms(z)


def _block_norms(cores):
    """Per-column norms of a block TT-vector, computed through a QR sweep.

    Squaring first (``diag(dot(w, w))``) would be cheaper and wrong: the cores
    of a residual are *not* small even when the residual is, so the sum of
    products cancels and the answer saturates at ``sqrt(eps) ||w's cores||``.
    Orthogonalizing left to right pushes the norm into the last core, where the
    block index still sits, and costs one QR sweep.
    """
    cores = list(cores)
    for k in range(len(cores) - 1):
        q, s = lo.left_orthogonalize(cores[k])
        cores[k] = q
        cores[k + 1] = _apply_left(s, cores[k + 1])
    last = bk.to_numpy(cores[-1])
    return np.sqrt(np.sum(np.abs(np.asarray(last)) ** 2, axis=(0, 1)))


def _symmetry_defect(m):
    """``||M - M^H|| / ||M||``; the local matrix must be Hermitian."""
    nrm = bk.norm(m)
    if nrm == 0:
        return 0.0
    return float(bk.norm(m - m.conj().T) / nrm)


def _local_eig_dense(left, acore, right, nblock, sym_tol):
    """The ``nblock`` smallest eigenpairs of the projected local matrix."""
    m = lo.local_matrix(left, acore, right)
    defect = _symmetry_defect(m)
    if defect > sym_tol:
        raise ValueError(
            f"the projected local matrix is not Hermitian (relative asymmetry "
            f"{defect:.2e} > {sym_tol:.1e}); eigb needs a symmetric/Hermitian A")
    w, v = bk.eigh(m)
    return w[:nblock], v[:, :nblock], 0.0


def _local_eig_lobpcg(left, acore, right, nblock, guess, tol, maxiter):
    """Matrix-free local eigensolve; numpy/scipy only (LOBPCG needs a host array).

    The interfaces and the matrix core are small, so pulling them to the host
    once per local solve is cheaper than a per-matvec round trip.  Returns the
    eigenpairs and the largest *measured* residual ``||B v - lam v||``.
    """
    from scipy.sparse.linalg import LinearOperator, lobpcg

    src = left
    left_np = bk.to_numpy(left)
    a_np = bk.to_numpy(acore)
    right_np = bk.to_numpy(right)
    r0, n, r1 = left_np.shape[0], a_np.shape[1], right_np.shape[0]
    size = r0 * n * r1

    def matmat(x):
        blocks = np.asarray(x).reshape((r0, n, r1, -1))
        return lo.local_matmat(left_np, a_np, right_np, blocks).reshape((size, -1))

    op = LinearOperator((size, size), matvec=lambda v: matmat(v.reshape((-1, 1)))[:, 0],
                        matmat=matmat, dtype=left_np.dtype)
    x0 = np.asarray(bk.to_numpy(guess)).reshape((size, nblock))
    x0, _ = np.linalg.qr(x0)
    with warnings.catch_warnings():
        # lobpcg warns when it exits on maxiter; we measure the residual below
        # and report the real number instead of relaying a vague warning.
        warnings.simplefilter("ignore")
        w, v = lobpcg(op, x0, largest=False, tol=tol, maxiter=maxiter)
    res = matmat(v) - v * w[None, :]
    res_max = float(np.max(np.linalg.norm(res, axis=0)))
    # the eigenvalues of a Hermitian problem are real: casting them to the
    # (possibly complex) dtype of the eigenvectors only earns a ComplexWarning
    # from the caller, which converts them back to float64
    return (bk.asarray(np.real(w), bk.real_dtype(bk.dtype_of(src)),
                       backend=bk.backend_of(src)),
            bk.asarray(v, bk.dtype_of(src), backend=bk.backend_of(src)), res_max)


def eigb(A, y0, eps, rmax=150, nswp=20, max_full_size=1000, verb=1,
         return_history=False, lobpcg_maxiter=200, sym_tol=None,
         check_residual=True, res_warn=None):
    """The ``B`` smallest eigenpairs of a symmetric TT-matrix.

    ``B = y0.r[-1]``: the last rank of the initial guess is the number of
    eigenvalues sought.  For the largest eigenvalues, call with ``-A``.

    Args:
        A: Symmetric (Hermitian) :class:`tt.matrix` with ``A.n == A.m``.
        y0: Initial guess, a block :class:`tt.vector` with ``y0.r[-1] == B``.
        eps: Relative accuracy: the truncation threshold of every local SVD is
            ``eps / sqrt(d)``, and the sweep stops once the relative drop of
            ``sum(lambda)`` over a full sweep is below ``eps``.
        rmax: Maximal TT rank of the block vector.
        nswp: Maximal number of full sweeps.
        max_full_size: Above this local size the local eigenproblem is solved
            matrix-free (LOBPCG) instead of densely.  Local problems smaller
            than ``5 B + 10`` always take the dense path -- LOBPCG is not
            defined for a block that large relative to the space.
        verb: 0 silent, 1 one line per sweep, 2 one line per local solve.
        return_history: also return the :class:`EigbHistory`.
        lobpcg_maxiter: iteration cap for the matrix-free local solver.
        sym_tol: relative asymmetry of a local matrix above which the run stops
            with an error instead of returning eigenvalues of ``(B + B^H)/2``.
            ``None`` (default) means ``sqrt(eps_machine)`` of the working dtype
            -- 1.5e-8 in float64, 3.5e-4 in float32.  A fixed 1e-8 would reject
            every float32 problem, whose projected local matrices are asymmetric
            at the 1e-7 level from rounding alone.
        check_residual: measure ``||A y_i - lam_i y_i||`` on the returned block
            (one TT matvec plus three dots, see :func:`block_residuals`) and put
            it in the history.  ``ermax`` cannot see a stalled iteration; this
            can, so leave it on unless the cost matters.
        res_warn: warn when the largest *backward error*
            ``||A y_i - lam_i y_i|| / ||A||_2`` exceeds this.  Pure reporting:
            the numbers are in ``history.res`` / ``history.res_rel`` /
            ``history.res_back`` whatever the threshold.  The denominator is an
            estimate (:func:`spectral_norm_estimate`) and costs about 12 TT
            matvecs, only when ``check_residual`` is on.
            ``None`` (default) means ``sqrt(eps)``, floored at
            ``8 * eps_machine`` of the working dtype -- that is the residual a
            converged run actually reaches, because the eigenvalue error is
            quadratic in the eigenvector error while the residual is linear.
            A *fixed* threshold is the wrong shape: 1e-2 left six silent decades
            between an ``eps=1e-8`` request and the warning, and that is exactly
            where a run whose rank never grows comes to rest -- ``eigb`` cannot
            increase the rank at ``B == 1`` (both local SVD groupings bound the
            new rank by ``B * r_old``), so a too-small guess rank stalls at a
            non-eigenvector and used to return quietly.  See
            ``docs/plans/eigenvalues.md``.

    Returns:
        ``(y, lam)``, or ``(y, lam, history)`` if ``return_history``.  ``y`` is
        the block TT-vector with ``y.r[-1] == B`` whose columns are orthonormal;
        ``lam`` is a numpy array of ``B`` Ritz values in ascending order.

    Raises:
        ValueError: on shape/rank mismatches, a non-Hermitian local matrix, or
            a local problem too small to carry ``B`` eigenvectors.

    Note:
        Non-convergence is *reported*, never hidden: ``history.converged`` is
        False and a ``RuntimeWarning`` carries the achieved indicator.  The
        converse is reported too: ``converged=True`` only says the Ritz values
        stopped moving, so the residual is measured as well and a large one
        warns even on a "converged" run.
    """
    if not isinstance(A, matrix):
        raise TypeError(f"eigb needs a tt.matrix, got {type(A)!r}")
    if not np.array_equal(A.n, A.m):
        raise ValueError(f"eigb needs a square TT-matrix, got n={A.n}, m={A.m}")
    if A.d != y0.d:
        raise ValueError(f"dimension mismatch: A.d={A.d}, y0.d={y0.d}")
    if not np.array_equal(np.asarray(A.n).ravel(), np.asarray(y0.n).ravel()):
        raise ValueError(f"mode mismatch: A.n={A.n}, y0.n={y0.n}")
    if y0.r[0] != 1:
        raise ValueError(f"the block index must sit on the right: y0.r[0]={y0.r[0]}")
    if int(nswp) < 1:
        raise ValueError(f"nswp must be at least 1, got {nswp}")

    t_start = time.time()
    d = int(y0.d)
    nblock = int(y0.r[-1])
    n = [int(v) for v in y0.n]
    dt = bk.result_dtype(A.dtype, y0.dtype)
    cores = [bk.asarray(c, dt) for c in y0.cores]
    acores = lo.operator_cores(A, cores[0], dt)
    if sym_tol is None:
        sym_tol = float(np.sqrt(bk.eps_of(dt)))
    if res_warn is None:
        res_warn = max(float(np.sqrt(eps)), 8.0 * float(bk.eps_of(dt)))
    hist = EigbHistory(eps=float(eps))

    if verb > 0:
        print(f"Solving a block eigenvalue problem\n"
              f"Looking for {nblock} eigenvalues with accuracy {eps:.1E}")

    # --- d == 1: no sweep, the "local" problem is the whole problem ----------
    if d == 1:
        left = lo.ones_interface(cores[0], dt)
        right = lo.ones_interface(cores[0], dt)
        lam, v, _ = _local_eig_dense(left, acores[0], right, nblock, sym_tol)
        y = vector.from_list([v.reshape((1, n[0], nblock))])
        hist.lam = np.asarray(bk.to_numpy(lam), dtype=np.float64)
        hist.converged = True
        hist.ranks = list(y.r)
        hist.nswp_done = 0
        _record_residual(hist, A, y, hist.lam, check_residual, res_warn, nswp)
        hist.time = time.time() - t_start
        out = (y, hist.lam)
        return out + (hist,) if return_history else out

    # --- initial orthogonalization, left to right ----------------------------
    # Cores 0..d-2 become left-orthonormal; the last core keeps the block index,
    # which from here on travels with the current site (the sweep-time right
    # boundary rank is 1, not B).
    left_int = [None] * (d + 1)
    right_int = [None] * (d + 1)
    left_int[0] = lo.ones_interface(cores[0], dt)
    for k in range(d - 1):
        q, s = lo.left_orthogonalize(cores[k])
        cores[k] = q
        cores[k + 1] = _apply_left(s, cores[k + 1])
        left_int[k + 1] = lo.phi_left(left_int[k], acores[k], cores[k], cores[k])
    q, _ = lo.left_orthogonalize(cores[d - 1])   # (r n, B) -> orthonormal guess
    if q.shape[2] < nblock:
        raise ValueError(
            f"the initial guess cannot carry {nblock} eigenvectors: the last "
            f"unfolding is {cores[d - 1].shape[0] * n[d - 1]}x{nblock} and has "
            f"rank {q.shape[2]}")
    blk = q.reshape((q.shape[0], n[d - 1], 1, nblock))
    right_int[d] = lo.ones_interface(cores[0], dt)

    r = [int(c.shape[0]) for c in cores] + [1]
    # the last core is now represented by `blk`; anything reading it before the
    # sweep has written it back is a bug, so make that read fail loudly
    cores[d - 1] = None
    eps2 = float(eps) / np.sqrt(d)

    # --- the sweep -----------------------------------------------------------
    i, direction, swp = d - 1, -1, 1
    ermax, fvold = 0.0, 0.0
    lam = None
    converged = False
    while swp <= nswp:
        t_step = time.time()
        size = r[i] * n[i] * r[i + 1]
        if size < nblock:
            raise ValueError(
                f"local problem at site {i} has size {size} < B={nblock}; "
                "increase the ranks of the initial guess")
        if size <= max_full_size or size < 5 * nblock + 10:
            solver = "dense"
            lam_t, v, res = _local_eig_dense(left_int[i], acores[i], right_int[i + 1],
                                             nblock, sym_tol)
        else:
            solver = "lobpcg"
            lam_t, v, res = _local_eig_lobpcg(
                left_int[i], acores[i], right_int[i + 1], nblock, blk,
                tol=eps2 / 10.0, maxiter=lobpcg_maxiter)
        hist.max_local_res = max(hist.max_local_res, res)
        lam = np.asarray(bk.to_numpy(lam_t), dtype=np.float64)
        blk = v.reshape((r[i], n[i], r[i + 1], nblock))

        fv = float(np.sum(lam))
        erloc = (fvold - fv) / abs(fv) if fv != 0 else 0.0
        ermax = max(ermax, erloc)
        fvold = fv

        # --- move the block to the neighbouring site -------------------------
        rank_new = r[i + 1] if direction > 0 else r[i]
        if direction < 0 and i > 0:
            # (r n r', B) -> (B r, n r'): the block index joins the LEFT index,
            # which is how the interface rank is allowed to grow to min(B r, n r')
            u, s, vh = bk.svd(rearrange(blk, "a i b B -> (B a) (i b)"))
            rnew = _truncation_rank(s, eps2, rmax, u.shape[1])
            cores[i] = vh[:rnew, :].reshape((rnew, n[i], r[i + 1]))
            us = (u[:, :rnew] * s[:rnew]).reshape((nblock, r[i], rnew))
            blk = _merge_left(cores[i - 1], us)     # (r_{i-1}, n_{i-1}, rnew, B)
            r[i] = rnew
            rank_new = rnew
            right_int[i] = lo.phi_right(right_int[i + 1], acores[i], cores[i], cores[i])
        elif direction > 0 and i < d - 1:
            u, s, vh = bk.svd(rearrange(blk, "a i b B -> (a i) (b B)"))
            rnew = _truncation_rank(s, eps2, rmax, u.shape[1])
            cores[i] = u[:, :rnew].reshape((r[i], n[i], rnew))
            sv = (s[:rnew].reshape((rnew, 1)) * vh[:rnew, :]).reshape(
                (rnew, r[i + 1], nblock))
            blk = _merge_right(sv, cores[i + 1])    # (rnew, n_{i+1}, r_{i+2}, B)
            r[i + 1] = rnew
            rank_new = rnew
            left_int[i + 1] = lo.phi_left(left_int[i], acores[i], cores[i], cores[i])

        hist.steps.append(dict(sweep=swp, site=i, direction=direction,
                               loc_size=size, solver=solver, lam=lam.copy(),
                               res=res, rank_new=rank_new,
                               time=time.time() - t_step))
        if verb > 1:
            print(f"swp: {swp} i: [{i + 1}/{d}] loc_size: {size} "
                  f"solver: {solver} res: {res:.3E} ermax: {ermax:.3E} "
                  f"dfv: {erloc:.3E}")

        # --- direction bookkeeping ------------------------------------------
        if direction > 0 and i == d - 2:
            direction, i = -1, d - 1
            hist.sweeps.append(dict(sweep=swp, ermax=ermax,
                                    max_rank=int(max(r[:d])), lam=lam.copy(),
                                    time=time.time() - t_start))
            if verb > 0:
                print(f"swp: {swp} er = {ermax:.5E} rmax:{max(r[:d])}")
            swp += 1
            if ermax < eps:
                converged = True
                break
            ermax = 0.0
        elif direction < 0 and i == 1:
            direction, i = 1, 0
        else:
            i += direction

    hist.nswp_done = swp - 1 if converged else nswp
    hist.converged = converged
    # the running `ermax` is reset at the start of every sweep; the number worth
    # reporting is the one of the last *completed* sweep
    hist.ermax = hist.sweeps[-1]["ermax"] if hist.sweeps else ermax
    ermax = hist.ermax
    if not converged:
        warnings.warn(
            f"eigb did not converge in {nswp} sweeps: the sum of the {nblock} "
            f"Ritz values still moved by {ermax:.3E} (relative) in the last "
            f"sweep, requested {eps:.3E}. The returned eigenpairs are the last "
            "iterate; check the residuals before using them.",
            RuntimeWarning, stacklevel=2)

    # the block sits on the last core, whose sweep-time right rank is 1
    cores[d - 1] = blk.reshape((r[d - 1], n[d - 1], nblock))
    y = vector.from_list(cores)
    hist.lam = lam
    hist.ranks = [int(v) for v in y.r]
    _record_residual(hist, A, y, lam, check_residual, res_warn, nswp)
    hist.time = time.time() - t_start
    if verb > 0:
        print(f"Total local solves: {len(hist.steps)}")
        if hist.res is not None:
            print(f"Eigenresiduals ||A y - lam y||: "
                  f"{np.array2string(hist.res, precision=3)}")
    out = (y, lam)
    return out + (hist,) if return_history else out


def _record_residual(hist, A, y, lam, check_residual, res_warn, nswp):
    """Measure the eigenresidual of the returned block and report a bad one."""
    if not check_residual:
        return
    res, znorm = block_residuals(A, y, lam)
    hist.res = res
    hist.res_rel = res / np.where(znorm > 0, znorm, 1.0)
    hist.anorm = spectral_norm_estimate(A)
    # a zero operator has every vector for an eigenvector at lam = 0; there is
    # nothing to scale by and nothing to warn about
    hist.res_back = res / (hist.anorm if hist.anorm > 0 else 1.0)
    worst = float(np.max(hist.res_back))
    if worst > res_warn:
        warnings.warn(
            f"eigb returned pairs with a backward error up to "
            f"{worst:.3E} (residual {float(np.max(res)):.3E} against "
            f"||A||~{hist.anorm:.3E}), above the "
            f"{res_warn:.3E} expected for eps={hist.eps:.1E}; the Ritz values "
            f"are not eigenvalues of A to that accuracy. The sweep indicator "
            f"({hist.ermax:.3E} over {nswp} allowed sweeps) cannot see this: an "
            "alternating iteration can stall at a point that is not an "
            "eigenvector -- a rank-deficient or zero initial guess, too small "
            "an rmax, or a local solver that did not converge "
            f"(largest local residual {hist.max_local_res:.3E}). The commonest "
            f"cause is a guess rank that is simply too small: eigb never grows "
            f"the rank beyond B * r_guess, and at B == 1 not at all, so it "
            f"converges inside the manifold it was handed. Start from a random "
            f"guess of larger rank (ranks used: {hist.ranks}), or raise rmax.",
            RuntimeWarning, stacklevel=3)


# --- small shape-shuffling helpers ------------------------------------------

def _truncation_rank(s, eps2, rmax, ncols):
    """Rank kept by a relative-``eps2`` truncation, capped by ``rmax``."""
    nrm = float(bk.norm(s))
    return max(1, min(_ops.chop(s, eps2 * nrm), int(rmax), int(ncols)))


def _apply_left(s, core):
    """``s @ core`` on the first index of a 3-index core."""
    r0, nk, r1 = core.shape
    return (s @ core.reshape((r0, nk * r1))).reshape((s.shape[0], nk, r1))


def _merge_left(core, us):
    """core ``(a, n, c)`` times ``us (B, c, rnew)`` -> block ``(a, n, rnew, B)``."""
    return einsum(core, us, "a n c, B c k -> a n k B")


def _merge_right(sv, core):
    """``sv (rnew, b, B)`` times core ``(b, n, c)`` -> block ``(rnew, n, c, B)``."""
    return einsum(sv, core, "k b B, b n c -> k n c B")
