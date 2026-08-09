"""Greedy DMRG cross interpolation -- Savostyanov's ttcross, ported.

This is the algorithm of Savostyanov, *Quasioptimality of maximum-volume cross
interpolation of tensors*, LAA 458:217-244, 2014, in the sequential form of the
reference implementation ``github.com/savostyanov/ttcross`` (``dmrgg.f90``); the
parallel-over-bonds form is Dolgov & Savostyanov, arXiv:1903.11554.  It is not
:func:`tt.algs.cross.rect_cross` with different constants; it is a different
construction, and the two are complementary:

* here the rank grows by **at most one per bond per sweep**, and the pivot is
  the entry of largest *residual* ``|A - interpolant|`` found by a rook search
  over one column and one row of the two-site superblock;
* there the rank grows by ``kickrank`` per micro-step and the pivots maximise
  the 2-volume of an orthonormal basis -- a proxy for the residual.

The greedy growth is far more parsimonious in function evaluations on smooth
tensors, which is what it was built for (high-dimensional quadrature); the
measured comparison against the Fortran original and against ``rect_cross`` is
in ``docs/plans/cross-approximation.md`` section 2.1a.

State, exactly as in the original
---------------------------------
Per bond ``p`` a list of pivot quadruples ``(i, j, k, q)``: ``i`` indexes the
pivot list of bond ``p-1``, ``j`` is the mode index at dimension ``p``, ``k``
the mode index at dimension ``p+1``, ``q`` indexes the pivot list of bond
``p+1``.  A full multi-index is recovered by walking the quadruples left and
right, so the index sets are **nested by construction** -- the property the
quasioptimality proof needs.  Per dimension one block ``C_p`` of raw function
values ``A(I_{p-1}, n_p, J_p)``; the cross matrix ``M_p = A(I_p, J_p)`` is a
row subset of ``C_p``, and the interpolant is
``C_1 M_1^{-1} C_2 M_2^{-1} ... C_d``.

Deliberate differences from ``dmrgg.f90``, stated plainly
---------------------------------------------------------
* The cross matrix is factorised by LAPACK LU with partial pivoting instead of
  the original's incrementally bordered LU without pivoting.  Same interpolant,
  same evaluation count; the factorisation of an ``r x r`` matrix is noise next
  to the function calls.
* ``fun`` follows this package's cross contract -- one call on an integer array
  of shape ``(batch, d)``, 0-based -- where the original calls a scalar Fortran
  function under OpenMP.  The batches are whole fibers, so vectorisation is
  structural, not cosmetic.
* The accuracy knob is exposed: the reference driver hard-codes
  ``accuracy = 500 * eps_machine``, which is where its published integrals
  saturate; here ``eps`` is the caller's.
* The lottery is seeded (``seed``), so a run is reproducible; the original
  draws from an unseeded generator.
* Ties in the residual argmax may break differently than Fortran's ``idamax``,
  so pivot *sequences* can differ from the original run to run; the
  interpolation quality is the property, not the pivot list.

References
----------
* D. V. Savostyanov, "Quasioptimality of maximum-volume cross interpolation of
  tensors", Linear Algebra Appl. 458:217-244, 2014, arXiv:1305.1818.
* S. Dolgov, D. V. Savostyanov, "Parallel cross interpolation for
  high-precision calculation of high-dimensional integrals", Comput. Phys.
  Commun. 246:106869, 2020, arXiv:1903.11554.
* Reference implementation: ``github.com/savostyanov/ttcross`` (GPL-2.0,
  read for this port, not copied).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from ..core.vector import vector
from .cross import _Counter, _evaluate, _to_backend, element
from .. import backend as bk

__all__ = ["dmrg_cross", "greedy_cross", "DmrgCrossHistory"]

#: An accepted pivot must beat the largest value seen so far by more than the
#: rounding floor... (``small_element`` of dmrgg.f90, float64 branch).
SMALL_ELEMENT = 10 * np.finfo(np.float64).eps

#: ...and must not collapse relative to the previous sweep's best pivot
#: (``small_pivot``): a pivot 1e5 times smaller than the last sweep's best is
#: read as "this bond is done", not as rank worth adding.
SMALL_PIVOT = 1e-5


@dataclass
class DmrgCrossHistory:
    """What the run knows about itself; recorded in full even with ``verbose=False``.

    Attributes:
        eps: Requested pivot threshold (``None`` if only ``rmax`` stopped it).
        pivoting: The rook-search depth that was used.
        sweeps: One dict per sweep with keys ``sweep, dir, pivotmax, amax,
            accepted, max_rank, fun_eval, time`` (``fun_eval`` is the running
            total; ``pivotmax`` is the largest accepted residual pivot of the
            sweep, ``nan`` if the sweep accepted nothing).
        fun_eval: Number of function values requested by the cross itself.
        fun_eval_check: Extra values spent on the held-out accuracy check.
        amax: Largest ``|A|`` entry seen anywhere -- the reference against
            which the stopping rule reads ``pivotmax``.
        converged: The three-strike pivot criterion fired
            (``pivotmax <= eps * amax`` on ``strike_limit`` consecutive
            sweeps).  A stop at ``rmax`` or ``nswp`` is not convergence and
            says so in ``stop_reason``.
        stop_reason: ``'eps'``, ``'rmax'`` or ``'nswp'``.
        strikes: Consecutive under-threshold sweeps at the moment of stopping.
        ranks: TT ranks of the returned tensor.
        err_check: Held-out relative error in the Frobenius norm, or ``None``
            when ``n_check == 0``.  A Monte Carlo measurement: it resolves what
            a uniform sample can hit and nothing finer.
        err_check_inf: Same sample, infinity norm -- the norm the pivot
            criterion actually speaks (this is ``dtt_accchk``'s pair of norms).
        err_check_worst: Multi-index of the worst held-out error.
        compiled: The numba fast path ran (``fun`` was a jitted dispatcher and
            the ``[fast]`` extra is installed); the numpy path computes the
            same thing, at interpreter speed.
        time: Wall-clock seconds.
    """

    eps: float | None = None
    pivoting: int = 1
    sweeps: list = field(default_factory=list)
    fun_eval: int = 0
    fun_eval_check: int = 0
    amax: float = 0.0
    converged: bool = False
    stop_reason: str = ""
    strikes: int = 0
    ranks: list = field(default_factory=list)
    err_check: float | None = None
    err_check_inf: float | None = None
    err_check_worst: list | None = None
    compiled: bool = False
    time: float = 0.0

    def __repr__(self):
        chk = "n/a" if self.err_check is None else f"{self.err_check:.2e}"
        return (f"DmrgCrossHistory(sweeps={len(self.sweeps)}, "
                f"converged={self.converged}, stop={self.stop_reason!r}, "
                f"err_check={chk}, fun_eval={self.fun_eval}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


# --- state -------------------------------------------------------------------

class _State:
    """Blocks, pivot sets and the derived cross matrices of one run.

    The owners of the truth are ``C`` (raw value blocks, one per dimension) and
    ``quad`` (pivot quadruples, one list per bond).  ``lp``/``rs`` -- the
    assembled left prefixes and right suffixes of the pivots -- and the LU
    factorisations are caches maintained next to them.
    """

    def __init__(self, n):
        self.n = n
        d = len(n)
        self.C = [None] * d                  # C[k]: (r_{k-1}, n_k, r_k) raw values
        self.quad = [[] for _ in range(d - 1)]   # per bond: list of (i, j, k, q)
        self.lp = [None] * (d - 1)           # lp[p]: (r_p, p+1) left prefixes
        self.rs = [None] * (d - 1)           # rs[p]: (r_p, d-p-1) right suffixes
        self._lu = {}                        # bond -> factorised M_p
        self._luarr = {}                     # bond -> (lu, piv0) for the kernel

    def r(self, p):
        """Rank of bond ``p``; the boundaries count as rank 1."""
        d = len(self.n)
        if p < 0 or p >= d - 1:
            return 1
        return len(self.quad[p])

    def left_prefixes(self, p):
        """(r_{p-1}, p) array of left multi-indices; one empty row at p = 0."""
        if p == 0:
            return np.empty((1, 0), dtype=np.int64)
        return self.lp[p - 1]

    def right_suffixes(self, p):
        """(r_{p+1}, d-p-2) array of right multi-indices; empty at the end."""
        d = len(self.n)
        if p >= d - 2:
            return np.empty((1, 0), dtype=np.int64)
        return self.rs[p + 1]

    def cross_matrix(self, p):
        """``M_p = A(I_p, J_p)`` factorised: ``(getrs, lu, piv)``.

        Raw LAPACK ``getrf``/``getrs`` instead of ``lu_factor``/``lu_solve``:
        the sweep calls this once per bond visit on a wide right-hand side,
        and the scipy wrapper overhead costs more than the solve.  It stays a
        *solve* -- an explicit ``M^{-1}`` applied as a GEMM was tried and
        reverted: multiplication by an inverse is not backward stable, the
        residual noise floor rises from ``eps`` to ``cond(M) * eps``, and the
        pivot-acceptance threshold (calibrated for a backward-stable residual)
        starts accepting duplicate pivots, which makes ``M`` singular.
        """
        if p not in self._lu:
            ii = np.array([t[0] for t in self.quad[p]], dtype=np.int64)
            jj = np.array([t[1] for t in self.quad[p]], dtype=np.int64)
            m = np.asfortranarray(self.C[p][ii, jj, :])
            getrf, getrs = sla.get_lapack_funcs(("getrf", "getrs"), (m,))
            lu, piv, _info = getrf(m, overwrite_a=True)
            self._lu[p] = (getrs, lu, piv)
        return self._lu[p]

    def solve(self, p, b, trans=0):
        """``M_p^{-1} b`` (or ``M_p^{-T} b``), backward stably."""
        getrs, lu, piv = self.cross_matrix(p)
        x, _info = getrs(lu, piv, b, trans=trans)
        return x

    def lu_arrays(self, p):
        """The factors as plain arrays for the compiled kernel.

        scipy's raw ``getrf`` wrapper already converts LAPACK's 1-based pivot
        indices to 0-based (verified against ``lu_factor``, which documents
        the convention) -- do not subtract 1 again.
        """
        if p not in self._luarr:
            _getrs, lu, piv = self.cross_matrix(p)
            self._luarr[p] = (np.ascontiguousarray(lu), piv.astype(np.int64))
        return self._luarr[p]

    def accept(self, p, ii, jj, kk, qq, acol, arow):
        """Append pivot ``(ii, jj, kk, qq)`` at bond ``p`` with its fibers."""
        r1, npp, _ = self.C[p].shape
        _, np2, r2 = self.C[p + 1].shape
        self.C[p] = np.concatenate(
            [self.C[p], acol.reshape(r1, npp, 1)], axis=2)
        self.C[p + 1] = np.concatenate(
            [self.C[p + 1], arow.reshape(1, np2, r2)], axis=0)
        self.quad[p].append((ii, jj, kk, qq))
        newl = np.concatenate([self.left_prefixes(p)[ii],
                               np.array([jj], dtype=np.int64)])
        newr = np.concatenate([np.array([kk], dtype=np.int64),
                               self.right_suffixes(p)[qq]])
        self.lp[p] = (newl[None, :] if self.lp[p] is None
                      else np.vstack([self.lp[p], newl]))
        self.rs[p] = (newr[None, :] if self.rs[p] is None
                      else np.vstack([self.rs[p], newr]))
        # Only M_p gained a row and a column; M_{p +- 1} index into the same
        # pivots as before and their factorisations stay valid.
        self.pop_lu(p)

    def pop_lu(self, p):
        self._lu.pop(p, None)
        self._luarr.pop(p, None)


# --- index assembly ----------------------------------------------------------

def _col_indices(st, p, kk, qq):
    """Multi-indices of the full column fiber ``(all i, all j | kk, qq)``.

    C-ordered ``(i outer, j inner)``, so the values reshape to
    ``(r_{p-1}, n_p)`` directly.
    """
    left = st.left_prefixes(p)
    r1 = left.shape[0]
    n = st.n[p]
    tail = np.concatenate([np.array([kk], dtype=np.int64),
                           st.right_suffixes(p)[qq]])
    out = np.empty((r1 * n, len(st.n)), dtype=np.int64)
    out[:, :p] = np.repeat(left, n, axis=0)
    out[:, p] = np.tile(np.arange(n, dtype=np.int64), r1)
    out[:, p + 1:] = tail
    return out


def _row_indices(st, p, ii, jj):
    """Multi-indices of the full row fiber ``(ii, jj | all k, all q)``.

    C-ordered ``(k outer, q inner)`` -> reshape ``(n_{p+1}, r_{p+1})``.
    """
    head = np.concatenate([st.left_prefixes(p)[ii],
                           np.array([jj], dtype=np.int64)])
    right = st.right_suffixes(p)
    r2 = right.shape[0]
    n2 = st.n[p + 1]
    out = np.empty((n2 * r2, len(st.n)), dtype=np.int64)
    out[:, :p + 1] = head
    out[:, p + 1] = np.repeat(np.arange(n2, dtype=np.int64), r2)
    out[:, p + 2:] = np.tile(right, (n2, 1))
    return out


def _point_indices(st, p, pts):
    """Multi-indices for arbitrary superblock points ``(i, j, k, q)``."""
    pts = np.asarray(pts, dtype=np.int64)
    out = np.empty((len(pts), len(st.n)), dtype=np.int64)
    out[:, :p] = st.left_prefixes(p)[pts[:, 0]]
    out[:, p] = pts[:, 1]
    out[:, p + 1] = pts[:, 2]
    out[:, p + 2:] = st.right_suffixes(p)[pts[:, 3]]
    return out


# --- the sweep ---------------------------------------------------------------

def _lottery(u1, u2, wcol, wrow):
    """Superblock points from uniform draws, by the original's index lottery.

    ``wcol``/``wrow`` are non-negative weights over the flattened row and
    column positions; positions already holding a pivot carry weight zero.
    The uniforms are drawn by the caller so that the numpy and the compiled
    path consume identical randomness.  Inverse-CDF, as the original's
    ``lottery2`` (cumsum + bisection); ``rng.choice(p=...)`` does the same
    thing an order of magnitude slower.
    """
    scol, srow = wcol.sum(), wrow.sum()
    ij = np.searchsorted(np.cumsum(wcol), u1 * scol, side="right")
    kq = np.searchsorted(np.cumsum(wrow), u2 * srow, side="right")
    return (np.minimum(ij, len(wcol) - 1), np.minimum(kq, len(wrow) - 1))


def _bond_pivot(fun, st, p, opts, counter, hist, start_with_row):
    """Rook-search one bond; return the pivot and its fibers, or ``None``.

    This is the ``piv >= 0`` branch of ``dmrgg.f90`` (lottery seed + rook
    alternation) and the ``piv = -1`` full search, with the original's exit
    logic kept exactly: an evaluation-budget exit after ``2 * pivoting``
    crossings, a fixed-point exit when the argmax stops moving, and fibers
    that are always consistent with the final quadruple.
    """
    n1, n2 = st.n[p], st.n[p + 1]
    r1, r2 = st.r(p - 1), st.r(p + 1)
    rp = st.r(p)
    Cm = st.C[p].reshape(r1 * n1, rp)
    Rm = st.C[p + 1].reshape(rp, n2 * r2)
    piv = opts["pivoting"]
    # The interpolant is applied one column / one row / one scattered batch at
    # a time -- never as the full ``M^{-1} R``: forming it costs
    # ``O(r^2 n r2)`` per bond visit where the original's incremental factors
    # pay ``O(r n)``, and it was the whole gap to the Fortran (measured;
    # docs/plans/cross-approximation.md 2.1b).

    def _amax(vals):
        hist.amax = max(hist.amax, float(np.abs(vals).max()))

    if piv == -1:                                # full search over the superblock
        pts = np.stack(np.meshgrid(
            np.arange(r1), np.arange(n1), np.arange(n2), np.arange(r2),
            indexing="ij"), axis=-1).reshape(-1, 4)
        a = _evaluate(fun, _point_indices(st, p, pts), counter)
        _amax(a)
        a = a.reshape(r1 * n1, n2 * r2)
        res = a - Cm @ st.solve(p, np.asfortranarray(Rm))
        flat = int(np.argmax(np.abs(res)))
        rowpos, colpos = divmod(flat, n2 * r2)
        ii, jj = divmod(rowpos, n1)
        kk, qq = divmod(colpos, r2)
        pivot = res[rowpos, colpos]
        acol = a[:, colpos]
        arow = a[rowpos, :]
        return pivot, ii, jj, kk, qq, acol, arow

    # -- lottery seed ---------------------------------------------------------
    wcol = np.ones(r1 * n1)
    wrow = np.ones(n2 * r2)
    for (i, j, k, q) in st.quad[p]:
        wcol[i * n1 + j] = 0.0
        wrow[k * r2 + q] = 0.0
    nlot = r1 + n1 + n2 + r2
    u1 = opts["rng"].random(nlot)
    u2 = opts["rng"].random(nlot)
    if wcol.sum() == 0.0 or wrow.sum() == 0.0:   # every position is a pivot
        return None

    if opts["fast"]:
        from . import _dmrg_fast as df
        lu, piv0 = st.lu_arrays(p)
        (status, pivot, ii, jj, kk, qq, acol, arow, neval, amax, badval,
         badidx) = df.bond_kernel(
            fun, st.C[p], st.C[p + 1], lu, piv0,
            np.ascontiguousarray(st.left_prefixes(p)),
            np.ascontiguousarray(st.right_suffixes(p)),
            u1, u2, wcol, wrow, piv, start_with_row, hist.amax)
        counter.n += int(neval)
        hist.amax = float(amax)
        if status == 1:
            return None
        if status == 2:
            raise ValueError(
                f"fun returned a non-finite value {badval!r} at multi-index "
                f"{[int(v) for v in badidx]}; cross cannot interpolate that")
        if status == 3:
            raise ValueError(
                "fun returned the wrong number of values for a batch; it must "
                "be vectorized over the first axis and return one value per "
                "row")
        return float(pivot), int(ii), int(jj), int(kk), int(qq), acol, arow

    ijpos, kqpos = _lottery(u1, u2, wcol, wrow)
    pts = np.stack([ijpos // n1, ijpos % n1, kqpos // r2, kqpos % r2], axis=1)
    b = _evaluate(fun, _point_indices(st, p, pts), counter)
    _amax(b)
    X = st.solve(p, np.asfortranarray(Rm[:, kqpos]))     # (rp, nlot)
    res = b - np.einsum("ls,sl->l", Cm[ijpos], X)
    best = int(np.argmax(np.abs(res)))
    ii, jj, kk, qq = (int(v) for v in pts[best])
    pivot = res[best]

    # -- fibers / rook alternation -------------------------------------------
    acol = arow = None
    if piv == 0:                                 # no rook: just take the fibers
        acol = _evaluate(fun, _col_indices(st, p, kk, qq), counter)
        arow = _evaluate(fun, _row_indices(st, p, ii, jj), counter)
        _amax(acol), _amax(arow)
        return pivot, ii, jj, kk, qq, acol, arow

    crs, done, skipcol = 0, False, start_with_row
    while not done:
        if not skipcol:
            acol = _evaluate(fun, _col_indices(st, p, kk, qq), counter)
            _amax(acol)
            crs += 1
            done = (arow is not None) and crs >= 2 * piv
            if not done:
                rescol = acol - Cm @ st.solve(p, Rm[:, kk * r2 + qq].copy())
                pos = int(np.argmax(np.abs(rescol)))
                i, j = divmod(pos, n1)
                done = (arow is not None) and (i, j) == (ii, jj)
                ii, jj = i, j
                pivot = rescol[pos]
        skipcol = False
        if not done:
            arow = _evaluate(fun, _row_indices(st, p, ii, jj), counter)
            _amax(arow)
            crs += 1
            done = (acol is not None) and crs >= 2 * piv
            if not done:
                resrow = arow - st.solve(p, Cm[ii * n1 + jj].copy(),
                                         trans=1) @ Rm
                pos = int(np.argmax(np.abs(resrow)))
                k, q = divmod(pos, r2)
                done = (acol is not None) and (k, q) == (kk, qq)
                kk, qq = k, q
                pivot = resrow[pos]
    return pivot, ii, jj, kk, qq, acol, arow


def _pick_fast_path(fun, d, pivoting):
    """Whether the compiled bond kernel can run this ``fun``.

    It can when ``fun`` is a numba dispatcher (then the kernel calls it without
    re-entering the interpreter), the ``[fast]`` extra is importable, the
    search is lottery/rook (``pivoting >= 0``; the exhaustive ``-1`` is a
    calibration path and stays numpy), and a one-batch probe from compiled
    code returns float64 -- the kernel is real-valued, complex funs take the
    numpy path.  A dispatcher the kernel cannot type falls back with a
    warning rather than an error: the numpy path computes the same thing.
    """
    if pivoting < 0 or d < 2:
        return False
    from . import _dmrg_fast as df
    if not (df.HAVE_NUMBA and df.is_jitted(fun)):
        return False
    try:
        out = df.probe(fun, np.zeros((2, d), dtype=np.int64))
    except Exception as exc:                     # numba typing errors vary
        warnings.warn(
            f"fun is numba-jitted but the compiled cross kernel cannot call "
            f"it ({type(exc).__name__}); falling back to the numpy path",
            RuntimeWarning, stacklevel=3)
        return False
    arr = np.asarray(out)
    return arr.dtype == np.float64 and arr.ndim == 1 and arr.shape[0] == 2


# --- entry point -------------------------------------------------------------

def dmrg_cross(fun, x0, eps=1e-6, rmax=None, pivoting=1, strike_limit=3,
               nswp=1000, verbose=False, n_check=0, check_seed=0, seed=0):
    """Greedy DMRG cross approximation of a black-box tensor.

    Args:
        fun: Vectorized black box: takes an integer array of shape
            ``(batch, d)`` (0-based multi-indices) and returns ``batch``
            values.  It must be a genuine function of the index.
        x0: A :class:`tt.vector` supplying mode sizes, backend and dtype (its
            ranks are ignored -- the greedy always starts from rank 1), or a
            plain sequence of mode sizes for a numpy/float64 run.
        eps: Pivot threshold of the stopping rule: the run stops once the
            largest accepted residual pivot of a sweep stays below
            ``eps * max|A|`` for ``strike_limit`` consecutive sweeps.  This is
            an infinity-norm-flavoured criterion on the *residual*, not a
            Frobenius-norm certificate on the answer; ``n_check`` measures the
            latter.  ``None`` disables the rule (then ``rmax`` must be given).
        rmax: Hard cap on the TT ranks; with ``rmax = r`` at most ``r - 1``
            sweeps run, since each raises a bond rank by at most one.
        pivoting: Rook-search depth: each accepted pivot spends at most
            ``2 * pivoting`` column/row fiber evaluations hunting the largest
            residual (this is ``PIV`` of the reference driver).  ``0`` accepts
            the lottery argmax without a rook search; ``-1`` searches the whole
            superblock exhaustively -- exact but ``O(r^2 n^2)`` evaluations
            per bond, for small problems and for calibration only.
        strike_limit: Consecutive under-threshold sweeps required to stop
            (the reference uses 3).
        nswp: Safety cap on sweeps; hitting it warns.
        verbose: One line per sweep.  The history is recorded regardless.
        n_check: Measure the held-out error on that many random points after
            the run (both norms, see :class:`DmrgCrossHistory`); costs that
            many extra evaluations.  Contradicting ``eps`` warns.
        check_seed: Seed of the held-out sample.
        seed: Seed of the pivot lottery -- the run is deterministic for a
            fixed seed, unlike the reference implementation.

    Returns:
        :class:`tt.vector` carrying a :class:`DmrgCrossHistory` as
        ``.history``.  The ranks are exactly what the greedy built -- there is
        no SVD and no rounding anywhere; call ``.round(...)`` afterwards if a
        compressed representation is wanted.

    Raises:
        ValueError: on a non-vectorized or non-finite ``fun`` (naming the
            multi-index), ``eps`` and ``rmax`` both absent, a negative ``eps``,
            ``rmax < 1``, ``pivoting < -1``, or ``strike_limit < 1``.

    Warns:
        RuntimeWarning: when the sweep cap ``nswp`` stops a run whose pivot
            criterion did not fire, and when the ``n_check`` measurement
            contradicts the ``eps`` the run claims to have reached.
    """
    if eps is None and rmax is None:
        raise ValueError("need a stopping rule: give eps, rmax, or both")
    if eps is not None and eps <= 0:
        raise ValueError(f"eps must be positive or None, got {eps}")
    if rmax is not None and int(rmax) < 1:
        raise ValueError(f"rmax must be at least 1 or None, got {rmax}")
    if int(pivoting) < -1:
        raise ValueError(f"pivoting must be >= -1, got {pivoting}")
    if int(strike_limit) < 1:
        raise ValueError(f"strike_limit must be >= 1, got {strike_limit}")

    if isinstance(x0, vector):
        n = [int(v) for v in x0.n]
        backend = bk.backend_of(x0.cores[0])
        dtype = bk.dtype_of(x0.cores[0])
    else:
        n = [int(v) for v in x0]
        backend = bk.get_backend()
        dtype = backend.dtype
    opts = {"pivoting": int(pivoting), "backend": backend, "dtype": dtype,
            "rng": np.random.default_rng(seed),
            "fast": _pick_fast_path(fun, len(n), int(pivoting))}
    d = len(n)
    t0 = time.time()
    counter = _Counter()
    hist = DmrgCrossHistory(eps=None if eps is None else float(eps),
                            pivoting=int(pivoting))

    if d == 1:
        idx = np.arange(n[0], dtype=np.int64).reshape((-1, 1))
        vals = _evaluate(fun, idx, counter)
        hist.amax = float(np.abs(vals).max())
        y = vector.from_list([_to_backend(vals.reshape((1, n[0], 1)), opts)])
        hist.converged, hist.stop_reason = True, "eps"
        hist.sweeps.append({"sweep": 0, "dir": "--", "pivotmax": 0.0,
                            "amax": hist.amax, "accepted": 0, "max_rank": 1,
                            "fun_eval": counter.n, "time": time.time() - t0})
        return _finish(fun, y, hist, counter, opts, n_check, check_seed, t0,
                       eps)

    st = _State(n)

    # -- initial cross: probe a few shifted diagonals, take the largest -------
    nn = min(n)
    snum = 8
    probe = np.empty((snum * nn, d), dtype=np.int64)
    for s in range(snum):
        for k in range(nn):
            probe[s * nn + k] = [(k + s * t) % n[t] for t in range(d)]
    vals = _evaluate(fun, probe, counter)
    hist.amax = float(np.abs(vals).max())
    ind0 = probe[int(np.argmax(np.abs(vals)))]

    for p in range(d - 1):
        st.quad[p] = [(0, int(ind0[p]), int(ind0[p + 1]), 0)]
        st.lp[p] = ind0[:p + 1][None, :].astype(np.int64)
        st.rs[p] = ind0[p + 1:][None, :].astype(np.int64)
    for k in range(d):
        fiber = np.tile(ind0, (n[k], 1))
        fiber[:, k] = np.arange(n[k])
        v = _evaluate(fun, fiber, counter)
        hist.amax = max(hist.amax, float(np.abs(v).max()))
        st.C[k] = v.reshape(1, n[k], 1)

    # ``pivotmax_prev`` guards against pivots collapsing between sweeps; the
    # original leaves it at the *unset* -1 after a sweep that accepted nothing,
    # which disarms that guard for the next sweep.  Kept faithfully.
    pivotmax_prev = hist.amax
    strikes = 0
    stop_reason = ""
    it = 0
    if rmax is not None and it + 1 >= int(rmax):
        stop_reason = "rmax"

    while not stop_reason:
        it += 1
        t_swp = time.time()
        lr = (it % 2 == 1)
        bonds = range(d - 1) if lr else range(d - 2, -1, -1)
        pivotmax = -1.0
        accepted = 0
        for p in bonds:
            got = _bond_pivot(fun, st, p, opts, counter, hist,
                              start_with_row=not lr)
            if got is None:
                continue
            pivot, ii, jj, kk, qq, acol, arow = got
            ok = (abs(pivot) > SMALL_ELEMENT * hist.amax
                  and abs(pivot) > SMALL_PIVOT * pivotmax_prev)
            if not ok:
                continue
            st.accept(p, ii, jj, kk, qq, acol, arow)
            accepted += 1
            pivotmax = max(pivotmax, abs(pivot))
        pivotmax_prev = pivotmax

        hist.sweeps.append({
            "sweep": it, "dir": ">>" if lr else "<<",
            "pivotmax": pivotmax if accepted else float("nan"),
            "amax": hist.amax, "accepted": accepted,
            "max_rank": max(st.r(p) for p in range(d - 1)),
            "fun_eval": counter.n, "time": time.time() - t_swp})
        if verbose:
            s = hist.sweeps[-1]
            print(f"{it:3d}{s['dir']} rank {s['max_rank']:3d} "
                  f"pivotmax {s['pivotmax']:.3e} evals {counter.n}")

        if eps is not None:
            strikes = strikes + 1 if pivotmax <= eps * hist.amax else 0
            if strikes >= int(strike_limit):
                stop_reason = "eps"
        if not stop_reason and rmax is not None and it + 1 >= int(rmax):
            stop_reason = "rmax"
        if not stop_reason and it >= int(nswp):
            stop_reason = "nswp"
            warnings.warn(
                f"dmrg_cross spent {nswp} sweeps without the pivot criterion "
                f"firing (pivotmax {pivotmax:.3e} vs eps*amax "
                f"{(eps or 0) * hist.amax:.3e}); returning the current "
                "interpolant", RuntimeWarning, stacklevel=2)

    hist.strikes = strikes
    hist.converged = stop_reason == "eps"
    hist.stop_reason = stop_reason
    hist.compiled = bool(opts["fast"])

    # -- assemble the TT: fold M_p^{-1} into the last axis of C_p -------------
    cores = []
    for p in range(d - 1):
        r1, npp, rp = st.C[p].shape
        core = st.solve(
            p, np.asfortranarray(st.C[p].reshape(r1 * npp, rp).T), trans=1).T
        cores.append(core.reshape(r1, npp, rp))
    cores.append(st.C[d - 1])
    y = vector.from_list([_to_backend(c, opts) for c in cores])
    return _finish(fun, y, hist, counter, opts, n_check, check_seed, t0, eps)


def _finish(fun, y, hist, counter, opts, n_check, check_seed, t0, eps):
    """Held-out check, bookkeeping, and the history attached to the result."""
    hist.fun_eval = counter.n
    hist.ranks = [int(v) for v in y.r]
    if n_check:
        n = [int(v) for v in y.n]
        rng = np.random.default_rng(check_seed)
        idx = np.stack([rng.integers(0, nk, size=int(n_check)) for nk in n],
                       axis=1).astype(np.int64)
        avals = _evaluate(fun, idx, _Counter())
        hist.fun_eval_check = int(n_check)
        bvals = np.asarray(bk.to_numpy(element(y, idx))).reshape(-1)
        diff = np.abs(avals - bvals)
        worst = int(np.argmax(diff))
        ainf = float(np.abs(avals).max())
        afro = float(np.linalg.norm(avals))
        hist.err_check_inf = float(diff[worst]) / max(ainf, 1e-300)
        hist.err_check = float(np.linalg.norm(diff)) / max(afro, 1e-300)
        hist.err_check_worst = idx[worst].tolist()
        if eps is not None and hist.converged and hist.err_check_inf > 100 * eps:
            warnings.warn(
                f"held-out check contradicts the pivot criterion: measured "
                f"relative error {hist.err_check_inf:.3e} (inf-norm on "
                f"{n_check} points) against eps={eps:.1e}",
                RuntimeWarning, stacklevel=3)
    hist.time = time.time() - t0
    y.history = hist
    return y


#: The name used in the greedy-cross literature for exactly this algorithm.
greedy_cross = dmrg_cross
