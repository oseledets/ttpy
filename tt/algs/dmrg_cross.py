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
        self._lu = {}                        # bond -> lu_factor of M_p

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
        """``M_p = A(I_p, J_p)`` -- a row subset of ``C[p]``, factorised."""
        if p not in self._lu:
            ii = np.array([t[0] for t in self.quad[p]], dtype=np.int64)
            jj = np.array([t[1] for t in self.quad[p]], dtype=np.int64)
            m = self.C[p][ii, jj, :]
            self._lu[p] = sla.lu_factor(m)
        return self._lu[p]

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
    left = st.left_prefixes(p)
    right = st.right_suffixes(p)
    out = np.empty((len(pts), len(st.n)), dtype=np.int64)
    for row, (i, j, k, q) in enumerate(pts):
        out[row, :p] = left[i]
        out[row, p] = j
        out[row, p + 1] = k
        out[row, p + 2:] = right[q]
    return out


# --- the sweep ---------------------------------------------------------------

def _lottery(rng, wcol, wrow, npnt):
    """``npnt`` superblock points drawn by the original's index lottery.

    ``wcol``/``wrow`` are non-negative weights over the flattened row and
    column positions; positions already holding a pivot carry weight zero.
    Returns ``None`` when a side has no admissible position left.
    """
    scol, srow = wcol.sum(), wrow.sum()
    if scol == 0.0 or srow == 0.0:
        return None
    ij = rng.choice(len(wcol), size=npnt, p=wcol / scol)
    kq = rng.choice(len(wrow), size=npnt, p=wrow / srow)
    return ij, kq


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
    lu = st.cross_matrix(p)
    W = sla.lu_solve(lu, Rm)                     # M_p^{-1} R -- (rp, n2*r2)
    piv = opts["pivoting"]

    def _amax(vals):
        hist.amax = max(hist.amax, float(np.abs(vals).max()))

    if piv == -1:                                # full search over the superblock
        pts = [(i, j, k, q) for i in range(r1) for j in range(n1)
               for k in range(n2) for q in range(r2)]
        a = _evaluate(fun, _point_indices(st, p, pts), counter)
        _amax(a)
        a = a.reshape(r1 * n1, n2 * r2)
        res = a - Cm @ W
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
    drawn = _lottery(opts["rng"], wcol, wrow, nlot)
    if drawn is None:                            # every position is a pivot
        return None
    ijpos, kqpos = drawn
    pts = [(ij // n1, ij % n1, kq // r2, kq % r2)
           for ij, kq in zip(ijpos, kqpos)]
    b = _evaluate(fun, _point_indices(st, p, pts), counter)
    _amax(b)
    res = b - np.einsum("ls,ls->l", Cm[ijpos], W[:, kqpos].T)
    best = int(np.argmax(np.abs(res)))
    ii, jj, kk, qq = pts[best]
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
                rescol = acol - Cm @ W[:, kk * r2 + qq]
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
                resrow = arow - Cm[ii * n1 + jj] @ W
                pos = int(np.argmax(np.abs(resrow)))
                k, q = divmod(pos, r2)
                done = (acol is not None) and (k, q) == (kk, qq)
                kk, qq = k, q
                pivot = resrow[pos]
    return pivot, ii, jj, kk, qq, acol, arow


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
            "rng": np.random.default_rng(seed)}
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

    # -- assemble the TT: fold M_p^{-1} into the last axis of C_p -------------
    cores = []
    for p in range(d - 1):
        r1, npp, rp = st.C[p].shape
        lu = st.cross_matrix(p)
        core = sla.lu_solve(lu, st.C[p].reshape(r1 * npp, rp).T, trans=1).T
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
