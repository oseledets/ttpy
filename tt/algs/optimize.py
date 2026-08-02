"""Global minimization by TT-cross: the minimal element of a tensor / a function.

The idea (Sozykin, Oseledets) is to run a cross-approximation sweep, but to
choose the interpolation indices for a *transformed* tensor instead of the
tensor itself.  At every site the sweep looks at a small block

    C[a, i, b] = A[ J_left[a], i, J_right[b] ],

remembers the smallest entry it has ever seen (``lam``), and then applies a
monotonically **decreasing** map ``s(., lam)`` to the block before selecting the
new indices.  Small values of ``A`` become large values of ``s``, so the
maximum-volume row selection -- which chases large entries -- concentrates the
index sets around the region where ``A`` is small.  The default map

    s(p, lam) = pi/2 - arctan((p - lam) / rho)

is bounded, so a single huge entry cannot dominate the selection, and it is
steep exactly at the current record, which is where the resolution is needed;
``rho`` sets that steepness.

Nothing here is a descent method: the returned point is the best entry the sweep
has *seen*, and it is reported together with the number of entries examined, so
"how much did this cost and how sure am I" is answerable.  The value returned is
always re-evaluated at the returned point, so the pair ``(val, point)`` is
self-consistent by construction rather than by hope.

References
----------
* I. Sozykin, I. Oseledets, "TT-cross based global optimization of
  multidimensional arrays" (the ``tt.optimize.tt_min`` module of legacy ttpy,
  which this file replaces).
* A. Mikhalev, I. Oseledets, "Rectangular maximum-volume submatrices and their
  applications", Linear Algebra Appl. 538 (2018) 187-211, arXiv:1502.07838 --
  the row selection used at every step.

Differences from the legacy ``tt/optimize/tt_min.py``
----------------------------------------------------
* The left and right index sets are two separate lists.  The legacy code stored
  both in one array ``Jy`` and relied on the sweep direction to decide which
  meaning a given entry currently has; that is correct but unreadable, and it
  cannot be checked.
* Both sweep directions truncate the smoothed block to ``rmax`` singular
  vectors before the row selection.  The legacy code did an SVD going right to
  left but a plain QR going left to right, which let the index sets grow to
  ~4x ``rmax`` on every second half-sweep.
* ``min_func`` evaluates the objective on a ``(P, d)`` array of points *always*,
  including the final re-evaluation at the record point (the legacy code passed
  a bare ``(d,)`` vector there, so a vectorized objective crashed at the very
  end of a successful run).
* Both functions return a history object with the record trail, the number of
  evaluations and the index-set sizes.

Implementation note: the sweep is numpy code.  It is index bookkeeping over
blocks of a few hundred entries plus one small SVD per site; the TT cores stay
on their own backend and only the ``(rL, n, rR)`` block is moved to host memory
(``min_tens``), which is also what lets a user's ``smooth_fun`` be plain numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from ..core.vector import vector
from . import _indexing as _idx
from .maxvol import rect_maxvol

__all__ = ["min_tens", "min_func", "MinHistory"]


@dataclass
class MinHistory:
    """What the search knows about itself; recorded in full even with ``verb=0``.

    Attributes:
        records: One dict per improvement of the record, with keys ``sweep``,
            ``site``, ``value``, ``point`` (multi-index), ``evaluations``
            (entries examined so far) and ``time``.
        evaluations: Number of tensor entries / function values examined,
            counted **with repetitions**: consecutive sweeps revisit
            overlapping blocks, so this is the work done, not the number of
            distinct points seen (which is at most this and at most
            ``prod(n)``).
        sweeps: Number of completed sweeps.
        index_sizes: ``d + 1`` pairs ``(left, right)`` -- the sizes of the left
            and right interface index sets held at the end of the run, the
            analogue of the TT ranks of the search.  ``None`` means the set was
            never built (the leftmost right-set and the rightmost left-set are
            never needed).
        value: The final value.
        point: The final point (a multi-index for :func:`min_tens`, a point of
            the grid for :func:`min_func`).
        consistency: ``|value - min(block)|``, i.e. the difference between the
            value re-evaluated at the returned point and the value the sweep
            saw in the block.  It is zero up to roundoff; a large number means
            the objective is not a function of its argument (random, stateful)
            and the result is meaningless.
        time: Wall-clock seconds.
    """

    records: list = field(default_factory=list)
    evaluations: int = 0
    sweeps: int = 0
    index_sizes: list = field(default_factory=list)
    value: float = float("nan")
    point: object = None
    consistency: float = float("nan")
    time: float = 0.0

    @property
    def max_index_set(self):
        """Largest interface index set built during the run."""
        sizes = [s for pair in self.index_sizes for s in pair if s is not None]
        return max(sizes) if sizes else 0

    def __repr__(self):
        return (f"MinHistory(value={self.value:.6g}, sweeps={self.sweeps}, "
                f"evaluations={self.evaluations}, "
                f"records={len(self.records)}, "
                f"max_index_set={self.max_index_set}, "
                f"consistency={self.consistency:.2e}, time={self.time:.2f}s)")


def _default_smooth(rho):
    """``s(p, lam) = pi/2 - arctan((p - lam)/rho)``: decreasing, bounded, steep at ``lam``."""
    if not rho > 0:
        raise ValueError(f"rho must be positive, got {rho}")

    def smooth_fun(p, lam):
        return 0.5 * np.pi - np.arctan((p - lam) / rho)

    return smooth_fun


def _select_rows(smoothed, rank_cap):
    """Rows of maximal 2-volume among the dominant ``rank_cap`` left singular vectors.

    ``smoothed`` is the ``(N, r)`` matrix whose rows are the candidate indices.
    Truncating to ``rank_cap`` columns first is what keeps the index sets from
    growing: the row selection returns at least as many rows as columns.
    """
    u = bk.svd(smoothed)[0]
    q = u[:, :max(1, min(rank_cap, u.shape[1]))]
    piv, _ = rect_maxvol(q)
    return np.asarray(piv, dtype=np.int64)


def _search(evaluate, n, rmax, nswp, smooth_fun, verb, rng, label):
    """The shared sweep of :func:`min_tens` and :func:`min_func`.

    Args:
        evaluate: ``evaluate(left, k, right) -> (rL, n_k, rR) numpy block`` with
            ``block[a, i, b]`` the value at multi-index
            ``[left[a], i, right[b]]``.
        n: List of mode sizes.
        rmax: Cap on the number of singular vectors kept at every site; the
            index sets themselves come out somewhat larger (the rectangular
            maxvol adds rows until the interpolation coefficients are bounded).
        nswp: Number of sweeps (one sweep = one pass over all sites).
        smooth_fun: ``smooth_fun(block, lam)``, decreasing in its first argument.
        verb: Print each new record.
        rng: ``numpy.random.Generator`` used for the initial index sets.
        label: Word used in the printed record lines.

    Returns:
        ``(value, multi_index, history)``.
    """
    d = len(n)
    if int(nswp) < 1:
        raise ValueError(
            f"nswp must be at least 1, got {nswp}: with no sweep the search "
            "has looked at no entry at all and there is nothing to return")
    if rmax is not None and int(rmax) < 1:
        raise ValueError(f"rmax must be at least 1 (or None for no cap), got {rmax}")
    hist = MinHistory()
    t0 = time.perf_counter()

    left = [None] * (d + 1)
    right = [None] * (d + 1)
    left[0] = np.zeros((1, 0), dtype=np.int64)
    right[d] = np.zeros((1, 0), dtype=np.int64)

    # Initial left index sets: a random subset of the candidates at each site.
    # Nothing is known about the tensor yet, so a random start is as good as any
    # -- and it is the only source of randomness in the whole method.
    for k in range(d - 1):
        cand = _idx.extend_left(left[k], n[k])
        take = cand.shape[0] if rmax is None else min(rmax, cand.shape[0])
        sel = np.sort(rng.permutation(cand.shape[0])[:take])
        left[k + 1] = cand[sel]

    best_val = np.inf
    best_idx = None
    site, direction, swp = d - 1, -1, 0

    while swp < nswp:
        block = np.asarray(evaluate(left[site], site, right[site + 1]))
        rl, nk, rr = block.shape
        if (rl, nk, rr) != (left[site].shape[0], n[site], right[site + 1].shape[0]):
            raise ValueError(
                f"evaluate returned a block of shape {block.shape}, expected "
                f"{(left[site].shape[0], n[site], right[site + 1].shape[0])}")
        hist.evaluations += block.size

        flat = block.ravel(order="F")
        pos = int(np.argmin(flat))
        if flat[pos] < best_val:
            best_val = float(flat[pos])
            a, rest = pos % rl, pos // rl
            i, b = rest % nk, rest // nk
            best_idx = np.concatenate(
                (left[site][a], [i], right[site + 1][b])).astype(np.int64)
            hist.records.append(dict(sweep=swp, site=site, value=best_val,
                                     point=best_idx.copy(),
                                     evaluations=hist.evaluations,
                                     time=time.perf_counter() - t0))
            if verb:
                print(f"New record: {best_val!r} point: {best_idx.tolist()} "
                      f"{label}: {hist.evaluations}")

        smoothed = np.asarray(smooth_fun(block, best_val), dtype=np.float64)
        if smoothed.shape != block.shape:
            raise ValueError(
                f"smooth_fun changed the block shape: {block.shape} -> {smoothed.shape}")

        if direction < 0 and site > 0:
            # Choose rR' rows of the (n rR, rL) unfolding: a new right index set
            # for modes site .. d-1.
            mat = smoothed.reshape((rl, nk * rr), order="F").T
            piv = _select_rows(mat, rl if rmax is None else min(rl, rmax))
            right[site] = _idx.extend_right(right[site + 1], nk)[piv]
        elif direction > 0 and site < d - 1:
            mat = smoothed.reshape((rl * nk, rr), order="F")
            piv = _select_rows(mat, rr if rmax is None else min(rr, rmax))
            left[site + 1] = _idx.extend_left(left[site], nk)[piv]

        site += direction
        if site == d or site == -1:
            direction = -direction
            site += direction
            swp += 1
            hist.sweeps = swp

    hist.index_sizes = [(None if left[k] is None else left[k].shape[0],
                         None if right[k] is None else right[k].shape[0])
                        for k in range(d + 1)]
    hist.time = time.perf_counter() - t0
    if best_idx is None:
        # Only reachable when no entry ever compared smaller than +inf, i.e.
        # every value examined was NaN.  Returning "the best seen" would then
        # mean returning nothing at all, dressed up as an answer.
        raise FloatingPointError(
            f"no finite value was found in {hist.evaluations} entries examined: "
            "the objective returned NaN everywhere the sweep looked")
    return best_val, best_idx, hist


def min_tens(tens, rmax=10, nswp=10, verb=True, smooth_fun=None, *,
             rho=1.0, seed=None, return_history=False):
    """Approximate minimal element of a TT tensor.

    Args:
        tens: A :class:`tt.vector` with boundary ranks 1.
        rmax: Cap on the number of singular vectors kept at every site;
            ``None`` means no cap.  The index sets end up somewhat larger (see
            :func:`_search`); their actual sizes are in
            ``history.index_sizes``.
        nswp: Number of sweeps.
        verb: Print every new record.
        smooth_fun: ``smooth_fun(block, lam)``, a decreasing function of the
            first argument applied elementwise; defaults to
            ``pi/2 - arctan((p - lam)/rho)``.
        rho: Steepness of the default ``smooth_fun`` (ignored if one is given).
            The legacy default is ``1.0``.
        seed: Seed for the random initial index sets; pass one to make a run
            reproducible.
        return_history: Also return the :class:`MinHistory`.

    Returns:
        ``(value, point)``, or ``(value, point, history)`` when
        ``return_history``.  ``point`` is a numpy integer multi-index and
        ``value`` is ``tens[point]``, re-evaluated -- not the value the sweep
        happened to hold.

    Note:
        This is a heuristic: with too few sweeps or too small ``rmax`` it simply
        returns the best entry it saw.  ``history.evaluations`` says how much
        work that took, and comparing it with ``prod(n)`` is the only honest
        statement about confidence available -- when the two are of the same
        order the sweep has effectively enumerated the tensor and the answer is
        the true minimum; when it is orders of magnitude smaller, the answer is
        a guess supported by the structure of the tensor.
    """
    if not isinstance(tens, vector):
        raise TypeError(f"min_tens expects a tt.vector, got {type(tens)!r}")
    cores = list(tens.cores)
    if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        raise ValueError("min_tens needs boundary ranks equal to 1, got "
                         f"r[0]={cores[0].shape[0]}, r[d]={cores[-1].shape[2]}")
    if bk.is_complex(cores[0]):
        raise TypeError("min_tens orders the entries, which a complex tensor "
                        "does not admit; take .real() or abs() first")
    n = [int(x) for x in tens.n]
    d = len(n)
    smooth = smooth_fun if smooth_fun is not None else _default_smooth(rho)
    rng = np.random.default_rng(seed)

    def evaluate(left, k, right):
        # left interface rows (rL, r_k) and right interface columns (r_{k+1}, rR),
        # then one contraction with the free core -- O(rL n r^2 + rL n rR r),
        # never one element at a time.
        lft = _idx.left_product(cores[:k], left)
        rgt = _idx.right_product(cores[k + 1:], right)
        blk = einsum(lft, cores[k], "p a, a n b -> p n b")
        blk = einsum(blk, rgt, "p n b, b q -> p n q")
        return bk.to_numpy(blk)

    if d == 1:
        vals = np.asarray(bk.to_numpy(cores[0])).reshape(-1)
        i = int(np.argmin(vals))
        hist = MinHistory(evaluations=vals.size, sweeps=0, index_sizes=[(1, None), (None, 1)],
                          value=float(vals[i]), point=np.array([i]),
                          consistency=0.0)
        return (hist.value, hist.point, hist) if return_history else (hist.value, hist.point)

    seen, point, hist = _search(evaluate, n, rmax, nswp, smooth, verb, rng,
                                "elements seen")
    value = float(np.real(bk.to_numpy(_idx.sample(cores, point.reshape(1, d)))[0]))
    hist.value, hist.point = value, point
    hist.consistency = abs(value - seen)
    if return_history:
        return value, point, hist
    return value, point


def min_func(fun, bounds_min, bounds_max, d=None, rmax=10, nswp=10, n0=64,
             rho=0.5, smooth_fun=None, verb=True, *, seed=None,
             return_history=False):
    """Approximate minimum of a function on a tensor-product grid.

    The grid is ``n0`` equispaced points per dimension between ``bounds_min``
    and ``bounds_max``; the minimum is searched with the same cross sweep as
    :func:`min_tens`, so the objective is only ever evaluated at
    ``history.evaluations`` of the ``n0**d`` grid nodes.

    Args:
        fun: The objective.  It is called as ``fun(X)`` with ``X`` a
            ``(P, d)`` array of points and must return ``P`` values.  It is
            *never* called on a single ``(d,)`` vector, including for the final
            re-evaluation at the record point.
        bounds_min, bounds_max: Box of the search.  Either two scalars together
            with ``d``, or two length-``d`` sequences (then ``d`` is inferred).
        d: Number of dimensions; required when the bounds are scalars.
        rmax: Cap on the number of singular vectors kept at every site;
            ``None`` means no cap.
        nswp: Number of sweeps.
        n0: Number of grid points per dimension.
        rho: Steepness of the default ``smooth_fun``: the smaller ``rho``, the
            more sharply the index selection concentrates on points near the
            current record.  Ignored when ``smooth_fun`` is given.
        smooth_fun: ``smooth_fun(block, lam)``, decreasing in the first argument.
        verb: Print every new record.
        seed: Seed for the random initial index sets.
        return_history: Also return the :class:`MinHistory`.

    Returns:
        ``(value, point)``, or ``(value, point, history)`` when
        ``return_history``.  ``point`` is a ``(d,)`` array of coordinates and
        ``value`` is ``fun(point)``, re-evaluated -- so the pair is always
        consistent, and ``history.consistency`` reports the difference against
        what the sweep saw (nonzero only for a non-deterministic objective).

    Note:
        The result is confined to the grid: the exact minimizer is found only up
        to the grid step ``(bounds_max - bounds_min) / (n0 - 1)``, and the value
        only up to what the function does over that step.
    """
    if d is None:
        a = np.asarray(bounds_min, dtype=np.float64).ravel()
        b = np.asarray(bounds_max, dtype=np.float64).ravel()
        if a.size != b.size or a.size < 1:
            raise ValueError(
                f"bounds_min and bounds_max must have equal length >= 1, "
                f"got {a.size} and {b.size}; pass d= for scalar bounds")
        d = a.size
    else:
        d = int(d)
        a = np.full(d, 1.0, dtype=np.float64) * np.asarray(bounds_min, dtype=np.float64)
        b = np.full(d, 1.0, dtype=np.float64) * np.asarray(bounds_max, dtype=np.float64)
    if np.any(b <= a):
        bad = int(np.flatnonzero(b <= a)[0])
        raise ValueError(f"empty search box in dimension {bad}: "
                         f"bounds_min={a[bad]} >= bounds_max={b[bad]}")

    n = [int(n0)] * d
    grid = [np.linspace(a[k], b[k], n[k]) for k in range(d)]
    smooth = smooth_fun if smooth_fun is not None else _default_smooth(rho)
    rng = np.random.default_rng(seed)

    def points(multi_index):
        """Grid coordinates of a ``(P, d)`` multi-index array."""
        out = np.empty(multi_index.shape, dtype=np.float64)
        for k in range(d):
            out[:, k] = grid[k][multi_index[:, k]]
        return out

    def call(multi_index):
        vals = np.asarray(fun(points(multi_index)), dtype=np.float64).ravel()
        if vals.size != multi_index.shape[0]:
            raise ValueError(
                f"fun returned {vals.size} values for {multi_index.shape[0]} "
                "points; it must map a (P, d) array to P values")
        return vals

    def evaluate(left, k, right):
        j = _idx.index_block(left, n[k], right)
        return call(j).reshape((left.shape[0], n[k], right.shape[0]), order="F")

    if d == 1:
        j = np.arange(n[0], dtype=np.int64).reshape(-1, 1)
        vals = call(j)
        i = int(np.argmin(vals))
        hist = MinHistory(evaluations=vals.size, sweeps=0, index_sizes=[(1, None), (None, 1)],
                          value=float(vals[i]), point=points(j[i:i + 1])[0],
                          consistency=0.0)
        return (hist.value, hist.point, hist) if return_history else (hist.value, hist.point)

    seen, index, hist = _search(evaluate, n, rmax, nswp, smooth, verb, rng,
                                "fevals")
    point = points(index.reshape(1, d))[0]
    value = float(call(index.reshape(1, d))[0])
    hist.evaluations += 1
    hist.value, hist.point = value, point
    hist.consistency = abs(value - seen)
    if return_history:
        return value, point, hist
    return value, point
