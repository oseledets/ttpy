"""Tensor completion from sparse samples by alternating least squares.

Given ``P`` known entries of an unknown tensor ``A`` -- a COO list of
multi-indices and values -- find a TT tensor ``X`` of prescribed rank that
minimizes

    J(X) = 1/2 sum_p ( X[i_p] - a_p )^2 .

The functional is quadratic and, crucially, *separately* quadratic in each core:
with all other cores frozen, the entry ``X[i_p]`` is a linear functional of the
free core,

    X[i_p] = L_p^T  X_k[:, i_p(k), :]  R_p ,

where ``L_p`` is the product of the cores left of ``k`` evaluated at ``i_p`` and
``R_p`` the product of the cores right of it.  Better still, the slices
``X_k[:, i, :]`` of one core do not interact: a sample only touches the slice
``i = i_p(k)``.  So one core update splits into ``n_k`` independent least
squares problems of size ``(number of samples hitting that slice) x r_k r_{k+1}``,
each with design matrix rows ``vec(L_p R_p^T)``.  That is the whole algorithm;
sweeping over ``k`` until the functional stops moving is ALS.

Because every sub-problem is solved to its exact minimum, ``J`` cannot increase:
monotonicity is a property of the method and therefore a test, not a hope.  (It
holds for ``alpha = 0``; a positive ``alpha`` truncates the small singular
values of the local systems, which regularizes an underdetermined slice at the
price of no longer being the exact minimizer -- see the argument description.)

What ALS cannot do is choose the rank, and with fewer samples than parameters
the answer is arbitrary in the unsampled directions.  The only meaningful
accuracy statement is therefore the error on entries the fit has never seen; a
small ``J`` on the training set means nothing by itself.

References
----------
* L. Grasedyck, M. Kluge, S. Kraemer, "Variants of alternating least squares
  tensor completion in the tensor train format", SIAM J. Sci. Comput. 37(5),
  2015, arXiv:1509.00311.
* M. Steinlechner, "Riemannian optimization for high-dimensional tensor
  completion", SIAM J. Sci. Comput. 38(5), 2016 (same functional, different
  minimizer).
* Replaces ``tt/completion/als.py`` of legacy ttpy (same public signature).

Differences from the legacy implementation
------------------------------------------
* It no longer modifies the caller's data: the legacy code divided
  ``cooP['values']`` by its norm in place and multiplied it back at the end, so
  an exception in between left the caller's array scaled.
* The design matrices are built for all samples at once with two interface
  sweeps instead of one python-level ``getRow`` call per (sample, slice); the
  cost per sweep drops from ``O(P d^2 r^2)`` python operations to ``O(P d r^2)``
  BLAS.
* A slice with no samples keeps its previous value instead of being zeroed.
  Zeroing changes ``X`` without changing ``J`` -- it silently destroys rank.
* ``alpha`` is actually used (the legacy call had it commented out), and
  ``time.clock`` -- removed from python 3.8 -- is gone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from einops import rearrange

from .. import backend as bk
from ..core import tools
from ..core.vector import vector
from . import _indexing as _idx

__all__ = ["ttSparseALS", "completion_functional", "CompletionHistory"]


@dataclass
class CompletionHistory:
    """What the run knows about itself; recorded in full even with ``verbose=0``.

    Attributes:
        fit: Value of the *relative* functional
            ``J(X) / ||values||^2`` after each sweep (the legacy ``'fit'``
            list).  Relative, so ``tol`` is scale free.
        sweepTime: Wall-clock seconds of each sweep.
        initTime: Seconds spent building the starting tensor.
        converged: Whether the functional actually reached ``tol``.  A run that
            stopped moving at a stationary point above ``tol`` is **not**
            converged, and says so.
        stop_reason: ``'tol'`` (reached the target), ``'stalled'`` (the
            functional stopped changing above the target -- an ALS stationary
            point, see the note on :func:`ttSparseALS`), ``'maxnsweeps'``.
        monotone: Whether ``fit`` never increased.  With ``alpha > 0`` the local
            solves are regularized, not exact, and this can be ``False``.
        empty_slices: Number of (core, slice) pairs that no sample touched in
            the last sweep; those slices are undetermined by the data and keep
            whatever the initial guess had.
        ranks: TT ranks of the returned tensor.
        time: Total wall-clock seconds.

    The legacy code returned a plain dict, so ``info['fit']`` also works.
    """

    fit: list = field(default_factory=list)
    sweepTime: list = field(default_factory=list)
    initTime: float = 0.0
    converged: bool = False
    stop_reason: str = ""
    monotone: bool = True
    empty_slices: int = 0
    ranks: list = field(default_factory=list)
    time: float = 0.0

    def __getitem__(self, key):
        """Legacy dict access (``info['fit']``)."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __repr__(self):
        last = f"{self.fit[-1]:.3e}" if self.fit else "n/a"
        return (f"CompletionHistory(sweeps={len(self.fit)}, fit={last}, "
                f"stop={self.stop_reason!r}, monotone={self.monotone}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


def _unpack(cooP, shape=None):
    """Validate the COO dict and return ``(indices, values)`` as fresh arrays."""
    if not isinstance(cooP, dict) or "indices" not in cooP or "values" not in cooP:
        raise ValueError(
            "cooP must be a dict with keys 'indices' ((P, d) integer array) "
            f"and 'values' ((P,) array); got {type(cooP)!r} with "
            f"{sorted(cooP) if isinstance(cooP, dict) else '-'}")
    indices = np.array(cooP["indices"], dtype=np.int64, copy=True)
    values = np.array(cooP["values"], dtype=np.float64, copy=True).ravel()
    if indices.ndim != 2:
        raise ValueError(f"cooP['indices'] must be 2d (P, d), got {indices.shape}")
    if indices.shape[0] != values.size:
        raise ValueError(
            f"cooP has {indices.shape[0]} indices but {values.size} values")
    if values.size == 0:
        raise ValueError("cooP is empty: nothing to fit")
    if shape is not None:
        shape = np.asarray(shape, dtype=np.int64).ravel()
        if shape.size != indices.shape[1]:
            raise ValueError(
                f"shape has {shape.size} modes, the indices have {indices.shape[1]}")
        bad = np.flatnonzero((indices < 0).any(axis=0) | (indices >= shape).any(axis=0))
        if bad.size:
            k = int(bad[0])
            raise IndexError(
                f"sample index out of range in mode {k}: values in "
                f"[{indices[:, k].min()}, {indices[:, k].max()}], mode size {shape[k]}")
    return indices, values


def completion_functional(x, cooP):
    """``J(X) = 1/2 sum_p (X[i_p] - a_p)^2`` on the sampled entries.

    Args:
        x: A :class:`tt.vector`.
        cooP: ``{'indices': (P, d) int array, 'values': (P,) array}``.

    Returns:
        float: the value of the functional.
    """
    indices, values = _unpack(cooP, x.n)
    pred = np.asarray(bk.to_numpy(_idx.sample(list(x.cores), indices)),
                      dtype=np.float64)
    return 0.5 * float(np.sum((pred - values) ** 2))


def ttSparseALS(cooP, shape, x0=None, ttRank=1, tol=1e-5, maxnsweeps=20,
                verbose=True, alpha=1e-2, *, seed=None):
    """TT completion from sparse samples by alternating least squares.

    Args:
        cooP: ``{'indices': (P, d) integer array, 'values': (P,) array}``.  Not
            modified.
        shape: Mode sizes of the tensor to complete.  Ignored when ``x0`` is
            given (the shape of ``x0`` wins), but still validated against the
            indices.
        x0: Starting TT tensor.  ``None`` means a random tensor of rank
            ``ttRank``, normalized to 1.
        ttRank: TT rank of the random start; ignored when ``x0`` is given.
        tol: Stop when the relative functional drops below ``tol``, or when it
            moves by less than ``tol`` between two sweeps.
        maxnsweeps: Maximum number of sweeps (a sweep updates every core once,
            left to right).
        verbose: Print one line per sweep.  The history is recorded either way.
        alpha: Relative cut-off for the small singular values of each local
            least squares problem (the ``rcond`` of the solve).  ``alpha <= 0``
            means "machine precision", i.e. the exact minimum-norm least
            squares solution -- the only setting for which the functional is
            guaranteed to be monotonically non-increasing.  A positive value
            regularizes slices that carry fewer samples than unknowns, at the
            price of that guarantee.
        seed: Seed for the random start; pass one to make a run reproducible.

    Returns:
        tuple: ``(x, info)`` -- the completed :class:`tt.vector` and a
        :class:`CompletionHistory` (also indexable like the legacy dict:
        ``info['fit']``).

    Note:
        A small ``info.fit`` says the sampled entries are reproduced; it says
        nothing about the rest of the tensor.  With ``P`` samples and
        ``sum_k r_k n_k r_{k+1}`` parameters, only the ratio of those two
        numbers makes the result meaningful, and the honest check is the error
        on entries that were not in ``cooP``.

    Note:
        ALS on this functional is not globally convergent: the problem is not
        jointly convex, and from a random start the sweep can run into a
        stationary point far above the target.  Measured on a rank-2 tensor of
        shape ``8x8x8x8`` (96 parameters) recovered at rank 2 with
        ``alpha = 0``: with ~820 distinct samples 4 of 6 random starts reached
        ``fit ~ 1e-15`` and the other 2 stalled around ``1e-1``; with ~1330
        distinct samples 6 of 6 reached ``1e-15``.  The run *reports* which of
        the two happened -- ``info.converged`` and ``info.stop_reason`` -- so
        the remedy (more samples, another ``seed``, a better ``x0``) is a
        decision the caller can actually make.  A stalled run is never returned
        as if it were a solution.
    """
    t0 = time.perf_counter()
    hist = CompletionHistory()

    if x0 is None:
        indices, values = _unpack(cooP, shape)
        x = tools.rand(np.asarray(shape, dtype=np.int64).ravel(), r=ttRank,
                       samplefunc=None if seed is None
                       else np.random.default_rng(seed).standard_normal)
        x = x.round(0.0)
        x = (1.0 / x.norm()) * x
    else:
        if not isinstance(x0, vector):
            raise TypeError(f"x0 must be a tt.vector, got {type(x0)!r}")
        x = x0.copy()
        indices, values = _unpack(cooP, x.n)
    d = x.d
    if indices.shape[1] != d:
        raise ValueError(
            f"the samples have {indices.shape[1]} modes, the tensor has {d}")
    hist.initTime = time.perf_counter() - t0

    cores = [np.array(bk.to_numpy(c), dtype=np.float64, copy=True) for c in x.cores]
    if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        raise ValueError("ttSparseALS needs boundary ranks equal to 1")
    n = [c.shape[1] for c in cores]

    # Scale the data to unit norm so that `tol` compares against a relative
    # quantity; the tensor is scaled back on the way out.
    norm_p = float(np.linalg.norm(values))
    if norm_p == 0.0:
        raise ValueError("all sampled values are zero: the problem is degenerate")
    scaled = values / norm_p

    rcond = None if alpha is None or alpha <= 0 else float(alpha)
    if verbose:
        print(f"Initialization: {hist.initTime:.3f} s, {indices.shape[0]} samples, "
              f"{sum(c.size for c in cores)} parameters")

    # Slice membership never changes; compute it once instead of once per sweep.
    members = [[np.flatnonzero(indices[:, k] == i) for i in range(n[k])]
               for k in range(d)]

    for sweep in range(maxnsweeps):
        t_sweep = time.perf_counter()
        empty = 0

        # Right interfaces for the whole sweep: cores k+1..d-1 are untouched
        # until the sweep reaches them, so these stay valid while we move right.
        rgt = [None] * (d + 1)
        rgt[d] = np.ones((1, indices.shape[0]))
        for k in range(d - 1, -1, -1):
            rgt[k] = _idx.right_step(cores[k], indices[:, k], rgt[k + 1])
        lft = np.ones((indices.shape[0], 1))

        for k in range(d):
            r1, nk, r2 = cores[k].shape
            core = cores[k].copy()
            for i in range(nk):
                rows = members[k][i]
                if rows.size == 0:
                    empty += 1
                    continue
                # design matrix: row p is vec(L_p R_p^T) with the left rank
                # running fastest, matching the F-ordered slice below
                design = rearrange(
                    np.einsum("pa,bp->pab", lft[rows], rgt[k + 1][:, rows]),
                    "p a b -> p (b a)")
                sol = np.linalg.lstsq(design, scaled[rows], rcond=rcond)[0]
                core[:, i, :] = rearrange(sol, "(b a) -> a b", a=r1)
            cores[k] = core
            lft = _idx.left_step(lft, core, indices[:, k])

        pred = lft.reshape(-1)
        fit = 0.5 * float(np.sum((pred - scaled) ** 2))
        hist.fit.append(fit)
        hist.sweepTime.append(time.perf_counter() - t_sweep)
        hist.empty_slices = empty
        if len(hist.fit) > 1 and fit > hist.fit[-2] * (1.0 + 1e-12) + 1e-300:
            hist.monotone = False
        if verbose:
            print(f"sweep {sweep + 1}/{maxnsweeps}\t fit value: {fit:.5e}\t "
                  f"time: {hist.sweepTime[-1]:.3f} s")
        if fit < tol:
            hist.converged, hist.stop_reason = True, "tol"
            break
        if len(hist.fit) > 1 and abs(fit - hist.fit[-2]) < tol:
            # The sweep stopped moving without reaching `tol`.  That is a
            # stationary point of the ALS iteration, not a solution, and it is
            # reported as such: `converged` stays False.
            hist.stop_reason = "stalled"
            break
    else:
        hist.stop_reason = "maxnsweeps"

    x = vector.from_list([bk.asarray(c) for c in cores])
    x = norm_p * x
    hist.ranks = [int(r) for r in x.r]
    hist.time = time.perf_counter() - t0
    if verbose:
        last = f"{hist.fit[-1]:.5e}" if hist.fit else "n/a"
        print(f"Total: {hist.time:.3f} s, stop: {hist.stop_reason}, fit: {last}")
    return x, hist
