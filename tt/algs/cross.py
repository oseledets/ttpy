"""TT-cross: build a TT tensor from a black-box function of the indices.

The tensor ``A[i_1, ..., i_d] = fun(i_1, ..., i_d)`` is never formed.  Only
``O(d n r^2)`` entries are ever asked for, on *fibers* selected by (rectangular)
maximum-volume submatrices.  ``fun`` is called on a whole batch of multi-indices
at a time -- vectorized evaluation is the entire point of the method.

Algorithm
---------
Alternating (left-to-right, then right-to-left) one-site cross with rectangular
maxvol row selection.  At site ``k`` the supercore

    A_k[alpha, i, beta] = fun(I_k[alpha], i, J_k[beta])

is evaluated on the current left index set ``I_k`` (multi-indices of modes
``0..k-1``) and right index set ``J_k`` (modes ``k+1..d-1``), an orthogonal basis
``Q`` of its column (resp. row) space is computed, and
:func:`tt.algs.maxvol.rect_maxvol` picks ``rho + kickrank ...`` rows of that
basis.  Those rows extend the index set for the next site; the interpolation
matrix ``C = Q pinv(Q[ind])`` becomes the new core.  The basis is deliberately
*not* truncated at the local accuracy -- see :func:`_left_basis` for what that
costs.  Ranks are controlled by ``rmax`` and by the final rounding to ``eps``.
The result is the classical cross interpolant

    X = A(:, J_0) A(I_1, J_0)^{-1} A(I_1, :, J_1) ... A(I_{d-1}, :)

which is *exact* whenever the tensor has TT ranks bounded by the index set sizes
and the sets are unisolvent.

References
----------
* I. V. Oseledets, E. E. Tyrtyshnikov, "TT-cross approximation for
  multidimensional arrays", Linear Algebra Appl. 432(1):70-88, 2010.
  https://doi.org/10.1016/j.laa.2009.07.024
* S. Dolgov, D. Savostyanov, "Alternating minimal energy methods for linear
  systems in higher dimensions", SIAM J. Sci. Comput. 36(5), 2014,
  arXiv:1301.6068 (greedy rank adaptation used here as ``kickrank``).
* A. Mikhalev, I. V. Oseledets, "Rectangular maximum-volume submatrices and
  their applications", Linear Algebra Appl. 538:187-211, 2018, arXiv:1502.07838.

Notes
-----
Index bookkeeping (the multi-index sets) is plain numpy ``int64``: it is integer
combinatorics, not linear algebra, and it never touches a GPU.  Everything that
*is* linear algebra goes through :mod:`tt.backend`, so the cores come out on the
backend of ``x0``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from einops import einsum, rearrange

from .. import backend as bk
from ..core import _ops
from ..core.vector import vector
from .maxvol import maxvol, rect_maxvol

__all__ = ["rect_cross", "cross", "greedy_cross", "CrossHistory", "element"]

_EMPTY = np.empty((1, 0), dtype=np.int64)


# --- history -----------------------------------------------------------------

@dataclass
class CrossHistory:
    """Everything the run knows about itself (R7: verbose=False still records).

    None of the error numbers is a bound.  Read them together:

    * ``err_rel`` is the relative change between the last two sweeps, the
      classical cross indicator.  It tracks the true error to about an order of
      magnitude while the ranks are still free to grow (measured on the QTT
      Coulomb kernel of the tests: 4.7e-4 reported against 2.0e-3 true), and it
      collapses to machine precision as soon as they are *not* free -- when
      ``rmax`` binds or ``kickrank`` is zero.  Both of those are flagged.
    * ``err_round`` is the exact relative error added by the final rounding of
      the interpolant to ``eps``.  It is a measurement, not an estimate.
    * ``err_check`` is a Monte Carlo measurement on points nobody looked at,
      the only number computed against ``fun`` itself.  When it is requested and
      it exceeds the requested accuracy, the run warns -- a measurement that
      contradicts ``eps`` is the strongest evidence available and staying quiet
      about it would make the whole check pointless.  It is still a random
      sample: a feature carried by a few entries (a spike) is invisible to all
      three numbers, and no amount of sampling changes that.

    Attributes:
        eps: Requested relative accuracy.
        sweeps: One dict per sweep with keys ``sweep, err_rel, err_abs, erank,
            max_rank, fun_eval, time`` (``fun_eval`` is the running total).
        fun_eval: Number of function values requested by the cross itself.
        fun_eval_check: Extra values spent on the held-out accuracy check.
        converged: The *stopping criterion* fired, i.e. the change between the
            last two sweeps fell below the threshold.  Not a certificate of
            accuracy -- see above and ``rmax_active``.
        err_rel: Relative change between the last two sweeps.
        err_round: Relative error added by the final ``round(eps)``.
        err_check: Relative error on ``n_check`` random held-out points, or
            ``None`` if it was not requested.
        ranks: TT ranks of the returned tensor.
        rmax_active: The returned ranks saturate the requested ``rmax``, so the
            approximation was rank-limited rather than accuracy-limited.
        time: Wall clock seconds.
    """

    eps: float
    sweeps: list = field(default_factory=list)
    fun_eval: int = 0
    fun_eval_check: int = 0
    converged: bool = False
    err_rel: float = float("nan")
    err_round: float = float("nan")
    err_check: float | None = None
    ranks: list = field(default_factory=list)
    rmax_active: bool = False
    time: float = 0.0

    def __repr__(self):
        chk = "n/a" if self.err_check is None else f"{self.err_check:.2e}"
        return (f"CrossHistory(sweeps={len(self.sweeps)}, "
                f"converged={self.converged}, err_rel={self.err_rel:.2e}, "
                f"err_round={self.err_round:.2e}, err_check={chk}, "
                f"fun_eval={self.fun_eval}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"rmax_active={self.rmax_active}, time={self.time:.2f}s)")


# --- index bookkeeping -------------------------------------------------------

def _merge(left, n, right):
    """All multi-indices ``(left[a], i, right[b])`` in C order of ``(a, i, b)``.

    The row order is ``row = (a * n + i) * r2 + b``, i.e. the *right* index runs
    fastest.  A batch of values returned by ``fun`` for these rows therefore
    reshapes directly to ``(r1, n, r2)`` with a plain C-order reshape.
    """
    r1, kl = left.shape
    r2, kr = right.shape
    out = np.empty((r1 * n * r2, kl + 1 + kr), dtype=np.int64)
    out[:, :kl] = np.repeat(left, n * r2, axis=0)
    out[:, kl] = np.tile(np.repeat(np.arange(n, dtype=np.int64), r2), r1)
    out[:, kl + 1:] = np.tile(right, (r1 * n, 1))
    return out


class _Counter:
    def __init__(self):
        self.n = 0


def _evaluate(fun, idx, counter):
    """Call ``fun`` on a batch of multi-indices; fail loudly on garbage."""
    vals = np.asarray(fun(idx))
    # A 0-d return is rejected even when the batch happens to hold one index:
    # otherwise a scalar-valued (non-vectorized) ``fun`` slips through on grids
    # with a mode of size one and the whole run is built on a constant.
    if vals.ndim == 0 or vals.size != idx.shape[0]:
        raise ValueError(
            f"fun returned {vals.size} values (shape {vals.shape}) for a batch "
            f"of {idx.shape[0]} multi-indices; it must be vectorized over the "
            "first axis and return one value per row")
    vals = vals.reshape(-1)
    if not np.all(np.isfinite(vals)):
        bad = int(np.flatnonzero(~np.isfinite(vals))[0])
        raise ValueError(
            f"fun returned a non-finite value {vals[bad]!r} at multi-index "
            f"{idx[bad].tolist()}; cross cannot interpolate that")
    counter.n += idx.shape[0]
    return vals


# --- linear algebra micro-steps ----------------------------------------------

def _to_backend(vals, opts):
    """Move a numpy batch of values onto the backend and dtype of ``x0``.

    The *width* (32/64 bit) is the one the caller chose with ``x0``; the value
    is promoted to complex if ``fun`` produced complex numbers, because
    discarding an imaginary part would be a wrong answer, not a conversion.
    """
    dtype = opts["dtype"]
    if np.iscomplexobj(vals):
        dtype = bk.complex_dtype(dtype)
    return bk.asarray(vals, dtype=dtype, backend=opts["backend"])


def _left_basis(mat, rmax):
    """Orthonormal basis of the column space of ``mat``, capped at ``rmax``.

    The basis is deliberately *not* truncated at the local accuracy.  Doing so
    (an obvious-looking economy, and the one the commented-out lines of the
    reference implementation warn about) makes the index sets shrink: the size
    of the next index set is the numerical rank of the current sampled block
    plus ``kickrank``, so a block that happens to be rank deficient -- which is
    the normal case, e.g. every ``f(i_1 + ... + i_d)`` has repeated fibers --
    resets the set to a smaller size than it had.  Left and right sets then cap
    each other and the ranks lock at a fixed point far above ``eps``: measured
    on ``1/(1 + i_1 + ... + i_4)``, ``n = 8``, ``eps = 1e-10``, that fixed point
    is a 12% relative error reported as converged.

    Rank control is therefore not done here.  It is done by ``rmax`` and by the
    final :meth:`round` at ``eps``, which is the single owner of "how many ranks
    does this accuracy need".
    """
    q, _s, _ = bk.svd(mat)
    rho = max(1, min(q.shape[1], rmax))
    return q[:, :rho]


def _select_rows(q, kickrank, rf, rmax, tau):
    """Rows of ``q`` with (nearly) maximal 2-volume, at least ``rho+kickrank``.

    ``maxvol``/``rect_maxvol`` are numpy-only index-selection routines owned by
    :mod:`tt.algs.maxvol`; the basis is handed to them as numpy and only the
    pivot list comes back, so the caller stays backend agnostic.
    """
    qn = np.asarray(bk.to_numpy(q))
    npts, rho = qn.shape
    if npts <= rho:
        return np.arange(npts, dtype=np.int64)
    max_k = int(min(npts, rho + kickrank + rf, max(rmax, rho)))
    add_k = int(max(0, min(kickrank, max_k - rho)))
    if add_k == 0:
        ind = maxvol(qn)[0]
    else:
        ind = rect_maxvol(qn, tau, maxK=max_k, min_add_K=add_k)[0]
    ind = np.asarray(ind, dtype=np.int64).reshape(-1)
    if ind.size < rho or ind.min() < 0 or ind.max() >= npts:
        raise ValueError(
            f"maxvol returned {ind.size} pivots in [{ind.min()}, {ind.max()}] "
            f"for a {npts}x{rho} basis; expected >= {rho} pivots in range")
    return ind


def _interp(q, ind):
    """``C = Q pinv(Q[ind])``: the least-squares interpolation matrix.

    ``C @ Q[ind] == Q`` exactly (``Q[ind]`` has full column rank after maxvol),
    so the cross interpolant reproduces the function on the selected fibers.
    Normal equations are used on purpose: ``bk.solve`` raises on a singular
    ``Q[ind]`` instead of quietly returning a minimum-norm guess, and it works
    on every backend (``lstsq`` drivers are not uniformly available on GPU).
    ``Q[ind]`` is well conditioned by construction of the maximum-volume
    submatrix, so squaring its condition number is harmless here.
    """
    qi = q[ind]
    qih = rearrange(qi.conj(), "i j -> j i")
    return q @ bk.solve(qih @ qi, qih)


# --- initial index sets ------------------------------------------------------

def _init_right_indices(cores):
    """Right index sets ``J[0..d-2]`` from the initial guess, by square maxvol."""
    d = len(cores)
    out = _ops.orthogonalize(cores, center=0)
    jset = [None] * d
    jset[d - 1] = _EMPTY
    carry = None
    for k in range(d - 1, 0, -1):
        c = out[k]
        if carry is not None:  # the right index of this core was re-selected
            c = einsum(c, carry, "a n b, b c -> a n c")
        r0, nk, _r1 = c.shape
        q, rmat = bk.qr(rearrange(c, "a n b -> (n b) a"))
        ind = _select_rows(q, 0, 0, r0, 1.05)
        jset[k - 1] = _merge(_EMPTY, nk, jset[k])[ind]
        # cores to the left must absorb the row selection so that the ranks
        # stay consistent with the index sets we just built.
        carry = rearrange(q[ind] @ rmat, "i j -> j i")
    return jset


# --- sweeps ------------------------------------------------------------------

def _sweep_lr(fun, iset, jset, n, opts, counter):
    """Left-to-right: refine the left index sets ``I[1..d-1]``."""
    d = len(n)
    for k in range(d - 1):
        r1, r2 = iset[k].shape[0], jset[k].shape[0]
        idx = _merge(iset[k], n[k], jset[k])
        vals = _evaluate(fun, idx, counter)
        sup = _to_backend(vals.reshape((r1 * n[k], r2)), opts)
        q = _left_basis(sup, opts["rmax"])
        ind = _select_rows(q, opts["kickrank"], opts["rf"], opts["rmax"],
                           opts["tau"])
        iset[k + 1] = _merge(iset[k], n[k], _EMPTY)[ind]


def _sweep_rl(fun, iset, jset, n, opts, counter):
    """Right-to-left: refine ``J[0..d-2]`` and build the cores of the answer."""
    d = len(n)
    cores = [None] * d
    for k in range(d - 1, 0, -1):
        r1, r2 = iset[k].shape[0], jset[k].shape[0]
        idx = _merge(iset[k], n[k], jset[k])
        vals = _evaluate(fun, idx, counter)
        sup = _to_backend(vals.reshape((r1, n[k] * r2)), opts)
        q = _left_basis(rearrange(sup, "a s -> s a"), opts["rmax"])
        ind = _select_rows(q, opts["kickrank"], opts["rf"], opts["rmax"],
                           opts["tau"])
        cmat = _interp(q, ind)
        cores[k] = rearrange(cmat, "(i b) g -> g i b", i=n[k], b=r2)
        jset[k - 1] = _merge(_EMPTY, n[k], jset[k])[ind]
    idx = _merge(iset[0], n[0], jset[0])
    vals = _evaluate(fun, idx, counter)
    cores[0] = _to_backend(vals.reshape((1, n[0], jset[0].shape[0])), opts)
    return cores


# --- evaluation of a TT tensor at scattered indices --------------------------

def element(x, idx):
    """Values of a TT tensor at a batch of multi-indices.

    Args:
        x: ``tt.vector`` (or a list of cores) with boundary ranks one.
        idx: integer array of shape ``(m, d)``.

    Returns:
        Backend array of ``m`` values.
    """
    cores = x.cores if isinstance(x, vector) else list(x)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != len(cores):
        raise ValueError(f"idx has shape {idx.shape}, expected (m, {len(cores)})")
    if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        raise ValueError("element() needs boundary ranks equal to one")
    # A negative index is not an index into a tensor, it is a numpy slicing
    # convention; letting it through would silently return the value at the
    # opposite end of the mode instead of failing.
    if idx.size and idx.min() < 0:
        bad = int(np.unravel_index(int(np.argmin(idx)), idx.shape)[0])
        raise ValueError(
            f"idx contains a negative entry at row {bad}: {idx[bad].tolist()}; "
            "indices must be in [0, n_k)")
    p = cores[0][0][idx[:, 0], :]
    for k in range(1, len(cores)):
        g = cores[k][:, idx[:, k], :]
        p = einsum(p, g, "m a, a m b -> m b")
    return p.reshape((-1,))


def _held_out_error(fun, x, n, n_check, rng, counter):
    """Relative error on random points that the cross never asked for."""
    idx = np.stack([rng.integers(0, nk, size=n_check) for nk in n], axis=1)
    exact = _evaluate(fun, idx, counter)
    approx = np.asarray(bk.to_numpy(element(x, idx))).reshape(-1)
    den = float(np.linalg.norm(exact))
    num = float(np.linalg.norm(approx - exact))
    return num / den if den > 0 else num


# --- the method --------------------------------------------------------------

def rect_cross(fun, x0, eps=1e-6, nswp=20, kickrank=1, rf=2, verbose=False,
               eps_abs=0.0, rmax=None, tau=1.1, n_check=0, check_seed=0,
               stop_fun=None, round_result=True):
    """Cross approximation of a black-box tensor, rectangular-maxvol flavour.

    Args:
        fun: Vectorized black box.  Takes an integer array of shape
            ``(batch, d)`` and returns ``batch`` values.  It must be a genuine
            function of the index (same index -> same value).
        x0: ``tt.vector`` used only for its mode sizes, its starting ranks and
            its backend/dtype; a random rank-2 guess is a fine choice.  The
            result comes back on that backend with that floating point width
            (a ``float32`` ``x0`` gives ``float32`` cores), promoted to complex
            if ``fun`` returns complex values.
        eps: Target relative accuracy.  Drives both the local truncation
            (``eps/sqrt(d)`` per core) and the stopping criterion.
        nswp: Maximum number of sweeps (one sweep = left-to-right and back).
        kickrank: Minimum number of extra rows the rectangular maxvol adds on
            top of the numerical rank at every micro-step.  This is the
            exploration budget *and* the only error detector the method has:
            ``kickrank=0`` makes it a plain fixed-rank cross which cannot see
            the part of the tensor its index sets do not span, and is therefore
            reported with a warning.
        rf: Additional slack for the rank growth: a micro-step may go up to
            ``rho + kickrank + rf`` rows.  (The legacy docstring calls it a
            growth *factor*, the legacy code uses it additively; we follow the
            code.)
        verbose: Print a line per sweep.  Independent of the recorded history.
        eps_abs: Absolute stopping threshold; the sweep stops when the change is
            below ``max(eps * ||x||, eps_abs)``.
        rmax: Hard cap on the TT ranks.  ``None`` means no cap.  If the answer
            saturates the cap, ``eps`` was not the binding constraint: the run
            warns and sets ``history.rmax_active``.
        tau: Rectangular maxvol tolerance (upper bound on the row norms of the
            coefficient matrix).
        n_check: If > 0, measure the true relative error on that many uniformly
            random held-out points after the last sweep.  Costs that many extra
            function evaluations, counted separately.
        check_seed: Seed of the held-out sample, for reproducibility.
        stop_fun: ``stop_fun(x_prev, x_new) -> bool`` replacing the default
            stopping criterion.
        round_result: Round the answer to ``eps`` at the end.  The exploration
            rows inflate the ranks by ``kickrank`` per edge; rounding removes
            them at an error already covered by ``eps``.

    Returns:
        ``tt.vector`` with an extra attribute ``history`` (:class:`CrossHistory`)
        holding the number of function evaluations, the per-sweep residuals and
        the convergence flag.

    Raises:
        ValueError: if ``fun`` is not vectorized or returns non-finite values.

    Warns:
        RuntimeWarning: if the sweeps ran out before the stopping criterion was
            met, if the ranks saturate ``rmax``, or if ``kickrank`` is zero.
            The tensor is still returned -- with the achieved accuracy in
            ``history`` -- because a wrong-but-quiet answer is the one thing we
            must not produce.

    Note:
        No cross method can certify its own accuracy: it only ever sees
        ``O(d n r^2)`` entries.  A tensor that is zero everywhere except on a
        few entries (a spike, a delta) is returned as *zero*, with
        ``converged=True``, ``err_rel=0`` and ``err_check=0``, because none of
        those numbers -- nor any number computed from samples -- can see an
        entry that was never sampled.  Cross is for functions with a decaying
        singular value spectrum; ``eps`` is a request, not a guarantee.
    """
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun)!r}")
    if not isinstance(x0, vector):
        raise TypeError(f"x0 must be a tt.vector, got {type(x0)!r}")
    if int(nswp) < 1:
        raise ValueError(f"nswp must be at least 1, got {nswp}")
    # ``rmax=0`` used to be swallowed by a truthiness test and silently meant
    # "no cap at all" -- the opposite of what anyone typing it wants.
    if rmax is not None and int(rmax) < 1:
        raise ValueError(f"rmax must be at least 1 or None, got {rmax}")
    t0 = time.time()
    n = [int(v) for v in x0.n]
    d = len(n)
    counter = _Counter()
    hist = CrossHistory(eps=float(eps))
    opts = {"kickrank": int(kickrank), "rf": int(rf), "tau": float(tau),
            "rmax": int(rmax) if rmax is not None else 10 ** 9,
            "backend": bk.backend_of(x0.cores[0]),
            "dtype": bk.dtype_of(x0.cores[0])}

    if d == 1:
        # Nothing to sweep over: the whole tensor is one fiber, so the "sweep"
        # is exact.  It still goes through the common tail below, otherwise the
        # history of a d=1 run would be a different object than every other run
        # (no sweep entry, ``n_check`` silently ignored).
        t_swp = time.time()
        idx = np.arange(n[0], dtype=np.int64).reshape((-1, 1))
        vals = _evaluate(fun, idx, counter)
        y = vector.from_list([_to_backend(vals.reshape((1, n[0], 1)), opts)])
        hist.converged, hist.err_rel = True, 0.0
        hist.sweeps.append({
            "sweep": 0, "err_rel": 0.0, "err_abs": 0.0, "erank": 1.0,
            "max_rank": 1, "fun_eval": counter.n,
            "time": time.time() - t_swp})
    else:
        iset = [_EMPTY] * d
        jset = _init_right_indices(x0.cores)
        xprev = x0
        y = None
        for swp in range(int(nswp)):
            t_swp = time.time()
            _sweep_lr(fun, iset, jset, n, opts, counter)
            y = vector.from_list(_sweep_rl(fun, iset, jset, n, opts, counter))
            nrm = y.norm()
            err_abs = (y - xprev).norm()
            err_rel = err_abs / nrm if nrm > 0 else err_abs
            hist.sweeps.append({
                "sweep": swp, "err_rel": float(err_rel),
                "err_abs": float(err_abs), "erank": float(y.erank),
                "max_rank": int(max(y.r)), "fun_eval": counter.n,
                "time": time.time() - t_swp})
            hist.err_rel = float(err_rel)
            if verbose:
                print(f"cross: swp {swp + 1}/{nswp} err_rel = {err_rel:.3e} "
                      f"erank = {y.erank:.1f} max_rank = {max(y.r)} "
                      f"fun_eval = {counter.n}")
            if stop_fun is not None:
                hist.converged = bool(stop_fun(xprev, y))
            else:
                hist.converged = bool(err_abs <= max(eps * nrm, eps_abs))
            if hist.converged:
                break
            xprev = y

    if round_result:
        yr = y.round(eps)
        nrm = y.norm()
        hist.err_round = float((yr - y).norm() / nrm) if nrm > 0 else 0.0
        y = yr
    else:
        hist.err_round = 0.0
    hist.ranks = [int(v) for v in y.r]
    hist.fun_eval = counter.n
    hist.rmax_active = (rmax is not None and d > 1
                        and max(hist.ranks) >= int(rmax))
    if n_check > 0:
        check = _Counter()
        hist.err_check = _held_out_error(
            fun, y, n, int(n_check), np.random.default_rng(check_seed), check)
        hist.fun_eval_check = check.n
    hist.time = time.time() - t0
    y.history = hist

    # Every reason the answer might not deserve its eps gets its own clause.
    # In particular a *converged* run whose ranks sit on the cap, or one with
    # the rank adaptation switched off, is exactly the case where the change
    # between sweeps is ~1e-16 and the true error is 1e-1: silence there would
    # be the worst possible output.
    notes = []
    if not hist.converged:
        notes.append(f"{nswp} sweeps did not reach eps={eps:.1e}, last "
                     f"relative change {hist.err_rel:.3e}")
    if hist.rmax_active:
        notes.append(f"the rank cap rmax={int(rmax)} is active, so eps="
                     f"{eps:.1e} is not certified")
    if opts["kickrank"] < 1:
        notes.append("kickrank=0 switches off the rank adaptation, so the "
                     "change between sweeps measures stagnation at a fixed "
                     "rank and says nothing about the error")
    if notes:
        msg = ("tt cross: " + "; ".join(notes)
               + f"; ranks {hist.ranks}, rounding error "
                 f"{hist.err_round:.3e}, {hist.fun_eval} function evaluations")
        if hist.err_check is not None:
            msg += (f"; measured relative error on {n_check} held-out points "
                    f"{hist.err_check:.3e}")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return y


def cross(fun, n=None, d=None, eps=1e-6, r=2, seed=0, **kwargs):
    """Cross approximation from mode sizes only (random rank-``r`` start).

    Args:
        fun: Vectorized black box, see :func:`rect_cross`.
        n: Mode sizes -- a list, or a single int together with ``d``.
        d: Number of modes when ``n`` is a scalar.
        eps: Target relative accuracy.
        r: Ranks of the random initial guess.
        seed: Seed of that guess, so a run is reproducible by default.
        **kwargs: Forwarded to :func:`rect_cross` (``nswp``, ``kickrank``,
            ``rmax``, ``n_check``, ``verbose``, ...).

    Returns:
        ``tt.vector`` with the ``history`` attribute of :func:`rect_cross`.
    """
    if not callable(fun) and callable(n):
        fun, n = n, fun  # legacy entry point tt.cross.cross(n, f)
    from ..core.tools import rand
    rng = np.random.default_rng(seed)
    x0 = rand(n, d, r, samplefunc=lambda size: rng.standard_normal(size))
    return rect_cross(fun, x0, eps=eps, **kwargs)


#: The rank adaptation here *is* the greedy one (rectangular maxvol adds the
#: rows with the largest interpolation residual), so ``tt.greedy_cross`` is the
#: same routine under the name used in the AMEn/greedy-cross literature.
greedy_cross = rect_cross
