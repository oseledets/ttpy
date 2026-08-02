"""Elementwise functions of several TT tensors, by cross approximation.

Given TT tensors ``X_1, ..., X_p`` with identical mode sizes and a vectorized
callable ``funs``, this module builds a TT approximation of

    Y[i_1, ..., i_d] = funs( X_1[i_1..i_d], ..., X_p[i_1..i_d] )

without ever forming ``Y`` densely.  ``funs`` may return several components per
point; then the last TT rank of the answer is that number of components
(the legacy ``d2`` convention, a "block" TT vector).

Why this is a *cross* problem
-----------------------------
``Y`` is a black-box tensor: any single entry costs ``p`` TT contractions, and
nothing else about it is known -- in particular no arithmetic on the cores of
the ``X_j`` can produce it (``exp``, ``1/x``, ``sqrt`` are not polynomial).  The
only tool that applies is a sampling method that asks for ``O(d n r^2)`` entries
on adaptively chosen fibers.  That is exactly :func:`tt.algs.cross.rect_cross`.

Design decision (one owner per truth)
-------------------------------------
This module contains **no TT sweep of its own**.  It is a thin adapter:

    black box  <-  funs(  [ element(X_j, idx) for j ]  )
    engine     <-  tt.algs.cross.rect_cross

:func:`tt.algs.cross.element` evaluates a TT tensor at a batch of multi-indices,
which is precisely the interface ``rect_cross`` demands, so the fit is exact and
the whole method is ~200 lines of bookkeeping instead of a second, independently
buggy DMRG sweep.  The legacy ``multifuncrs2`` fused the two steps (it carried
interface matrices ``Rx`` for every input so that a superblock of ``X_j`` was
assembled by two ``gemm``s instead of ``d`` of them).  That saves a factor of
``d`` in the sampling of the inputs and is the only thing lost here; the number
of *user function* evaluations -- the expensive part in practice -- is the same.

Differences from the legacy Fortran-era code, stated plainly
------------------------------------------------------------
* rank enrichment is rectangular maxvol (Mikhalev-Oseledets) rather than the
  AMEn residual kick, so ``kickrank`` means "extra rows above the numerical rank
  at every micro-step", not "rank of a separately tracked residual tensor".
  ``kickrank2`` (extra *random* rows) has no counterpart and is rejected.
* the basis is always orthogonalized by an SVD, so ``do_qr`` is a no-op; it is
  accepted for signature compatibility and recorded in ``history``.
* ``pcatype='uchol'`` (incomplete Cholesky enrichment) is not implemented and
  raises instead of silently doing something else.
* a zero result is never returned quietly: it is either verified against fresh
  random samples or reported.

References
----------
* I. V. Oseledets, E. E. Tyrtyshnikov, "TT-cross approximation for
  multidimensional arrays", Linear Algebra Appl. 432(1):70-88, 2010.
* S. V. Dolgov, D. V. Savostyanov, "Alternating minimal energy methods for
  linear systems in higher dimensions", SIAM J. Sci. Comput. 36(5):A2248-A2271,
  2014, arXiv:1301.6068 -- ``amen_cross`` / ``multifuncrs2`` are from this line
  of work.
* A. Mikhalev, I. V. Oseledets, "Rectangular maximum-volume submatrices and
  their applications", Linear Algebra Appl. 538:187-211, 2018, arXiv:1502.07838.

What ``eps`` actually buys (measured, not promised)
---------------------------------------------------
``eps`` enters in exactly two places inside :func:`tt.algs.cross.rect_cross`,
and neither is an error *bound*: it is the threshold of the stopping rule (the
relative change between two sweeps) and the accuracy of the final rounding of
the interpolant.  The sweeps themselves do *not* truncate locally -- see
``tt.algs.cross._left_basis`` for what that costs -- so how close the achieved
error lands to ``eps`` is a property of the cross engine, not of this adapter.
Measured on ``1/(1+t)``, ``t = (i+1)/2^d`` on a binary QTT grid, relative error
on 2000 held-out points (``n_check``), float64, default ``kickrank=5``, as
achieved divided by requested:

    d        eps=1e-6      eps=1e-10
    10          0.43          0.27
    20          0.19          0.35
    40          0.19          0.11

All six runs reported ``history.converged is True`` and warned about nothing --
correctly, the stopping criterion *was* met.  That the ratio stays below one is
a measurement on one smooth function, not a promise: for a function whose
relevant fibers the sampling never visits (a spike on a few entries) every one
of those numbers would be optimistic, and so would the stopping rule.  ``eps``
is a knob; ``history.err_check`` (i.e. passing ``n_check``) is the only honest
measurement of the error actually obtained.

Notes
-----
The samples handed to ``funs`` are plain numpy arrays of shape
``(batch, len(X))``, whatever backend the TT tensors live on: ``funs`` is user
code written against numpy (that is the legacy contract, and ``numpy`` is the
only array API every user function can be assumed to speak).  Everything else --
the cross sweeps, the cores of the answer -- stays on the backend of ``X[0]``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from ..core.vector import vector
from .cross import CrossHistory, element, rect_cross

__all__ = ["multifuncrs", "multifuncrs2", "MultifuncrsHistory"]

#: Random points used to decide whether an exactly-zero answer is the truth or
#: a symptom of degenerate starting indices.
_ZERO_PROBE = 64


# --- history -----------------------------------------------------------------

@dataclass
class MultifuncrsHistory:
    """What the run knows about itself; filled in even when ``verb=0``.

    Attributes:
        eps: Requested relative accuracy.
        d2: Number of components returned by ``funs`` (last TT rank of ``y``).
        attempts: How many restarts were spent (1 = no restart needed).
        funs_calls: Number of calls to the user's ``funs``.
        funs_values: Number of points ``funs`` was asked about, probes included.
        probe_values: Points spent on the shape/dtype probe alone.
        cross: The underlying :class:`tt.algs.cross.CrossHistory` -- per-sweep
            relative change, ranks, timing, convergence flag.  ``sweeps``,
            ``converged``, ``err_rel``, ``err_check``, ``err_round`` and
            ``rmax_active`` are forwarded from it as properties.
        ranks: TT ranks of the returned tensor (last one is ``d2``).
        time: Wall clock seconds for the whole call.
        ignored_options: Legacy options accepted but without effect here.
    """

    eps: float
    d2: int = 1
    attempts: int = 1
    funs_calls: int = 0
    funs_values: int = 0
    probe_values: int = 0
    cross: CrossHistory | None = None
    ranks: list = field(default_factory=list)
    time: float = 0.0
    ignored_options: list = field(default_factory=list)

    @property
    def sweeps(self):
        """Per-sweep records of the cross engine."""
        return [] if self.cross is None else self.cross.sweeps

    @property
    def converged(self):
        """Whether the sweeps met the stopping criterion before ``nswp``."""
        return False if self.cross is None else self.cross.converged

    @property
    def err_rel(self):
        """Relative change between the last two sweeps (an estimate, not a bound)."""
        return float("nan") if self.cross is None else self.cross.err_rel

    @property
    def err_check(self):
        """Measured relative error on held-out points, or ``None``."""
        return None if self.cross is None else self.cross.err_check

    @property
    def rmax_active(self):
        """Whether the answer sits on the rank cap, i.e. ``eps`` is not certified.

        ``converged`` is ``True`` in exactly that case as well -- the sweeps
        stop moving because they *cannot* move -- so a caller who tests only
        ``converged`` would read a capped, inaccurate answer as a good one.  The
        run also warns; this is the same fact, reachable programmatically.
        """
        return False if self.cross is None else self.cross.rmax_active

    @property
    def err_round(self):
        """Relative error added by the final rounding to ``eps`` (a measurement)."""
        return float("nan") if self.cross is None else self.cross.err_round

    def __repr__(self):
        return (f"MultifuncrsHistory(sweeps={len(self.sweeps)}, "
                f"converged={self.converged}, err_rel={self.err_rel:.2e}, "
                f"d2={self.d2}, funs_calls={self.funs_calls}, "
                f"funs_values={self.funs_values}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"rmax_active={self.rmax_active}, time={self.time:.2f}s)")


# --- input handling ----------------------------------------------------------

def _check_inputs(X):
    """Normalize ``X`` to a list of TT vectors with identical mode sizes."""
    if isinstance(X, vector):
        raise TypeError("X must be a list of tt.vector, not a single tt.vector; "
                        "wrap it: multifuncrs2([x], funs, ...)")
    xs = list(X)
    if not xs:
        raise ValueError("X is empty: there is nothing to take a function of")
    for j, x in enumerate(xs):
        if not isinstance(x, vector):
            raise TypeError(f"X[{j}] is {type(x)!r}, expected a tt.vector")
        if x.cores[0].shape[0] != 1 or x.cores[-1].shape[2] != 1:
            raise ValueError(
                f"X[{j}] has boundary ranks {x.r[0]}, {x.r[-1]}; multifuncrs "
                "needs ordinary TT vectors (boundary ranks equal to one)")
    n0 = [int(v) for v in xs[0].n]
    for j, x in enumerate(xs[1:], start=1):
        nj = [int(v) for v in x.n]
        if nj != n0:
            raise ValueError(
                f"X[{j}] has mode sizes {nj}, X[0] has {n0}; multifuncrs is "
                "elementwise and needs identical grids")
    return xs, n0


def _sample(xs, idx):
    """Values of every input at a batch of multi-indices: ``(batch, len(xs))``."""
    cols = [np.asarray(bk.to_numpy(element(x, idx))).reshape(-1) for x in xs]
    return np.stack(cols, axis=1)


def _as_values(vals, batch, d2):
    """Check the shape of what ``funs`` returned and cast it to ``(batch, d2)``."""
    vals = np.asarray(vals)
    if vals.ndim == 1:
        got = 1
    elif vals.ndim == 2:
        got = vals.shape[1]
    else:
        raise ValueError(
            f"funs returned an array of shape {vals.shape}; expected "
            f"(batch,) or (batch, d2)")
    if vals.shape[0] != batch or got != d2:
        raise ValueError(
            f"funs returned shape {vals.shape} for a batch of {batch} points "
            f"and d2={d2}; expected ({batch}, {d2}) or ({batch},) when d2=1")
    return vals.reshape((batch, d2))


class _Meter:
    """Counts what the user's function was actually asked to compute."""

    def __init__(self):
        self.calls = 0
        self.values = 0

    def note(self, m):
        self.calls += 1
        self.values += int(m)


def _probe(xs, funs, n, d2, seed, meter):
    """One cheap call to ``funs`` to learn its component count and dtype.

    The number of components cannot be guessed from the signature and must not
    be assumed: getting it wrong silently reshapes the answer.  So ask.
    """
    rng = np.random.default_rng(seed)
    m = int(min(8, np.prod(np.asarray(n, dtype=np.int64))))
    idx = np.stack([rng.integers(0, nk, size=m) for nk in n], axis=1)
    vals = np.asarray(funs(_sample(xs, idx)))
    meter.note(m)
    if vals.ndim == 1:
        got = 1
    elif vals.ndim == 2:
        got = int(vals.shape[1])
    else:
        raise ValueError(
            f"funs returned an array of shape {vals.shape}; it must map "
            "(batch, len(X)) to (batch,) or (batch, d2)")
    if vals.shape[0] != m:
        raise ValueError(
            f"funs returned {vals.shape[0]} rows for a batch of {m} points; "
            "it must be vectorized over the first axis")
    if d2 is not None and int(d2) != got:
        raise ValueError(
            f"d2={int(d2)} was requested but funs returns {got} components")
    if not np.all(np.isfinite(vals)):
        raise ValueError(
            "funs returned a non-finite value on the initial probe; cross "
            "cannot interpolate that (is the domain right, e.g. 1/x near 0?)")
    complex_out = bool(np.iscomplexobj(vals)) or any(x.is_complex for x in xs)
    dtype = "complex128" if complex_out else bk.canon_dtype(
        np.result_type(vals.dtype, np.float64))
    return got, dtype


def _make_blackbox(xs, funs, d2, d, meter):
    """The scalar black box that :func:`rect_cross` samples.

    For ``d2 > 1`` the components live on an extra, artificial mode of size
    ``d2`` appended to the grid, so the engine sees an ordinary ``(d+1)``-
    dimensional tensor.  Rows that share a spatial index are de-duplicated
    before ``funs`` is called, so the user's function is still evaluated once
    per point, not once per (point, component).
    """
    if d2 == 1:
        def fun(idx):
            idx = np.asarray(idx, dtype=np.int64)
            vals = _as_values(funs(_sample(xs, idx)), idx.shape[0], 1)
            meter.note(idx.shape[0])
            return vals[:, 0]
        return fun

    def fun(idx):
        idx = np.asarray(idx, dtype=np.int64)
        spatial, comp = idx[:, :d], idx[:, d]
        uniq, inv = np.unique(spatial, axis=0, return_inverse=True)
        vals = _as_values(funs(_sample(xs, uniq)), uniq.shape[0], d2)
        meter.note(uniq.shape[0])
        return vals[inv.reshape(-1), comp]
    return fun


# --- initial guess -----------------------------------------------------------

def _guess_cores(modes, ranks, dtype, like, seed):
    """Random starting cores on the backend of ``like``.

    The randomness is drawn with numpy and only then moved to the backend, so
    that ``seed`` means the same thing everywhere.  ``bk.randn`` accepts an
    ``rng`` but the torch backend ignores it, which would make ``seed`` a silent
    no-op on GPU -- a promise the docstring must not make and break.
    """
    rng = np.random.default_rng(seed)
    backend = bk.backend_of(like)
    cores = []
    for k, nk in enumerate(modes):
        shape = (int(ranks[k]), int(nk), int(ranks[k + 1]))
        a = rng.standard_normal(shape)
        if str(dtype).startswith("complex"):
            a = a + 1j * rng.standard_normal(shape)
        cores.append(bk.asarray(a, dtype, backend=backend))
    return cores


def _initial_guess(y0, modes, d2, r0, dtype, like, seed):
    """Starting TT for the cross: ``d`` modes, plus the component mode if any."""
    d = len(modes) - (1 if d2 > 1 else 0)
    if y0 is None:
        ranks = [1] + [int(r0)] * (len(modes) - 1) + [1]
        return vector.from_list(_guess_cores(modes, ranks, dtype, like, seed))
    if not isinstance(y0, vector):
        raise TypeError(f"y0 is {type(y0)!r}, expected a tt.vector")
    ny = [int(v) for v in y0.n]
    if ny != modes[:d]:
        raise ValueError(f"y0 has mode sizes {ny}, the inputs have {modes[:d]}")
    # bk.copy, not ``c.copy()``: torch tensors spell it ``clone`` and the bare
    # method call crashed every torch run that passed a y0.
    cores = [bk.copy(c) for c in y0.cores]
    if d2 == 1:
        if cores[-1].shape[2] != 1:
            raise ValueError(
                f"y0 has last rank {cores[-1].shape[2]} but funs returns a "
                "single component; the last rank must be 1")
        return vector.from_list(cores)
    tail = cores[-1].shape[2]
    if tail == d2:
        # y0 already carries the components in its last rank: expose them as the
        # extra mode by an exact identity core.
        cores.append(bk.eye(d2, dtype=bk.dtype_of(cores[-1]),
                            like=cores[-1]).reshape((d2, d2, 1)))
    elif tail == 1:
        cores.extend(_guess_cores([d2], [1, 1], dtype, like, seed))
    else:
        raise ValueError(
            f"y0 has last rank {tail}; for a {d2}-component funs it must be "
            f"{d2} (the components) or 1 (a spatial-only guess)")
    return vector.from_list(cores)


# --- the engine --------------------------------------------------------------

def _run(X, funs, eps, nswp, kickrank, y0, rmax, verb, name, *, d2, rf, tau,
         eps_exit, n_check, seed, restart_it, do_qr, kicktype, pcatype,
         trunctype, kickrank2, r0):
    t0 = time.time()
    xs, n = _check_inputs(X)
    d = len(n)
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if int(nswp) < 1:
        raise ValueError(f"nswp must be at least 1, got {nswp}")
    if int(kickrank) < 0:
        raise ValueError(f"kickrank must be non-negative, got {kickrank}")
    # A rank cap that cannot be met is a contradiction, not a request for "no
    # cap": reinterpreting rmax <= 0 as "unlimited" would silently run a job the
    # caller believes is capped.
    if rmax is not None and int(rmax) < 1:
        raise ValueError(
            f"rmax must be at least 1 (or None for no cap), got {rmax}")
    if int(r0) < 1:
        raise ValueError(f"r0 (ranks of the random initial guess) must be at "
                         f"least 1, got {r0}")

    ignored = []
    if do_qr:
        ignored.append("do_qr (the local basis is always orthogonalized by SVD)")
    if pcatype not in ("svd",):
        raise ValueError(
            f"pcatype={pcatype!r} is not implemented; only 'svd' is. "
            "The incomplete-Cholesky enrichment of the legacy code has no "
            "counterpart in the rectangular-maxvol cross used here.")
    if trunctype not in ("fro",):
        raise ValueError(
            f"trunctype={trunctype!r} is not implemented; only 'fro' is "
            "(the truncation is Frobenius-optimal by construction).")
    if kicktype not in ("amr-two", "rect"):
        raise ValueError(
            f"kicktype={kicktype!r} is not implemented; the enrichment here is "
            "rectangular maxvol ('rect', accepted also as the legacy "
            "'amr-two').")
    if int(kickrank2) != 0:
        raise ValueError(
            f"kickrank2={kickrank2} (extra random enrichment) is not "
            "implemented; the rectangular maxvol picks its extra rows by "
            "volume, not at random.")

    meter = _Meter()
    k2, dtype = _probe(xs, funs, n, d2, seed, meter)
    probe_values = meter.values
    hist = MultifuncrsHistory(eps=eps, d2=k2, probe_values=probe_values,
                              ignored_options=ignored)

    fun = _make_blackbox(xs, funs, k2, d, meter)
    modes = n + ([k2] if k2 > 1 else [])
    like = xs[0].cores[0]
    rmax_eff = int(rmax) if rmax is not None else None

    stop_fun = None
    if eps_exit is not None and float(eps_exit) != eps:
        eps_exit = float(eps_exit)

        def stop_fun(xprev, ycur):
            nrm = float(ycur.norm())
            return float((ycur - xprev).norm()) <= eps_exit * max(nrm, 1e-300)

    y = None
    attempts = max(1, int(restart_it) + 1)
    for attempt in range(attempts):
        # A restart must be a *different* run.  Re-using y0 would replay the
        # identical deterministic sweep and burn the attempt for nothing, so
        # only the first attempt honours the supplied guess.
        start = y0 if attempt == 0 else None
        x0 = _initial_guess(start, modes, k2, r0, dtype, like,
                            seed + 1013 * attempt)
        y = rect_cross(fun, x0, eps=eps, nswp=int(nswp),
                       kickrank=int(kickrank), rf=int(rf), verbose=False,
                       rmax=rmax_eff, tau=float(tau), n_check=int(n_check),
                       check_seed=seed, stop_fun=stop_fun, round_result=True)
        hist.attempts = attempt + 1
        if float(y.norm()) != 0.0:
            break
        # An exactly-zero answer is either the truth or a degenerate index set.
        # Decide it by fresh samples instead of guessing (U: no hidden unknown).
        rng = np.random.default_rng(seed + 7919 * (attempt + 1))
        idx = np.stack([rng.integers(0, nk, size=_ZERO_PROBE) for nk in n], axis=1)
        probe = _as_values(funs(_sample(xs, idx)), _ZERO_PROBE, k2)
        meter.note(_ZERO_PROBE)
        if not np.any(probe):
            warnings.warn(
                f"{name}: the result is exactly zero and so are "
                f"{_ZERO_PROBE} random samples of funs; returning the zero "
                "tensor", RuntimeWarning, stacklevel=3)
            break
        if attempt == attempts - 1:
            raise ValueError(
                f"{name}: the cross collapsed to the zero tensor although "
                f"funs is nonzero on random samples (max |funs| = "
                f"{np.max(np.abs(probe)):.3e}) after {attempts} attempt(s). "
                "Increase restart_it or supply a better y0.")

    hist.cross = y.history
    if k2 > 1:
        cores = list(y.cores)
        # Fold the artificial component mode back into the last TT rank, which
        # is the legacy block-TT layout: cores[-1] is (r, d2, 1) by construction.
        merged = einsum(cores[d - 1], cores[d][:, :, 0], "a n b, b j -> a n j")
        y = vector.from_list(cores[:d - 1] + [merged])
    hist.ranks = [int(v) for v in y.r]
    hist.funs_calls = meter.calls
    hist.funs_values = meter.values
    hist.time = time.time() - t0
    y.history = hist

    if verb >= 1:
        for rec in hist.sweeps:
            print(f"={name}= sweep {rec['sweep'] + 1}, dy: {rec['err_rel']:.3e},"
                  f" erank: {rec['erank']:.4g}, max_rank: {rec['max_rank']},"
                  f" fun_eval: {rec['fun_eval']}")
        print(f"={name}= done: converged={hist.converged}, "
              f"ranks={hist.ranks}, funs calls={hist.funs_calls}, "
              f"funs values={hist.funs_values}, time={hist.time:.2f}s")
    return y


# --- public entry points -----------------------------------------------------

def multifuncrs2(X, funs, eps=1e-6, nswp=10, kickrank=5, y0=None, rmax=999999,
                 verb=1, do_qr=False, restart_it=0, *, d2=None, rf=2, tau=1.1,
                 eps_exit=None, n_check=0, seed=0, kicktype="amr-two",
                 pcatype="svd", trunctype="fro", kickrank2=0, r0=2):
    """Cross approximation of a (vector-)function of several TT tensors.

    Computes ``Y[i] = funs(X_1[i], ..., X_p[i])`` in the TT format to relative
    accuracy ``eps``, sampling the inputs and ``funs`` only on the fibers that
    the cross selects.

    Args:
        X: List of ``tt.vector`` with identical mode sizes.
        funs: Vectorized callable.  Receives a numpy array ``V`` of shape
            ``(batch, len(X))`` -- ``V[m, j] = X_j`` at the ``m``-th sampled
            multi-index -- and returns ``(batch,)`` or ``(batch, d2)``.  It must
            be a genuine function of its argument (same input, same output).
        eps: Target relative accuracy in the Frobenius norm.  It is a knob, not
            a bound -- see "What ``eps`` actually buys" in the module docstring
            for the measured error/eps ratio as a function of ``d``.  The one
            failure this method cannot detect by itself is a feature carried by
            a few entries: for ``funs`` equal to ``1`` at a single point of a
            ``6^5`` grid and ``1e-3`` elsewhere, the run returns the constant
            ``1e-3`` (relative error 0.995) with ``converged=True`` and a
            relative change between sweeps of ``1e-15``.  Only ``n_check``
            large enough to hit the feature sees it (3000 points did, 20 did
            not); nothing else can.
        nswp: Maximum number of cross sweeps (one sweep = forward and back).
        kickrank: Rank-increasing parameter: extra rows the rectangular maxvol
            adds on top of the numerical rank at every micro-step.  ``0`` turns
            the method into a fixed-rank cross.
        y0: Initial guess (``tt.vector``).  Its ranks seed the index sets.  For
            a multi-component ``funs`` its last rank may be ``d2`` (the legacy
            way of declaring the number of components) or ``1``.
        rmax: Hard cap on the TT ranks of the answer; ``None`` means no cap.
            Values below 1 raise instead of being read as "no cap".  It caps
            the ``d - 1`` internal ranks only: when ``funs`` returns ``d2 > 1``
            components the last rank of the result *is* ``d2`` by definition of
            the block-TT layout and is not capped (``rmax=2`` with three
            components returns ranks ``[1, 2, 2, 2, 3]``).  Capping it would
            throw components away, which is never what a rank budget means.
        verb: 0 silent, >= 1 prints the sweep table after the run.  The history
            is recorded either way; ``verb`` never changes the numerics.
        do_qr: Accepted for signature compatibility; no effect, the local basis
            is always orthogonalized by an SVD (recorded in
            ``history.ignored_options``).
        restart_it: Number of extra attempts with a fresh random initial guess
            if the cross collapses onto the zero tensor while ``funs`` is
            demonstrably nonzero.  A restart always draws a new random guess:
            replaying ``y0`` would reproduce the same deterministic sweep.
        d2: Expected number of components of ``funs``.  ``None`` means "ask
            ``funs``"; if given and wrong, the call raises.  Note that with
            several components ``eps`` is a budget for the *stacked* tensor
            (as in the legacy code): a component carrying a fraction ``w`` of
            the joint norm is only accurate to about ``eps/w`` relative to
            itself.  Measured on five components spanning two orders of
            magnitude, eps=1e-9: the smallest one (0.7% of the joint norm) came
            out at 1.2e-8.  Call the method once per component if you need a
            per-component relative accuracy.
        rf: Extra slack for the rank growth, on top of ``kickrank``.
        tau: Rectangular maxvol tolerance.
        eps_exit: Stopping threshold on the relative change between sweeps, if
            different from ``eps``.
        n_check: Measure the true relative error on that many uniformly random
            held-out points after the last sweep; ends up in
            ``history.err_check``.  Costs that many extra ``funs`` values.  It
            is a Monte Carlo estimate: it resolves what a uniform sample of
            that size can hit, and nothing finer.
            For a one-dimensional ``X`` the cross evaluates the whole tensor,
            so the measurement is exact and comes out as ``0.0`` rather than as
            an estimate.
        seed: Seed of the random initial guess and of the held-out sample.
        kicktype: Legacy switch; ``'amr-two'`` and ``'rect'`` both mean the
            rectangular-maxvol enrichment actually used.  Anything else raises.
        pcatype: Only ``'svd'`` is implemented; ``'uchol'`` raises.
        trunctype: Only ``'fro'`` is implemented.
        kickrank2: Legacy random enrichment; only ``0`` is implemented.
        r0: Ranks of the random initial guess when ``y0 is None``.

    Returns:
        ``tt.vector`` approximating ``funs(X...)``, carrying a ``history``
        attribute (:class:`MultifuncrsHistory`).  When ``funs`` returns ``d2``
        components, the last TT rank of the result is ``d2``.

    Raises:
        ValueError: on inconsistent mode sizes, on a ``funs`` that is not
            vectorized or returns the wrong shape / non-finite values, on
            unimplemented legacy options, and when the cross collapses to zero
            although ``funs`` is not.
        TypeError: if ``X`` is not a list of ``tt.vector``.

    Warns:
        RuntimeWarning: (from the cross engine) when ``nswp`` sweeps did not
            reach ``eps``.  The tensor is returned anyway, with the achieved
            accuracy in ``history`` -- a quietly wrong answer is the one thing
            this must not produce.

    References:
        S. V. Dolgov, D. V. Savostyanov, SIAM J. Sci. Comput. 36(5), 2014,
        arXiv:1301.6068; I. V. Oseledets, E. E. Tyrtyshnikov, Linear Algebra
        Appl. 432(1):70-88, 2010.

    Example:
        >>> import numpy as np, tt
        >>> x = tt.xfun(4, 5) * (1.0 / 1024) + tt.ones(4, 5)   # 1 <= x <= 2
        >>> y = tt.multifuncrs2([x], lambda v: 1.0 / v[:, 0], 1e-10, verb=0)
        >>> bool(abs(y.full() - 1.0 / x.full()).max() < 1e-8)
        True
    """
    return _run(X, funs, eps, nswp, kickrank, y0, rmax, verb, "multifuncrs2",
                d2=d2, rf=rf, tau=tau, eps_exit=eps_exit, n_check=n_check,
                seed=seed, restart_it=restart_it, do_qr=do_qr,
                kicktype=kicktype, pcatype=pcatype, trunctype=trunctype,
                kickrank2=kickrank2, r0=r0)


def multifuncrs(X, funs, eps=1e-6, nswp=10, kickrank=5, y0=None, rmax=999999,
                verb=1, *, d2=None, do_qr=False, restart_it=0, rf=2, tau=1.1,
                eps_exit=None, n_check=0, seed=0, kicktype="amr-two",
                pcatype="svd", trunctype="fro", kickrank2=0, r0=2):
    """Cross approximation of a (vector-)function of several TT tensors.

    The legacy toolbox shipped two routines: ``multifuncrs`` (two-site DMRG
    cross with an ``amr-two`` kick) and ``multifuncrs2`` (one-site AMEn cross).
    They differ only in *how the ranks are grown*, not in what they compute, and
    this rewrite grows ranks in exactly one way -- rectangular maxvol inside
    :func:`tt.algs.cross.rect_cross`.  Keeping two copies of that sweep would
    give two things to debug and one of them would rot, so ``multifuncrs`` is
    the same engine under the legacy name and signature.

    See :func:`multifuncrs2` for the full argument documentation; the only
    difference is that ``do_qr`` and ``restart_it`` are keyword-only here,
    matching the legacy positional order (which ended at ``verb``).

    Returns:
        ``tt.vector`` with a ``history`` attribute
        (:class:`MultifuncrsHistory`).
    """
    return _run(X, funs, eps, nswp, kickrank, y0, rmax, verb, "multifuncrs",
                d2=d2, rf=rf, tau=tau, eps_exit=eps_exit, n_check=n_check,
                seed=seed, restart_it=restart_it, do_qr=do_qr,
                kicktype=kicktype, pcatype=pcatype, trunctype=trunctype,
                kickrank2=kickrank2, r0=r0)
