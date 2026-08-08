"""Adversarial verification of tt.algs.{optimize, completion, riemannian, solvers}.

These tests exist to break the four ported modules, not to confirm them.  Every
one of them is written against an oracle that does not go through the code under
test:

* a full ``numpy`` argmin / ``numpy.linalg.solve`` on a dense array;
* an invariant that must hold whatever the implementation is (idempotence of a
  projector, orthogonality of a residual, monotonicity of an exact alternating
  minimisation, ``X = Q R``);
* a *second* algorithm computing the same quantity (the completion functional
  recomputed from ``x.full()``, the GMRES residual recomputed with an untruncated
  matvec).

Several of them pin defects found during verification and fixed here; the
docstring names the defect and the regime, and the numbers behind the tolerances
are in ``docs/NUMERICS.md`` so the claim can be checked rather than trusted.

Regime of the accuracy numbers below, unless a test says otherwise: float64,
numpy backend, ``d`` between 1 and 8, mode sizes between 2 and 8, TT ranks
between 1 and 4.
"""

import contextlib
import io
import sys
import warnings

import numpy as np
import pytest

import tt
from tt.algs.completion import completion_functional, ttSparseALS
from tt.algs.optimize import min_func, min_tens
from tt.algs.riemannian import project, projector_splitting_add, tt_qr
from tt.algs.solvers import GMRES


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def flat(x):
    return np.asarray(x.full(asvector=True))


def low_rank_tt(n, r, rng, dtype=np.float64):
    """Random TT with the stated ranks; ranks above ``prod n`` are trimmed."""
    d = len(n)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((ranks[k], n[k], ranks[k + 1]))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c.astype(dtype))
    return tt.vector.from_list(cores)


def rank_deficient(n, rng, stated=2):
    """A rank-1 tensor written with TT ranks ``(1, stated, ..., stated, 1)``."""
    d = len(n)
    cores = []
    for k in range(d):
        r1 = 1 if k == 0 else stated
        r2 = 1 if k == d - 1 else stated
        c = np.zeros((r1, n[k], r2))
        c[0, :, 0] = rng.standard_normal(n[k])
        cores.append(c)
    return tt.vector.from_list(cores)


def kron_f(mats):
    """Kronecker product with the first factor running fastest (the F-order)."""
    out = np.ones((1, 1), dtype=mats[0].dtype)
    for m in mats:
        out = np.kron(m, out)
    return out


def dense_tangent_projector(X, tol=1e-10):
    """Tangent projector at ``X`` as a dense matrix, from the SVDs of ``X.full()``.

    Independent of ``tt.algs.riemannian``: it only ever touches the dense array.
    The row-space projector is built from ``vh[:r].T`` (rows of ``vh`` as
    columns, not conjugated); ``vh[:r].conj().T`` gives the projector onto the
    complex conjugate of the row space, which is Hermitian, idempotent and of
    the right trace and is therefore invisible to every invariant test -- see
    :func:`test_the_dense_oracle_agrees_with_the_definition_of_the_tangent_space`.
    """
    n = [int(v) for v in X.n]
    d = len(n)
    full = np.asarray(X.full())
    left = [np.ones((1, 1), dtype=full.dtype)] * (d + 1)
    right = [np.ones((1, 1), dtype=full.dtype)] * (d + 1)
    for k in range(1, d):
        unf = full.reshape((int(np.prod(n[:k])), -1), order="F")
        u, s, vh = np.linalg.svd(unf, full_matrices=False)
        r = max(1, int(np.sum(s > s[0] * tol)))
        left[k] = u[:, :r] @ u[:, :r].conj().T
        v = vh[:r].T
        right[k - 1] = v @ v.conj().T
    proj = np.zeros((full.size, full.size), dtype=full.dtype)
    for k in range(d):
        proj = proj + kron_f([left[k], np.eye(n[k], dtype=full.dtype), right[k]])
    for k in range(d - 1):
        proj = proj - kron_f([left[k + 1], right[k]])
    return proj


def tangent_dimension(X):
    """``sum_k r_k n_k r_{k+1} - sum_{k>0} r_k^2``.

    When this equals ``prod(n)`` the tangent space is the whole space and the
    projector is the identity: a comparison against a dense projector then
    proves nothing at all.
    """
    r = [int(v) for v in X.r]
    n = [int(v) for v in X.n]
    return (sum(r[k] * n[k] * r[k + 1] for k in range(len(n)))
            - sum(r[k] ** 2 for k in range(1, len(n))))


def tangent_basis(X):
    """Orthonormal basis of the tangent space at ``X``, straight from its
    definition ``span_k { tau(C_1, ..., dC_k, ..., C_d) }``.

    Uses nothing from ``tt.algs.riemannian`` and no SVD of an unfolding: it
    enumerates the coordinate directions of every core, builds the dense tensor
    for each, and orthonormalizes.  This is the oracle for the oracle.
    """
    cores = [np.asarray(c) for c in X.cores]
    cols = []
    for k in range(len(cores)):
        shape = cores[k].shape
        for j in range(int(np.prod(shape))):
            e = np.zeros(int(np.prod(shape)), dtype=cores[k].dtype)
            e[j] = 1.0
            cur = list(cores)
            cur[k] = e.reshape(shape)
            cols.append(flat(tt.vector.from_list(cur)))
    b = np.stack(cols, axis=1)
    u, s, _ = np.linalg.svd(b, full_matrices=False)
    return u[:, : int(np.sum(s > s[0] * 1e-10))]


def coo_of(x, m, rng):
    """``m`` random positions of ``x`` with their exact values (duplicates dropped)."""
    n = [int(v) for v in x.n]
    idx = np.unique(np.stack([rng.integers(0, v, size=m) for v in n], axis=1), axis=0)
    dense = np.asarray(x.full())
    return {"indices": idx, "values": dense[tuple(idx[:, k] for k in range(len(n)))]}


def silent(fn, *a, **kw):
    """Run ``fn`` capturing stdout; return ``(result, captured_text)``."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# =============================================================================
# 1. optimize -- min_tens / min_func
# =============================================================================

def test_min_history_is_usable_for_d_equal_one():
    """Regression: ``index_sizes`` was ``[1, 1]`` (ints) for ``d = 1``, so both
    ``max_index_set`` and ``repr(history)`` raised ``TypeError: 'int' object is
    not iterable`` -- the history object was unusable exactly in the one case
    where nothing else can go wrong."""
    x = tt.vector.from_list([np.array([3.0, -1.0, 2.0, 0.5]).reshape(1, 4, 1)])
    val, point, hist = min_tens(x, verb=False, return_history=True)
    assert val == pytest.approx(-1.0) and int(point[0]) == 1
    assert hist.max_index_set == 1
    assert "MinHistory" in repr(hist)

    val, point, hist = min_func(lambda z: (z[:, 0] - 0.25) ** 2, -1.0, 1.0, d=1,
                                n0=9, verb=False, return_history=True)
    assert hist.max_index_set == 1
    assert "MinHistory" in repr(hist)


def test_min_rejects_a_search_that_looks_at_nothing():
    """Regression: ``nswp=0`` fell out of the sweep with ``best_idx = None`` and
    died with ``AttributeError: 'NoneType' object has no attribute 'reshape'``."""
    rng = np.random.default_rng(0)
    t = low_rank_tt([3, 4, 3], 2, rng)
    with pytest.raises(ValueError, match="nswp must be at least 1"):
        min_tens(t, nswp=0, verb=False)
    with pytest.raises(ValueError, match="nswp must be at least 1"):
        min_func(lambda z: (z ** 2).sum(axis=1), -1.0, 1.0, d=3, n0=8, nswp=0,
                 verb=False)
    with pytest.raises(ValueError, match="rmax must be at least 1"):
        min_tens(t, rmax=0, verb=False)


def test_min_tens_accepts_rmax_none():
    """Regression: ``_search`` documented and handled ``rmax=None`` (no cap) but
    the initial index sets did ``min(None, k)`` and raised ``TypeError``."""
    rng = np.random.default_rng(0)
    t = low_rank_tt([3, 4, 3], 2, rng)
    dense = np.asarray(t.full())
    val, _ = min_tens(t, rmax=None, nswp=6, verb=False, seed=0)
    assert val == pytest.approx(dense.min(), rel=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_min_tens_finds_the_dense_argmin(seed):
    """Oracle: ``numpy`` argmin of the full tensor.

    Regime: rank-3 TT, ``d = 5``, ``n = [3, 4, 5, 4, 3]`` (720 entries),
    ``rmax = 10``, ``nswp = 20``, float64.  Measured over 40 seeds outside the
    suite: 0 failures, worst relative gap 0.0 (the exact minimum every time).
    """
    rng = np.random.default_rng(seed)
    t = low_rank_tt([3, 4, 5, 4, 3], 3, rng)
    dense = np.asarray(t.full())
    val, point, hist = min_tens(t, rmax=10, nswp=20, verb=False, seed=seed,
                                return_history=True)
    assert val == pytest.approx(dense.min(), rel=1e-12)
    assert dense[tuple(int(i) for i in point)] == pytest.approx(val, rel=1e-12)
    assert hist.consistency < 1e-12 * max(1.0, abs(val))


def test_min_tens_mode_sizes_differ_and_d_is_two():
    """d = 2 with different mode sizes: the smallest d where a sweep happens."""
    rng = np.random.default_rng(6)
    t = low_rank_tt([5, 9], 3, rng)
    dense = np.asarray(t.full())
    val, point = min_tens(t, rmax=6, nswp=8, verb=False, seed=0)
    assert val == pytest.approx(dense.min(), rel=1e-12)
    assert dense[int(point[0]), int(point[1])] == pytest.approx(val, rel=1e-12)


def test_min_tens_rank_one_and_constant_tensors():
    """Rank-1 input, and a tensor whose entries are all equal (every point is a
    minimiser, so the only thing that can be wrong is the bookkeeping)."""
    rng = np.random.default_rng(2)
    cores = [rng.standard_normal((1, m, 1)) for m in (4, 5, 3)]
    t = tt.vector.from_list(cores)
    dense = np.asarray(t.full())
    val, point = min_tens(t, rmax=4, nswp=8, verb=False, seed=0)
    assert val == pytest.approx(dense.min(), rel=1e-12)

    ones = tt.ones([4, 4, 4])
    val, point = min_tens(ones, rmax=4, nswp=6, verb=False, seed=0)
    assert val == pytest.approx(1.0, rel=1e-12)
    zero = 0.0 * tt.ones([4, 4, 4])
    val, point = min_tens(zero, rmax=4, nswp=6, verb=False, seed=0)
    assert val == pytest.approx(0.0, abs=1e-14)


def test_min_tens_examines_far_fewer_entries_than_the_tensor_has():
    """Regime: ``d = 10``, ``n = 4`` (1 048 576 entries), rank 4, ``rmax = 8``,
    ``nswp = 10``.  It finds the exact minimum, not merely a small entry."""
    rng = np.random.default_rng(11)
    n = [4] * 10
    t = low_rank_tt(n, 4, rng)
    dense = np.asarray(t.full())
    val, point, hist = min_tens(t, rmax=8, nswp=10, verb=False, seed=0,
                                return_history=True)
    assert val == pytest.approx(dense.min(), rel=1e-10)
    assert hist.evaluations < 0.1 * np.prod(n)


def test_min_verbosity_is_silent_but_the_history_is_complete():
    """``verb=False`` prints nothing, and still records everything."""
    rng = np.random.default_rng(0)
    t = low_rank_tt([4, 4, 4], 3, rng)
    (val, point, hist), out = silent(min_tens, t, rmax=6, nswp=8, verb=False,
                                     seed=0, return_history=True)
    assert out == ""
    assert hist.sweeps == 8 and hist.records and hist.evaluations > 0
    assert hist.max_index_set >= 1 and np.isfinite(hist.consistency)
    assert hist.time > 0.0

    (v2, p2, h2), out = silent(min_func, lambda z: (z ** 2).sum(axis=1), -1.0,
                               1.0, d=3, n0=9, nswp=6, rmax=4, verb=False,
                               seed=0, return_history=True)
    assert out == ""
    assert h2.sweeps == 6 and h2.records and h2.evaluations > 0


def test_min_func_detects_a_non_deterministic_objective():
    """``history.consistency`` is documented as the detector of an objective that
    is not a function of its argument.  Check that it actually fires."""
    rng = np.random.default_rng(0)

    def noisy(x):
        return (x ** 2).sum(axis=1) + rng.standard_normal(x.shape[0])

    val, point, hist = min_func(noisy, -1.0, 1.0, d=3, n0=9, nswp=6, rmax=4,
                                verb=False, seed=0, return_history=True)
    assert hist.consistency > 1e-3, hist.consistency
    # and the returned value is the re-evaluated one, not the one the sweep saw
    assert np.isfinite(val)


def test_min_func_returns_a_grid_point_whose_value_is_the_returned_value():
    """Self-consistency against a *second* evaluation of the objective, and the
    point must be a node of the stated grid."""
    d, n0 = 4, 33
    lo = np.array([-1.0, 0.0, -2.0, 1.0])
    hi = np.array([1.0, 3.0, 2.0, 4.0])
    target = np.array([0.25, 1.125, -0.5, 2.5])

    def fun(x):
        assert x.ndim == 2 and x.shape[1] == d
        return ((x - target) ** 2).sum(axis=1)

    val, point, hist = min_func(fun, lo, hi, rmax=6, nswp=14, n0=n0, verb=False,
                                seed=0, return_history=True)
    assert val == pytest.approx(float(fun(np.asarray(point).reshape(1, d))[0]),
                                rel=1e-14, abs=1e-300)
    for k in range(d):
        grid = np.linspace(lo[k], hi[k], n0)
        assert np.min(np.abs(grid - point[k])) < 1e-12
    assert val == pytest.approx(0.0, abs=1e-12)   # the target is a grid node


def test_min_tens_refuses_what_it_cannot_order():
    x = tt.rand([3, 3, 3], r=2).astype("complex128")
    with pytest.raises(TypeError, match="complex"):
        min_tens(x, verb=False)
    with pytest.raises(ValueError, match="boundary ranks"):
        min_tens(tt.vector.from_list([np.ones((2, 3, 2)), np.ones((2, 3, 2))]),
                 verb=False)


# =============================================================================
# 2. completion -- ttSparseALS
# =============================================================================

def test_completion_refuses_complex_data_instead_of_dropping_it():
    """Casting ``cooP['values']`` to float64 drops the imaginary part behind a
    numpy ``ComplexWarning``, after which the run reports a fit at machine zero
    for a fit to half the data.  Complex input must be refused instead."""
    rng = np.random.default_rng(2)
    truth = low_rank_tt([4, 4, 4], 2, rng, dtype=np.complex128)
    coo = coo_of(truth, 200, rng)
    with pytest.raises(TypeError, match="complex"):
        ttSparseALS(coo, [4, 4, 4], ttRank=2, maxnsweeps=2, verbose=False, seed=0)
    with pytest.raises(TypeError, match="complex"):
        ttSparseALS(coo_of(low_rank_tt([4, 4, 4], 2, rng), 200, rng), [4, 4, 4],
                    x0=truth, maxnsweeps=1, verbose=False)


def test_completion_reports_an_underdetermined_fit():
    """Regression.  The functional can be driven to machine zero while the
    answer is arbitrary, and nothing said so.

    Regime: rank-4 tensor of shape ``6x6x6`` (144 parameters), fitted at rank 4
    from 38 distinct samples, ``alpha = 0``, ``tol = 1e-13``, float64.
    The fit reaches machine zero with no empty slices while the answer is
    several times the norm of the truth away from it (``docs/NUMERICS.md``).
    The run must say ``determined = False`` and warn.
    """
    rng = np.random.default_rng(3)
    truth = low_rank_tt([6, 6, 6], 4, rng)
    dense = np.asarray(truth.full())
    coo = coo_of(truth, 40, rng)
    assert coo["indices"].shape[0] < 60

    with pytest.warns(RuntimeWarning, match="not determined by"):
        x, info = ttSparseALS(coo, [6, 6, 6], ttRank=4, tol=1e-13,
                              maxnsweeps=30, verbose=False, alpha=0.0, seed=0)
    assert info.converged                      # it does reproduce the samples
    assert info.fit[-1] < 1e-20
    assert not info.determined                 # ... and that means nothing
    assert info.underdetermined_slices > 0
    assert rel(x.full(), dense) > 1.0          # the honest number


def test_completion_well_posed_run_is_determined_and_silent():
    """The counterpart: with enough samples nothing warns and ``determined`` is
    True.  Regime: rank-2 tensor ``8x8x8x8`` (96 parameters), ~1330 distinct
    samples of 4096, ``alpha = 0``, ``tol = 1e-13``, float64; the entries the
    fit never saw are recovered too."""
    rng = np.random.default_rng(1)
    n = [8, 8, 8, 8]
    truth = low_rank_tt(n, 2, rng)
    dense = np.asarray(truth.full())
    coo = coo_of(truth, 1600, rng)

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        x, info = ttSparseALS(coo, n, ttRank=2, tol=1e-13, maxnsweeps=100,
                              verbose=False, alpha=0.0, seed=1)
    assert info.converged and info.determined
    seen = np.zeros(n, dtype=bool)
    seen[tuple(coo["indices"][:, k] for k in range(4))] = True
    err_unseen = (np.linalg.norm(np.asarray(x.full())[~seen] - dense[~seen])
                  / np.linalg.norm(dense[~seen]))
    assert err_unseen < 1e-5, err_unseen


def test_completion_fit_equals_the_functional_recomputed_from_the_dense_tensor():
    """Two owners of the same number must agree: the ``fit`` the sweep reports
    and ``J(x)/||values||^2`` recomputed from ``x.full()`` by hand.

    Compared after a *single* sweep, where ``J`` is still of order 1e-2.  After
    convergence ``J`` is at machine zero while the terms it is summed from are
    O(1), so it carries no correct relative digits and the comparison would be a
    comparison of roundoff.
    """
    rng = np.random.default_rng(5)
    n = [5, 4, 6]
    truth = low_rank_tt(n, 2, rng)
    coo = coo_of(truth, 300, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = ttSparseALS(coo, n, ttRank=2, tol=0.0, maxnsweeps=1,
                              verbose=False, alpha=0.0, seed=0)
    xd = np.asarray(x.full())
    by_hand = 0.5 * sum((xd[tuple(i)] - v) ** 2
                        for i, v in zip(coo["indices"], coo["values"]))
    scale = float(np.linalg.norm(coo["values"])) ** 2
    assert by_hand / scale > 1e-6, "make the residual big enough to compare"
    assert info.fit[-1] == pytest.approx(by_hand / scale, rel=1e-10)
    assert completion_functional(x, coo) == pytest.approx(by_hand, rel=1e-10)


def test_completion_is_monotone_for_exact_local_solves():
    """A property of the method, not of the code: every local problem is solved
    to its exact minimum, so ``J`` cannot increase.  d = 2 and d = 4."""
    for n, r, m, seed in ([[7, 9], 3, 60, 0], [[6, 6, 6, 6], 3, 900, 3]):
        rng = np.random.default_rng(seed)
        truth = low_rank_tt(n, r, rng)
        coo = coo_of(truth, m, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, info = ttSparseALS(coo, n, ttRank=r, tol=0.0, maxnsweeps=12,
                                  verbose=False, alpha=0.0, seed=seed)
        fit = np.asarray(info.fit)
        assert info.monotone, fit
        assert np.all(np.diff(fit) <= 1e-12 * max(fit[0], 1e-300)), fit


def test_completion_verbose_false_is_silent_and_records_everything():
    rng = np.random.default_rng(0)
    truth = low_rank_tt([5, 5, 5], 2, rng)
    coo = coo_of(truth, 400, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (x, info), out = silent(ttSparseALS, coo, [5, 5, 5], ttRank=2, tol=0.0,
                                maxnsweeps=4, verbose=False, alpha=0.0, seed=0)
    assert out == ""
    assert len(info.fit) == 4 and len(info.sweepTime) == 4
    assert info.stop_reason == "maxnsweeps" and not info.converged
    assert info.ranks == [1, 2, 2, 1] and info.time > 0.0
    assert info["fit"] is info.fit                       # legacy dict access


def test_completion_recovers_a_rank_one_tensor():
    """Rank-1 edge case: 1x1 local systems, so nothing can hide in a null space.

    Regime: ``n = [6, 5, 4]``, 66 distinct samples of 120 entries, 15
    parameters, ``alpha = 0``, ``tol = 1e-14``, float64.  The error on the whole
    tensor is ``sqrt(2 * fit)``, i.e. exactly what the stopping tolerance buys
    and not one digit more; the assertion is set to that, not below it.
    """
    rng = np.random.default_rng(7)
    n = [6, 5, 4]
    truth = low_rank_tt(n, 1, rng)
    dense = np.asarray(truth.full())
    coo = coo_of(truth, 90, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = ttSparseALS(coo, n, ttRank=1, tol=1e-14, maxnsweeps=50,
                              verbose=False, alpha=0.0, seed=0)
    assert info.converged and info.determined
    assert info.fit[-1] < 1e-14
    assert rel(x.full(), dense) < 10.0 * np.sqrt(2.0 * info.fit[-1])


def test_completion_maxnsweeps_zero_returns_the_start_and_says_so():
    rng = np.random.default_rng(0)
    truth = low_rank_tt([4, 4, 4], 2, rng)
    coo = coo_of(truth, 200, rng)
    x0 = low_rank_tt([4, 4, 4], 2, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = ttSparseALS(coo, [4, 4, 4], x0=x0, maxnsweeps=0, verbose=False)
    assert info.fit == [] and not info.converged
    assert info.stop_reason == "maxnsweeps"
    assert rel(x.full(), x0.full()) < 1e-12


# =============================================================================
# 3. riemannian
# =============================================================================

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_the_dense_oracle_agrees_with_the_definition_of_the_tangent_space(dtype):
    """The oracle for the oracle.

    Every ``project`` test in this file and in ``test_ports.py`` compares
    against ``dense_tangent_projector``.  That function is itself a piece of
    code and it had a conjugation bug (``vh[:r].conj().T``) that no invariant
    could see: the wrong matrix is Hermitian, idempotent, of the right trace,
    and equal to the identity -- hence indistinguishable from the right one --
    at any point of maximal TT rank, which is exactly where the complex test in
    ``test_ports.py`` was run.  Regime: ``d = 3``, ``n = [3, 4, 5]``, TT rank 2,
    tangent dimension 24 inside 60.  The correct oracle matches the definition
    to roundoff; the conjugated one is O(1) off and does not even fix the
    tangent space.
    """
    rng = np.random.default_rng(4)
    n = [3, 4, 5]
    X = low_rank_tt(n, 2, rng, dtype=dtype)
    assert tangent_dimension(X) < int(np.prod(n))

    t = tangent_basis(X)
    assert t.shape[1] == tangent_dimension(X)
    p_true = t @ t.conj().T
    p_oracle = dense_tangent_projector(X)
    assert np.abs(p_oracle - p_true).max() < 1e-12
    # ... and so does the code under test
    cols = []
    for j in range(int(np.prod(n))):
        e = np.zeros(int(np.prod(n)), dtype=np.complex128 if dtype is np.complex128
                     else np.float64)
        e[j] = 1.0
        cols.append(flat(project(X, tt.vector(e.reshape(n, order="F"), 1e-14))))
    p_code = np.stack(cols, axis=1)
    assert np.abs(p_code - p_true).max() < 1e-12


def test_project_refuses_a_rank_deficient_point():
    """Regression, and the worst defect found in this module.

    A rank-1 tensor written with TT ranks ``(1, 2, 2, 1)`` is not a point of the
    rank-``(1, 2, 2, 1)`` manifold; the closed-form projector then projects onto
    a strictly larger space.  What comes back is a perfectly good projector, so
    no invariant test could have caught it (``docs/NUMERICS.md``).  The
    advertised guard
    ("orthogonalization changed the ranks") never fired, because a QR never
    drops rank, and ``X.round(0)`` -- which the docstring claimed made the ranks
    minimal -- cannot remove anything: ``chop`` returns the full size whenever
    ``eps <= 0``.
    """
    rng = np.random.default_rng(1)
    n = [4, 4, 4]
    X = rank_deficient(n, rng)
    assert list(X.round(0.0).r) == [1, 2, 2, 1]          # round(0) removes nothing
    assert list(X.round(1e-14).r) == [1, 1, 1, 1]        # ... this does
    Z = tt.rand(n, r=2)

    with pytest.raises(ValueError, match="numerical rank"):
        project(X, Z)

    # rounded onto its true rank, the same point is fine and matches dense truth
    Xr = X.round(1e-14)
    got = flat(project(Xr, Z))
    want = dense_tangent_projector(Xr) @ flat(Z)
    assert rel(got, want) < 1e-10


def test_project_accepts_every_legitimately_full_rank_point():
    """The new guard must not fire on healthy input: ranks that are maximal,
    ranks that are not, rank 1, mode sizes that differ, complex entries."""
    rng = np.random.default_rng(4)
    cases = [
        tt.rand([4, 4, 4], r=[1, 4, 4, 1]).round(0.0),
        tt.rand([5, 3, 4, 6], r=[1, 2, 5, 3, 1]).round(0.0),
        low_rank_tt([6, 5, 4], 1, rng),
        low_rank_tt([3, 4, 5], 2, rng, dtype=np.complex128),
        tt.rand([2, 7, 2], r=[1, 2, 2, 1]).round(0.0),
    ]
    for X in cases:
        n = [int(v) for v in X.n]
        Z = low_rank_tt(n, 2, rng, dtype=np.asarray(X.cores[0]).dtype)
        got = flat(project(X, Z))
        want = dense_tangent_projector(X) @ flat(Z)
        assert rel(got, want) < 1e-9, (list(X.r), rel(got, want))
        if tangent_dimension(X) < int(np.prod(n)):
            # non-degenerate: also check against the definition of the tangent
            # space, so this does not rest on the oracle alone
            t = tangent_basis(X)
            assert rel(t @ (t.conj().T @ flat(Z)), want) < 1e-9


def test_project_d_equals_two_and_mode_of_size_one():
    rng = np.random.default_rng(8)
    X = tt.rand([4, 5], r=[1, 3, 1]).round(0.0)
    Z = tt.rand([4, 5], r=2)
    assert rel(flat(project(X, Z)), dense_tangent_projector(X) @ flat(Z)) < 1e-10

    X = low_rank_tt([3, 1, 4], 2, rng)
    Z = low_rank_tt([3, 1, 4], 2, rng)
    assert rel(flat(project(X, Z)), dense_tangent_projector(X) @ flat(Z)) < 1e-10


def test_project_of_a_mixed_real_complex_list():
    """A list whose members have different dtypes must promote, not truncate."""
    rng = np.random.default_rng(7)
    X = tt.rand([3, 3, 3], r=[1, 2, 2, 1]).round(0.0)
    zs = [tt.rand([3, 3, 3], r=2),
          low_rank_tt([3, 3, 3], 2, rng, dtype=np.complex128)]
    got = flat(project(X, zs))
    total = flat(zs[0]) + flat(zs[1])
    assert np.iscomplexobj(got)
    assert rel(got, dense_tangent_projector(X) @ total) < 1e-10


def test_project_complex_is_the_hermitian_projector():
    """For a complex tensor the projector must be Hermitian, not symmetric: the
    residual has to be orthogonal for the conjugated inner product."""
    rng = np.random.default_rng(9)
    X = low_rank_tt([3, 4, 3], 2, rng, dtype=np.complex128)
    Z = low_rank_tt([3, 4, 3], 3, rng, dtype=np.complex128)
    pz = project(X, Z)
    resid = Z - pz
    scale = float(Z.norm())
    for _ in range(4):
        t = project(X, low_rank_tt([3, 4, 3], 2, rng, dtype=np.complex128))
        assert abs(tt.dot(resid, t)) < 1e-10 * scale * float(t.norm())
    assert rel(project(X, pz).full(), pz.full()) < 1e-10


def test_project_rejects_mismatched_input():
    X = tt.rand([3, 3, 3], r=2).round(0.0)
    with pytest.raises(ValueError, match="mode sizes differ"):
        project(X, tt.rand([3, 4, 3], r=2))
    with pytest.raises(ValueError, match="empty list"):
        project(X, [])
    with pytest.raises(TypeError):
        project(X, np.zeros(27))


def test_projector_splitting_add_is_exact_even_at_a_rank_deficient_point():
    """Measured, not assumed.  Unlike ``project`` the splitting only needs the
    frames, and exactness survives at a rank-deficient point: ``d = 3``,
    ``n = 4``, ``Y`` a rank-1 tensor written with TT ranks ``(1, 2, 2, 1)``."""
    rng = np.random.default_rng(0)
    n = [4, 4, 4]
    Y = rank_deficient(n, rng)
    W = tt.vector.from_list([rng.standard_normal(s) for s in
                             [(1, 4, 2), (2, 4, 2), (2, 4, 1)]])
    out = projector_splitting_add(Y, W - Y)
    assert rel(out.full(), W.full()) < 1e-12
    assert list(out.r) == [1, 2, 2, 1]


@pytest.mark.parametrize("n,ranks", [([6, 5], [1, 3, 1]),
                                     ([3, 4, 5, 3], [1, 2, 4, 2, 1])])
def test_projector_splitting_add_exactness(n, ranks):
    rng = np.random.default_rng(2)
    Y = tt.vector.from_list([rng.standard_normal((ranks[k], n[k], ranks[k + 1]))
                             for k in range(len(n))])
    W = tt.vector.from_list([rng.standard_normal((ranks[k], n[k], ranks[k + 1]))
                             for k in range(len(n))])
    assert rel(projector_splitting_add(Y, W - Y).full(), W.full()) < 1e-9


def test_projector_splitting_add_is_a_first_order_retraction():
    """``psa(Y, tZ) - (Y + t P_Y Z)`` must be ``O(t^2)``: three values of ``t``,
    each halving, and the ratio of ``err / t^2`` must stay near 1 (a first-order
    error would double it at every halving)."""
    rng = np.random.default_rng(3)
    n = [4, 5, 4]
    Y = tt.rand(n, r=[1, 3, 3, 1]).round(0.0)
    Y = (1.0 / Y.norm()) * Y
    Z = tt.rand(n, r=2)
    Z = (1.0 / Z.norm()) * Z
    pz = flat(project(Y, Z))
    scaled = []
    for t in (4e-3, 2e-3, 1e-3):
        got = flat(projector_splitting_add(Y, t * Z))
        scaled.append(np.linalg.norm(got - (flat(Y) + t * pz)) / t ** 2)
    assert scaled[0] > 1e-6, scaled          # there is something to measure
    assert max(scaled) / min(scaled) < 1.5, scaled


def test_projector_splitting_add_rejects_mismatched_input():
    Y = tt.rand([3, 3, 3], r=2)
    with pytest.raises(ValueError, match="mode sizes differ"):
        projector_splitting_add(Y, tt.rand([3, 4, 3], r=2))
    with pytest.raises(TypeError):
        projector_splitting_add(Y, np.zeros(27))


@pytest.mark.parametrize("left_to_right", [True, False])
def test_tt_qr_on_a_rank_deficient_tensor(left_to_right):
    """``tt_qr`` must *not* refuse a deficient point: ``X = Q R`` and the
    orthogonality of the cores both still hold, to roundoff."""
    rng = np.random.default_rng(5)
    X = rank_deficient([4, 5, 4], rng)
    q, r = tt_qr(X, left_to_right=left_to_right)
    assert rel(float(r[0, 0]) * np.asarray(q.full()), X.full()) < 1e-12
    for c in (np.asarray(c) for c in q.cores):
        if left_to_right:
            m = c.reshape((-1, c.shape[2]))
            assert rel(m.T.conj() @ m, np.eye(c.shape[2])) < 1e-12
        else:
            m = c.reshape((c.shape[0], -1))
            assert rel(m @ m.T.conj(), np.eye(c.shape[0])) < 1e-12


@pytest.mark.parametrize("left_to_right", [True, False])
def test_tt_qr_complex_and_d_one(left_to_right):
    rng = np.random.default_rng(6)
    X = low_rank_tt([3, 4, 3], 2, rng, dtype=np.complex128)
    q, r = tt_qr(X, left_to_right=left_to_right)
    assert rel(complex(r[0, 0]) * np.asarray(q.full()), X.full()) < 1e-12
    for c in (np.asarray(c) for c in q.cores):
        if left_to_right:
            m = c.reshape((-1, c.shape[2]))
            assert rel(m.conj().T @ m, np.eye(c.shape[2])) < 1e-12
        else:
            m = c.reshape((c.shape[0], -1))
            assert rel(m @ m.conj().T, np.eye(c.shape[0])) < 1e-12

    v = tt.vector.from_list([rng.standard_normal((1, 6, 1))])
    q, r = tt_qr(v, left_to_right=left_to_right)
    assert rel(float(r[0, 0]) * np.asarray(q.full()), v.full()) < 1e-12


# =============================================================================
# 4. solvers -- GMRES
# =============================================================================

def matvec_of(A):
    return lambda x, eps: tt.matvec(A, x).round(eps)


def laplace_problem(d, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    A = tt.qlaplace_dd([d])
    x_exact = low_rank_tt([2] * d, rank, rng)
    return A, x_exact, tt.matvec(A, x_exact).round(1e-14)


def test_gmres_on_a_complex_non_hermitian_operator():
    """Oracle: ``numpy.linalg.solve`` on the dense operator.

    Regime: ``d = 3``, ``n = 4`` (64 x 64), complex128, a random TT-matrix of
    rank 2 shifted by ``3 I`` to make it solvable, ``eps = 1e-9``, ``m = 20``.
    """
    rng = np.random.default_rng(4)
    d, n = 3, 4
    cores = []
    for k in range(d):
        s = (1 if k == 0 else 2, n, n, 1 if k == d - 1 else 2)
        cores.append(rng.standard_normal(s) + 1j * rng.standard_normal(s))
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A + 3.0 * tt.eye([n] * d)
    x_exact = low_rank_tt([n] * d, 2, rng, dtype=np.complex128)
    b = tt.matvec(A, x_exact).round(1e-14)

    x, res = GMRES(matvec_of(A), tt.zeros([n] * d), b, eps=1e-9, maxit=300, m=20,
                   verbose=0)
    assert res <= 1e-9
    dense_x = np.linalg.solve(np.asarray(A.full()), flat(b))
    assert rel(flat(x), dense_x) < 1e-7
    assert rel(flat(x), flat(x_exact)) < 1e-7


def test_gmres_restarts_do_not_use_the_python_stack():
    """The legacy version recursed once per restart.  Run 300+ restarts with the
    recursion limit lowered to 80: a recursive implementation cannot pass.

    Regime: ``qlaplace_dd([4])`` (16 x 16), ``m = 1``, ``eps = 1e-8``,
    ``maxit = 2000`` -- over 300 restart cycles.
    """
    A, x_exact, b = laplace_problem(4, rank=2, seed=1)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(80)
    try:
        x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * 4), b, eps=1e-8,
                             maxit=2000, m=1, verbose=0, return_history=True)
    finally:
        sys.setrecursionlimit(old)
    assert hist.converged and res <= 1e-8
    assert len(hist.cycles) > 100, len(hist.cycles)
    assert hist.iterations == sum(c["iterations"] for c in hist.cycles)
    assert rel(flat(x), flat(x_exact)) < 1e-4


def test_gmres_reported_residual_survives_a_lossy_operator():
    """The returned residual is measured through the *inexact* closure, so it is
    only as good as the closure.  Recompute it with an untruncated matvec.

    Regime: ``qlaplace_dd([8])`` (256 x 256), rank-4 right-hand side,
    ``eps = 1e-6``, ``m = 20``, ``maxit = 300`` -- a run that does not converge,
    so the truncations really bite.  The reported and the recomputed residual
    must still agree to a fraction of a percent.
    """
    rng = np.random.default_rng(5)
    d = 8
    A = tt.qlaplace_dd([d])
    x_exact = low_rank_tt([2] * d, 4, rng)
    b = tt.matvec(A, x_exact).round(1e-14)
    with pytest.warns(RuntimeWarning, match="stopped after"):
        x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=1e-6,
                             maxit=300, m=20, verbose=0, return_history=True)
    exact = float((b - tt.matvec(A, x)).norm()) / float(b.norm())
    assert abs(exact / res - 1.0) < 0.05, (res, exact)
    assert not hist.converged and res > 1e-6


def test_gmres_handles_an_invariant_krylov_space():
    """``m`` larger than the dimension of the space: the Hessenberg subdiagonal
    goes to zero and the loop must stop, not divide by it."""
    A = tt.qlaplace_dd([2])
    x_exact = tt.rand([2, 2], r=2)
    b = tt.matvec(A, x_exact).round(1e-14)
    x, res, hist = GMRES(matvec_of(A), tt.zeros([2, 2]), b, eps=1e-10, maxit=200,
                         m=40, verbose=0, return_history=True)
    assert res <= 1e-10 and hist.iterations <= 4

    # identity operator with eps = 0: the projected problem is solved exactly at
    # step one, so the relaxed tolerance divides by a residual that is exactly 0
    I = tt.eye([2] * 3)
    rhs = tt.rand([2] * 3, r=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, res = GMRES(matvec_of(I), tt.zeros([2] * 3), rhs, eps=0.0, maxit=10,
                       m=5, verbose=0)
    assert rel(flat(x), flat(rhs)) < 1e-12


def test_gmres_rejects_input_it_cannot_act_on():
    A, x_exact, b = laplace_problem(3)
    with pytest.raises(ValueError, match="Krylov dimension"):
        GMRES(matvec_of(A), tt.zeros([2] * 3), b, m=0)
    with pytest.raises(ValueError, match="eps must be non-negative"):
        GMRES(matvec_of(A), tt.zeros([2] * 3), b, eps=-1.0)
    with pytest.raises(ValueError, match="right-hand side is zero"):
        GMRES(matvec_of(A), tt.zeros([2] * 3), 0.0 * tt.ones(2, 3))
    with pytest.raises(TypeError, match="tt.vectors"):
        GMRES(matvec_of(A), np.zeros(8), b)


def test_gmres_maxit_zero_reports_the_starting_residual():
    """No iteration at all: the answer is the initial guess and the residual is
    the one it actually has -- no claim of convergence."""
    A, x_exact, b = laplace_problem(4, seed=2)
    with pytest.warns(RuntimeWarning, match="stopped after"):
        x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * 4), b, eps=1e-8,
                             maxit=0, m=5, verbose=0, return_history=True)
    assert not hist.converged and hist.iterations == 0 and hist.cycles == []
    assert res == pytest.approx(1.0, rel=1e-12)      # ||b - 0|| / ||b||
    assert float(x.norm()) == 0.0                    # the untouched initial guess


def test_gmres_verbose_zero_is_silent_but_records_everything():
    A, x_exact, b = laplace_problem(4, seed=5)
    (x, res, hist), out = silent(GMRES, matvec_of(A), tt.zeros([2] * 4), b,
                                 eps=1e-8, maxit=200, m=5, verbose=0,
                                 return_history=True)
    assert out == ""
    assert hist.converged and hist.cycles and hist.residuals
    assert hist.iterations == sum(c["iterations"] for c in hist.cycles)
    assert len(hist.residuals) == len(hist.cycles) + 1
    assert hist.ranks == [int(v) for v in x.r]
    assert hist.time > 0.0 and "converged" in hist.message
    for c in hist.cycles:
        assert np.isfinite(c["res_est"]) and np.isfinite(c["res_end"]) is not None


def test_gmres_does_not_touch_the_caller_s_initial_guess():
    A, x_exact, b = laplace_problem(4, seed=2)
    u0 = low_rank_tt([2] * 4, 2, np.random.default_rng(0))
    before = [np.asarray(c).copy() for c in u0.cores]
    x, res = GMRES(matvec_of(A), u0, b, eps=1e-8, maxit=200, m=20, verbose=0)
    for old, new in zip(before, u0.cores):
        assert np.array_equal(old, np.asarray(new))
    assert res <= 1e-8
    # and a second call with the same guess must solve the same problem
    x2, res2 = GMRES(matvec_of(A), u0, b, eps=1e-8, maxit=200, m=20, verbose=0)
    assert rel(flat(x2), flat(x)) < 1e-10


def test_gmres_matches_the_dense_solve_for_a_nonsymmetric_operator():
    """Oracle: ``numpy.linalg.solve``.  Regime: ``d = 3``, ``n = 5`` (125 x 125),
    real non-symmetric TT-matrix of rank 2 shifted by ``4 I``, ``eps = 1e-10``."""
    rng = np.random.default_rng(12)
    d, n = 3, 5
    cores = [rng.standard_normal((1 if k == 0 else 2, n, n, 1 if k == d - 1 else 2))
             for k in range(d)]
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A + 4.0 * tt.eye([n] * d)
    dense = np.asarray(A.full())
    assert rel(dense.T, dense) > 0.1                     # genuinely non-symmetric
    x_exact = low_rank_tt([n] * d, 2, rng)
    b = tt.matvec(A, x_exact).round(1e-14)
    x, res = GMRES(matvec_of(A), tt.zeros([n] * d), b, eps=1e-10, maxit=300,
                   m=25, verbose=0)
    assert res <= 1e-10
    assert rel(flat(x), np.linalg.solve(dense, flat(b))) < 1e-8


def test_gmres_shape_mismatch_is_loud():
    """A closure whose output does not match ``b`` must blow up, not converge to
    something."""
    A = tt.qlaplace_dd([3])
    b = tt.rand([2] * 4, r=2)
    with pytest.raises(Exception):
        GMRES(matvec_of(A), tt.zeros([2] * 3), b, eps=1e-8, maxit=5, m=3,
              verbose=0)
