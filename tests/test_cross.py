"""TT-cross checked against dense truth and against the tensor it is sampling.

Oracles used here:
  * a dense numpy evaluation of the black box on the whole grid (small cases);
  * the exact TT tensor that the black box samples, compared through the frozen
    core arithmetic (``(y - x).norm() / x.norm()``);
  * held-out random points that the algorithm never asked for.

Never the legacy implementation.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.cross import cross, element, rect_cross


# --- helpers -----------------------------------------------------------------

def eval_tt_dense(cores, idx):
    """Values of a TT tensor at multi-indices, written out by hand.

    Independent of ``tt.algs.cross.element`` on purpose: this is what defines the
    black box in the tests, so it must not share code with the thing under test.
    """
    cores = [np.asarray(c) for c in cores]
    p = np.ones((idx.shape[0], 1), dtype=cores[0].dtype)
    for k, c in enumerate(cores):
        p = np.einsum("ma,amb->mb", p, c[:, idx[:, k], :])
    return p[:, 0]


def rand_tt(n, r, seed, complex_=False):
    rng = np.random.default_rng(seed)
    n = list(n)
    d = len(n)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((rr[k], n[k], rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c)
    return tt.vector.from_list(cores)


def counted(f):
    """Wrap a black box so the test can count the values it really handed out."""
    box = {"n": 0}

    def wrapped(idx):
        idx = np.asarray(idx)
        assert idx.ndim == 2, f"fun must get a (batch, d) array, got {idx.shape}"
        box["n"] += idx.shape[0]
        return f(idx)

    wrapped.count = box
    return wrapped


def all_indices(n):
    """Every multi-index of a grid, in the C order used by ``vector.full()``."""
    flat = np.arange(int(np.prod(n)))
    return np.stack(np.unravel_index(flat, tuple(n)), axis=1)


# --- element() ---------------------------------------------------------------

def test_element_matches_full():
    x = rand_tt([3, 4, 2, 5], r=3, seed=1)
    idx = all_indices(x.n)
    got = np.asarray(element(x, idx)).reshape(tuple(x.n))
    assert np.linalg.norm(got - np.asarray(x.full())) < 1e-12

    got = np.asarray(element(x, idx))
    ref = eval_tt_dense(x.cores, idx)
    assert np.linalg.norm(got - ref) < 1e-12


def test_element_rejects_wrong_shape():
    x = rand_tt([3, 3], r=2, seed=2)
    with pytest.raises(ValueError):
        element(x, np.zeros((5, 3), dtype=int))


# --- exact recovery of a low-rank tensor -------------------------------------

def test_recovers_exact_low_rank_tensor():
    n, r = [6] * 10, 3
    xt = rand_tt(n, r, seed=7)
    fun = counted(lambda idx: eval_tt_dense(xt.cores, idx))

    y = cross(fun, n, eps=1e-12, nswp=10, n_check=500, seed=3)
    h = y.history

    err = (y - xt).norm() / xt.norm()
    assert err < 1e-10, f"relative TT error {err:.3e}"
    assert h.err_check < 1e-10, f"held-out error {h.err_check:.3e}"
    assert h.converged
    # the whole point: far fewer evaluations than the 6**10 = 6.0e7 entries
    assert h.fun_eval < 1e-3 * 6 ** 10
    assert fun.count["n"] == h.fun_eval + h.fun_eval_check
    assert max(h.ranks) <= r + 1  # rounding must remove the exploration ranks


def test_recovers_complex_low_rank_tensor():
    n, r = [4] * 6, 2
    xt = rand_tt(n, r, seed=11, complex_=True)
    fun = counted(lambda idx: eval_tt_dense(xt.cores, idx))

    y = cross(fun, n, eps=1e-12, nswp=10, seed=5)
    assert y.is_complex
    err = (y - xt).norm() / xt.norm()
    assert err < 1e-10, f"relative TT error {err:.3e}"


def test_rank_one_from_a_sum_of_indices():
    """f = sum of indices is TT rank 2; check against the dense tensor."""
    n = [4] * 5
    fun = counted(lambda idx: idx.sum(axis=1).astype(float))
    y = cross(fun, n, eps=1e-10, nswp=8, seed=1)
    ref = fun(all_indices(n)).reshape(tuple(n))
    err = np.linalg.norm(np.asarray(y.full()) - ref) / np.linalg.norm(ref)
    assert err < 1e-10, f"relative error {err:.3e}"
    assert max(y.history.ranks) <= 3


# --- smooth function on a QTT grid, dense oracle -----------------------------

M = 5                       # bits per variable
NVAR = 3
D_QTT = M * NVAR
WEIGHT = 2 ** np.arange(M)  # mode 1 is the fastest index -> least significant bit


def qtt_coulomb(idx):
    """1 / (x + y + z + 1) on a 2^M x 2^M x 2^M grid in [0, 1]^3, QTT indexed."""
    idx = np.asarray(idx)
    g = [(idx[:, v * M:(v + 1) * M] @ WEIGHT + 0.5) / 2 ** M for v in range(NVAR)]
    return 1.0 / (g[0] + g[1] + g[2] + 1.0)


QTT_REF = qtt_coulomb(all_indices([2] * D_QTT)).reshape((2,) * D_QTT)


def dense_err(y, ref):
    return float(np.linalg.norm(np.asarray(y.full()) - ref) / np.linalg.norm(ref))


def test_qtt_smooth_function_against_dense():
    fun = counted(qtt_coulomb)
    y = cross(fun, [2] * D_QTT, eps=1e-8, nswp=20, n_check=400, seed=2)
    h = y.history
    err = dense_err(y, QTT_REF)
    assert err < 1e-6, f"dense relative error {err:.3e}, history {h}"
    assert h.converged
    # the reported accuracy must agree with the dense truth it claims to measure
    assert h.err_check == pytest.approx(err, rel=0.5), f"{h.err_check} vs {err}"
    assert h.fun_eval < 2 ** D_QTT  # cheaper than filling the grid


def test_reported_accuracy_tracks_the_dense_truth_at_a_loose_eps():
    """At eps=1e-2 the reported numbers must be in the same league as the truth.

    Truncating the local bases at the local accuracy lets the index sets shrink
    and pins the ranks, at which point the iteration freezes and its own change
    indicator becomes meaningless.  With the truncation removed (see
    ``cross._left_basis``) the ranks stay free and the change between sweeps is
    again an indicator of the error, to within an order of magnitude
    (``docs/NUMERICS.md``).  The held-out measurement must agree with the dense
    truth much more tightly.
    """
    y = cross(qtt_coulomb, [2] * D_QTT, eps=1e-2, nswp=20, n_check=400, seed=2)
    h = y.history
    err = dense_err(y, QTT_REF)
    assert err < 1e-2, f"eps=1e-2 was requested, got {err:.3e}"
    assert h.err_rel < 100 * err, f"err_rel {h.err_rel:.2e} vs true {err:.2e}"
    assert h.err_check == pytest.approx(err, rel=0.5), f"{h.err_check} vs {err}"


def test_error_decreases_with_eps():
    errs, evals = [], []
    for eps in (1e-2, 1e-5, 1e-10):
        y = cross(qtt_coulomb, [2] * D_QTT, eps=eps, nswp=20, seed=2)
        errs.append(dense_err(y, QTT_REF))
        evals.append(y.history.fun_eval)
    assert errs[0] > errs[1] > errs[2], f"errors {errs}"
    assert errs[2] < 1e-9, f"errors {errs}"
    assert errs[0] < 1e-1, f"errors {errs}"
    assert evals[0] < evals[2], f"fun_eval {evals}"


def test_sin_over_x_on_a_plain_grid():
    n = [32] * 4
    def fun(idx):
        s = 1.0 + idx.astype(float).sum(axis=1) / 16.0
        return np.sin(s) / s
    y = cross(fun, n, eps=1e-9, nswp=15, seed=4)
    ref = fun(all_indices(n)).reshape(tuple(n))
    err = np.linalg.norm(np.asarray(y.full()) - ref) / np.linalg.norm(ref)
    # The stopping criterion is the change between sweeps -- an estimate of the
    # error, not a bound -- so the tolerance carries a factor.  It is 5, not the
    # 100 this test used to carry: measured 5.5e-10 for eps=1e-9 (numpy,
    # float64, identical to two digits for seeds 0..4), i.e. the method lands
    # *below* the requested accuracy here and a 100x window would have hidden a
    # 20-fold regression.
    assert err < 5 * 1e-9, f"relative error {err:.3e}"
    assert y.history.fun_eval < 0.2 * 32 ** 4


# --- honesty on a tensor that is not low-rank --------------------------------

def test_noise_is_reported_as_failure():
    n = [6] * 6
    rng = np.random.default_rng(0)
    table = rng.standard_normal(6 ** 6)

    def fun(idx):
        return table[np.ravel_multi_index(tuple(idx.T), tuple(n))]

    with pytest.warns(RuntimeWarning, match="did not reach eps"):
        y = cross(fun, n, eps=1e-8, nswp=4, rmax=20, n_check=500, seed=1)
    h = y.history
    assert h.converged is False
    assert h.err_rel > 1e-3, f"reported change {h.err_rel:.3e}"
    assert h.err_check > 0.1, f"held-out error {h.err_check:.3e}"
    assert max(h.ranks) <= 20
    assert len(h.sweeps) == 4


def test_rank_cap_is_honoured_and_reported():
    n = [5] * 8
    xt = rand_tt(n, r=8, seed=13)
    fun = lambda idx: eval_tt_dense(xt.cores, idx)
    with pytest.warns(RuntimeWarning, match="rank cap"):
        y = cross(fun, n, eps=1e-10, nswp=4, rmax=4, seed=1)
    assert max(y.history.ranks) <= 4
    assert y.history.converged is False
    err = (y - xt).norm() / xt.norm()
    assert err > 1e-6, "a rank-4 tensor cannot reproduce a rank-8 one"


# --- history and API ---------------------------------------------------------

def test_history_is_recorded_when_silent():
    n = [4] * 5
    xt = rand_tt(n, r=2, seed=17)
    y = cross(lambda idx: eval_tt_dense(xt.cores, idx), n, eps=1e-10, verbose=False)
    h = y.history
    assert h.fun_eval > 0 and h.time >= 0.0
    assert [s["sweep"] for s in h.sweeps] == list(range(len(h.sweeps)))
    assert all(s["fun_eval"] > 0 and s["max_rank"] > 0 for s in h.sweeps)
    assert h.sweeps[-1]["err_rel"] == pytest.approx(h.err_rel)
    assert h.err_check is None          # not requested -> not invented
    assert list(h.ranks) == list(y.r)
    assert "CrossHistory" in repr(h)


def test_rect_cross_takes_an_initial_guess():
    n = [4] * 6
    xt = rand_tt(n, r=3, seed=19)
    x0 = tt.rand(n, r=2)
    y = rect_cross(lambda idx: eval_tt_dense(xt.cores, idx), x0, eps=1e-11)
    assert (y - xt).norm() / xt.norm() < 1e-10


def test_legacy_argument_order():
    n = [4, 5, 3]
    fun = lambda idx: 1.0 + idx.astype(float).prod(axis=1)
    y = cross(n, fun, eps=1e-10)                    # legacy tt.cross.cross(n, f)
    ref = fun(all_indices(n)).reshape(tuple(n))
    assert np.linalg.norm(np.asarray(y.full()) - ref) / np.linalg.norm(ref) < 1e-10


def test_single_mode():
    y = cross(lambda idx: idx[:, 0].astype(float) ** 2, [7])
    assert y.d == 1
    assert np.allclose(np.asarray(y.full()).reshape(-1), np.arange(7.0) ** 2)
    assert y.history.fun_eval == 7
    assert y.history.converged


def test_lazy_attribute_on_tt_namespace():
    """ttpy 1.x had `tt.cross` as a MODULE (`from tt.cross import cross`)."""
    import tt.cross as cross_module

    assert tt.rect_cross is rect_cross
    assert tt.cross is cross_module
    assert tt.cross.cross is cross
    assert cross_module.rect_cross is rect_cross


# --- loud failures -----------------------------------------------------------

def test_non_vectorized_fun_raises():
    with pytest.raises(ValueError, match="vectorized"):
        cross(lambda idx: 1.0, [3, 3, 3])


def test_non_finite_values_raise():
    def fun(idx):
        # the very first fiber sweeps i_0 over the whole mode, so this is hit
        return np.where(idx[:, 0] >= 2, np.inf, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        cross(fun, [4, 4, 4])


def test_bad_arguments_raise():
    with pytest.raises(TypeError):
        rect_cross("not a function", tt.rand([3, 3], r=2))
    with pytest.raises(TypeError):
        rect_cross(lambda idx: idx.sum(axis=1), [3, 3])
    with pytest.raises(ValueError):
        rect_cross(lambda idx: idx.sum(axis=1), tt.rand([3, 3], r=2), nswp=0)


def test_rect_maxvol_does_not_scold_a_caller_for_its_own_budget():
    """``maxK`` reached is not a surprise when the caller chose ``maxK``.

    This warning was 80 of the 93 RuntimeWarnings the suite emitted: it fired
    every time ``cross._select_rows`` set a rank budget and then respected it.
    That early stop costs nothing -- identical errors and identical ranks across
    the whole range of ``rf`` (``docs/NUMERICS.md``). The other
    non-convergence branch (``tol < 1``, which no row set can satisfy) is a real
    mistake and must still warn.
    """
    from tt.algs.maxvol import rect_maxvol

    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 6))

    with pytest.warns(RuntimeWarning, match="stopped at K"):
        piv_warned = rect_maxvol(a, 1.05, maxK=8, min_add_K=2)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        piv_quiet = rect_maxvol(a, 1.05, maxK=8, min_add_K=2,
                                warn_budget=False)[0]

    # silencing the warning must not change a single pivot
    assert np.array_equal(np.sort(piv_warned), np.sort(piv_quiet))

    # The other branch is a different mistake -- tol < 1 is unachievable by any
    # row set -- and warn_budget must not reach it. It shows up without a
    # budget, where "stopped" cannot mean "hit the caller's maxK".
    with pytest.warns(RuntimeWarning, match="tol < 1|can never achieve"):
        rect_maxvol(a, 0.5, min_add_K=2, warn_budget=False)


def test_cross_kickrank2_adds_rows_and_keeps_the_answer():
    """Uniformly random extra pivots: more evaluations, no loss of accuracy.

    They exist for the failure mode where the greedy index sets reach a fixed
    point while an unsampled region still carries the error -- every internal
    indicator reads machine precision while the answer is wrong
    (``docs/NUMERICS.md``; the reproducer is in
    docs/plans/cross-approximation.md, and it is numpy-version sensitive, so it
    is documented rather than pinned here). What *is* stable and worth pinning:
    the knob costs evaluations and never hurts a well-behaved problem.
    """
    d, n = 5, 8
    calls = {"k": 0}

    def fun(idx):
        calls["k"] += len(idx)
        s = idx.sum(1) / (n - 1.0)
        return np.sin(s) / (1.0 + s)

    grid = np.stack(np.meshgrid(*[np.arange(n)] * d, indexing="ij"), -1)
    exact = fun(grid.reshape(-1, d))
    calls["k"] = 0

    seen = {}
    for k2 in (0, 3):
        calls["k"] = 0
        y = cross(fun, n, d, eps=1e-10, r=2, seed=0, kickrank=2, kickrank2=k2)
        got = np.asarray(y.full(asvector=True))
        err = np.linalg.norm(got - exact) / np.linalg.norm(exact)
        seen[k2] = (err, calls["k"])
        assert err < 1e-8, f"kickrank2={k2}: {err:.3E}"

    assert seen[3][1] > seen[0][1], "the extra rows cost nothing -- not applied?"
