"""TT-cross checked against dense truth and against the tensor it is sampling.

Oracles used here:
  * a dense numpy evaluation of the black box on the whole grid (small cases);
  * the exact TT tensor that the black box samples, compared through the frozen
    core arithmetic (``(y - x).norm() / x.norm()``);
  * held-out random points that the algorithm never asked for.

Never the legacy implementation.
"""

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


def test_reported_accuracy_is_honest_when_the_sweeps_stall():
    """eps=1e-2 stops with err_rel ~ 1e-16 while the true error is ~1e-2.

    The change between sweeps is *not* an error bound: once the truncation pins
    the ranks, the iteration stops moving.  The held-out measurement is the one
    number a user may trust, so it has to agree with the dense truth.
    """
    y = cross(qtt_coulomb, [2] * D_QTT, eps=1e-2, nswp=20, n_check=400, seed=2)
    h = y.history
    err = dense_err(y, QTT_REF)
    assert err > 10 * h.err_rel, "this case is only interesting if err_rel lies"
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
    # the stopping criterion is the change between sweeps, an estimate of the
    # error and not a bound: landing within two orders of magnitude of eps is
    # what the method promises (measured: 1.3e-8 for eps=1e-9)
    assert err < 100 * 1e-9, f"relative error {err:.3e}"
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
