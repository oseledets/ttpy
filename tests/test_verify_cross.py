"""Adversarial verification of ``tt.algs.cross``.

Every accuracy assertion here is against one of:
  * the dense numpy array of the black box on the whole grid (small grids), or
  * the exact TT tensor that the black box samples, compared with the frozen
    core arithmetic, or
  * an invariant that does not involve the module under test (dtype, index
    ranges, determinism, printed output).

Never against the module's own output, never against the legacy package.

Regime for all accuracy numbers below: numpy backend, float64 unless the test
says otherwise, grids and eps written in the test body.
"""

import io
import contextlib
import warnings

import numpy as np
import pytest

import tt
from tt import backend as bk
from tt.algs.cross import cross, element, rect_cross


# --- oracles -----------------------------------------------------------------

def all_indices(n):
    """Every multi-index of the grid, in the C order of ``vector.full()``."""
    flat = np.arange(int(np.prod(n)))
    return np.stack(np.unravel_index(flat, tuple(n)), axis=1)


def dense_of(fun, n):
    """The whole tensor, as a dense numpy array (small grids only)."""
    return np.asarray(fun(all_indices(n))).reshape(tuple(n))


def rel_err(y, ref):
    return float(np.linalg.norm(np.asarray(y.full()) - ref) / np.linalg.norm(ref))


def eval_tt_dense(cores, idx):
    """Hand-written TT evaluation; shares no code with ``cross.element``."""
    cores = [np.asarray(c) for c in cores]
    p = np.ones((idx.shape[0], 1), dtype=cores[0].dtype)
    for k, c in enumerate(cores):
        p = np.einsum("ma,amb->mb", p, c[:, idx[:, k], :])
    return p[:, 0]


def rand_tt(n, r, seed, complex_=False):
    rng = np.random.default_rng(seed)
    rr = [1] + [r] * (len(n) - 1) + [1]
    cores = []
    for k in range(len(n)):
        c = rng.standard_normal((rr[k], n[k], rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c)
    return tt.vector.from_list(cores)


# --- the regression that matters: does cross actually reach eps? -------------

SMOOTH = {
    # name: (mode sizes, black box).  All of them have a dense TT-SVD
    # representation at 1e-10 with ranks well below the mode products, so
    # "cross cannot do better" is never the excuse.
    "coulomb_5^5": ([5] * 5, lambda i: 1.0 / (1.0 + i.astype(float).sum(axis=1))),
    "coulomb_8^4": ([8] * 4, lambda i: 1.0 / (1.0 + i.astype(float).sum(axis=1))),
    "sqrt_10^4": ([10] * 4, lambda i: np.sqrt(1.0 + i.astype(float).sum(axis=1))),
    "inv_square_7^4": ([7] * 4,
                       lambda i: 1.0 / (1.0 + (i.astype(float) ** 2).sum(axis=1))),
    "gauss_9^4": ([9] * 4,
                  lambda i: np.exp(-((i.astype(float) - 4.0) ** 2).sum(axis=1) / 9.0)),
}


@pytest.mark.parametrize("name", sorted(SMOOTH))
@pytest.mark.parametrize("eps", [1e-4, 1e-10])
def test_smooth_function_reaches_the_requested_accuracy(name, eps):
    """The accuracy actually delivered, against the dense array.

    Regression for the failure that made this module untrustworthy: with the
    local bases truncated at ``eps/sqrt(d)`` the index sets could shrink, the
    ranks locked at a fixed point and the run reported ``converged=True`` with
    ``err_rel=1e-16`` at a true relative error of 1.2e-1 on ``coulomb_8^4``,
    eps=1e-10.

    The tolerance is ``3 * eps`` and that number is not arbitrary.  Measured on
    this box (numpy, float64, seeds 0..4, the ratios are seed independent to
    two digits) the worst ``err / eps`` over the whole bank is 0.70
    (``sqrt_10^4`` at eps=1e-4); the pre-fix code reached 5.2 on that same
    entry and 1.2e+3 on ``coulomb_8^4`` at eps=1e-10.  A looser tolerance --
    the 30x this test used to carry -- lets half of the pre-fix failures
    through, which is why it is 3 and not 30.
    """
    n, fun = SMOOTH[name]
    ref = dense_of(fun, n)
    y = cross(fun, n, eps=eps, nswp=20, seed=0)
    err = rel_err(y, ref)
    # dense TT-SVD of the same tensor: proof that the accuracy is reachable
    best = tt.vector(ref, eps=eps)
    best_err = float(np.linalg.norm(np.asarray(best.full()) - ref)
                     / np.linalg.norm(ref))
    assert best_err < 3 * eps, f"the oracle itself missed eps: {best_err:.2e}"
    assert err < 3 * eps, (
        f"{name}: cross gave {err:.3e} for eps={eps:.0e} "
        f"(dense TT-SVD reaches {best_err:.3e} with ranks {list(best.r)}); "
        f"history {y.history}")


def test_converged_flag_is_not_handed_out_at_a_large_error():
    """If the run says ``converged`` on a smooth function, it must be right.

    Not a tautology: the flag is computed from the change between sweeps, the
    error from the dense array.  This is exactly the pairing that used to be
    broken (converged=True at a 12% error).
    """
    for name, (n, fun) in SMOOTH.items():
        ref = dense_of(fun, n)
        y = cross(fun, n, eps=1e-8, nswp=20, seed=0)
        err = rel_err(y, ref)
        if y.history.converged:
            assert err < 1e-5, f"{name}: converged=True at a true error {err:.2e}"


# --- loud reporting when the answer cannot deserve its eps -------------------

def test_rank_cap_is_reported_even_when_the_sweeps_converge():
    """rmax=2 on a tensor that needs 9: the run settles, so it *looks* converged.

    The change between sweeps drops to 1e-16 because the ranks cannot move, and
    the true error is 11%.  Silence here would be the worst possible output.
    """
    n = [5] * 5
    fun = lambda i: 1.0 / (1.0 + i.astype(float).sum(axis=1))
    ref = dense_of(fun, n)
    with pytest.warns(RuntimeWarning, match="rank cap"):
        y = cross(fun, n, eps=1e-10, nswp=6, rmax=2, seed=0)
    err = rel_err(y, ref)
    assert err > 1e-3, f"the case is only interesting if the cap hurts: {err:.2e}"
    assert y.history.rmax_active is True
    assert max(y.history.ranks) <= 2


def test_kickrank_zero_is_reported():
    """kickrank=0 removes the only error detector, so it cannot be silent."""
    n = [5] * 6
    xt = rand_tt(n, r=4, seed=3)
    fun = lambda i: eval_tt_dense(xt.cores, i)
    with pytest.warns(RuntimeWarning, match="kickrank=0"):
        y = cross(fun, n, eps=1e-10, nswp=6, kickrank=0, r=2, seed=0)
    err = (y - xt).norm() / xt.norm()
    assert err > 1e-3, f"fixed-rank cross should miss a rank-4 tensor: {err:.2e}"


def test_no_warning_on_a_healthy_run():
    """The warnings above are worth nothing if they cry wolf on a good run."""
    n = [4] * 5
    xt = rand_tt(n, r=2, seed=17)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = cross(lambda i: eval_tt_dense(xt.cores, i), n, eps=1e-10, seed=0)
    assert (y - xt).norm() / xt.norm() < 1e-10


def test_eps_zero_is_unreachable_and_says_so():
    n = [4] * 4
    xt = rand_tt(n, r=2, seed=1)
    with pytest.warns(RuntimeWarning, match="did not reach eps"):
        y = cross(lambda i: eval_tt_dense(xt.cores, i), n, eps=0.0, nswp=3, seed=0)
    assert y.history.converged is False


def test_spike_is_the_documented_blind_spot():
    """A delta is returned as *zero* with every indicator reading zero.

    This is inherent: cross sees O(d n r^2) entries and a delta lives on one.
    The test exists so that the limitation is a pinned, visible property of the
    package rather than a surprise in someone's paper -- and so that a future
    'improvement' that claims to detect it has to prove it here.
    """
    n = [8] * 5
    target = np.array([3, 5, 1, 6, 2])
    fun = lambda i: (np.all(np.asarray(i) == target, axis=1)).astype(float)
    y = cross(fun, n, eps=1e-8, nswp=10, n_check=500, seed=0)
    h = y.history
    ref = dense_of(fun, n)
    err = rel_err(y, ref)
    assert err == pytest.approx(1.0), f"expected the zero tensor, err {err:.3e}"
    assert h.converged is True          # nothing in the sampled data says otherwise
    assert h.err_check == 0.0           # 500 random points all miss the spike
    assert float(y.norm()) == 0.0


# --- dtype / backend contract ------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_result_dtype_follows_x0(dtype):
    """The docstring promises the width of ``x0``; check it, do not assume it."""
    n = [6] * 4
    x0 = tt.vector.from_list([np.asarray(c, dtype=dtype)
                              for c in tt.rand(n, r=2).cores])
    fun = lambda i: 1.0 / (1.0 + i.astype(np.float64).sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = rect_cross(fun, x0, eps=1e-4, nswp=8)
    assert np.asarray(y.cores[0]).dtype == np.dtype(dtype)
    assert all(np.asarray(c).dtype == np.dtype(dtype) for c in y.cores)
    err = rel_err(y, dense_of(fun, n))
    assert err < 1e-3, f"{dtype}: err {err:.2e}"


def test_complex_values_promote_a_real_x0():
    """Dropping an imaginary part would be a wrong answer, not a conversion."""
    n = [6] * 4
    x0 = tt.vector.from_list([np.asarray(c, dtype="float32")
                              for c in tt.rand(n, r=2).cores])

    def fun(i):
        s = 1.0 + np.asarray(i).astype(float).sum(axis=1)
        return np.exp(1j * s) / s

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = rect_cross(fun, x0, eps=1e-4, nswp=10)
    assert np.asarray(y.cores[0]).dtype == np.dtype("complex64")
    assert rel_err(y, dense_of(fun, n)) < 1e-3


def test_complex_smooth_function_against_dense():
    n = [8] * 4

    def fun(i):
        s = 1.0 + np.asarray(i).astype(float).sum(axis=1) / 8.0
        return np.exp(1j * s) / s

    ref = dense_of(fun, n)
    y = cross(fun, n, eps=1e-10, nswp=15, n_check=300, seed=1)
    err = rel_err(y, ref)
    assert err < 1e-8, f"complex dense error {err:.3e}"
    assert y.history.err_check == pytest.approx(err, rel=1.0)


def test_element_on_complex_cores():
    x = rand_tt([3, 4, 2], r=3, seed=2, complex_=True)
    idx = all_indices(x.n)
    got = np.asarray(element(x, idx))
    assert got.dtype == np.dtype("complex128")
    assert np.linalg.norm(got - eval_tt_dense(x.cores, idx)) < 1e-12


def test_torch_backend_is_actually_exercised_not_just_assumed():
    """The module claims to be backend agnostic; run it on torch and check.

    No CUDA needed -- the dispatch path is the same on a CPU torch tensor, and
    that is what was never verified.  Measured on b300 with torch 2.13.0+cpu:
    float64 gives 1.8e-7 for eps=1e-6, complex promotion gives 8.3e-10 for
    eps=1e-8, both against the dense numpy array.
    """
    torch = pytest.importorskip("torch")
    n = [6] * 4
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    ref = dense_of(fun, n)
    cores = [torch.as_tensor(np.asarray(c), dtype=torch.float64, device="cpu")
             for c in tt.rand(n, r=2).cores]
    y = rect_cross(fun, tt.vector.from_list(cores), eps=1e-6, nswp=10,
                   n_check=200)
    assert isinstance(y.cores[0], torch.Tensor), "the answer left the backend"
    assert all(isinstance(c, torch.Tensor) for c in y.cores)
    got = np.asarray(bk.to_numpy(y.full()))
    err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert err < 3e-6, f"torch float64: {err:.3e}"
    assert y.history.err_check == pytest.approx(err, rel=0.5)

    # element() must survive the trip too: it is what err_check is computed with
    idx = all_indices(n)
    vals = np.asarray(bk.to_numpy(element(y, idx))).reshape(tuple(n))
    assert np.linalg.norm(vals - got) < 1e-12

    def cfun(i):
        s = 1.0 + np.asarray(i).astype(float).sum(axis=1) / 6.0
        return np.exp(1j * s) / s

    cores = [torch.as_tensor(np.asarray(c), dtype=torch.float64, device="cpu")
             for c in tt.rand(n, r=2).cores]
    z = rect_cross(cfun, tt.vector.from_list(cores), eps=1e-8, nswp=10)
    assert z.cores[0].dtype == torch.complex128
    cref = dense_of(cfun, n)
    assert float(np.linalg.norm(np.asarray(bk.to_numpy(z.full())) - cref)
                 / np.linalg.norm(cref)) < 1e-7


# --- shapes and degenerate grids ---------------------------------------------

def test_two_modes():
    n = [5, 7]
    xt = rand_tt(n, r=3, seed=4)
    y = cross(lambda i: eval_tt_dense(xt.cores, i), n, eps=1e-11, nswp=10, seed=0)
    assert (y - xt).norm() / xt.norm() < 1e-10
    assert list(y.n) == n


def test_one_mode_matches_the_whole_fiber():
    fun = lambda i: np.cos(np.asarray(i)[:, 0].astype(float))
    y = cross(fun, [9], eps=1e-12)
    assert y.d == 1
    assert np.allclose(np.asarray(y.full()).reshape(-1), np.cos(np.arange(9.0)))
    assert y.history.fun_eval == 9        # exactly one evaluation per entry


def test_mode_of_size_one():
    n = [3, 1, 4]
    fun = lambda i: 1.0 + np.asarray(i)[:, 0] + 2.0 * np.asarray(i)[:, 2]
    y = cross(fun, n, eps=1e-11, seed=0)
    assert list(y.n) == n
    assert rel_err(y, dense_of(fun, n)) < 1e-10


def test_mode_sizes_all_different_and_a_non_symmetric_function():
    """Catches any transposition of the mode order: every mode enters differently."""
    n = [3, 4, 5, 6]

    def fun(i):
        i = np.asarray(i).astype(float)
        return ((1.0 + i[:, 0]) * np.exp(-i[:, 1] / 3.0)
                + np.cos(i[:, 2]) * (1.0 + i[:, 3] ** 2))

    y = cross(fun, n, eps=1e-11, nswp=12, seed=0)
    assert list(y.n) == n
    assert rel_err(y, dense_of(fun, n)) < 1e-10


def test_zero_tensor():
    n = [4] * 4
    y = cross(lambda i: np.zeros(np.asarray(i).shape[0]), n, eps=1e-8,
              n_check=50, seed=0)
    assert float(y.norm()) == 0.0
    assert y.history.converged is True
    assert y.history.err_check == 0.0


def test_rank_one_constant():
    n = [4] * 5
    y = cross(lambda i: np.full(np.asarray(i).shape[0], 3.0), n, eps=1e-11, seed=0)
    assert rel_err(y, np.full(tuple(n), 3.0)) < 1e-12
    assert max(y.history.ranks) == 1, f"a constant is rank 1, got {y.history.ranks}"


def test_x0_ranks_larger_than_the_grid_allows():
    n = [2] * 8
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float) @ 2.0 ** np.arange(8))
    y = rect_cross(fun, tt.rand(n, r=30), eps=1e-10, nswp=8)
    assert rel_err(y, dense_of(fun, n)) < 1e-9


# --- contract with the black box ---------------------------------------------

def test_fun_only_ever_sees_valid_indices():
    n = [4, 6, 3, 5]
    seen = {"bad": None, "dtype": set()}

    def fun(idx):
        idx = np.asarray(idx)
        seen["dtype"].add(idx.dtype.kind)
        if idx.ndim != 2 or idx.shape[1] != len(n):
            seen["bad"] = f"shape {idx.shape}"
        elif idx.min() < 0 or np.any(idx.max(axis=0) >= np.array(n)):
            seen["bad"] = f"out of range {idx.max(axis=0)}"
        return 1.0 / (1.0 + idx.astype(float).sum(axis=1))

    cross(fun, n, eps=1e-10, nswp=8, seed=0)
    assert seen["bad"] is None, seen["bad"]
    assert seen["dtype"] == {"i"}, f"indices must stay integral, got {seen['dtype']}"


def test_scalar_return_is_rejected_even_for_a_batch_of_one():
    """A non-vectorized ``fun`` used to slip through when every mode had size 1."""
    with pytest.raises(ValueError, match="vectorized"):
        cross(lambda idx: 1.0, [1, 1, 1])
    with pytest.raises(ValueError, match="vectorized"):
        cross(lambda idx: np.float64(2.0), [1, 2, 1])


def test_wrong_length_return_is_rejected():
    with pytest.raises(ValueError, match="vectorized"):
        cross(lambda idx: np.zeros(np.asarray(idx).shape[0] + 1), [3, 3, 3])


def test_nan_anywhere_is_rejected_not_absorbed():
    n = [4, 4, 4]

    def fun(idx):
        idx = np.asarray(idx)
        v = 1.0 + idx.astype(float).sum(axis=1)
        v[(idx[:, 0] == 3) & (idx[:, 1] == 3)] = np.nan
        return v

    with pytest.raises(ValueError, match="non-finite"):
        cross(fun, n, eps=1e-10, nswp=5, seed=0)


def test_run_is_reproducible_for_a_fixed_seed():
    n = [5] * 5
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    a = cross(fun, n, eps=1e-8, seed=1)
    b = cross(fun, n, eps=1e-8, seed=1)
    # bitwise, on the cores: ``(a - b).norm()`` would only prove that the QR
    # sweep of the difference rounds to 1e-15, which proves nothing.
    assert [np.asarray(c).shape for c in a.cores] == \
           [np.asarray(c).shape for c in b.cores]
    for ca, cb in zip(a.cores, b.cores):
        assert np.array_equal(np.asarray(ca), np.asarray(cb))
    assert a.history.fun_eval == b.history.fun_eval


# --- history / explainability -------------------------------------------------

def test_silent_run_still_records_everything():
    n = [4] * 5
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        y = cross(fun, n, eps=1e-10, verbose=False, n_check=100, seed=0)
    assert buf.getvalue() == "", "verbose=False must not print"
    h = y.history
    assert len(h.sweeps) >= 1
    assert h.fun_eval > 0 and h.fun_eval_check == 100
    assert h.time > 0.0
    assert np.isfinite(h.err_rel) and np.isfinite(h.err_round)
    assert h.err_round <= max(10 * h.eps, 1e-12), (
        f"the final rounding cost {h.err_round:.2e} at eps={h.eps:.0e}")
    assert list(h.ranks) == [int(v) for v in y.r]
    assert h.rmax_active is False

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cross(fun, n, eps=1e-10, verbose=True, seed=0)
    assert "err_rel" in buf.getvalue()


def test_fun_eval_counts_every_call_and_nothing_else():
    n = [4] * 5
    box = {"n": 0}

    def fun(idx):
        box["n"] += np.asarray(idx).shape[0]
        return 1.0 / (1.0 + np.asarray(idx).astype(float).sum(axis=1))

    y = cross(fun, n, eps=1e-8, n_check=123, seed=0)
    assert box["n"] == y.history.fun_eval + y.history.fun_eval_check
    assert y.history.fun_eval_check == 123


# --- the measurement must not be quieter than the guess -----------------------

def test_held_out_measurement_that_contradicts_eps_is_reported():
    """converged=True, ranks free, nothing looks wrong -- and the answer is 100% wrong.

    ``f = 1`` on the upper corner block and 0 elsewhere: every fiber the cross
    starts from is identically zero, so it returns the zero tensor, the change
    between sweeps is exactly 0 and the run "converges" in two sweeps.  The
    2000 held-out points the caller paid for measure a relative error of 1.0.
    Before the fix this run produced *no warning at all*: the one number that
    was actually measured against ``fun`` was recorded and never surfaced.
    """
    n = [8] * 4
    fun = lambda i: (np.all(np.asarray(i) >= 4, axis=1)).astype(float)
    with pytest.warns(RuntimeWarning, match="held-out points"):
        y = cross(fun, n, eps=1e-8, nswp=10, n_check=2000, seed=0)
    h = y.history
    assert h.converged is True          # the indicator really does say "fine"
    assert h.err_rel == 0.0
    assert h.err_check == pytest.approx(1.0)
    assert rel_err(y, dense_of(fun, n)) == pytest.approx(1.0)


def test_a_good_run_with_a_check_stays_quiet():
    """The warning above is worthless if it fires on a run that met its eps."""
    n = [5] * 5
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = cross(fun, n, eps=1e-6, nswp=20, n_check=500, seed=0)
    assert y.history.err_check < 1e-6
    assert rel_err(y, dense_of(fun, n)) < 3e-6


# --- argument validation ------------------------------------------------------

@pytest.mark.parametrize("rmax", [0, -1])
def test_absurd_rmax_is_rejected_not_reinterpreted(rmax):
    """``rmax=0`` used to mean "no cap" (falsy) and ``rmax=-1`` meant rank 1.

    Measured before the fix: ``rmax=0`` returned ranks [1,5,8,5,1] (uncapped),
    ``rmax=-3`` returned the rank-1 tensor at a 6.0e-1 relative error.  Both are
    the library guessing what the caller meant.
    """
    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    with pytest.raises(ValueError, match="rmax"):
        cross(fun, [5] * 4, eps=1e-10, rmax=rmax)


def test_element_rejects_a_negative_index():
    """numpy would wrap it around and hand back the value from the far end."""
    x = rand_tt([3, 4, 5], r=2, seed=8)
    with pytest.raises(ValueError, match="negative"):
        element(x, np.array([[-1, 0, 0]]))
    with pytest.raises(IndexError):
        element(x, np.array([[3, 0, 0]]))


# --- one-mode runs keep the same history contract as every other run ----------

def test_d1_run_honours_n_check_and_records_a_sweep():
    """A d=1 run used to return a differently-shaped history: no sweep entry and
    ``n_check`` silently ignored (``err_check=None`` after paying for it)."""
    fun = lambda i: np.cos(np.asarray(i)[:, 0].astype(float))
    y = cross(fun, [9], eps=1e-12, n_check=5)
    h = y.history
    assert h.err_check == pytest.approx(0.0, abs=1e-14)
    assert h.fun_eval_check == 5
    assert h.fun_eval == 9              # the whole fiber, once
    assert len(h.sweeps) == 1
    assert h.sweeps[0]["fun_eval"] == 9
    assert h.converged is True
    assert h.rmax_active is False
    assert np.allclose(np.asarray(y.full()).reshape(-1), np.cos(np.arange(9.0)))


def test_d1_with_kickrank_zero_does_not_cry_wolf():
    """There is no rank adaptation to switch off when there is one mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cross(lambda i: np.asarray(i)[:, 0].astype(float), [6], kickrank=0)


# --- the stopping criterion the caller asked for ------------------------------

def test_stop_fun_replaces_the_criterion_and_is_obeyed():
    seen = []

    def stop(prev, new):
        seen.append((float(prev.norm()), float(new.norm())))
        return len(seen) >= 2

    fun = lambda i: 1.0 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    y = rect_cross(fun, tt.rand([5] * 4, r=2), eps=1e-14, nswp=10, stop_fun=stop)
    assert len(seen) == 2, "stop_fun must be consulted once per sweep"
    assert len(y.history.sweeps) == 2, "and it must actually stop the loop"
    assert y.history.converged is True


def test_eps_abs_can_stop_a_run_that_eps_cannot():
    """A tiny tensor: the relative criterion is unreachable, the absolute one is."""
    fun = lambda i: 1e-12 / (1.0 + np.asarray(i).astype(float).sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = rect_cross(fun, tt.rand([5] * 4, r=2), eps=1e-16, eps_abs=1e-10,
                       nswp=10)
    assert y.history.converged is True
    assert len(y.history.sweeps) < 10


def test_round_result_false_keeps_the_exploration_ranks():
    """The ranks the interpolant really has, before the cosmetic rounding."""
    n = [5] * 5
    xt = rand_tt(n, r=3, seed=21)
    fun = lambda i: eval_tt_dense(xt.cores, i)
    raw = cross(fun, n, eps=1e-11, nswp=8, seed=0, round_result=False)
    rounded = cross(fun, n, eps=1e-11, nswp=8, seed=0, round_result=True)
    assert max(raw.history.ranks) > max(rounded.history.ranks)
    assert raw.history.err_round == 0.0
    assert rounded.history.err_round <= 1e-11 * 10
    # both must represent the same tensor to the requested accuracy
    for y in (raw, rounded):
        assert float((y - xt).norm() / xt.norm()) < 1e-10


def test_evaluations_stay_far_below_the_grid_in_high_dimension():
    """The whole point of the method, measured: d=20 QTT grid of 2^20 entries."""
    d = 20
    w = 2.0 ** np.arange(d) % 7

    def fun(idx):
        return 1.0 / (1.0 + np.asarray(idx).astype(float) @ w)

    y = cross(fun, [2] * d, eps=1e-8, nswp=15, seed=0)
    h = y.history
    assert h.converged
    assert h.fun_eval < 0.2 * 2 ** d, f"{h.fun_eval} evaluations of 2^20"
    # O(d n r^2) is the claim; check the constant is not absurd
    r = max(h.ranks)
    assert h.fun_eval < 40 * d * 2 * r ** 2, f"{h.fun_eval} vs d n r^2 = {d * 2 * r**2}"
