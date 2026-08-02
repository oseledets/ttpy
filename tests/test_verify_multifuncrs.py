"""Adversarial verification of :mod:`tt.algs.multifuncrs`.

Everything here is checked against a dense array built from the *rule* that
defines the input tensor, never against ``x.full()`` of the tensor the method
was given and never against the legacy implementation.  Two independent input
families are used so that a bug in one hand-written core list cannot hide:

* ``sum_tensor``:   ``x[i] = shift + sum_k t_k(i_k)``          (TT rank 2)
* ``prod_tensor``:  ``x[i] = prod_k (1 + t_k(i_k))``           (TT rank 1)

The rest of the oracles are invariants: exact reproduction of a linear
combination, rank caps, the exact count of user-function evaluations, and the
held-out measurement ``history.err_check`` cross-checked against the dense error
in a regime where the two can actually disagree.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.cross import CrossHistory
from tt.algs.multifuncrs import multifuncrs, multifuncrs2


# --- input families ----------------------------------------------------------

def sum_tensor(n, shift=2.0, seed=0):
    """``x[i] = shift + sum_k t_k(i_k)``: TT cores by hand, dense by rule."""
    n = list(n)
    d = len(n)
    rng = np.random.default_rng(seed)
    t = [rng.uniform(0.2, 1.0, size=nk) for nk in n]

    dense = np.full(n, float(shift))
    for k in range(d):
        shape = [1] * d
        shape[k] = n[k]
        dense = dense + t[k].reshape(shape)

    if d == 1:
        return tt.vector.from_list([(shift + t[0]).reshape((1, n[0], 1))]), dense
    cores = []
    c0 = np.zeros((1, n[0], 2))
    c0[0, :, 0] = t[0]
    c0[0, :, 1] = 1.0
    cores.append(c0)
    for k in range(1, d - 1):
        ck = np.zeros((2, n[k], 2))
        ck[0, :, 0] = 1.0
        ck[1, :, 0] = t[k]
        ck[1, :, 1] = 1.0
        cores.append(ck)
    cl = np.zeros((2, n[d - 1], 1))
    cl[0, :, 0] = 1.0
    cl[1, :, 0] = shift + t[d - 1]
    cores.append(cl)
    return tt.vector.from_list(cores), dense


def prod_tensor(n, seed=0):
    """``x[i] = prod_k (1 + t_k(i_k))``: rank 1, an independent second family."""
    n = list(n)
    d = len(n)
    rng = np.random.default_rng(seed)
    t = [1.0 + rng.uniform(0.2, 1.0, size=nk) for nk in n]
    dense = np.ones(n)
    for k in range(d):
        shape = [1] * d
        shape[k] = n[k]
        dense = dense * t[k].reshape(shape)
    cores = [t[k].reshape((1, n[k], 1)) for k in range(d)]
    return tt.vector.from_list(cores), dense


def rel_err(approx, exact):
    approx = np.asarray(tt.backend.to_numpy(approx))
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def component(y, j):
    cores = list(y.cores)
    return tt.vector.from_list(cores[:-1] + [cores[-1][:, :, j:j + 1]])


# --- dimension edge cases ----------------------------------------------------

@pytest.mark.parametrize("n", [[7], [5, 7], [3, 5, 4, 2, 6]])
def test_edge_dimensions_and_ragged_modes(n):
    """d=1, d=2 and mode sizes that differ per mode, against dense truth."""
    eps = 1e-10
    x, dense = sum_tensor(n, shift=2.0, seed=3)
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
    assert [int(v) for v in y.n] == n
    err = rel_err(y.full(), 1.0 / dense)
    assert err <= 10 * eps, f"n={n}: relative error {err:.3e}"


def test_d1_with_vector_valued_funs():
    """A single mode plus the artificial component mode: the fold must survive."""
    x, dense = sum_tensor([9], shift=2.0, seed=4)
    y = multifuncrs2([x], lambda v: np.stack([np.sqrt(v[:, 0]),
                                              np.log(v[:, 0])], axis=1),
                     eps=1e-10, verb=0)
    assert y.d == 1 and int(y.r[-1]) == 2
    assert rel_err(component(y, 0).full(), np.sqrt(dense)) < 1e-12
    assert rel_err(component(y, 1).full(), np.log(dense)) < 1e-12


def test_two_inputs_with_ragged_modes():
    """Two different tensors on a non-cubic grid, non-separable function."""
    n = [4, 7, 3, 5]
    eps = 1e-10
    a, ad = sum_tensor(n, shift=2.0, seed=5)
    b, bd = prod_tensor(n, seed=6)
    z = multifuncrs2([a, b], lambda v: np.exp(-v[:, 0]) / (1.0 + v[:, 1] ** 2),
                     eps=eps, verb=0)
    err = rel_err(z.full(), np.exp(-ad) / (1.0 + bd ** 2))
    assert err <= 10 * eps, f"relative error {err:.3e}"


def test_sample_columns_follow_the_input_order():
    """``V[:, j]`` must be ``X_j`` -- a permutation here would be invisible in
    any symmetric test function."""
    n = [5, 6, 7]
    a, ad = sum_tensor(n, shift=2.0, seed=7)
    b, bd = sum_tensor(n, shift=9.0, seed=8)
    assert abs(ad - bd).min() > 1.0
    pick0 = multifuncrs2([a, b], lambda v: v[:, 0], eps=1e-12, verb=0)
    pick1 = multifuncrs2([a, b], lambda v: v[:, 1], eps=1e-12, verb=0)
    assert rel_err(pick0.full(), ad) < 1e-13
    assert rel_err(pick1.full(), bd) < 1e-13


def test_rank_one_and_zero_inputs():
    """Degenerate inputs: a constant tensor and the exact zero tensor."""
    const = tt.ones([4, 4, 4]) * 3.0
    y = multifuncrs2([const], lambda v: np.sqrt(v[:, 0]), eps=1e-12, verb=0)
    assert rel_err(y.full(), np.full((4, 4, 4), np.sqrt(3.0))) < 1e-14
    assert max(int(v) for v in y.r) == 1

    z = multifuncrs2([tt.zeros([4, 4, 4])], lambda v: np.exp(v[:, 0]),
                     eps=1e-12, verb=0)
    assert rel_err(z.full(), np.ones((4, 4, 4))) < 1e-14


def test_many_inputs_many_components():
    """p=4 inputs, d2=5 components of very different magnitude.

    ``eps`` is a budget for the *stacked* block tensor, exactly as in the legacy
    code, so it is the joint error that must be under 10*eps.  A component whose
    norm is a fraction ``w`` of the joint norm can only be expected to reach
    ``10*eps/w`` in relative terms -- measured here: component 2 (``1/X_0``,
    0.7% of the joint norm) comes out at 1.2e-08 for eps=1e-09, i.e. 12*eps
    globally but 0.09 of its own budget.  Anyone who needs per-component
    relative accuracy must call the method once per component.
    """
    n = [5] * 4
    eps = 1e-9
    xs, ds = [], []
    for i in range(4):
        xi, di = sum_tensor(n, shift=2.0 + i, seed=10 + i)
        xs.append(xi)
        ds.append(di)

    def f(v):
        return np.stack([v[:, 0] + v[:, 1], v[:, 2] * v[:, 3], 1.0 / v[:, 0],
                         np.exp(-v[:, 1]), np.sqrt(v[:, 2] + v[:, 3])], axis=1)

    y = multifuncrs2(xs, f, eps=eps, verb=0)
    assert int(y.r[-1]) == 5
    truth = [ds[0] + ds[1], ds[2] * ds[3], 1.0 / ds[0], np.exp(-ds[1]),
             np.sqrt(ds[2] + ds[3])]
    num = den = 0.0
    for j, ex in enumerate(truth):
        got = np.asarray(tt.backend.to_numpy(component(y, j).full()))
        num += float(np.linalg.norm(got - ex)) ** 2
        den += float(np.linalg.norm(ex)) ** 2
    joint = np.sqrt(num / den)
    assert joint <= 10 * eps, f"joint relative error {joint:.3e}"
    for j, ex in enumerate(truth):
        w = float(np.linalg.norm(ex)) / np.sqrt(den)
        err = rel_err(component(y, j).full(), ex)
        assert err <= 10 * eps / w, (
            f"component {j}: {err:.3e} > {10 * eps / w:.3e} "
            f"(its share of the norm is {w:.4f})")


# --- complex -----------------------------------------------------------------

def test_complex_input_tensor():
    """The inputs themselves are complex, not just the output of funs."""
    eps = 1e-10
    x, dense = sum_tensor([5, 5, 5], shift=2.0, seed=11)
    cx = tt.vector.from_list([c.astype(np.complex128) for c in x.cores]) * (1 + 0.5j)
    cd = dense * (1 + 0.5j)
    assert cx.is_complex
    y = multifuncrs2([cx], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
    assert y.is_complex
    assert rel_err(y.full(), 1.0 / cd) <= 10 * eps


def test_funs_becomes_complex_after_the_probe():
    """The probe sees real values, the sweep does not: the imaginary part must
    not be silently dropped."""
    x, dense = sum_tensor([6] * 4, shift=2.0, seed=12)
    cut = np.sort(dense.reshape(-1))[3]      # 3 entries end up negative
    eps = 1e-8
    y = multifuncrs2([x], lambda v: np.emath.sqrt(v[:, 0] - cut), eps=eps,
                     verb=0)
    assert y.is_complex, "a complex answer came back as a real tensor"
    exact = np.emath.sqrt(dense - cut)
    assert np.iscomplexobj(exact) and np.abs(exact.imag).max() > 0
    err = rel_err(y.full(), exact)
    assert err <= 1e-6, f"relative error {err:.3e}"


# --- rank control and argument validation ------------------------------------

@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_rmax_is_never_exceeded(cap):
    x, dense = sum_tensor([6] * 5, shift=2.0, seed=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-14, rmax=cap,
                         verb=0)
    assert max(int(v) for v in y.r) <= cap, f"ranks {list(y.r)} for rmax={cap}"
    assert np.isfinite(rel_err(y.full(), 1.0 / dense))


@pytest.mark.parametrize("kw", [{"rmax": 0}, {"rmax": -3}, {"r0": 0},
                                {"r0": -1}])
def test_meaningless_rank_arguments_raise(kw):
    """rmax=0 used to be read as 'no cap' and r0=0 crashed inside maxvol."""
    x, _ = sum_tensor([5, 5, 5], seed=14)
    with pytest.raises(ValueError, match="rmax|r0"):
        multifuncrs2([x], lambda v: v[:, 0], eps=1e-6, verb=0, **kw)


def test_rmax_none_means_no_cap():
    x, dense = sum_tensor([6] * 4, shift=2.0, seed=15)
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-12, rmax=None, verb=0)
    assert rel_err(y.full(), 1.0 / dense) <= 1e-11


def test_y0_with_a_wrong_component_tail_raises():
    x, _ = sum_tensor([6] * 4, shift=2.0, seed=16)
    two = multifuncrs2([x], lambda v: np.stack([v[:, 0], 1.0 / v[:, 0]], 1),
                       eps=1e-6, verb=0)
    assert int(two.r[-1]) == 2
    with pytest.raises(ValueError, match="last rank 2"):
        multifuncrs2([x], lambda v: np.stack(
            [v[:, 0], 1.0 / v[:, 0], v[:, 0] ** 2], 1),
            eps=1e-6, y0=two, verb=0)


def test_y0_carrying_the_components_is_accepted():
    """The ``tail == d2`` branch (an identity core is appended) is live code."""
    x, dense = sum_tensor([5, 5, 5], shift=2.0, seed=17)

    def f(v):
        return np.stack([np.sqrt(v[:, 0]), np.log(v[:, 0])], axis=1)

    coarse = multifuncrs2([x], f, eps=1e-4, verb=0)
    assert int(coarse.r[-1]) == 2
    fine = multifuncrs2([x], f, eps=1e-11, y0=coarse, verb=0)
    assert rel_err(component(fine, 0).full(), np.sqrt(dense)) <= 1e-10
    assert rel_err(component(fine, 1).full(), np.log(dense)) <= 1e-10


# --- bookkeeping -------------------------------------------------------------

def test_funs_counters_are_exact():
    """history.funs_calls / funs_values must be the truth, probes and the
    held-out check included."""
    x, _ = sum_tensor([6] * 5, shift=2.0, seed=18)
    seen = {"calls": 0, "values": 0}

    def counted(v):
        seen["calls"] += 1
        seen["values"] += v.shape[0]
        return 1.0 / v[:, 0]

    y = multifuncrs2([x], counted, eps=1e-8, verb=0, n_check=100)
    assert y.history.funs_calls == seen["calls"]
    assert y.history.funs_values == seen["values"]
    assert y.history.probe_values > 0
    assert y.history.ignored_options == []


def test_verb0_is_silent_even_when_the_engine_complains(capsys):
    x, _ = sum_tensor([6] * 5, shift=2.0, seed=19)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-14, nswp=1,
                         kickrank=0, r0=1, verb=0)
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err == ""
    assert not y.history.converged
    assert len(y.history.sweeps) == 1
    assert np.isfinite(y.history.err_rel)


def test_d1_documents_that_err_check_is_not_measured():
    """d=1 returns the exact tensor and the engine skips the held-out check.

    The docstring says so; this pins code and documentation together.  If the
    engine ever starts measuring it, the docstring must be updated with it.
    """
    x, _ = sum_tensor([9], shift=2.0, seed=20)
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-10, verb=0,
                     n_check=50)
    assert y.history.err_check is None
    assert "err_check" in multifuncrs2.__doc__ and "None" in multifuncrs2.__doc__


# --- the zero-collapse guard (unreachable from a natural funs) ---------------

def _fake_zero_cross(monkeypatch, seen):
    """Replace the engine by one that always returns the zero tensor."""
    def fake(fun, x0, **kw):
        seen.append(x0)
        y = tt.zeros([int(v) for v in x0.n])
        y.history = CrossHistory(eps=kw.get("eps", 0.0))
        y.history.converged = True
        return y
    monkeypatch.setattr("tt.algs.multifuncrs.rect_cross", fake)


def test_collapse_to_zero_raises_when_funs_is_not_zero(monkeypatch):
    seen = []
    _fake_zero_cross(monkeypatch, seen)
    x, _ = sum_tensor([5, 5, 5], shift=2.0, seed=21)
    with pytest.raises(ValueError, match="collapsed to the zero tensor"):
        multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-6, verb=0,
                     restart_it=2)
    assert len(seen) == 3, f"restart_it=2 must give 3 attempts, got {len(seen)}"


def test_restart_draws_a_fresh_guess_even_when_y0_is_given(monkeypatch):
    """A restart that replays y0 is a wasted attempt: the sweep is
    deterministic, so the second run would return the same zero tensor."""
    x, _ = sum_tensor([5, 5, 5], shift=2.0, seed=22)
    y0 = tt.rand([5, 5, 5], r=2)
    seen = []
    _fake_zero_cross(monkeypatch, seen)
    with pytest.raises(ValueError, match="collapsed"):
        multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-6, verb=0, y0=y0,
                     restart_it=2)
    assert len(seen) == 3
    flat = [np.concatenate([np.asarray(c).reshape(-1) for c in g.cores])
            for g in seen]
    assert np.allclose(flat[0], np.concatenate(
        [np.asarray(c).reshape(-1) for c in y0.cores]))
    assert not np.array_equal(flat[1], flat[0])
    assert not np.array_equal(flat[2], flat[1])


def test_collapse_to_zero_with_a_zero_funs_only_warns(monkeypatch):
    seen = []
    _fake_zero_cross(monkeypatch, seen)
    x, _ = sum_tensor([5, 5, 5], shift=2.0, seed=23)
    with pytest.warns(RuntimeWarning, match="exactly zero"):
        y = multifuncrs2([x], lambda v: 0.0 * v[:, 0], eps=1e-6, verb=0,
                         restart_it=2)
    assert float(y.norm()) == 0.0
    assert len(seen) == 1, "a genuinely zero function must not be retried"


# --- accuracy regimes: what eps actually buys --------------------------------

@pytest.mark.parametrize("d,eps,factor", [(10, 1e-6, 10.0), (10, 1e-10, 10.0),
                                          (20, 1e-6, 10.0), (20, 1e-10, 50.0)])
def test_qtt_reciprocal_held_out_accuracy(d, eps, factor):
    """1/(1+t) on a 2^d binary QTT grid, error on 2000 held-out points.

    ``factor`` is the tolerance the module docstring claims for this regime:
    within 10*eps up to d=20 at eps=1e-6, and 19*eps (measured) at d=20,
    eps=1e-10.  float64 throughout.
    """
    x = tt.xfun(2, d) * (1.0 / 2 ** d) + tt.ones(2, d) * (1.0 / 2 ** d)
    y = multifuncrs2([x], lambda v: 1.0 / (1.0 + v[:, 0]), eps=eps, verb=0,
                     n_check=2000)
    assert y.history.converged
    assert y.history.err_check is not None
    assert y.history.err_check <= factor * eps, (
        f"d={d}, eps={eps:.0e}: measured {y.history.err_check:.3e}")


def test_qtt_small_d_against_the_dense_grid():
    """d=12: cheap enough to compare with all 4096 entries, and it checks the
    index convention (mode 1 is the fastest index) at the same time."""
    d, eps = 12, 1e-9
    grid = (np.arange(2 ** d) + 1.0) / 2 ** d
    x = tt.xfun(2, d) * (1.0 / 2 ** d) + tt.ones(2, d) * (1.0 / 2 ** d)
    y = multifuncrs2([x], lambda v: 1.0 / (1.0 + v[:, 0]), eps=eps, verb=0)
    got = np.asarray(y.full(asvector=True)).reshape(-1)
    assert rel_err(got, 1.0 / (1.0 + grid)) <= 10 * eps


def test_high_d_reports_converged_and_the_module_says_eps_is_no_bound():
    """d=40: the run reports converged and issues no warning, and the achieved
    accuracy is *not* tied to eps.

    Measured on two truncation rules of :func:`tt.algs.cross.rect_cross`:
    err/eps = 6.4e3 (eps/sqrt(d) local truncation) and 0.11 (rank-capped
    truncation) at eps=1e-10 -- a 5e4 spread for the same call.  So the only
    thing a caller may rely on is the measurement, and the module docstring must
    say so; that coupling is what this test protects.
    """
    d, eps = 40, 1e-10
    x = tt.xfun(2, d) * (1.0 / 2 ** d) + tt.ones(2, d) * (1.0 / 2 ** d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = multifuncrs2([x], lambda v: 1.0 / (1.0 + v[:, 0]), eps=eps, verb=0,
                         n_check=2000)
    assert not [w for w in caught if "did not reach" in str(w.message)], (
        "the run claims convergence, so it must not also warn about it")
    assert y.history.converged
    assert y.history.err_check is not None
    assert y.history.err_check <= 1e-5, (
        f"d=40 is far worse than ever measured: {y.history.err_check:.3e}")
    import tt.algs.multifuncrs as m
    assert "*bound*" in m.__doc__, "the module must say eps is not a bound"
    assert "n_check" in m.__doc__ and "err_check" in multifuncrs2.__doc__


# --- loud failure ------------------------------------------------------------

def test_non_finite_only_in_the_sweep_names_the_index():
    """The probe misses it; the engine must still refuse, and say where."""
    x, dense = sum_tensor([6] * 5, shift=2.0, seed=24)
    top = dense.max()
    with np.errstate(divide="ignore"):
        with pytest.raises(ValueError, match="non-finite value.*at multi-index"):
            multifuncrs2([x], lambda v: 1.0 / (v[:, 0] - top), eps=1e-8, verb=0)


@pytest.mark.parametrize("f,msg", [
    (lambda v: np.float64(1.0), "shape"),
    (lambda v: np.zeros((v.shape[0] + 1,)), "vectorized"),
    (lambda v: np.zeros((v.shape[0], 2, 2)), "shape"),
])
def test_malformed_funs_output_raises(f, msg):
    x, _ = sum_tensor([6] * 4, shift=2.0, seed=25)
    with pytest.raises(ValueError, match=msg):
        multifuncrs2([x], f, eps=1e-6, verb=0)


def test_matrix_in_the_input_list_raises():
    x, _ = sum_tensor([6] * 5, shift=2.0, seed=26)
    with pytest.raises(TypeError, match="expected a tt.vector"):
        multifuncrs2([x, tt.eye([6] * 5)], lambda v: v[:, 0], eps=1e-6, verb=0)


def test_no_bare_except_or_assert_in_the_module():
    """R2/R4: no swallowed exception, no assert used for input validation."""
    import inspect

    import tt.algs.multifuncrs as m
    src = inspect.getsource(m)
    assert "except" not in src, "the module must not catch anything"
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("assert "), f"assert as validation: {line}"


# --- legacy entry point ------------------------------------------------------

def test_multifuncrs_accepts_the_legacy_positional_call():
    """multifuncrs(X, funs, eps, nswp, kickrank, y0, rmax, verb) positionally."""
    x, dense = sum_tensor([6] * 4, shift=2.0, seed=27)
    y = multifuncrs([x], lambda v: 1.0 / v[:, 0], 1e-10, 10, 5, None, 999999, 0)
    assert rel_err(y.full(), 1.0 / dense) <= 1e-9


def test_multifuncrs2_accepts_the_legacy_positional_call():
    x, dense = sum_tensor([6] * 4, shift=2.0, seed=28)
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], 1e-10, 10, 5, None, 999999,
                     0, False, 0)
    assert rel_err(y.full(), 1.0 / dense) <= 1e-9


# --- torch -------------------------------------------------------------------

def _to_torch(x, torch, dev):
    return tt.vector.from_list(
        [torch.as_tensor(np.asarray(c), device=dev, dtype=torch.float64)
         for c in x.cores])


def test_torch_y0_does_not_crash():
    """Regression: ``[c.copy() for c in y0.cores]`` raised AttributeError on a
    torch tensor, so every torch run with an initial guess died."""
    torch = pytest.importorskip("torch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, dense = sum_tensor([5] * 4, shift=2.0, seed=29)
    xt = _to_torch(x, torch, dev)
    coarse = multifuncrs2([xt], lambda v: 1.0 / v[:, 0], eps=1e-3, verb=0)
    fine = multifuncrs2([xt], lambda v: 1.0 / v[:, 0], eps=1e-11, y0=coarse,
                        verb=0)
    assert fine.backend.name == "torch"
    assert rel_err(fine.full(), 1.0 / dense) <= 1e-10


def test_torch_vector_valued_and_complex():
    torch = pytest.importorskip("torch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, dense = sum_tensor([5] * 4, shift=2.0, seed=30)
    xt = _to_torch(x, torch, dev)
    eps = 1e-10
    y = multifuncrs2([xt], lambda v: np.stack([np.sqrt(v[:, 0]),
                                               np.log(v[:, 0])], axis=1),
                     eps=eps, verb=0)
    assert y.backend.name == "torch" and int(y.r[-1]) == 2
    assert rel_err(component(y, 0).full(), np.sqrt(dense)) <= 10 * eps
    assert rel_err(component(y, 1).full(), np.log(dense)) <= 10 * eps

    z = multifuncrs2([xt], lambda v: np.exp(1j * v[:, 0]), eps=eps, verb=0)
    assert z.backend.name == "torch" and z.is_complex
    assert rel_err(z.full(), np.exp(1j * dense)) <= 10 * eps
