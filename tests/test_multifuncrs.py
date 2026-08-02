"""multifuncrs / multifuncrs2 checked against dense elementwise truth.

The oracle is always a full numpy array built independently of the TT machinery:
the test defines ``x`` by the *rule* ``x[i] = shift + sum_k t_k(i_k)``, builds
the dense array from that rule with numpy broadcasting, and separately builds
the TT cores of the same rule by hand.  ``funs`` applied to the dense array is
the ground truth; nothing here consults the legacy implementation, and the
dense truth never goes through ``x.full()``.

Secondary oracles: rank bounds (``rmax``), the recorded history, and the
measured held-out error ``history.err_check``.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.multifuncrs import MultifuncrsHistory, multifuncrs, multifuncrs2


# --- the test tensor ---------------------------------------------------------

def make_sum_tensor(n, shift=2.0, seed=0):
    """A rank-2 TT and its dense twin for ``x[i] = shift + sum_k t_k(i_k)``.

    Returns:
        (tt.vector, dense ndarray of shape ``n``, list of the ``t_k``).

    The TT cores are written out by hand from the same ``t_k`` that build the
    dense array, so the two are the same mathematical object arrived at by two
    independent routes.
    """
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
        cores = [(shift + t[0]).reshape((1, n[0], 1))]
        return tt.vector.from_list(cores), dense, t

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
    return tt.vector.from_list(cores), dense, t


def rel_err(approx, exact):
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


N = [6, 6, 6, 6, 6]


@pytest.fixture(scope="module")
def xdata():
    return make_sum_tensor(N, shift=2.0, seed=1)


@pytest.fixture(scope="module")
def ydata():
    return make_sum_tensor(N, shift=1.0, seed=2)


def component(y, j):
    """Component ``j`` of a block-TT answer (last TT rank = number of components)."""
    cores = list(y.cores)
    return tt.vector.from_list(cores[:-1] + [cores[-1][:, :, j:j + 1]])


# --- one-argument functions against the dense truth --------------------------

@pytest.mark.parametrize("name,f,eps", [
    ("exp", lambda v: np.exp(v), 1e-8),
    ("inv", lambda v: 1.0 / v, 1e-8),
    ("sqrt", lambda v: np.sqrt(v), 1e-8),
])
def test_single_argument_matches_dense(xdata, name, f, eps):
    x, dense, _ = xdata
    assert dense.min() > 2.0, "the test grid must stay well away from zero"
    y = multifuncrs2([x], lambda v: f(v[:, 0]), eps=eps, verb=0)
    err = rel_err(y.full(), f(dense))
    assert err <= 10 * eps, f"{name}: relative error {err:.3e} > {10 * eps:.1e}"
    assert y.history.converged, f"{name}: {y.history}"


def test_two_arguments_matches_dense(xdata, ydata):
    """x / (1 + y^2): two inputs, genuinely non-separable."""
    x, xd, _ = xdata
    y, yd, _ = ydata
    eps = 1e-8
    z = multifuncrs2([x, y], lambda v: v[:, 0] / (1.0 + v[:, 1] ** 2),
                     eps=eps, verb=0)
    exact = xd / (1.0 + yd ** 2)
    err = rel_err(z.full(), exact)
    assert err <= 10 * eps, f"relative error {err:.3e}"


def test_three_arguments_linear_combination(xdata, ydata):
    """A linear combination is exactly reproducible: an invariant, not a tolerance."""
    x, xd, _ = xdata
    y, yd, _ = ydata
    z = multifuncrs2([x, y, x], lambda v: 2 * v[:, 0] - 3 * v[:, 1] + v[:, 2],
                     eps=1e-12, verb=0)
    exact = 2 * xd - 3 * yd + xd
    assert rel_err(z.full(), exact) < 1e-11


# --- accuracy control --------------------------------------------------------

def test_accuracy_improves_as_eps_tightens(xdata):
    """1/x has geometrically decaying TT ranks: tighter eps must buy accuracy."""
    x, dense, _ = xdata
    exact = 1.0 / dense
    errs, ranks = [], []
    for eps in (1e-3, 1e-6, 1e-9):
        y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
        errs.append(rel_err(y.full(), exact))
        ranks.append(int(max(y.r)))
        assert errs[-1] <= 10 * eps, f"eps={eps:.0e}: error {errs[-1]:.3e}"
    assert errs[1] < errs[0], f"errors {errs}"
    assert errs[2] < errs[1], f"errors {errs}"
    assert ranks[0] <= ranks[1] <= ranks[2], f"ranks {ranks}"


def test_rmax_is_respected_and_costs_accuracy(xdata):
    x, dense, _ = xdata
    exact = 1.0 / dense
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        capped = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-12,
                              rmax=3, verb=0)
        free = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-12,
                            rmax=30, verb=0)
    assert max(capped.r) <= 3, f"ranks {list(capped.r)} exceed rmax=3"
    assert max(free.r) <= 30
    e_capped, e_free = rel_err(capped.full(), exact), rel_err(free.full(), exact)
    assert e_free < e_capped, f"capped {e_capped:.3e}, free {e_free:.3e}"
    assert e_free < 1e-10, f"uncapped run only reached {e_free:.3e}"


def test_non_convergence_is_loud(xdata):
    """One sweep cannot reach 1e-12; the engine must say so, not stay quiet."""
    x, _, _ = xdata
    with pytest.warns(RuntimeWarning, match="did not reach"):
        y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-12, nswp=1,
                         kickrank=0, r0=1, verb=0)
    assert not y.history.converged


def test_err_check_is_measured(xdata):
    """err_check must track the real error, and be able to disagree.

    Measured at eps=1e-8 both numbers are ~1e-10, so an ``abs=1e-7`` comparison
    would pass for *any* err_check: the assertion has to be relative, and it has
    to be made in a regime where the error is large enough to be resolved.
    """
    x, dense, _ = xdata
    exact = 1.0 / dense ** 4          # ranks decay slowly: a loose eps hurts
    for eps in (1e-1, 1e-2, 1e-3):
        y = multifuncrs2([x], lambda v: 1.0 / v[:, 0] ** 4, eps=eps, verb=0,
                         n_check=3000)
        true = rel_err(y.full(), exact)
        chk = y.history.err_check
        assert chk is not None
        assert true > 1e-5, f"eps={eps:.0e}: error {true:.2e} too small to resolve"
        assert chk == pytest.approx(true, rel=0.25), (
            f"eps={eps:.0e}: held-out estimate {chk:.3e} vs dense truth "
            f"{true:.3e}")


# --- reporting ---------------------------------------------------------------

def test_verb0_prints_nothing_but_records_history(xdata, capsys):
    x, _, _ = xdata
    y = multifuncrs2([x], lambda v: np.exp(v[:, 0]), eps=1e-8, verb=0)
    out = capsys.readouterr()
    assert out.out == "", f"verb=0 printed {out.out!r}"
    assert out.err == "", f"verb=0 printed to stderr {out.err!r}"
    h = y.history
    assert isinstance(h, MultifuncrsHistory)
    assert len(h.sweeps) >= 1
    assert h.funs_calls > 0 and h.funs_values > 0
    assert h.probe_values > 0
    assert h.ranks == [int(v) for v in y.r]
    assert set(h.sweeps[0]) >= {"sweep", "err_rel", "erank", "max_rank",
                                "fun_eval", "time"}
    assert h.attempts == 1
    assert isinstance(repr(h), str)


def test_verb1_prints(xdata, capsys):
    x, _, _ = xdata
    multifuncrs2([x], lambda v: np.exp(v[:, 0]), eps=1e-6, verb=1)
    out = capsys.readouterr().out
    assert "multifuncrs2" in out and "sweep" in out


# --- vector-valued funs (the legacy d2 convention) ---------------------------

def test_vector_valued_funs(xdata):
    """funs -> (batch, 2): the answer carries the components in its last rank."""
    x, dense, _ = xdata
    eps = 1e-9

    def f(v):
        return np.stack([np.sqrt(v[:, 0]), np.log(v[:, 0])], axis=1)

    y = multifuncrs2([x], f, eps=eps, verb=0)
    assert y.r[-1] == 2, f"last rank {y.r[-1]}, expected 2"
    assert y.history.d2 == 2
    assert rel_err(component(y, 0).full(), np.sqrt(dense)) <= 10 * eps
    assert rel_err(component(y, 1).full(), np.log(dense)) <= 10 * eps


def test_vector_valued_funs_calls_are_deduplicated(xdata):
    """The component mode must not multiply the number of user evaluations."""
    x, _, _ = xdata
    calls = {"values": 0}

    def f(v):
        calls["values"] += v.shape[0]
        return np.stack([np.sqrt(v[:, 0]), np.log(v[:, 0])], axis=1)

    y = multifuncrs2([x], f, eps=1e-8, verb=0)
    scalar = y.history.cross.fun_eval  # scalar entries the engine asked for
    # ``< scalar`` alone would pass on a single de-duplicated row in the whole
    # run.  Measured here: 9910 scalar entries for 7408 user points, a ratio of
    # 1.34 out of the ideal 2 (the cross does not put every spatial index next
    # to both components in the same batch, so the ideal is not reachable).
    assert scalar >= 1.25 * calls["values"], (
        f"funs was called on {calls['values']} points for {scalar} scalar "
        "entries; de-duplication over the component mode is not working")
    assert y.history.funs_values == calls["values"]


def test_d2_mismatch_raises(xdata):
    x, _, _ = xdata
    with pytest.raises(ValueError, match="d2=3 was requested"):
        multifuncrs2([x], lambda v: np.stack([v[:, 0], v[:, 0]], axis=1),
                     eps=1e-6, d2=3, verb=0)


# --- the two entry points ----------------------------------------------------

def test_multifuncrs_and_multifuncrs2_agree(xdata):
    x, dense, _ = xdata
    eps = 1e-8
    a = multifuncrs([x], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
    b = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
    exact = 1.0 / dense
    assert rel_err(a.full(), exact) <= 10 * eps
    assert rel_err(b.full(), exact) <= 10 * eps
    assert (a - b).norm() / b.norm() <= 20 * eps


def test_exposed_on_the_package(xdata):
    assert tt.multifuncrs2 is multifuncrs2
    assert tt.multifuncrs is multifuncrs


# --- options -----------------------------------------------------------------

def test_initial_guess_is_used(xdata):
    x, dense, _ = xdata
    exact = 1.0 / dense
    y0 = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-4, verb=0)
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-10, y0=y0, verb=0)
    assert rel_err(y.full(), exact) <= 1e-9


def test_initial_guess_wrong_modes_raises(xdata):
    x, _, _ = xdata
    bad = tt.rand([6, 6, 6], r=2)
    with pytest.raises(ValueError, match="mode sizes"):
        multifuncrs2([x], lambda v: v[:, 0], eps=1e-6, y0=bad, verb=0)


def test_do_qr_is_recorded_as_ignored(xdata):
    x, _, _ = xdata
    y = multifuncrs2([x], lambda v: np.exp(v[:, 0]), eps=1e-6, verb=0,
                     do_qr=True)
    assert any("do_qr" in s for s in y.history.ignored_options)


@pytest.mark.parametrize("kw,msg", [
    ({"pcatype": "uchol"}, "pcatype"),
    ({"trunctype": "cheb"}, "trunctype"),
    ({"kicktype": "rand"}, "kicktype"),
    ({"kickrank2": 3}, "kickrank2"),
])
def test_unimplemented_legacy_options_raise(xdata, kw, msg):
    x, _, _ = xdata
    with pytest.raises(ValueError, match=msg):
        multifuncrs2([x], lambda v: v[:, 0], eps=1e-6, verb=0, **kw)


def test_eps_exit_controls_the_stopping_rule(xdata):
    x, _, _ = xdata
    y = multifuncrs2([x], lambda v: 1.0 / v[:, 0], eps=1e-10, eps_exit=1e-2,
                     verb=0)
    assert y.history.converged
    assert len(y.history.sweeps) <= 3


# --- complex ------------------------------------------------------------------

def test_complex_valued_funs(xdata):
    x, dense, _ = xdata
    eps = 1e-8
    y = multifuncrs2([x], lambda v: np.exp(1j * v[:, 0]), eps=eps, verb=0)
    assert y.is_complex
    assert rel_err(y.full(), np.exp(1j * dense)) <= 10 * eps


# --- loud failures -----------------------------------------------------------

def test_not_vectorized_funs_raises(xdata):
    x, _, _ = xdata
    with pytest.raises(ValueError, match="vectorized|returned"):
        multifuncrs2([x], lambda v: np.exp(v[0, 0]), eps=1e-6, verb=0)


def test_non_finite_funs_raises(xdata):
    x, dense, _ = xdata
    # 1/(x - x_max) is +-inf exactly at the maximum of the grid
    with pytest.raises(ValueError, match="non-finite"):
        multifuncrs2([x], lambda v: 1.0 / (v[:, 0] - dense.max()),
                     eps=1e-6, verb=0)


def test_mode_size_mismatch_raises(xdata):
    x, _, _ = xdata
    other = tt.rand([6, 6, 6, 6, 7], r=2)
    with pytest.raises(ValueError, match="identical grids"):
        multifuncrs2([x, other], lambda v: v[:, 0] * v[:, 1], eps=1e-6, verb=0)


def test_single_tensor_not_in_a_list_raises(xdata):
    x, _, _ = xdata
    with pytest.raises(TypeError, match="list of tt.vector"):
        multifuncrs2(x, lambda v: v[:, 0], eps=1e-6, verb=0)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="X is empty"):
        multifuncrs2([], lambda v: v[:, 0], eps=1e-6, verb=0)


def test_zero_function_is_reported_not_hidden(xdata):
    x, _, _ = xdata
    with pytest.warns(RuntimeWarning, match="exactly zero"):
        y = multifuncrs2([x], lambda v: 0.0 * v[:, 0], eps=1e-6, verb=0)
    assert float(y.norm()) == 0.0


# --- reproducibility and backends --------------------------------------------

def test_seed_is_honoured(xdata):
    """The default run is bit-for-bit reproducible.

    Measured: changing ``seed`` does *not* change the answer here (seed=0, 17,
    12345 give identical cores to the last bit on this problem) -- the initial
    guess only seeds the first index sets and is washed out after one sweep.
    So the claim under test is reproducibility, plus "another seed still lands
    on the same accuracy", not "another seed is another run".
    """
    x, _, _ = xdata
    f = lambda v: 1.0 / v[:, 0]                                   # noqa: E731
    a = multifuncrs2([x], f, eps=1e-6, verb=0)
    b = multifuncrs2([x], f, eps=1e-6, verb=0)
    # cores compared entry by entry: (a - b).norm() would only tell us that the
    # cancellation is at roundoff level, not that the two runs are the same run.
    assert [c.shape for c in a.cores] == [c.shape for c in b.cores]
    for ca, cb in zip(a.cores, b.cores):
        assert np.array_equal(np.asarray(ca), np.asarray(cb)), \
            "the default run must be bit-for-bit reproducible"
    c = multifuncrs2([x], f, eps=1e-6, seed=17, verb=0)
    assert rel_err(c.full(), 1.0 / xdata[1]) <= 1e-5


def test_torch_backend_gives_the_same_answer(xdata):
    torch = pytest.importorskip("torch")
    x, dense, _ = xdata
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = tt.vector.from_list(
        [torch.as_tensor(np.asarray(c), device=dev, dtype=torch.float64)
         for c in x.cores])
    eps = 1e-9
    y = multifuncrs2([xt], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)
    assert y.backend.name == "torch"
    got = np.asarray(tt.backend.to_numpy(y.full()))
    assert rel_err(got, 1.0 / dense) <= 10 * eps


# --- a QTT-sized sanity check ------------------------------------------------

def test_qtt_grid_reciprocal():
    """1/(1+t) on a 2^12 QTT grid: d=12 binary modes, dense truth 4096 entries."""
    d = 12
    x, dense, _ = None, None, None
    grid = (np.arange(2 ** d) + 1.0) / 2 ** d          # t in (0, 1]
    xt = tt.xfun(2, d) * (1.0 / 2 ** d) + tt.ones(2, d) * (1.0 / 2 ** d)
    eps = 1e-9
    y = multifuncrs2([xt], lambda v: 1.0 / (1.0 + v[:, 0]), eps=eps, verb=0)
    got = np.asarray(y.full(asvector=True)).reshape(-1)
    exact = 1.0 / (1.0 + grid)
    assert rel_err(got, exact) <= 10 * eps
    assert max(y.r) <= 12, f"ranks {list(y.r)} unexpectedly large for 1/(1+t)"
