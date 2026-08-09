"""Greedy DMRG cross (``tt.algs.dmrg_cross``) against dense truth.

Every accuracy claim is checked against a dense array built from the rule that
defines the input, or against a closed-form integral (Bailey's Ising constants,
the same references the algorithm's own test driver uses).  Nothing is compared
with ``rect_cross`` output except where agreement of two independent engines is
itself the property under test.  The measured parity with the Fortran original
is in ``docs/plans/cross-approximation.md`` section 2.1a; the assertions here
are looser than those numbers, so a platform difference cannot flake them.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.dmrg_cross import DmrgCrossHistory, dmrg_cross, greedy_cross


def rel(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def sum_tensor(n, shift=2.0, seed=0):
    """``x[i] = shift + sum_k t_k(i_k)``: exact TT rank 2, dense by rule."""
    rng = np.random.default_rng(seed)
    t = [rng.uniform(0.2, 1.0, size=nk) for nk in n]
    dense = np.full(n, float(shift))
    for k, nk in enumerate(n):
        shape = [1] * len(n)
        shape[k] = nk
        dense = dense + t[k].reshape(shape)

    def fun(idx):
        idx = np.asarray(idx, dtype=np.int64)
        return shift + sum(t[k][idx[:, k]] for k in range(len(n)))

    return fun, dense


def ising_c(m, n):
    """The C_m Ising integrand on the Gauss-Legendre grid, weights baked in.

    Exactly the discretized tensor of ``test_crs_ising.f90`` (KIND 'c'): the
    reference value below is Bailey's constant, quoted in that driver.
    """
    d = m - 1
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = (x + 1.0) / 2.0
    scale = float(n // 2)
    ws = (w / 2.0) * scale

    def fun(idx):
        idx = np.asarray(idx, dtype=np.int64)
        t = nodes[idx]
        v = 1.0 + np.cumprod(t[:, ::-1], axis=1).sum(axis=1)
        u = 1.0 + np.cumprod(t, axis=1).sum(axis=1)
        return 2.0 / (v * u) * np.prod(ws[idx], axis=1)

    return fun, d, scale


C6_TRUE = 0.648634209031007075263149843450351690889772509481627995615


# --- exactness ---------------------------------------------------------------

def test_exact_rank_2_is_recovered_at_rank_2():
    """A rank-2 tensor comes back at rank exactly 2, to roundoff.

    The greedy cannot overshoot on an exactly low-rank input: once the rank is
    reached the residual is zero everywhere, every pivot is rejected, and the
    strike rule fires.  Both halves -- the ranks and the stop -- are asserted.
    """
    n = [6, 5, 7, 6, 5]
    fun, dense = sum_tensor(n, seed=1)
    y = dmrg_cross(fun, n, eps=1e-12)
    assert y.history.converged and y.history.stop_reason == "eps"
    assert max(y.history.ranks) == 2
    assert rel(y.full(), dense) < 1e-13


def test_interpolation_is_exact_on_the_cross():
    """The interpolant must reproduce ``fun`` exactly on its own pivots.

    This is the defining property of cross interpolation and it survives the
    LAPACK-solve assembly (the port's one structural difference from the
    original's incremental LU).  Checked on every held-out point being exact
    for a tensor the greedy captures exactly.
    """
    n = [5] * 4
    fun, dense = sum_tensor(n, seed=2)
    y = dmrg_cross(fun, n, eps=1e-12, n_check=200)
    assert y.history.err_check_inf < 1e-13


def test_smooth_function_reaches_the_requested_accuracy():
    """``1/(2 + sum t_k)`` at eps=1e-10, against the dense array."""
    n = [8] * 5
    base, dense = sum_tensor(n, seed=3)

    def fun(idx):
        return 1.0 / base(idx)

    y = dmrg_cross(fun, n, eps=1e-10, n_check=1000)
    assert y.history.converged
    assert rel(y.full(), 1.0 / dense) < 1e-8
    assert y.history.err_check < 1e-8


def test_ising_c6_quadrature_digits():
    """C_6 at n=17 quadrature points: the integral, not just the tensor.

    The quadrature itself limits the accuracy at n=17, so the assertion is on
    digits the discretization can deliver -- the point is that the greedy
    finds the tensor, and the contraction gives the known constant.
    """
    fun, d, scale = ising_c(6, 17)
    y = dmrg_cross(fun, [17] * d, eps=1e-10)
    val = float(tt.dot(y, tt.ones(17, d))) / scale ** d
    assert abs(1.0 - val / C6_TRUE) < 1e-6
    assert y.history.fun_eval < 60_000


@pytest.mark.parametrize("pivoting", [-1, 0, 1, 3])
def test_every_pivoting_strategy_works(pivoting):
    """Full search, bare lottery and two rook depths all reach the answer."""
    n = [5] * 4
    base, dense = sum_tensor(n, seed=4)

    def fun(idx):
        return np.exp(-base(idx))

    y = dmrg_cross(fun, n, eps=1e-9, pivoting=pivoting)
    assert rel(y.full(), np.exp(-dense)) < 1e-7


# --- engine-vs-engine --------------------------------------------------------

def test_agrees_with_rect_cross_on_a_smooth_tensor():
    """Two independent engines, one tensor, both within eps of dense truth."""
    from tt.algs.cross import rect_cross
    n = [7] * 4
    base, dense = sum_tensor(n, seed=5)

    def fun(idx):
        return np.sqrt(base(idx))

    yg = dmrg_cross(fun, n, eps=1e-10)
    yr = rect_cross(fun, tt.rand(n, r=2), eps=1e-10)
    ref = np.sqrt(dense)
    assert rel(yg.full(), ref) < 1e-8
    assert rel(yr.full(), ref) < 1e-8


# --- contract and refusal ----------------------------------------------------

def test_greedy_cross_is_this_engine_now():
    """``tt.greedy_cross`` resolves to the real greedy algorithm."""
    assert tt.greedy_cross is dmrg_cross
    assert greedy_cross is dmrg_cross


def test_determinism_and_seed():
    """Same seed, same cores; the lottery is the only randomness."""
    n = [6] * 4
    fun, _ = sum_tensor(n, seed=6)
    a = dmrg_cross(fun, n, eps=1e-10, seed=11)
    b = dmrg_cross(fun, n, eps=1e-10, seed=11)
    assert rel(a.full(), b.full()) == 0.0


def test_complex_function():
    n = [5] * 4
    base, dense = sum_tensor(n, seed=7)

    def fun(idx):
        return np.exp(1j * base(idx))

    y = dmrg_cross(fun, n, eps=1e-9)
    assert rel(y.full(), np.exp(1j * dense)) < 1e-7


def test_constant_tensor_stays_rank_1():
    y = dmrg_cross(lambda idx: np.full(len(idx), 3.0), [4] * 5, eps=1e-12)
    assert max(y.history.ranks) == 1
    assert y.history.converged
    assert rel(y.full(), np.full([4] * 5, 3.0)) < 1e-14


def test_d1_is_evaluated_exactly():
    vals = np.arange(1.0, 8.0)
    y = dmrg_cross(lambda idx: vals[np.asarray(idx)[:, 0]], [7], eps=1e-12)
    assert rel(np.asarray(y.full(asvector=True)), vals) < 1e-15
    assert y.history.converged


def test_rmax_stops_and_is_not_called_converged():
    n = [8] * 4
    base, _ = sum_tensor(n, seed=8)
    y = dmrg_cross(lambda idx: 1.0 / base(idx), n, eps=1e-14, rmax=3)
    assert y.history.stop_reason == "rmax"
    assert not y.history.converged
    assert max(y.history.ranks) <= 3


def test_non_finite_value_is_refused_with_the_index_named():
    n = [6] * 4
    base, dense = sum_tensor(n, shift=0.0, seed=9)
    top = dense.max()

    def fun(idx):
        with np.errstate(divide="ignore"):
            return 1.0 / (base(idx) - top)

    with pytest.raises(ValueError, match="non-finite value.*at multi-index"):
        dmrg_cross(fun, n, eps=1e-8)


def test_nonsense_arguments_are_refused():
    fun = lambda idx: np.ones(len(idx))
    with pytest.raises(ValueError, match="stopping rule"):
        dmrg_cross(fun, [4, 4], eps=None, rmax=None)
    with pytest.raises(ValueError, match="eps"):
        dmrg_cross(fun, [4, 4], eps=-1e-8)
    with pytest.raises(ValueError, match="rmax"):
        dmrg_cross(fun, [4, 4], rmax=0)
    with pytest.raises(ValueError, match="pivoting"):
        dmrg_cross(fun, [4, 4], pivoting=-2)
    with pytest.raises(ValueError, match="strike_limit"):
        dmrg_cross(fun, [4, 4], strike_limit=0)
    with pytest.raises(ValueError, match="vectorized"):
        dmrg_cross(lambda idx: 1.0, [4, 4], eps=1e-6)


def test_history_is_complete():
    n = [6] * 4
    fun, _ = sum_tensor(n, seed=10)
    y = dmrg_cross(fun, n, eps=1e-10, n_check=100)
    h = y.history
    assert isinstance(h, DmrgCrossHistory)
    assert h.fun_eval > 0 and h.fun_eval_check == 100
    assert h.ranks == [int(v) for v in y.r]
    assert len(h.sweeps) >= h.strikes >= 1
    assert h.amax > 0
    assert h.err_check is not None and h.err_check_inf is not None
    assert len(h.err_check_worst) == len(n)
    # the per-sweep records carry the running evaluation total
    assert h.sweeps[-1]["fun_eval"] <= h.fun_eval
