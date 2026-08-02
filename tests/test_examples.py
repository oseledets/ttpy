"""End-to-end scenarios, taken from the legacy `examples/` directory.

The legacy examples only printed their results; here every one of them is
checked against an oracle that does not come from ttpy: an analytic formula, a
dense computation, or an exact identity.  These are the acceptance tests for
"the package still does what people used it for".

Modules that are not built yet are skipped, not silently passed.
"""

import numpy as np
import pytest

import tt


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


# --- examples/test_amen.py ---------------------------------------------------

def test_amen_solve_laplacian_residual():
    """Legacy: d=12 QTT Laplacian, right-hand side of ones. Oracle: the residual."""
    amen = pytest.importorskip("tt.algs.amen")
    d = 12
    A = tt.qlaplace_dd([d])
    f = tt.ones(2, d)
    x = amen.amen_solve(A, f, f, 1e-6, verb=0)
    res = (tt.matvec(A, x) - f).norm() / f.norm()
    assert res < 1e-6, f"AMEn returned a solution with residual {res:.3e}"


def test_amen_solve_matches_dense_for_small_d():
    amen = pytest.importorskip("tt.algs.amen")
    d = 6
    A = tt.qlaplace_dd([d])
    f = tt.ones(2, d)
    x = amen.amen_solve(A, f, f, 1e-10, verb=0)
    ref = np.linalg.solve(A.full(), np.asarray(f.full(asvector=True)))
    assert rel(x.full(asvector=True), ref) < 1e-8


# --- examples/test_cross.py --------------------------------------------------

def test_multifuncrs_sinc_integral_is_pi_over_two():
    """Legacy: sin(x)/x on 2^d QTT points; the sum must approach pi/2.

    Oracle: the analytic value of the improper integral, plus the tail estimate.
    """
    mf = pytest.importorskip("tt.algs.multifuncrs")
    d = 30
    n = 2 ** d
    b = 1e3
    h = b / (n + 1)
    x = (tt.xfun(2, d) + tt.ones(2, d)) * h
    y = mf.multifuncrs([x], lambda v: np.sin(v) / v, 1e-6,
                       y0=tt.ones(2, d), verb=0)
    approx = float(tt.dot(y, tt.ones(2, d)) * h)
    # int_0^b sin(x)/x dx = pi/2 - cos(b)/b + O(1/b^2); the midpoint sum adds O(h^2)
    assert abs(approx - np.pi / 2) < 2e-3, approx


# --- examples/test_multifuncrs.py --------------------------------------------

def test_multifuncrs2_sum_reproduces_exact_addition():
    """funs = sum over the arguments, so the answer is exactly a + b."""
    mf = pytest.importorskip("tt.algs.multifuncrs")
    a = tt.rand([3, 5, 7, 11], 4, [1, 4, 6, 5, 1])
    b = tt.rand([3, 5, 7, 11], 4, [1, 2, 4, 3, 1])
    c = mf.multifuncrs2([a, b], lambda v: np.sum(v, axis=1), eps=1e-8, verb=0)
    assert rel(c.full(), (a + b).full()) < 1e-6


# --- examples/test_eigb.py ---------------------------------------------------

def test_eigb_matches_analytic_laplacian_eigenvalues():
    """Legacy: 8 smallest eigenvalues of an 8-dimensional QTT Laplacian.

    Oracle: for the 1D discrete Laplacian with Dirichlet conditions on N points
    the eigenvalues are 4 sin^2(pi k / (2(N+1))); in d dimensions they are sums.
    """
    eigb_mod = pytest.importorskip("tt.algs.eigb")
    d, f = 4, 3                      # 2^4 points per axis, 3 axes: N = 16, 4096 total
    nblock = 4
    A = tt.qlaplace_dd([d] * f)
    N = 2 ** d
    lam1 = 4 * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1))) ** 2
    grid = sum(np.ix_(*([lam1] * f)))          # all sums lam1[k1]+lam1[k2]+lam1[k3]
    exact = np.sort(grid.ravel())[:nblock]

    ranks = [1] + [8] * (d * f - 1) + [nblock]
    x0 = tt.rand([2] * (d * f), d * f, ranks)
    y, lam = eigb_mod.eigb(A, x0, 1e-6, verb=0)
    assert np.max(np.abs(np.sort(np.asarray(lam)) - exact)) < 1e-4 * exact[-1], (
        f"eigenvalues {np.sort(np.asarray(lam))} vs analytic {exact}")


# --- examples/test_common.py -------------------------------------------------

def test_gmres_solves_the_laplacian():
    solvers = pytest.importorskip("tt.algs.solvers")
    d = 8
    A = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d)
    x, res = solvers.GMRES(lambda v, eps: tt.matvec(A, v).round(eps),
                           tt.rand(2, d, r=1) * 0.0, rhs, eps=1e-8, maxit=200, m=20)
    assert (tt.matvec(A, x) - rhs).norm() / rhs.norm() < 1e-6


def test_cross_recovers_a_low_rank_tensor():
    cross = pytest.importorskip("tt.algs.cross")
    d, n = 6, 4
    ref = tt.rand([n] * d, d, r=3)
    dense = np.asarray(ref.full())

    calls = {"n": 0}

    def fun(idx):
        idx = np.asarray(idx, dtype=int)
        calls["n"] += idx.shape[0]
        return dense[tuple(idx.T)]

    got = cross.cross(fun, [n] * d, eps=1e-10)
    assert rel(got.full(), dense) < 1e-8
    assert calls["n"] < n ** d, "cross must not evaluate the whole tensor"
