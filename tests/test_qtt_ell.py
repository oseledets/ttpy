"""The QTT elliptic kit and the BPX preconditioner.

Oracles, in order of preference: a closed-form formula from [BK20], the dense
matrix built by numpy, and the analytic spectrum of the discrete Laplacian.
Nothing here is checked against another routine of ours except where the point
*is* that two of our routines agree (the explicit BPX assembly against the
per-level sum built from :func:`prolongation` -- that one is the definition).

[BK20] M. Bachmayr, V. Kazeev, Found. Comput. Math. 20 (2020) 1175-1236.
"""

from __future__ import annotations

import numpy as np
import pytest

import tt
from tt.algs.amen import amen_solve
from tt.algs.qtt_ell import bpx, bpx_theta, prolongation, solve_direct_1d


def dense(a):
    return np.asarray(a.full())


def laplace_reference(d, bc):
    """``tridiag(-1, 2, -1)`` with a Neumann end replaced by 1."""
    n = 2 ** d
    ref = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    if bc[0] == "N":
        ref[0, 0] = 1.0
    if bc[1] == "N":
        ref[n - 1, n - 1] = 1.0
    return ref


def prolongation_reference(l, d):
    """``P_{l,L}`` written out from [BK20] Lemma 4 -- the independent oracle.

    ``2^((l-L)/2) [ I_{2^l} (x) eta_{L-l} + S_{2^l} (x) (xi - eta)_{L-l} ]``
    with ``eta_k = 2^-k (1, ..., 2^k)`` and ``xi_k = (1, ..., 1)``.
    """
    ihat = np.eye(2 ** l)
    shat = np.eye(2 ** l, k=-1)
    k = d - l
    eta = 2.0 ** (-k) * np.arange(1, 2 ** k + 1)
    xi = np.ones(2 ** k)
    return 2.0 ** ((l - d) / 2) * (np.kron(ihat, eta.reshape(-1, 1))
                                   + np.kron(shat, (xi - eta).reshape(-1, 1)))


# --- the construction kit ----------------------------------------------------

@pytest.mark.parametrize("d", [3, 5, 8])
def test_qdiff_and_qtri_ones_are_exact_inverses(d):
    """``T (I - S) = I``: this is why a 1D problem needs no iteration at all."""
    m, t = tt.qdiff(d), tt.qtri_ones(d)
    assert max(m.r) == 2 and max(t.r) == 2
    prod = dense((t @ m).round(1e-14))
    assert np.abs(prod - np.eye(2 ** d)).max() < 1e-13
    assert np.abs(dense(tt.qdiff(d, "forward")) - dense(m).T).max() < 1e-14


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("bc", ["DD", "DN", "ND"])
def test_qlaplace_dn_matches_the_dense_operator(d, bc):
    a = tt.qlaplace_dn(d, bc)
    assert np.abs(dense(a) - laplace_reference(d, bc)).max() < 1e-13
    assert max(a.r) <= 4


@pytest.mark.parametrize("d", [4, 6, 8])
def test_qlaplace_dn_smallest_eigenvalue_is_the_analytic_one(d):
    """Dirichlet-Neumann on ``N`` nodes: ``lam_min = 4 sin^2(pi/(2(2N+1)))``.

    The tolerance is *absolute*, scaled by ``||A||_2 <= 4``, and deliberately
    not relative: ``lam_min`` is 3.8e-05 already at ``d = 8`` and shrinks like
    ``4^-d``, so a relative 1e-12 would demand an absolute 3.8e-17 -- below the
    accuracy any eigensolver can give for a matrix of norm 4. Measured errors
    are 1.3e-15 (``d = 6``) and 2.1e-15 (``d = 8``), i.e. a few eps of the norm,
    which is the right thing to hold it to.
    """
    n = 2 ** d
    lam = np.linalg.eigvalsh(dense(tt.qlaplace_dn(d, "DN")))[0]
    exact = 4.0 * np.sin(np.pi / (2 * (2 * n + 1))) ** 2
    assert abs(lam - exact) < 1e-13 * 4.0


def test_qlaplace_dn_refuses_the_singular_case():
    """'NN' has the constants in its kernel; returning it silently is a trap."""
    with pytest.raises(ValueError, match="singular"):
        tt.qlaplace_dn(4, "NN")
    with pytest.raises(ValueError):
        tt.qlaplace_dn(4, "XY")
    with pytest.raises(ValueError, match="order"):
        tt.qlaplace_dn([3, 3], "DN", order="whatever")


def test_the_two_index_layouts_are_the_same_operator():
    """Dimension-major and level-major differ by a permutation, nothing else.

    They must not be mixed -- an operator in one layout and a preconditioner in
    the other compose to silent nonsense -- so the property that matters is that
    each is *internally* the same operator, which the spectrum shows.
    """
    d, ndim = 3, 2
    a_dim = tt.qlaplace_dn([d] * ndim, "DN", order="dim")
    a_lvl = tt.qlaplace_dn([d] * ndim, "DN", order="level")
    assert max(a_dim.r) <= 5 and max(a_lvl.r) <= 8
    w_dim = np.linalg.eigvalsh(dense(a_dim))
    w_lvl = np.linalg.eigvalsh(dense(a_lvl))
    assert np.abs(w_dim - w_lvl).max() < 1e-10
    assert tt.level_major_order([3, 3]) == [0, 3, 1, 4, 2, 5]


# --- the preconditioner ------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 4])
def test_prolongation_matches_the_closed_form(d):
    for l in range(d + 1):
        p = prolongation(l, d)
        assert max(p.r) == 2
        assert np.abs(dense(p) - prolongation_reference(l, d)).max() < 1e-13


@pytest.mark.parametrize("d", [2, 3, 4, 5])
@pytest.mark.parametrize("weight", [1, 2])
def test_bpx_equals_the_level_sum_it_stands_for(d, weight):
    """The definition: ``C_L = sum_l 2^(-w l) P_l P_l^T``.

    :func:`bpx` never forms this sum -- that is its whole point -- so the sum is
    exactly the right thing to check it against.
    """
    total = np.zeros((2 ** d, 2 ** d))
    for l in range(d + 1):
        p = dense(prolongation(l, d))
        total += 2.0 ** (-weight * l) * (p @ p.T)
    got = dense(bpx(d, 1, weight=weight, scaled=False))
    assert np.abs(got - total).max() < 1e-12 * np.abs(total).max()


@pytest.mark.parametrize("d", [3, 6, 12, 24, 40])
def test_bpx_rank_does_not_grow_with_the_number_of_levels(d):
    """Rank exactly ``2 * 4^D`` -- the reason the assembly is an automaton.

    Summing ``d + 1`` TT matrices and rounding would give the same operator with
    intermediate ranks growing in ``d``; this one is built, not accumulated.
    """
    c = bpx(d, 1)
    assert max(c.r) == 8
    assert list(c.r) == [1] + [8] * (d - 1) + [1]


def test_bpx_refuses_arguments_it_cannot_honour():
    with pytest.raises(ValueError, match="weight"):
        bpx(4, 1, weight=3)
    with pytest.raises(ValueError):
        bpx(0, 1)
    with pytest.raises(ValueError):
        bpx(4, 0)


def test_bpx_bounds_the_condition_number_while_the_operator_loses_it():
    """The claim of [BK20], measured: ``kappa(C A C)`` stays put as ``d`` grows.

    Reference numbers, reproduced here independently of the way they were first
    obtained (a per-level sum rather than the automaton): ``kappa(B) = 5.6674``
    at ``d = 4`` and ``10.6617`` at ``d = 10``, with ``lam_min(B) -> 2``, while
    ``kappa(A)`` goes from 4.4e+02 to 1.7e+06 over the same range.
    """
    kappa_a, kappa_b, lam_min = {}, {}, {}
    for d in (4, 6, 8, 10):
        a = dense(tt.qlaplace_dn(d, "DN"))
        c = dense(bpx(d, 1, weight=1, scaled=True))
        wa = np.linalg.eigvalsh(a)
        wb = np.linalg.eigvalsh(c @ a @ c)
        kappa_a[d], kappa_b[d], lam_min[d] = wa[-1] / wa[0], wb[-1] / wb[0], wb[0]

    assert kappa_a[10] / kappa_a[4] > 3e3          # the operator degrades
    assert kappa_b[10] / kappa_b[4] < 2.0          # the preconditioned one does not
    assert abs(kappa_b[4] - 5.6674) < 1e-3
    assert abs(kappa_b[10] - 10.6617) < 1e-3
    assert abs(lam_min[10] - 2.0) < 1e-4
    assert all(kappa_b[d] < 12.0 for d in kappa_b)


# --- the fused form ----------------------------------------------------------

@pytest.mark.parametrize("d", [3, 5, 8])
def test_theta_squared_is_the_preconditioned_operator(d):
    """``Theta^T Theta == C A C``, which is the whole point of the fused form.

    The same matrix, assembled without ever representing the triple product.
    """
    got = dense((bpx_theta(d).T @ bpx_theta(d)).round(1e-14))
    a = dense(tt.qlaplace_dn(d, "DN"))
    c = dense(bpx(d, 1, weight=1, scaled=True))
    want = c @ a @ c
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


@pytest.mark.parametrize("d", [4, 8, 16, 30])
def test_theta_rank_is_flat_and_small(d):
    """Rank 6 for ``Theta`` and 17 for ``Theta^T Theta``, whatever ``d`` is.

    Contrast with the triple product, whose rank was measured at 96, 135, 185
    for ``d = 10, 14, 18`` -- growing, which is what made it slow *and* what
    made its entries cancel.
    """
    th = bpx_theta(d)
    assert max(th.r) == 6
    assert max((th.T @ th).round(1e-14).r) <= 17


def test_bpx_theta_refuses_more_than_one_dimension():
    with pytest.raises(NotImplementedError, match="D = 1"):
        bpx_theta(4, D=2)
    with pytest.raises(ValueError):
        bpx_theta(0)


def test_the_preconditioner_earns_its_keep():
    """The claim, end to end: same problem, same tolerance, both solvers.

    ``-u'' = 1`` with ``u(0) = 0``, ``u'(1) = 0`` on ``2^18`` nodes. Without a
    preconditioner AMEn exhausts its sweeps and still has five wrong digits;
    with one it converges in a handful and is at machine precision. Measured at
    ``d = 30``: 30 sweeps / 23.3 s / relative error 1.03 against 7 sweeps /
    0.51 s / 1.8e-13.
    """
    d = 18
    n = 2 ** d
    h = 1.0 / n
    a = tt.qlaplace_dn(d, "DN")
    rhs = (tt.ones(2, d) - 0.5 * tt.unit(2, d, j=n - 1)) * (h * h)
    x = (tt.xfun(2, d) + tt.ones(2, d)) * h
    exact = (x - 0.5 * (x * x)).round(1e-14)

    # and the unpreconditioned run says so itself, which is half the point:
    # a solver that quietly returned this iterate would be the real problem
    with pytest.warns(UserWarning, match="did NOT reach"):
        plain = amen_solve(a, rhs, tt.ones(2, d), 1e-10, nswp=30, verb=0)
    err_plain = float((plain - exact).norm() / exact.norm())

    th = bpx_theta(d)
    c = bpx(d, 1, weight=1, scaled=True)
    w, info = amen_solve((th.T @ th).round(1e-14),
                         tt.matvec(c, rhs).round(1e-12), tt.ones(2, d), 1e-10,
                         nswp=30, verb=0, return_info=True)
    got = tt.matvec(c, w).round(1e-12)
    err_prec = float((got - exact).norm() / exact.norm())

    assert err_prec < 1e-10, f"preconditioned error {err_prec:.3E}"
    assert err_plain > 1e-7, f"unpreconditioned error {err_plain:.3E} -- too good?"
    assert info.nswp_done <= 12, f"took {info.nswp_done} sweeps"


# --- the 1D direct solve -----------------------------------------------------

@pytest.mark.parametrize("d", [6, 12, 20])
def test_solve_direct_1d_is_exact_for_the_constant_load(d):
    """``-u'' = 1``, ``u(0) = 0``, ``u'(1) = 0``, whose solution is ``x - x^2/2``.

    No iteration and no preconditioner: ``T = qtri_ones`` inverts ``qdiff``
    exactly, so this is two matvecs.
    """
    n = 2 ** d
    h = 1.0 / n
    rhs = (tt.ones(2, d) - 0.5 * tt.unit(2, d, j=n - 1)) * (h * h)
    u = solve_direct_1d(rhs, d)

    x = (tt.xfun(2, d) + tt.ones(2, d)) * h        # nodes (i+1)h, i = 0..N-1
    exact = (x - 0.5 * (x * x)).round(1e-14)
    err = float((u - exact).norm() / exact.norm())
    assert err < 1e-12, f"d={d}: relative nodal error {err:.3E}"
    assert max(u.r) <= 4


def test_solve_direct_1d_agrees_with_a_dense_solve_on_a_general_load():
    d = 8
    rng = np.random.default_rng(0)
    f_dense = rng.standard_normal(2 ** d)
    f = tt.vector(f_dense.reshape([2] * d, order="F"), 1e-14)
    got = np.asarray(solve_direct_1d(f, d).full()).reshape(-1, order="F")
    want = np.linalg.solve(dense(tt.qlaplace_dn(d, "DN")), f_dense)
    assert np.linalg.norm(got - want) < 1e-9 * np.linalg.norm(want)
