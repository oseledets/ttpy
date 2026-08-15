"""Interpolatory KSL (``tt.algs.ksl_deim``), ported from Dektor's PR #102.

Every check is against dense truth (``scipy.linalg.expm``,
``scipy.integrate.solve_ivp``) or a mathematical invariant (an eigenvector is
a fixed direction of the flow; ranks are preserved).  The manifolds are chosen
*full* (the TT-rank bound of the mode sizes), so there is no modelling error
to hide behind and what is measured is the discretization alone: the scheme is
built from explicit-Euler substeps, so unlike the orthogonal-projector KSL it
is first order even on a full manifold -- the order tests pin exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.integrate import solve_ivp

import tt
from tt.algs.ksl import ksl
from tt.algs.ksl_deim import ksl_deim


# --- helpers -----------------------------------------------------------------

def sym_tt_matrix(d, n, ranks, seed):
    """Random symmetric TT-matrix, spectrally normalized to ``||A||_2 = 1``."""
    rng = np.random.default_rng(seed)
    cores = [rng.standard_normal((ranks[k], n, n, ranks[k + 1]))
             for k in range(d)]
    m = tt.matrix.from_list(cores)
    m = (m + m.T).round(1e-14)
    return (1.0 / np.linalg.norm(np.asarray(m.full()), 2)) * m


def rand_tt(n, ranks, seed, scale=1.0):
    """Seeded random TT-vector of unit norm times ``scale``."""
    rng = np.random.default_rng(seed)
    y = tt.vector.from_list(
        [rng.standard_normal((ranks[k], n, ranks[k + 1]))
         for k in range(len(ranks) - 1)])
    return y * (scale / float(y.norm()))


def integrate(A, Nf, y0, T, nsteps):
    y = y0
    for _ in range(nsteps):
        y = ksl_deim(A, Nf, y, T / nsteps)
    return y


def dense_of(y):
    return np.asarray(y.full(asvector=True))


# The full manifold for d = 3 modes of size 4: every tensor has TT-ranks
# bounded by (4, 4), so the fixed-rank manifold is the whole space and the
# only error is the time discretization.
D, N, FULL_RANKS = 3, 4, [1, 4, 4, 1]


# --- (a) linear case: first order against dense expm --------------------------

def test_linear_first_order_against_dense_expm():
    """``Nf = 0``: errors at tau, tau/2, tau/4 shrink by the ratio ~2.

    The reference is ``expm(T A) y0`` -- on the full manifold the whole gap is
    the splitting/Euler error of the scheme, which the module documents as
    first order.  Ranks must come back untouched: the method cannot adapt them.
    """
    A = sym_tt_matrix(D, N, FULL_RANKS, seed=0)
    dense = np.asarray(A.full())
    y0 = rand_tt(N, FULL_RANKS, seed=10)
    T = 0.5
    exact = sla.expm(T * dense) @ dense_of(y0)

    errs = []
    for nsteps in (8, 16, 32):
        y = integrate(A, lambda v: 0, y0, T, nsteps)
        assert list(y.r) == list(y0.r)          # fixed-rank integrator
        errs.append(np.linalg.norm(dense_of(y) - exact) / np.linalg.norm(exact))
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(errs[:-1] > errs[1:]), f"errors do not decrease: {errs}"
    assert np.abs(orders - 1.0).max() < 0.25, f"orders {orders}, errs {errs}"


# --- (b) nonlinear case: first order against solve_ivp ------------------------

def test_nonlinear_first_order_against_dense_solve_ivp():
    """``dy/dt = A y + y.^2`` (entrywise square) against a dense ODE solve.

    Also pins the ``Nf`` contract: it is called on *numpy arrays of entries*
    of the iterate at the sampled fibers (3-way arrays on forward steps,
    matrices on backward steps), never on a TT object.
    """
    A = sym_tt_matrix(D, N, FULL_RANKS, seed=1)
    dense = np.asarray(A.full())
    y0 = rand_tt(N, FULL_RANKS, seed=11, scale=0.5)   # small: no blow-up
    T = 0.25

    seen_shapes = set()

    def Nf(v):
        assert isinstance(v, np.ndarray)     # the documented contract
        seen_shapes.add(v.ndim)
        return v ** 2

    ref = solve_ivp(lambda t, u: dense @ u + u ** 2, [0.0, T], dense_of(y0),
                    rtol=1e-11, atol=1e-13, t_eval=[T]).y[:, -1]

    errs = []
    for nsteps in (8, 16, 32):
        y = integrate(A, Nf, y0, T, nsteps)
        errs.append(np.linalg.norm(dense_of(y) - ref) / np.linalg.norm(ref))
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(errs[:-1] > errs[1:]), f"errors do not decrease: {errs}"
    assert np.abs(orders - 1.0).max() < 0.25, f"orders {orders}, errs {errs}"
    assert seen_shapes == {2, 3}     # both backward (matrix) and forward fibers


# --- (c) eigenvector invariance (Dektor's demo) --------------------------------

def test_eigenvector_direction_is_invariant():
    """The demo's oracle: an eigenvector of the QTT Laplacian keeps direction.

    ``A y = lam y`` makes every sampled fiber of the right-hand side
    proportional to the matching fiber of ``y``, so ten steps change the
    direction only by roundoff, and the norm grows by exactly the explicit
    Euler factor ``(1 + tau lam)^10`` -- which is asserted too, because it
    pins the substeps as Euler updates, not exponentials.

    The demo's 3D Laplacian is replaced by the 1D one: ``eigb`` here returns
    the 3D eigenvector with its true interior unit ranks at the dimension
    boundaries, which the original code does not support (module docstring).
    """
    d = 8
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(21)
    y0, lam = tt.eigb.eigb(A, tt.rand(2, d, r=2,
                                      samplefunc=rng.standard_normal),
                           1e-8, verb=0)

    tau, nsteps = 1e-2, 10
    y = y0
    for _ in range(nsteps):
        y = ksl_deim(A, lambda v: 0, y, tau)

    cos = tt.dot(y, y0) / (y.norm() * y0.norm())
    assert abs(cos - 1.0) < 1e-12
    assert abs(float(y.norm() / y0.norm())
               - (1.0 + tau * lam[0]) ** nsteps) < 1e-8


# --- (d) agreement with the orthogonal-projector KSL ---------------------------

def test_agrees_with_ksl_on_a_linear_problem():
    """Both integrators land near ``expm`` on the full manifold, hence near
    each other.  KSL is exact there; ksl_deim carries its O(tau) Euler error,
    so the mutual distance is bounded by that error, not by roundoff.
    """
    d, n = 5, 2
    ranks = [1, 2, 4, 4, 2, 1]
    A = sym_tt_matrix(d, n, [1, 2, 2, 2, 2, 1], seed=7)
    dense = np.asarray(A.full())
    y0 = rand_tt(n, ranks, seed=41)
    T, nsteps = 0.2, 64
    exact = sla.expm(T * dense) @ dense_of(y0)

    y_deim = integrate(A, lambda v: 0, y0, T, nsteps)
    y_ksl = y0
    for _ in range(4):
        y_ksl = ksl(A, y_ksl, T / 4, verb=0, scheme="symm",
                    local_tol=1e-13, check_rank=False)

    err_ksl = np.linalg.norm(dense_of(y_ksl) - exact) / np.linalg.norm(exact)
    err_deim = np.linalg.norm(dense_of(y_deim) - exact) / np.linalg.norm(exact)
    assert err_ksl < 1e-11                       # exact up to roundoff
    assert err_deim < 5e-3                       # first order in T/nsteps
    mutual = float((y_deim - y_ksl).norm() / y_ksl.norm())
    assert mutual < 5e-3
    assert mutual > 1e-8     # and honestly not the same integrator


# --- the package export --------------------------------------------------------

def test_package_export():
    from tt.algs import ksl_deim as mod
    assert tt.ksl_deim is mod.ksl_deim
