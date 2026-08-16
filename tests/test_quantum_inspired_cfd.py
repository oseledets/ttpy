"""Acceptance for the quantum-inspired (QTT) incompressible NS solver.

The example lives in ``examples/quantum_inspired_cfd``.  These tests pin the
solver against oracles outside the tensor world: the analytic Taylor-Green
vortex, an identical dense finite-difference scheme, and the exact periodic
differentiation the operators are supposed to reproduce.
"""

import os
import sys
import warnings

import numpy as np
import pytest

import tt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "examples", "quantum_inspired_cfd"))
qtt_ns = pytest.importorskip("qtt_ns")
import run as ns_run  # noqa: E402


def _zorder_grid(vec, d):
    N = 2 ** d
    z = np.arange(N * N)
    ix = np.zeros(N * N, int)
    iy = np.zeros(N * N, int)
    for b in range(d):
        ix |= ((z >> (2 * b)) & 1) << b
        iy |= ((z >> (2 * b + 1)) & 1) << b
    G = np.zeros((N, N))
    G[ix, iy] = np.asarray(vec.full(asvector=True)).ravel()
    return G


def test_zorder_periodic_operators_match_dense_differentiation():
    """The z-order 8th-order derivative/Laplacian MPOs are exact vs np.roll."""
    d = 4
    ops = qtt_ns.Operators(d, box=2 * np.pi, order=8)
    h = ops.h
    f = ops.field(lambda x, y: np.sin(x) * np.cos(2 * y), eps=1e-12)
    G = _zorder_grid(f, d)
    w = qtt_ns._D1_WEIGHTS[8]
    dGx = sum(c * (np.roll(G, -k, 0) - np.roll(G, k, 0)) for k, c in w.items()) / h
    got = _zorder_grid(tt.matvec(ops.Dx, f), d)
    assert np.linalg.norm(got - dGx) / np.linalg.norm(dGx) < 1e-12


def test_projection_makes_the_velocity_divergence_free():
    d = 5
    ops = qtt_ns.Operators(d, box=2 * np.pi, order=8)
    u = ops.field(lambda x, y: np.sin(x) * np.cos(y) + 0.3 * np.cos(2 * x))
    v = ops.field(lambda x, y: np.cos(3 * y) * np.sin(x))
    project = qtt_ns.make_projector(ops, solver="amen", eps=1e-11,
                                    rmax=40, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, v = project(u, v)
        div = qtt_ns._divergence(ops, u, v, 1e-12, 40)
    assert div.norm() < 1e-5


def test_taylor_green_energy_follows_the_analytic_decay():
    """E(t)/E0 must track exp(-4 nu t): the exact TGV solution."""
    res = ns_run.taylor_green(d=6, nu=0.05, T=0.4, chi=30, dense_check=False)
    assert np.max(np.abs(res["energy"] - res["analytic"])) < 1e-5
    assert res["ranks"].max() <= 8   # TGV is genuinely low rank


def test_qtt_matches_identical_dense_scheme_with_advection():
    """Isolates the rank-truncation error: same stencil, same projection."""
    res = ns_run.taylor_green(d=5, nu=0.05, T=0.3, chi=30, dense_check=True)
    assert res["dense_err"] < 1e-7
