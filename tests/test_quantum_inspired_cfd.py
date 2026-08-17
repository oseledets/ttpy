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
    # Tikhonov-regularized projection leaves div ~ reg*||phi|| ~ 1e-5, orders
    # below the flow scale -- the incompressibility a projection solver needs.
    assert div.norm() < 5e-5


def test_lobpcg_projection_converges_on_the_spd_poisson():
    """The projection Poisson is stored SPD (-(Dx^2+Dy^2)), so the fixed-rank
    lobpcg energy minimization converges -- guards against the sign/definiteness
    regression that froze it (max_dx=0) on the negative-semidefinite form."""
    d = 5
    ops = qtt_ns.Operators(d, box=2 * np.pi, order=8)
    u = ops.field(lambda x, y: np.sin(x) * np.cos(y) + 0.3 * np.cos(2 * x))
    v = ops.field(lambda x, y: np.cos(3 * y) * np.sin(x))
    project = qtt_ns.make_projector(ops, solver="lobpcg", eps=1e-10,
                                    rmax=30, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, v = project(u, v)
        div = qtt_ns._divergence(ops, u, v, 1e-12, 30)
    assert div.norm() < 5e-5


def test_taylor_green_energy_follows_the_analytic_decay():
    """E(t)/E0 must track exp(-4 nu t): the exact TGV solution."""
    res = ns_run.taylor_green(d=6, nu=0.05, T=0.4, chi=30, dense_check=False)
    assert np.max(np.abs(res["energy"] - res["analytic"])) < 1e-5
    assert res["ranks"].max() <= 8   # TGV is genuinely low rank


def test_qtt_matches_identical_dense_scheme_with_advection():
    """Isolates the rank-truncation error: same stencil, same projection."""
    res = ns_run.taylor_green(d=5, nu=0.05, T=0.3, chi=30, dense_check=True)
    assert res["dense_err"] < 1e-7


# --- vorticity-streamfunction solver -----------------------------------------

vorticity = pytest.importorskip("qtt_vorticity")


def test_vorticity_tgv_decay_is_exact_with_ksl():
    """TGV vorticity omega=-2 cos x cos y decays as exp(-2 nu t); the KSL
    viscous integrator makes the enstrophy decay exp(-4 nu t) machine-exact."""
    d = 6
    ops = qtt_ns.Operators(d, box=2 * np.pi, order=8)
    om = ops.field(lambda x, y: -2 * np.cos(x) * np.cos(y))
    nu, dt, n = 0.05, 0.1 * ops.h, 30
    e0 = float(tt.dot(om, om))
    g = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            om, g = vorticity.step_ksl(ops, om, dt, nu, eps=1e-9, rmax=40,
                                       guess=g)
    ratio = float(tt.dot(om, om)) / e0
    assert abs(ratio - np.exp(-4 * nu * n * dt)) < 1e-9


def test_vorticity_agrees_with_velocity_pressure_solver():
    """The scalar vorticity solver and the velocity-pressure solver evolve the
    same flow to the same vorticity field."""
    d = 6
    ops = qtt_ns.Operators(d, box=2 * np.pi, order=8)
    psi = ops.field(lambda x, y: np.cos(x) * np.cos(y)
                    + 0.3 * np.cos(2 * x) * np.cos(y))
    u0 = tt.matvec(ops.Dy, psi).round(1e-10)
    v0 = (tt.matvec(ops.Dx, psi) * (-1.0)).round(1e-10)
    om0 = (tt.matvec(ops.Dx, v0) - tt.matvec(ops.Dy, u0)).round(1e-10)
    nu, dt, n = 0.02, 0.08 * ops.h, 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = qtt_ns.make_projector(ops, eps=1e-9, rmax=40, tol=1e-9)
        u, v = project(u0, v0)
        for _ in range(n):
            u, v = qtt_ns.step(ops, u, v, dt, nu, project, eps=1e-9, rmax=40)
        om_vel = (tt.matvec(ops.Dx, v) - tt.matvec(ops.Dy, u)).round(1e-10)
        om, g = om0, None
        for _ in range(n):
            om, g = vorticity.step_rk2(ops, om, dt, nu, eps=1e-9, rmax=40,
                                       guess=g)
    assert (om - om_vel).norm() / om_vel.norm() < 1e-4


# --- 3D solver ---------------------------------------------------------------

ns3d = pytest.importorskip("qtt_ns3d")


def test_3d_abc_beltrami_decays_analytically():
    """The ABC flow is Beltrami (curl V = V), so its advection is a pure
    gradient absorbed by the pressure and it decays as V(t)=V0 exp(-nu t):
    the kinetic energy must follow exp(-2 nu t).  Validates the octal 3D
    operators, the divergence-free projection, and the advection cancellation."""
    d = 4
    ops = ns3d.Operators3D(d, box=2 * np.pi, order=8)
    u = ops.field(lambda x, y, z: np.sin(z) + np.cos(y))
    v = ops.field(lambda x, y, z: np.sin(x) + np.cos(z))
    w = ops.field(lambda x, y, z: np.sin(y) + np.cos(x))
    nu, dt, n = 0.1, 0.1 * ops.h, 20

    def energy(u, v, w):
        return tt.dot(u, u) + tt.dot(v, v) + tt.dot(w, w)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = ns3d.make_projector(ops, eps=1e-9, rmax=30, tol=1e-9)
        u, v, w = project(u, v, w)
        assert ns3d.divergence(ops, u, v, w, 1e-12, 30).norm() < 1e-9
        e0 = energy(u, v, w)
        for _ in range(n):
            u, v, w = ns3d.step(ops, u, v, w, dt, nu, project, eps=1e-9, rmax=30)
    ratio = energy(u, v, w) / e0
    assert abs(ratio - np.exp(-2 * nu * n * dt)) < 1e-5
