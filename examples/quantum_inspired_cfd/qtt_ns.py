"""Incompressible Navier-Stokes in 2D, carried entirely in the QTT format.

This is the "quantum-inspired" turbulence solver of Gourianov et al.,
*A quantum-inspired approach to exploit turbulence structures*, Nature Comput.
Sci. 2 (2022) 30-37 -- reclaimed as what it is, quantized tensor train.  Each
velocity component lives on a ``2^d x 2^d`` periodic grid, encoded in the
**interleaved (z-order) quaternary QTT**: the two grid-index bits at level ``k``
share one mode of size 4, so a bond of the train separates *length scales*
rather than directions.  The bond dimension chi is the interscale correlation
the paper measures; capping it (``rmax``) is the whole method.

We solve

    dV/dt + (V . grad) V = - grad p + nu * Lap V,   div V = 0

by a projection (fractional-step) method: advance velocity by advection and
viscosity, then project onto the divergence-free manifold by solving one
pressure Poisson equation -- with the toolbox's own QTT solvers
(``amen_solve`` or the fixed-rank ``lobpcg_solve``), which is the point of
doing it here.

Everything is real, periodic, and never leaves the tensor-train format.
"""

import numpy as np

import tt
from tt.algs.amen import amen_solve


# --- finite-difference weights (Fornberg central stencils) -------------------

_D1_WEIGHTS = {
    2: {1: 0.5},
    8: {1: 4.0 / 5.0, 2: -1.0 / 5.0, 3: 4.0 / 105.0, 4: -1.0 / 280.0},
}
_D2_WEIGHTS = {
    2: {0: -2.0, 1: 1.0},
    8: {0: -205.0 / 72.0, 1: 8.0 / 5.0, 2: -1.0 / 5.0,
        3: 8.0 / 315.0, 4: -1.0 / 560.0},
}


def _periodic_shift(d, k):
    """The k-fold periodic shift on 2^d points (circulant), as a QTT matrix."""
    return tt.Toeplitz(tt.unit(2, d, j=k % (2 ** d)), d, kind="C")


def _periodic_d1(d, h, order):
    """Central first-derivative MPO on 2^d periodic points, scaled by 1/h."""
    w = _D1_WEIGHTS[order]
    op = None
    for k, c in w.items():
        term = _periodic_shift(d, -k) - _periodic_shift(d, k)   # antisymmetric
        term = term * (c / h)
        op = term if op is None else op + term
    return op.round(1e-13)


def _periodic_d2(d, h, order):
    """Central second-derivative MPO on 2^d periodic points, scaled by 1/h^2."""
    w = _D2_WEIGHTS[order]
    eye = tt.eye(2, d)
    op = eye * (w[0] / h ** 2)
    for k in range(1, max(w) + 1):
        term = (_periodic_shift(d, -k) + _periodic_shift(d, k)) * (w[k] / h ** 2)
        op = op + term
    return op.round(1e-13)


# --- operators, in the interleaved z-order layout ----------------------------

class Operators:
    """The z-order derivative and Laplacian MPOs on the 2^d x 2^d torus."""

    def __init__(self, d, box=2.0 * np.pi, order=8, eps=1e-12):
        self.d = d
        self.N = 2 ** d
        self.box = box
        self.h = box / self.N
        self.eps = eps
        eye = tt.eye(2, d)
        h = self.h
        # central high-order stencils drive advection and viscosity
        d1 = _periodic_d1(d, h, order)
        d2 = _periodic_d2(d, h, order)
        # x = first (low) bits, y = second (high) bits -- see tt.zmeshgrid
        self.Dx = tt.zkron(d1, eye).round(eps)
        self.Dy = tt.zkron(eye, d1).round(eps)
        self.Lap = (tt.zkron(d2, eye) + tt.zkron(eye, d2)).round(eps)
        # The projection Poisson operator is the *composition* of the same
        # central first-difference operators the divergence and gradient use,
        # Lap_proj = Dx Dx + Dy Dy.  Because it is exactly div(grad(.)) in this
        # discretization, a projected field is discretely divergence-free to
        # the accuracy of the pressure solve -- and, being high-order central,
        # it keeps the projection at the order of the momentum stencil.
        self.Lap_proj = (self.Dx @ self.Dx + self.Dy @ self.Dy).round(eps)
        self.ex, self.ey = tt.zmeshgrid(d)
        # mean-pinning term e e^T / N^2 (rank 1): makes the periodic Poisson
        # operator nonsingular by fixing the otherwise-free constant mode.
        ones_mat = tt.matrix.from_list([np.ones((1, 4, 4, 1)) for _ in range(d)])
        self.Lap_reg = (self.Lap_proj + ones_mat * (1.0 / self.N ** 2)).round(eps)

    def field(self, fun, eps=1e-10):
        """Sample ``fun(x, y)`` into a z-order QTT vector on the grid."""
        h = self.h
        return tt.multifuncrs(
            [self.ex, self.ey],
            lambda P: fun(P[:, 0] * h, P[:, 1] * h),
            eps=eps, verb=0)


# --- the projection (fractional-step) time integrator ------------------------

def _divergence(ops, u, v, eps, rmax):
    return (tt.matvec(ops.Dx, u) + tt.matvec(ops.Dy, v)).round(eps, rmax=rmax)


def make_projector(ops, solver="amen", eps=1e-8, rmax=40, tol=1e-8):
    """Return ``project(u, v) -> (u, v)`` enforcing ``div V = 0`` in QTT.

    Solves ``Lap phi = div V`` with the requested toolbox solver and subtracts
    ``grad phi``.  ``solver="amen"`` (default) is the rank-adaptive
    ``amen_solve``: its local preconditioner carries the periodic Laplacian's
    ``O(4^d)`` conditioning without trouble to large ``d``.  ``solver="lobpcg"``
    (fixed-rank ``lobpcg_solve``) stays on the bounded-rank manifold but has no
    strong preconditioner, so on this ill-conditioned periodic operator it
    stalls -- it would want a BPX-style preconditioner (cf. ``bpx_elliptic``).
    """
    phi_guess = [None]

    def feasible(d, rmax):
        """A left/right-feasible rank profile for a mode-4 QTT of d cores."""
        prof = [min(rmax, 4 ** min(k, d - k)) for k in range(d + 1)]
        cores = [np.random.default_rng(k).standard_normal((prof[k], 4, prof[k + 1]))
                 for k in range(d)]
        x = tt.vector.from_list(cores)
        return x * (1.0 / x.norm())

    def solve_poisson(rhs):
        if solver == "amen":
            return amen_solve(ops.Lap_reg, rhs, phi_guess[0], tol,
                              nswp=20, verb=0, rmax=rmax, local_prec="c")
        if solver == "lobpcg":
            x0 = phi_guess[0] if phi_guess[0] is not None else feasible(ops.d, rmax)
            return tt.lobpcg_solve(ops.Lap_reg, rhs, x0, tol, nswp=60, verb=0)
        raise ValueError(solver)

    def project(u, v):
        div = _divergence(ops, u, v, eps, rmax)
        phi = solve_poisson(div)
        phi_guess[0] = phi
        u = (u - tt.matvec(ops.Dx, phi)).round(eps, rmax=rmax)
        v = (v - tt.matvec(ops.Dy, phi)).round(eps, rmax=rmax)
        return u, v

    return project


def _rhs(ops, u, v, nu, eps, rmax):
    """The convection-diffusion right-hand side (no pressure)."""
    ux, uy = tt.matvec(ops.Dx, u), tt.matvec(ops.Dy, u)
    vx, vy = tt.matvec(ops.Dx, v), tt.matvec(ops.Dy, v)
    adv_u = (u * ux + v * uy).round(eps, rmax=rmax)
    adv_v = (u * vx + v * vy).round(eps, rmax=rmax)
    du = (tt.matvec(ops.Lap, u) * nu - adv_u).round(eps, rmax=rmax)
    dv = (tt.matvec(ops.Lap, v) * nu - adv_v).round(eps, rmax=rmax)
    return du, dv


def step(ops, u, v, dt, nu, project, eps=1e-8, rmax=40):
    """One second-order (Heun) projection step, all in QTT at bond <= rmax."""
    du1, dv1 = _rhs(ops, u, v, nu, eps, rmax)
    u1 = (u + du1 * dt).round(eps, rmax=rmax)
    v1 = (v + dv1 * dt).round(eps, rmax=rmax)
    u1, v1 = project(u1, v1)
    du2, dv2 = _rhs(ops, u1, v1, nu, eps, rmax)
    u2 = (u + (du1 + du2) * (0.5 * dt)).round(eps, rmax=rmax)
    v2 = (v + (dv1 + dv2) * (0.5 * dt)).round(eps, rmax=rmax)
    return project(u2, v2)
