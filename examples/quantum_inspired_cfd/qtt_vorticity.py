"""2D incompressible Navier-Stokes in the vorticity-streamfunction form, QTT.

The velocity-pressure projection method (``qtt_ns.py``) pays two pressure
Poisson solves per step and carries two vector components.  In two dimensions
the vorticity-streamfunction form removes the pressure and the incompressibility
constraint entirely -- it is a single scalar transport-diffusion equation:

    d omega/dt = nu * Lap omega - (V . grad) omega,   Lap psi = -omega,
    V = (d psi/dy, -d psi/dx).

Incompressibility is automatic (V is a curl).  This is also the natural shape
for the projector-splitting KSL integrator ``tt.ksl``: the stiff *linear*
viscous part d omega/dt = nu Lap omega is integrated exactly on the fixed-rank
manifold, and the advection is an explicit substep (Strang splitting).  Only one
streamfunction Poisson solve is needed per velocity evaluation.

Reuses the z-order operators of ``qtt_ns.Operators``.
"""

import numpy as np

import tt
from tt.algs.amen import amen_solve
from tt.algs.ksl import ksl

import qtt_ns as q


def velocity(ops, omega, eps, rmax, tol=1e-8, guess=None):
    """Solve Lap psi = -omega and return V = (d psi/dy, -d psi/dx)."""
    # ops.Lap_reg = -(Dx^2+Dy^2)+reg (SPD) ~ -Lap; so Lap_reg psi = omega gives
    # -Lap psi = omega, i.e. Lap psi = -omega -- the streamfunction equation.
    psi = amen_solve(ops.Lap_reg, omega, guess, tol, nswp=20, verb=0,
                     rmax=rmax, local_prec="c")
    u = tt.matvec(ops.Dy, psi)
    v = (tt.matvec(ops.Dx, psi) * (-1.0))
    return u.round(eps, rmax=rmax), v.round(eps, rmax=rmax), psi


def advection(ops, omega, u, v, eps, rmax, y0=None, fused_from=16):
    """(V . grad) omega at bounded rank.

    ``u*ox + v*oy`` in the explicit route has intermediate rank ``r_u r_ox``
    before rounding, and that r^2 SVD is the cost of the step.  ``tt.hadamard_sum``
    builds the same combination already compressed, in one sweep, at ``O(n^2 r^4)``
    instead of ``O(n r^6)`` -- measured 9x faster at bond 32 and 21-51x at bond 64.
    Below bond ``fused_from`` the explicit product is cheap enough that the fused
    route's fixed overhead is not worth it.
    """
    ox = tt.matvec(ops.Dx, omega)
    oy = tt.matvec(ops.Dy, omega)
    if max(max(u.r), max(ox.r)) >= fused_from:
        return tt.hadamard_sum([[u, ox], [v, oy]], eps=eps, rmax=rmax)
    return (u * ox + v * oy).round(eps, rmax=rmax)


def rhs(ops, omega, nu, eps, rmax, guess=None, adv_y0=None):
    u, v, psi = velocity(ops, omega, eps, rmax, guess=guess)
    adv = advection(ops, omega, u, v, eps, rmax, y0=adv_y0)
    visc = tt.matvec(ops.Lap, omega) * nu
    return (visc - adv).round(eps, rmax=rmax), psi, adv


def step_rk2(ops, omega, dt, nu, eps=1e-8, rmax=60, guess=None, cache=None):
    """Explicit Heun step on the full vorticity RHS (baseline integrator).

    ``cache`` is a persistent dict; pass the same one every step to warm-start
    the cross-approximated advection (`advection`) from the previous step.
    """
    y0 = cache.get("adv") if cache is not None else None
    k1, psi, a1 = rhs(ops, omega, nu, eps, rmax, guess, adv_y0=y0)
    w1 = (omega + k1 * dt).round(eps, rmax=rmax)
    k2, _, a2 = rhs(ops, w1, nu, eps, rmax, psi, adv_y0=a1)
    w2 = (omega + (k1 + k2) * (0.5 * dt)).round(eps, rmax=rmax)
    if cache is not None:
        cache["adv"] = a2
    return w2, psi


def step_ksl(ops, omega, dt, nu, eps=1e-8, rmax=60, guess=None):
    """Strang split: half advection (explicit) / viscous (KSL, exact) / half.

    The viscous flow d omega/dt = nu Lap omega is integrated by ``tt.ksl`` --
    the projector-splitting retraction onto the fixed-rank manifold -- so the
    stiff linear part is exact and the rank stays put; advection is an explicit
    half-step around it.
    """
    A = ops.Lap * nu

    def adv_half(w, guess):
        u, v, psi = velocity(ops, w, eps, rmax, guess=guess)
        a1 = advection(ops, w, u, v, eps, rmax)
        wh = (w - a1 * (0.5 * dt)).round(eps, rmax=rmax)
        u2, v2, _ = velocity(ops, wh, eps, rmax, guess=psi)
        a2 = advection(ops, wh, u2, v2, eps, rmax)
        return (w - (a1 + a2) * (0.25 * dt)).round(eps, rmax=rmax), psi

    omega, psi = adv_half(omega, guess)
    omega = ksl(A, omega, dt, verb=0, rmax=rmax)
    omega, psi = adv_half(omega, psi)
    return omega, psi
