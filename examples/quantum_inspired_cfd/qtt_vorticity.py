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


def advection(ops, omega, u, v, eps, rmax):
    """(V . grad) omega, bounded rank."""
    ox = tt.matvec(ops.Dx, omega)
    oy = tt.matvec(ops.Dy, omega)
    return (u * ox + v * oy).round(eps, rmax=rmax)


def rhs(ops, omega, nu, eps, rmax, guess=None):
    u, v, psi = velocity(ops, omega, eps, rmax, guess=guess)
    adv = advection(ops, omega, u, v, eps, rmax)
    visc = tt.matvec(ops.Lap, omega) * nu
    return (visc - adv).round(eps, rmax=rmax), psi


def step_rk2(ops, omega, dt, nu, eps=1e-8, rmax=60, guess=None):
    """Explicit Heun step on the full vorticity RHS (baseline integrator)."""
    k1, psi = rhs(ops, omega, nu, eps, rmax, guess)
    w1 = (omega + k1 * dt).round(eps, rmax=rmax)
    k2, _ = rhs(ops, w1, nu, eps, rmax, psi)
    w2 = (omega + (k1 + k2) * (0.5 * dt)).round(eps, rmax=rmax)
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
