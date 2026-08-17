"""Incompressible Navier-Stokes in 3D, carried entirely in the QTT format.

The three-dimensional companion of ``qtt_ns.py``.  Each velocity component lives
on a ``2^d x 2^d x 2^d`` periodic grid in the **interleaved (z-order) octal QTT**:
the three grid-index bits at level ``k`` share one mode of size 8, so a bond of
the train still separates length scales.  The operators are the same central
periodic stencils, interleaved by a nested ``tt.zkron``.

In three dimensions the vorticity is a vector (vortex stretching), so there is no
scalar streamfunction shortcut -- we keep the velocity-pressure form of
``qtt_ns`` and its Chorin projection, now over three components and a 3D pressure
Poisson.
"""

import numpy as np

import tt
from tt.algs.amen import amen_solve

from qtt_ns import _periodic_d1, _periodic_d2


def _zk3(a, b, c):
    """Interleave three mode-2 operators into one mode-8 z-order operator."""
    return tt.zkron(tt.zkron(a, b), c)


class Operators3D:
    """z-order derivative and Laplacian MPOs on the 2^d x 2^d x 2^d torus."""

    def __init__(self, d, box=2.0 * np.pi, order=8, eps=1e-12, reg=1e-6):
        self.d = d
        self.N = 2 ** d
        self.box = box
        self.h = box / self.N
        self.eps = eps
        eye = tt.eye(2, d)
        h = self.h
        d1 = _periodic_d1(d, h, order)
        d2 = _periodic_d2(d, h, order)
        self.Dx = _zk3(d1, eye, eye).round(eps)
        self.Dy = _zk3(eye, d1, eye).round(eps)
        self.Dz = _zk3(eye, eye, d1).round(eps)
        self.Lap = (_zk3(d2, eye, eye) + _zk3(eye, d2, eye)
                    + _zk3(eye, eye, d2)).round(eps)
        self.Lap_proj = ((self.Dx @ self.Dx + self.Dy @ self.Dy
                          + self.Dz @ self.Dz) * (-1.0)).round(eps)
        lin, one = tt.xfun(2, d), tt.ones(2, d)
        self.ex = tt.zkronv(tt.zkronv(lin, one), one)
        self.ey = tt.zkronv(tt.zkronv(one, lin), one)
        self.ez = tt.zkronv(tt.zkronv(one, one), lin)
        eye8 = tt.matrix.from_list([np.eye(8).reshape(1, 8, 8, 1)
                                    for _ in range(d)])
        self.Lap_reg = (self.Lap_proj + eye8 * reg).round(eps)

    def field(self, fun, eps=1e-10):
        h = self.h
        return tt.multifuncrs(
            [self.ex, self.ey, self.ez],
            lambda P: fun(P[:, 0] * h, P[:, 1] * h, P[:, 2] * h),
            eps=eps, verb=0)


def divergence(ops, u, v, w, eps, rmax):
    return (tt.matvec(ops.Dx, u) + tt.matvec(ops.Dy, v)
            + tt.matvec(ops.Dz, w)).round(eps, rmax=rmax)


def make_projector(ops, eps=1e-8, rmax=40, tol=1e-8):
    guess = [None]

    def project(u, v, w):
        div = divergence(ops, u, v, w, eps, rmax)
        phi = amen_solve(ops.Lap_reg, (div * (-1.0)).round(eps, rmax=rmax),
                         guess[0], tol, nswp=20, verb=0, rmax=rmax,
                         local_prec="c")
        guess[0] = phi
        u = (u - tt.matvec(ops.Dx, phi)).round(eps, rmax=rmax)
        v = (v - tt.matvec(ops.Dy, phi)).round(eps, rmax=rmax)
        w = (w - tt.matvec(ops.Dz, phi)).round(eps, rmax=rmax)
        return u, v, w

    return project


def _adv(ops, a, u, v, w, eps, rmax):
    """(V . grad) a = u a_x + v a_y + w a_z, bounded rank."""
    ax = tt.matvec(ops.Dx, a)
    ay = tt.matvec(ops.Dy, a)
    az = tt.matvec(ops.Dz, a)
    if max(max(u.r), max(ax.r)) >= 16:
        # already-compressed three-term combination (see tt.hadamard_sum)
        return tt.hadamard_sum([[u, ax], [v, ay], [w, az]], eps=eps, rmax=rmax)
    return (u * ax + v * ay + w * az).round(eps, rmax=rmax)


def _rhs(ops, u, v, w, nu, eps, rmax):
    du = (tt.matvec(ops.Lap, u) * nu - _adv(ops, u, u, v, w, eps, rmax))
    dv = (tt.matvec(ops.Lap, v) * nu - _adv(ops, v, u, v, w, eps, rmax))
    dw = (tt.matvec(ops.Lap, w) * nu - _adv(ops, w, u, v, w, eps, rmax))
    return (du.round(eps, rmax=rmax), dv.round(eps, rmax=rmax),
            dw.round(eps, rmax=rmax))


def step(ops, u, v, w, dt, nu, project, eps=1e-8, rmax=40):
    """One second-order (Heun) projection step in 3D."""
    du1, dv1, dw1 = _rhs(ops, u, v, w, nu, eps, rmax)
    u1 = (u + du1 * dt).round(eps, rmax=rmax)
    v1 = (v + dv1 * dt).round(eps, rmax=rmax)
    w1 = (w + dw1 * dt).round(eps, rmax=rmax)
    u1, v1, w1 = project(u1, v1, w1)
    du2, dv2, dw2 = _rhs(ops, u1, v1, w1, nu, eps, rmax)
    u2 = (u + (du1 + du2) * (0.5 * dt)).round(eps, rmax=rmax)
    v2 = (v + (dv1 + dv2) * (0.5 * dt)).round(eps, rmax=rmax)
    w2 = (w + (dw1 + dw2) * (0.5 * dt)).round(eps, rmax=rmax)
    return project(u2, v2, w2)
