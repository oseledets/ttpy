"""The multicomponent Smoluchowski coagulation model.

    dn(v,t)/dt = 1/2 int_0^v K(v-u; u) n(v-u) n(u) du  -  n(v) int K(u; v) n(u) du

discretized by the second-order predictor-corrector scheme of S. A. Matveev,
D. A. Zheltkov, E. E. Tyrtyshnikov, A. P. Smirnov, *Tensor train versus Monte
Carlo for the multicomponent Smoluchowski coagulation equation*, J. Comput.
Phys. 316:164-179, 2016 (doi:10.1016/j.jcp.2016.04.025).

This is a *model*, not a tensor-train primitive, so it lives with the example
rather than in the package: the reusable half -- the lower-triangular
trapezoidal convolution that makes the coalescence integral affordable -- is
:func:`tt.algs.convolution.trapezoidal_convolution`, and everything here is
the physics on top of it.

Kernels are given separably, as a list of rank-1 pairs ``(kv, ku)`` with
``K(u; v) = sum_a kv_a(v) ku_a(u)``: constant and additive are provided.  The
ballistic kernel of the paper is *not* separable and is not implemented.

Honest limits: second order in ``h`` and ``tau`` and no higher; ranks are held
only by rounding; the grid is uniform (the FFT convolution *is* the uniform
grid); mass leaves through the top of the box, which is the truncated model,
not a bug; nothing enforces ``n >= 0``.
"""

from __future__ import annotations

import numpy as np

import tt
from tt.algs.convolution import (component_sum, trapezoidal_convolution,
                                 trapezoidal_weights)

__all__ = ["constant_kernel", "additive_kernel", "coagulation_rhs",
           "predictor_corrector_step", "solve"]


def constant_kernel(n_modes):
    """``K == 1``: a single rank-1 term.  The kernel of the analytic test."""
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    e = tt.ones(modes)
    return [(e, e)]


def additive_kernel(n_modes, h):
    """``K(u; v) = sum_i u_i + sum_i v_i``: two rank-1 terms."""
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    e = tt.ones(modes)
    s = component_sum(modes, h)
    return [(e, s), (s, e)]


# --- the convolution (the paper's Algorithm 1) -------------------------------

def coagulation_rhs(n, kernel, h, eps=1e-10, rmax=None):
    """The Smoluchowski right-hand side ``L1 - n * L2`` in TT.

    Args:
        n: the current distribution, a TT tensor on the uniform grid.
        kernel: list of rank-1 terms ``[(kv_a, ku_a), ...]`` representing
            ``K(u; v) = sum_a kv_a(v) ku_a(u)``; each factor a TT tensor with
            the mode sizes of ``n``.
        h: grid step, scalar or one per mode.
        eps, rmax: accuracy and rank cap of the intermediate roundings.

    Returns:
        ``(1/2) sum_a conv(kv_a n, ku_a n) - n sum_a kv_a int ku_a n``,
        rounded.
    """
    if not kernel:
        raise ValueError("kernel is empty: give at least one rank-1 term")

    gain = None
    for kv, ku in kernel:
        term = trapezoidal_convolution((kv * n).round(eps, rmax),
                                       (ku * n).round(eps, rmax), h, eps)
        gain = term if gain is None else (gain + term).round(eps, rmax)
    gain = gain * 0.5

    w = trapezoidal_weights(n.n, h)
    sink = None
    for kv, ku in kernel:
        mass = float(tt.dot(w, ku * n))
        term = kv * mass
        sink = term if sink is None else (sink + term).round(eps, rmax)

    return (gain - n * sink).round(eps, rmax)


def predictor_corrector_step(n, kernel, tau, h, eps=1e-10, rmax=None):
    """One step of the paper's eq. (5): explicit midpoint, second order.

        n_{1/2} = n + (tau/2) F(n),      n_new = n + tau F(n_{1/2}).

    Two right-hand sides per step; the intermediate and the result are
    rounded to ``eps`` (and capped at ``rmax`` if given).
    """
    half = (n + (tau / 2.0) * coagulation_rhs(n, kernel, h, eps, rmax)
            ).round(eps, rmax)
    return (n + tau * coagulation_rhs(half, kernel, h, eps, rmax)
            ).round(eps, rmax)


def solve(n0, kernel, h, tau, nsteps, eps=1e-10, rmax=None, callback=None):
    """Integrate the coagulation equation for ``nsteps`` steps of size ``tau``.

    Args:
        n0: initial distribution on the uniform grid, a TT tensor.
        kernel: separable kernel, see :func:`coagulation_rhs`.
        h: grid step, scalar or one per mode.
        tau: time step.
        nsteps: number of steps.
        eps: rounding accuracy carried through every operation.
        rmax: optional hard rank cap for the roundings.
        callback: ``callback(step, t, n)`` after every step (``step`` is
            1-based, ``t = step * tau``); use it to record densities, ranks
            or wall time.  ``callback(0, 0.0, n0)`` is called before the
            first step.

    Returns:
        The distribution at ``t = nsteps * tau``.
    """
    n = n0
    if callback is not None:
        callback(0, 0.0, n)
    for step in range(1, int(nsteps) + 1):
        n = predictor_corrector_step(n, kernel, tau, h, eps, rmax)
        if callback is not None:
            callback(step, step * tau, n)
    return n
