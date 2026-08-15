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
``K(u; v) = sum_a kv_a(v) ku_a(u)``.  Constant and additive are separable by
hand; the ballistic kernel of the paper's eq. (17) is *not*, and is built
numerically by :func:`ballistic_kernel` -- a TT-cross of the ``2d``-dimensional
``K`` cut along the bond that separates the ``u`` modes from the ``v`` modes.

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
from tt.algs.cross import cross, element
from tt.core.vector import vector

__all__ = ["constant_kernel", "additive_kernel", "ballistic_kernel",
           "coagulation_rhs", "predictor_corrector_step", "solve"]


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


def _ballistic_values(su, sv):
    """The paper's eq. (17) as a function of the two component sums.

    ``K = (su^{1/3} + sv^{1/3})^2 sqrt(1/su + 1/sv)`` -- the collision cross
    section of two spheres of volume ``su``, ``sv`` times their relative
    thermal velocity.  Vectorized, broadcasting.
    """
    return ((su ** (1.0 / 3.0) + sv ** (1.0 / 3.0)) ** 2
            * np.sqrt(1.0 / su + 1.0 / sv))


def ballistic_kernel(n_modes, h, eps=1e-6, rmax=None, floor=None, info=None):
    """The ballistic kernel of the paper's eq. (17), split by TT-cross.

        K(u; v) = ((sum_i u_i)^{1/3} + (sum_i v_i)^{1/3})^2
                  * sqrt(1/sum_i u_i + 1/sum_i v_i)

    is not separable in ``u`` and ``v``, so unlike :func:`constant_kernel` and
    :func:`additive_kernel` it cannot be written down as a handful of rank-1
    pairs.  It is *approximated* by one instead, to accuracy ``eps``:

    1. the ``2d``-dimensional array ``K[i_1..i_d, j_1..j_d]`` on the grid
       ``u_k = i_k h_k``, ``v_k = j_k h_k`` is built by
       :func:`tt.algs.cross.cross` -- a black-box cross over the index, which
       never forms the ``N^{2d}`` entries;
    2. the resulting train is cut at the single bond that separates the two
       groups of modes.  **Mode order: the first ``d`` modes are ``u``, the
       last ``d`` are ``v``.**  If the rank of that bond is ``R``, then
       slicing the last ``u``-core and the first ``v``-core by the bond index
       ``a`` gives exactly

           ku_a = from_list(cores[:d-1] + [cores[d-1][:, :, a:a+1]])
           kv_a = from_list([cores[d][a:a+1, :, :]] + cores[d+1:])

       and ``sum_a ku_a(u) kv_a(v) == K(u; v)`` *identically*, entry by entry:
       the cut is a regrouping of the same TT contraction, not a second
       approximation.  All the error is the cross's, and it is measured (see
       ``info``).

    Why ``R`` is small, and why it barely grows with ``N``: ``K`` is a
    function of two scalars only, ``K = F(S_u, S_v)`` with ``S = sum_i v_i``,
    so the rank of the ``u|v`` bond is the epsilon-rank of the two-variable
    matrix ``F(s, t)`` and nothing else -- a property of ``F``, not of the
    grid.  Measured here: ``R = 6, 7, 7, 7, 7, 7`` for ``N = 100 .. 3200``,
    and ``R <= 8`` for ``d = 2..5``, against the paper's ``R = 19..23``
    (Table 6) at the same ``eps = 1e-6``.  The direct SVD of ``F`` sampled on
    the grid of sums puts the floor at 8, so the cross is finding it.

    The singularity, and what ``floor`` does
    ----------------------------------------
    ``K`` is *infinite* whenever ``sum_i u_i = 0`` or ``sum_i v_i = 0``: a
    particle of zero mass has zero volume and infinite thermal velocity.  That
    is the model, not a bug -- and the grid node ``i = 0`` sits exactly on it.

    ``floor`` is the resolution of this: the sums are clipped from below,
    ``su = max(sum_i u_i, floor)``, ``sv = max(sum_i v_i, floor)``.  The
    default is ``floor = min(h)``, i.e. *a particle whose mass is below one
    grid node is treated as one grid node*, which is the smallest mass the
    grid can represent at all.  ``K`` then stays finite everywhere, the cross
    has a bounded function to approximate, and the modification touches only
    the single hyperplane the grid cannot resolve anyway.

    The paper does not do this.  It moves the grid off the singularity
    instead, starting the volume axis at some ``V_min > 0`` and declaring the
    "full dissipation of sufficiently small particles" (its eq. (4)) --
    particles below ``V_min`` are removed from the system rather than clipped.
    The two devices are not the same physics: theirs deletes the mass below
    the cutoff, ours keeps it and understates its collision rate.  On the
    density at ``t = 1`` the difference is under 0.5% (see the module page),
    but it is a modelling choice and it is stated here rather than hidden.

    Args:
        n_modes: mode sizes of the ``d``-dimensional grid, e.g. ``[N, N]``.
        h: grid step, scalar or one per mode.
        eps: target relative accuracy of the cross.
        rmax: optional hard cap on the TT ranks of ``K``.
        floor: lower clip of both component sums; ``None`` means ``min(h)``.
        info: optional dict, filled with ``rank`` (the bond rank ``R``),
            ``err`` and ``err_max`` (relative Frobenius and worst-case
            pointwise error of the returned pairs against eq. (17), measured
            on 4000 random grid nodes nobody looked at), ``floor``,
            ``ranks`` (all TT ranks of ``K``) and ``fun_eval``.

    Returns:
        A list of ``R`` pairs ``(kv_a, ku_a)``, the same interface as the
        other kernels here.
    """
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    d = modes.size
    hs = np.asarray(h, dtype=np.float64).ravel()
    if hs.size == 1:
        hs = np.repeat(hs, d)
    if hs.size != d:
        raise ValueError(f"h has {hs.size} entries, the grid has {d} modes")
    if floor is None:
        floor = float(np.min(hs))
    floor = float(floor)
    if not floor > 0.0:
        raise ValueError(f"floor must be positive, got {floor}: the kernel is "
                         "infinite at zero mass, see the docstring")

    def fun(idx):
        idx = np.asarray(idx, dtype=np.float64)
        return _ballistic_values(np.maximum(idx[:, :d] @ hs, floor),
                                 np.maximum(idx[:, d:] @ hs, floor))

    K = cross(fun, [int(m) for m in modes] * 2, eps=eps, rmax=rmax, seed=0)
    cores = [np.asarray(c) for c in vector.to_list(K)]
    R = cores[d - 1].shape[2]

    kernel = [(vector.from_list([cores[d][a:a + 1, :, :]] + cores[d + 1:]),
               vector.from_list(cores[:d - 1] + [cores[d - 1][:, :, a:a + 1]]))
              for a in range(R)]

    if info is not None:
        rng = np.random.default_rng(20160525)
        iu = rng.integers(0, modes, size=(4000, d))
        iv = rng.integers(0, modes, size=(4000, d))
        ref = _ballistic_values(np.maximum(iu @ hs, floor),
                                np.maximum(iv @ hs, floor))
        got = np.zeros(ref.shape)
        for kv, ku in kernel:
            got += np.asarray(element(ku, iu)) * np.asarray(element(kv, iv))
        info.update(rank=R, ranks=[int(r) for r in K.r], floor=floor,
                    err=float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
                    err_max=float(np.max(np.abs(got - ref) / np.abs(ref))),
                    fun_eval=int(K.history.fun_eval))
    return kernel


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
