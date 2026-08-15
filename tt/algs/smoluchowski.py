"""Multicomponent Smoluchowski coagulation in TT: the low-rank convolution.

The equation, for a size distribution ``n(v, t)`` of particles carrying
``d`` conserved components ``v = (v_1, ..., v_d)``, is

    dn(v)/dt = (1/2) int_0^{v_1}..int_0^{v_d} K(v-u; u) n(v-u) n(u) du
               - n(v) int_0^inf..int_0^inf K(u; v) n(u) du,

a gain term ``L1`` (particles of "size" ``v`` made by merging ``u`` and
``v - u``) minus a loss term ``n * L2`` (particles of size ``v`` eaten by
anything).  On a uniform grid with ``N`` nodes per component the direct sum
for ``L1`` costs ``O(N^{2d})`` -- 10^8 for the modest ``d = 2, N = 100``,
and hopeless past ``d = 3``.  This module is the tensor-train answer of
Matveev-Zheltkov-Tyrtyshnikov-Smirnov (2016): keep ``n`` as a TT tensor and
never leave the format.

The mechanism, in one paragraph
-------------------------------
Two facts compose.  (1) The gain term is a *lower-triangular convolution*,
so on a uniform grid it is a plain circular convolution of zero-padded
arrays and the FFT computes it in ``O(N log N)`` per axis -- and the
trapezoidal quadrature weights are not an obstacle, because halving the
``v_k = 0`` slice of *both* factors turns the plain discrete convolution
into exactly the trapezoidal one (the ``1/2`` endpoint weights of
``int_0^{v_k}`` land on ``j = 0`` and ``j = i`` automatically).  (2) The
FFT, the elementwise product, the truncation to the first ``N`` entries and
the zeroing of the ``v_k = 0`` slice are all *per-mode* operations, so each
one acts on a single TT core and leaves the train alone.  What is left is
the elementwise product of two TT tensors, whose cores are the Kronecker
products of the factors' cores over the rank indices -- ranks multiply, and
one rounding puts them back.  That is :func:`trapezoidal_convolution`, the
paper's Algorithm 1, at ``O(d R^4 N log N)`` per call.

The kernel has to enter separably.  It is passed as a list of rank-1 terms
``[(kv_1, ku_1), ...]`` meaning ``K(u; v) = sum_a kv_a(v) ku_a(u)``, each
factor a :class:`tt.vector` on the same grid: the gain term then splits into
one convolution per term, and the loss term into one *scalar* quadrature per
term (``L2(v) = sum_a kv_a(v) * int ku_a(u) n(u) du``) -- the ``O(N^{2d})``
double integral of the sink collapses to ``O(d N R^2)``.  The constant and
additive kernels are exactly of this form (:func:`constant_kernel`,
:func:`additive_kernel`, ranks 1 and 2).

Time stepping is the paper's eq. (5), the explicit midpoint predictor-
corrector, in :func:`predictor_corrector_step`.

What this costs you, honestly
-----------------------------
* **Second order, and no better.**  Trapezoidal quadrature in ``v`` and the
  midpoint rule in ``t`` give ``O(h^2 + tau^2)``.  On the paper's own
  reference run (``d = 2``, ``K = 1``, ``V_max = 10``) the relative error of
  the *solution* is ~1.4e-1 at ``N = 100`` and ~2.3e-3 at ``N = 500``; the
  total density ``N(t)`` is far more accurate than the density profile
  because its leading quadrature errors cancel.  If you need three digits in
  the profile, you need a fine grid, not a smaller ``eps``.
* **Ranks grow and are cut by rounding, not bounded a priori.**  Every
  Hadamard product multiplies ranks (``R_f R_g`` for the convolution before
  rounding), so ``eps`` is load-bearing: it is what keeps ``R`` at the 7..13
  the paper reports rather than at ``R^2`` per step.  There is no theorem
  here promising the rank stays small -- it stays small for the smooth,
  nearly-separable distributions these kernels produce, and the ranks
  reported by :func:`solve` are the only honest monitor.
* **Uniform grid ``0..V_max`` only.**  The FFT convolution *is* the uniform
  grid; a graded mesh would need a different quadrature and a different
  algorithm.
* **Mass leaves through the top of the box.**  Particles grown past
  ``V_max`` are simply not represented, so ``sum v n`` decays.  That is the
  truncated model, not a bug in the solver, and it is why the additive
  kernel (which pushes mass up fast) needs a generous ``V_max``.
* **No negativity control.**  Nothing in the scheme keeps ``n >= 0``; TT
  rounding can put small negative values into the tail.  The paper does not
  fix this either.
* **Ballistic and other non-separable kernels are not provided.**  Feeding
  them in requires a separable approximation (e.g. by cross approximation of
  the ``2d``-dimensional ``K``), which this module does not build for you.

References
----------
* S. A. Matveev, D. A. Zheltkov, E. E. Tyrtyshnikov, A. P. Smirnov, "Tensor
  train versus Monte Carlo for the multicomponent Smoluchowski coagulation
  equation", J. Comput. Phys. 316:164-179, 2016,
  doi:10.1016/j.jcp.2016.04.025.  Algorithm 1 (the TT trapezoidal
  convolution) and eq. (5) (the predictor-corrector) are implemented here;
  eq. (18) is the analytic solution the tests check against.
* M. H. Lee, "On the validity of the coagulation equation and the nature of
  runaway growth", Icarus 143:74-86, 2000 -- the one-component version of
  the FFT convolution trick.
"""

from __future__ import annotations

import numpy as np

from .. import backend as bk
from ..core import tools
from ..core.vector import vector

__all__ = [
    "trapezoidal_convolution",
    "coagulation_rhs",
    "predictor_corrector_step",
    "solve",
    "trapezoidal_weights",
    "constant_kernel",
    "additive_kernel",
    "component_sum",
]


# --- grid helpers ------------------------------------------------------------

def _steps(h, d):
    """Broadcast ``h`` to one grid step per mode."""
    hs = np.asarray(h, dtype=np.float64).ravel()
    if hs.size == 1:
        hs = np.repeat(hs, d)
    if hs.size != d:
        raise ValueError(f"h has {hs.size} entries, the tensor has {d} modes")
    return hs


def trapezoidal_weights(n_modes, h):
    """Rank-1 TT of the trapezoidal weights of ``int_0^{V_max} .. dv``.

    ``h`` on the interior nodes, ``h/2`` on both ends of every axis, so that
    ``tt.dot(trapezoidal_weights(n), f)`` is the trapezoidal approximation of
    the integral of ``f`` over the whole box.
    """
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    hs = _steps(h, modes.size)
    cores = []
    for nk, hk in zip(modes, hs):
        if nk < 2:
            raise ValueError("the trapezoidal rule needs at least 2 nodes")
        w = np.full((1, int(nk), 1), float(hk))
        w[0, 0, 0] = w[0, -1, 0] = 0.5 * float(hk)
        cores.append(w)
    return vector.from_list(cores)


def component_sum(n_modes, h):
    """``S(v) = v_1 + ... + v_d`` on the uniform grid, as a rank-2 TT vector.

    The grid of mode ``k`` is ``0, h_k, ..., (n_k - 1) h_k``.
    """
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    hs = _steps(h, modes.size)
    d = modes.size
    grids = [hk * np.arange(int(nk), dtype=np.float64)
             for nk, hk in zip(modes, hs)]
    if d == 1:
        return vector.from_list([grids[0].reshape((1, -1, 1))])
    cores = []
    first = np.ones((1, int(modes[0]), 2))
    first[0, :, 0] = grids[0]
    cores.append(first)
    for k in range(1, d - 1):
        cur = np.zeros((2, int(modes[k]), 2))
        cur[0, :, 0] = 1.0
        cur[1, :, 0] = grids[k]
        cur[1, :, 1] = 1.0
        cores.append(cur)
    last = np.ones((2, int(modes[-1]), 1))
    last[1, :, 0] = grids[-1]
    cores.append(last)
    return vector.from_list(cores)


# --- kernels -----------------------------------------------------------------

def constant_kernel(n_modes):
    """``K == 1``: a single rank-1 term.  The kernel of the analytic test."""
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    e = tools.ones(modes)
    return [(e, e)]


def additive_kernel(n_modes, h):
    """``K(u; v) = sum_i u_i + sum_i v_i``: two rank-1 terms."""
    modes = np.asarray(n_modes, dtype=np.int64).ravel()
    e = tools.ones(modes)
    s = component_sum(modes, h)
    return [(e, s), (s, e)]


# --- the convolution (the paper's Algorithm 1) -------------------------------

def trapezoidal_convolution(f, g, h, eps=1e-10):
    """Lower-triangular trapezoidal convolution of two TT tensors.

    Computes, for every grid point ``v`` of the common uniform grid,

        C(v) = int_0^{v_1} .. int_0^{v_d} f(v - u) g(u) du

    by the tensor product of the one-dimensional trapezoidal rules -- i.e.
    ``C[i] = h^d sum_{j <= i} w_j^{(i)} f[i - j] g[j]`` with the endpoint
    weights ``w_0^{(i)} = w_i^{(i)} = 1/2`` -- and returns it as a
    :class:`tt.vector`.  ``C`` vanishes wherever any ``v_k = 0``, as the
    integral does.

    This is Algorithm 1 of the reference: per mode, halve the ``v_k = 0``
    slice of both cores (which *is* the trapezoidal rule, see the module
    docstring), zero-pad the mode to ``2 n_k``, FFT along it; then form the
    elementwise product of the two trains (cores multiply over the rank
    indices, ranks multiply); then per mode inverse-FFT, scale by ``h_k``,
    keep the first ``n_k`` entries and zero the ``v_k = 0`` slice.

    Args:
        f, g: TT tensors with identical mode sizes.
        h: grid step, scalar or one per mode.
        eps: relative accuracy of the final rounding.

    Returns:
        The convolution, rounded to ``eps``.  Its ranks are at most
        ``r_f * r_g`` before that rounding.

    Cost: ``O(d R^4 N log N)`` for ranks ``R`` and ``N`` nodes per mode --
    ``2 R^2`` forward transforms of length ``2N``, ``R^4`` inverse ones, and
    the rounding of a rank-``R^2`` train.
    """
    fc, gc = vector.to_list(f), vector.to_list(g)
    if len(fc) != len(gc):
        raise ValueError(f"f has {len(fc)} modes, g has {len(gc)}")
    d = len(fc)
    hs = _steps(h, d)
    fn = [np.asarray(bk.to_numpy(c)) for c in fc]
    gn = [np.asarray(bk.to_numpy(c)) for c in gc]
    for k in range(d):
        if fn[k].shape[1] != gn[k].shape[1]:
            raise ValueError(
                f"mode {k}: f has {fn[k].shape[1]} nodes, g has {gn[k].shape[1]}")
    real_in = all(np.isrealobj(c) for c in fn) and all(np.isrealobj(c) for c in gn)

    out = []
    for k in range(d):
        nk = fn[k].shape[1]
        a = np.array(fn[k], dtype=np.complex128)
        b = np.array(gn[k], dtype=np.complex128)
        # trapezoidal endpoint weights, folded into the factors
        a[:, 0, :] *= 0.5
        b[:, 0, :] *= 0.5
        # linear (not circular) convolution: pad the mode to double length
        a = np.fft.fft(a, n=2 * nk, axis=1)
        b = np.fft.fft(b, n=2 * nk, axis=1)
        # elementwise product of the two trains: Kronecker over the ranks,
        # (f-index major, g-index minor) on both sides, consistently per core
        q = np.einsum("aib,cid->acibd", a, b)
        q = q.reshape((a.shape[0] * b.shape[0], 2 * nk,
                       a.shape[2] * b.shape[2]))
        q = np.fft.ifft(q, axis=1) * hs[k]
        q = q[:, :nk, :].copy()          # drop the wrap-around tail
        q[:, 0, :] = 0.0                 # int_0^0 = 0
        out.append(q if not real_in else q.real)

    return vector.from_list(tools._like(out, f)).round(eps)


# --- the right-hand side -----------------------------------------------------

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
        mass = float(tools.dot(w, ku * n))
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
