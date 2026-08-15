"""Lower-triangular convolution of TT tensors on a uniform grid.

    q(v) = int_0^{v_1} .. int_0^{v_d} f(v - u) g(u) du,   v on a uniform grid

evaluated at every node of the grid at once, by the trapezoidal rule, in
``O(d R_f^2 R_g^2 N log N)``: each mode is zero-padded to ``2N`` and taken
through an FFT, the two trains are multiplied mode-wise, and the inverse
transform is cut back to the first ``N`` entries.  The direct quadrature
would cost ``O(N^{2d})``.

This is the convolution of Algorithm 1 of S. A. Matveev, D. A. Zheltkov,
E. E. Tyrtyshnikov, A. P. Smirnov, *Tensor train versus Monte Carlo for the
multicomponent Smoluchowski coagulation equation*, J. Comput. Phys.
316:164-179, 2016 (doi:10.1016/j.jcp.2016.04.025), lifted out of that
application because a lower-triangular convolution is not specific to
coagulation: the same call serves any Volterra-type convolution on a
uniform tensor grid.  The coagulation model that motivated it lives in
``examples/smoluchowski/``.

The ranks multiply before they are rounded: ``q`` starts at ``R_f R_g`` and
is truncated to ``eps``.  There is no theorem here -- watch the returned
ranks, they are the only monitor.
"""

from __future__ import annotations

import numpy as np

from .. import backend as bk
from ..core import tools
from ..core.vector import vector

__all__ = ["trapezoidal_convolution", "trapezoidal_weights", "component_sum"]


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
