"""Interpolatory KSL: dynamical TT approximation on DEIM cross fibers.

Ported from Alec Dektor's ``tt.ksl_deim`` (oseledets/ttpy PR #102).

One step of dynamical low-rank approximation for

    dy/dt = A y + Nf(y),    y(0) = y0,

where ``A`` is a TT-matrix and ``Nf`` a *pointwise* nonlinearity.  The
classical KSL integrator projects the right-hand side onto the tangent space
of the fixed-rank manifold *orthogonally*, which requires the full vector
``A y + Nf(y)`` -- unaffordable when ``Nf`` is a genuine nonlinearity with no
TT structure.  This integrator replaces the orthogonal tangent projector by an
*interpolatory* one (Dektor's collocation method): the right-hand side is only
ever **evaluated on cross fibers** -- entries indexed by left-nested index
sets ``I_k`` and right-nested index sets ``J_k`` chosen by QDEIM
(column-pivoted QR of the unfolded cores, Drmac & Gugercin 2016) -- and each
substep reconstructs a core update from those samples through pseudo-inverses
of the sampled interface matrices ``L_k`` (left interface evaluated at
``I_k``) and ``R_k`` (right interface at ``J_k``).

The sweep is the KS splitting of the first-order KSL: left to right, a
forward (explicit-Euler) update of core ``k`` from fiber samples,

    y_k <- y_k + tau * pinv(L_{k-1}) [A y + Nf(y)]_(I_{k-1} x n_k x J_{k+1})
                       pinv(R_{k+1}),

then QR of the updated core and a backward step on the connecting matrix,

    S <- S - tau * pinv(L_k) [A y + Nf(y)]_(I_k x J_{k+1}) pinv(R_{k+1}),

with the left-nested sets ``I_k`` regenerated from the freshly
left-orthogonalized cores as the sweep advances (so ``I_k`` extends ``I_{k-1}``
and ``J_k`` extends ``J_{k+1}`` -- the nesting is what makes every sample an
actual entry of the current iterate).

Honest limitations
------------------
* **First order only.**  The substeps are explicit Euler forward/backward
  updates, not exponential steps, and there is no Strang variant: the global
  error is O(tau) even on a manifold that contains the exact trajectory
  (unlike the orthogonal-projector KSL, which is exact there).  Being
  explicit, it also inherits the Euler stability restriction on ``tau``.
* **The projector is oblique.**  ``pinv(L)`` and ``pinv(R)`` interpolate at
  the QDEIM-selected fibers; this is *not* an orthogonal projection, and the
  approximation error carries the usual DEIM amplification factor
  ``||pinv(...)||`` on top of the best-approximation error.
* TT ranks are fixed by ``y0`` and never adapted; real arithmetic is assumed
  (the index-selection QRs use plain transposes, not conjugate transposes).
* **Interior TT-ranks of 1 are not supported** (inherited from the original):
  the interface evaluation squeezes sampled cores, and a unit interior rank
  drops one axis too many.  E.g. the ``eigb`` eigenvector of a 3D QTT
  Laplacian is a pure tensor product across the physical dimensions, with
  rank 1 at the dimension boundaries -- round such inputs up or perturb them.

Semantics of ``Nf`` (exactly as in the original)
------------------------------------------------
``Nf`` is never called on a TT object.  It receives a numpy array of
**entries of the current iterate** ``y`` evaluated at the cross fibers -- a
3-way array of fiber values for the forward steps, a matrix for the backward
steps -- and must act entrywise on it (returning an array of the same shape,
or a scalar such as ``0`` for the linear case).  This is what makes the method
collocation: ``(A y + Nf(y))`` at a sampled multi-index equals
``(A y)_index + Nf(y_index)`` because ``Nf`` is pointwise.

References
----------
1. A. Dektor, "Collocation methods for nonlinear differential equations on
   low-rank manifolds", Linear Algebra and its Applications, 2025,
   pp. 143-184.  https://doi.org/10.1016/j.laa.2024.11.001
2. A. Dektor, L. Einkemmer, "Interpolatory dynamical low-rank approximation
   for the 3+3d Boltzmann-BGK equation", arXiv:2411.15990, 2024.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from ..core.tools import matvec
from ..core.vector import vector

__all__ = ["ksl_deim"]


def ksl_deim(A, Nf, y0, tau):
    """One first-order interpolatory projector-splitting step for ``y(tau)``.

    Integrates ``dy/dt = A y + Nf(y)`` from ``y0`` over one step ``tau`` on
    the fixed-rank TT manifold of ``y0``; see the module docstring for the
    scheme and its limitations (first order, oblique projectors, fixed ranks).

    :param A: matrix in the TT-format
    :type A: tt.matrix
    :param Nf: pointwise nonlinearity; called on numpy arrays of entries of
        ``y`` sampled at cross fibers and applied entrywise (``lambda v: 0``
        for a linear problem, ``lambda v: v ** 2`` for ``y^2``, ...) -- see
        "Semantics of Nf" in the module docstring
    :type Nf: callable
    :param y0: initial condition in the TT-format; its ranks define the
        manifold and are kept
    :type y0: tt.vector
    :param tau: timestep
    :type tau: float
    :rtype: tt.vector

    :Example:
        >>> import tt
        >>> from tt.algs.ksl_deim import ksl_deim
        >>> a = tt.qlaplace_dd([8])
        >>> y0, ev = tt.eigb.eigb(a, tt.rand(2, 8, 2), 1e-8, verb=0)
        >>> y1 = ksl_deim(a, lambda v: 0, y0, 1e-2)
        >>> abs(tt.dot(y1, y0) / (y1.norm() * y0.norm()) - 1) < 1e-8
        True
    """
    r, n, d = y0.r, y0.n, y0.d
    y = [np.asarray(c) for c in vector.to_list(y0)]
    y, J = nested_J(y)
    I = d * [None]

    # ----------------- left-to-right sweep ----------------- #
    # first core forward step
    Ax_e = eval_rhs(A, Nf, y, I, J, -1, 1)
    R = right_eval(y, J, 1)
    y[0] = y[0] + tau * np.tensordot(Ax_e, np.linalg.pinv(R), axes=((2), (0)))

    # backward step
    cr = np.reshape(y[0], (r[0] * n[0], r[1]))
    cr, S = np.linalg.qr(cr)
    y[0] = np.reshape(cr, (r[0], n[0], r[1]))       # left-orth core 0
    cr2 = y[1]                                      # save for later
    y[1] = np.tensordot(S, y[1], axes=((1), (0)))   # temp. update to core 1
    I[0] = nested_Ik(y, n, r, I, 0)
    ax_e = eval_rhs(A, Nf, y, I, J, 0, 1)
    L = left_eval(y, I, 0)
    S = S - tau * (np.linalg.pinv(L) @ ax_e @ np.linalg.pinv(R))
    y[1] = np.tensordot(S, cr2, axes=((1), (0)))

    # interior cores
    for k in range(1, d - 1, 1):
        # forward step
        ax_e = eval_rhs(A, Nf, y, I, J, k - 1, k + 1)
        L = left_eval(y, I, k - 1)
        R = right_eval(y, J, k + 1)
        y[k] = y[k] + tau * np.tensordot(np.tensordot(np.linalg.pinv(L), ax_e, axes=((1,), (0,))),
                          np.linalg.pinv(R), axes=((2,), (0,)))

        # backward step
        cr = np.reshape(y[k], (r[k] * n[k], r[k + 1]))
        cr, S = np.linalg.qr(cr)
        y[k] = np.reshape(cr, (r[k], n[k], r[k + 1]))
        cr2 = y[k + 1]    # save for later
        y[k + 1] = np.tensordot(S, y[k + 1], axes=((1), (0)))
        I[k] = nested_Ik(y, n, r, I, k)
        ax_e = eval_rhs(A, Nf, y, I, J, k, k + 1)
        L = left_eval(y, I, k)
        S = S - tau * (np.linalg.pinv(L) @ ax_e @ np.linalg.pinv(R))
        y[k + 1] = np.tensordot(S, cr2, axes=((1), (0)))

    # final core forward step
    ax_e = eval_rhs(A, Nf, y, I, J, d - 2, d)
    L = left_eval(y, I, d - 2)
    y[d - 1] = y[d - 1] + tau * np.tensordot(np.linalg.pinv(L), ax_e,
                                             axes=((1), (0)))

    return vector.from_list(y)


def qdeim(M, r):
    """The first ``r`` QDEIM pivot rows of ``M`` (column-pivoted QR of M^T)."""
    row_inds = scipy.linalg.qr(M.transpose(), pivoting=True, mode='economic')[2]
    return row_inds[0:r]


def left_eval(x, I, k):
    """Evaluate the left ``k+1`` cores of TT ``x`` at the multi-indices ``I[k]``.

    :param x: cores of a vector in the TT-format
    :type x: list
    :param I: left nested multi-indices
    :type I: list
    :param k: index of the left interface to evaluate
    :type k: int
    :rtype: matrix of shape ``(len(I[k]), r[k+1])``
    """
    L = x[0][:, I[k][:, 0], :]
    L = np.squeeze(L)

    for i in range(1, k + 1, 1):
        L = np.tensordot(L, x[i][:, I[k][:, i], :], axes=((1), (0)))
        L = np.diagonal(L).T
    return L


def nested_Ik(x, n, r, I, k):
    """One left-nested index set ``I_k`` extending ``I_{k-1}`` by QDEIM.

    Pivots are taken from the unfolding of the left-orthogonal part
    (interface at ``I_{k-1}`` times core ``k``); each new multi-index is an
    old one with one mode index appended, which keeps the sets nested.

    :param x: cores of a vector in the TT-format
    :type x: list
    :param n: mode sizes
    :type n: array
    :param r: TT-ranks
    :type r: array
    :param I: left nested multi-indices built so far
    :type I: list
    :param k: which set to build
    :type k: int
    :rtype: integer matrix of shape ``(r[k+1], k+1)``
    """
    if k == 0:
        cr = np.reshape(x[0], (r[0] * n[0], r[1]))
        Ik = qdeim(cr, r[1])
        Ik = Ik[:, np.newaxis]
    else:
        L = left_eval(x, I, k - 1)   # evaluate left part of tt at preceding multi-indices
        T = np.tensordot(L, x[k], axes=((1), (0)))
        T = np.reshape(T, (r[k] * n[k], r[k + 1]))

        p = qdeim(T, r[k + 1])
        [p1, p2] = np.unravel_index(p, (r[k], n[k]))   # split indices
        p2 = p2[:, np.newaxis]
        Ik = np.hstack((I[k - 1][p1, :], p2))   # nested multi-indices
    return Ik


def nested_I(x):
    """Left-orthogonalize ``x`` and build all left nested index sets ``I``.

    Not used by :func:`ksl_deim` (which rebuilds ``I_k`` core by core as the
    sweep left-orthogonalizes); kept as in the original for standalone use.

    :param x: cores of a vector in the TT-format
    :type x: list
    :rtype: (list of cores, list of index sets)
    """
    d = len(x)
    n = np.zeros(d, dtype=np.int32)
    r = np.ones(d + 1, dtype=np.int32)
    for i in range(d):
        [_, n[i], r[i + 1]] = x[i].shape

    I = d * [None]
    for k in range(d - 1):
        # left-orth core k
        cr = np.reshape(x[k], (r[k] * n[k], r[k + 1]))
        cr, R = np.linalg.qr(cr)
        x[k] = np.reshape(cr, (r[k], n[k], r[k + 1]))
        x[k + 1] = np.tensordot(R, x[k + 1], axes=((1), (0)))

        I[k] = nested_Ik(x, n, r, I, k)   # nested multi-index set
    return x, I


def right_eval(x, J, k):
    """Evaluate cores ``k..d-1`` of TT ``x`` at the multi-indices ``J[k]``.

    :param x: cores of a vector in the TT-format
    :type x: list
    :param J: right nested multi-indices
    :type J: list
    :param k: index of the right interface to evaluate
    :type k: int
    :rtype: matrix of shape ``(r[k], len(J[k]))``
    """
    d = len(x)
    R = x[k][:, J[k][:, 0], :]

    p = 1
    for i in range(k + 1, d):
        R = np.tensordot(R, x[i][:, J[k][:, p], :], axes=((2), (0)))
        R = np.diagonal(R, axis1=1, axis2=2)
        R = np.transpose(R, (0, 2, 1))
        p += 1
    return np.squeeze(R)


def nested_Jk(x, n, r, J, k):
    """One right-nested index set ``J_k`` extending ``J_{k+1}`` by QDEIM.

    :param x: cores of a vector in the TT-format
    :type x: list
    :param n: mode sizes
    :type n: array
    :param r: TT-ranks
    :type r: array
    :param J: right nested multi-indices built so far
    :type J: list
    :param k: which set to build
    :type k: int
    :rtype: integer matrix of shape ``(r[k], d-k)``
    """
    d = len(x)
    if k == d - 1:
        cr = np.reshape(x[d - 1], (r[d - 1], n[d - 1] * r[d]))
        Jk = qdeim(cr.T, r[d - 1])
        Jk = Jk[:, np.newaxis]
    else:
        R = right_eval(x, J, k + 1)   # evaluate right part of tt at preceding multi-indices
        T = np.tensordot(x[k], R, axes=((2), (0)))
        T = np.reshape(T, (r[k], n[k] * r[k + 1]))
        p = qdeim(T.T, r[k])                             # qDEIM
        [p1, p2] = np.unravel_index(p, (n[k], r[k + 1]))   # split indices
        p1 = p1[:, np.newaxis]
        Jk = np.hstack((p1, J[k + 1][p2, :]))   # nested multi-index set
    return Jk


def nested_J(x):
    """Right-orthogonalize ``x`` and build all right nested index sets ``J``.

    :param x: cores of a vector in the TT-format
    :type x: list
    :rtype: (list of cores, list of index sets)
    """
    d = len(x)
    n = np.zeros(d, dtype=np.int32)
    r = np.ones(d + 1, dtype=np.int32)
    for i in range(d):
        [_, n[i], r[i + 1]] = x[i].shape

    J = d * [None]
    for k in range(d - 1, 0, -1):
        # right-orth core k
        cr = np.reshape(x[k], (r[k], n[k] * r[k + 1]))
        cr, R = np.linalg.qr(cr.T)
        x[k] = np.reshape(cr.T, (r[k], n[k], r[k + 1]))
        x[k - 1] = np.tensordot(x[k - 1], R.T, axes=((2), (0)))

        J[k] = nested_Jk(x, n, r, J, k)   # nested multi-index set
    return x, J


def eval_rhs(A, Nf, y, I, J, k1, k2):
    """Evaluate ``A y + Nf(y)`` on the cross fibers defined by ``I, J, k1, k2``.

    ``k2 - k1 == 2`` samples the 3-way fiber tensor
    ``(I[k1], :, J[k2])`` used by the forward core steps (``k1 == -1`` and
    ``k2 == d`` drop the missing side at the boundary); ``k2 - k1 == 1``
    samples the matrix ``(I[k1], J[k2])`` used by the backward S steps.
    ``Nf`` is applied entrywise to the sampled values of ``y`` -- this is the
    only place the nonlinearity is ever evaluated.

    :param A: matrix in the TT-format
    :type A: tt.matrix
    :param Nf: pointwise nonlinearity (see module docstring)
    :type Nf: callable
    :param y: cores of the current iterate
    :type y: list
    :param I: left nested indices
    :type I: list
    :param J: right nested indices
    :type J: list
    :param k1: left-nested index set used for evaluation
    :type k1: int
    :param k2: right-nested index set used for evaluation
    :type k2: int
    """
    d = len(y)
    Ay = matvec(A, vector.from_list(y))
    Ay = [np.asarray(c) for c in vector.to_list(Ay)]

    if k2 - k1 == 2:   # 3d tensor forward step
        k = k1 + 1
        if k == 0:
            RAy = right_eval(Ay, J, k + 1)
            Ay_e = np.tensordot(Ay[k], RAy, axes=((2), (0)))

            Ry = right_eval(y, J, k + 1)
            y_e = np.tensordot(y[k], Ry, axes=((2), (0)))
        elif k == d - 1:
            LAy = left_eval(Ay, I, k - 1)
            Ay_e = np.tensordot(LAy, Ay[k], axes=((1), (0)))

            Ly = left_eval(y, I, k - 1)
            y_e = np.tensordot(Ly, y[k], axes=((1), (0)))
        else:
            LAy = left_eval(Ay, I, k - 1)
            RAy = right_eval(Ay, J, k + 1)
            Ay_e = np.tensordot(np.tensordot(LAy, Ay[k], axes=((1,), (0,))),
                          RAy, axes=((2,), (0,)))

            Ly = left_eval(y, I, k - 1)
            Ry = right_eval(y, J, k + 1)
            y_e = np.tensordot(np.tensordot(Ly, y[k], axes=((1,), (0,))),
                          Ry, axes=((2,), (0,)))

    elif k2 - k1 == 1:   # 2d tensor backward step
        LAy = left_eval(Ay, I, k1)
        RAy = right_eval(Ay, J, k2)
        Ay_e = LAy @ RAy

        Ly = left_eval(y, I, k1)
        Ry = right_eval(y, J, k2)
        y_e = Ly @ Ry
    else:
        raise ValueError('Invalid evaluation indices k_1,k_2')

    return Ay_e + Nf(y_e)
