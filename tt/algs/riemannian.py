"""The tangent space of the fixed-rank TT manifold: projection and retraction.

The set of TT tensors of a fixed rank ``r`` is a smooth embedded submanifold of
the full tensor space.  Every first-order method on it (Riemannian gradient
descent, dynamical low-rank approximation, TT completion by Riemannian
optimization) needs exactly two operations, and they are what this module
provides:

* :func:`project` -- the orthogonal projection ``P_X Z`` of an arbitrary tensor
  onto the tangent space at ``X``.  In closed form (Lubich, Oseledets,
  Vandereycken 2015, Thm 3.1)

      P_X = sum_{k=1}^{d} P_{<k} (x) I_k (x) P_{>k}
          - sum_{k=1}^{d-1} P_{<=k} (x) P_{>k}

  with ``P_{<k}`` the orthogonal projector onto the column space of the ``k-1``
  st unfolding of ``X`` and ``P_{>k}`` the one onto its row space.  The result
  is again a TT tensor, of rank at most ``2 r``: the ``k``-th term contributes
  a "delta core" sandwiched between the left-orthogonal frame of ``X`` on its
  left and the right-orthogonal frame on its right, and all ``d`` terms fit
  into one core list of doubled ranks.

* :func:`projector_splitting_add` -- the retraction ``Y <- Y + delta`` computed
  by the Lie-Trotter splitting of the tangent-space projector (LOV 2015,
  section 4.2).  The result has exactly the ranks of ``Y``.  Its distinguishing
  property is *exactness*: if ``Y + delta`` happens to lie on the manifold, the
  splitting returns it exactly, not approximately -- which is also the sharpest
  available test of an implementation.

:func:`tt_qr` is the supporting QR of a TT tensor (orthogonalize all cores in
one direction, keep the remaining triangular factor).

Everything is written for both real and complex tensors: the frames enter every
contraction conjugated, which is what makes ``P_X`` the orthogonal projector for
the Hermitian inner product.  For real input the conjugations are no-ops and the
formulas coincide with the legacy code.

References
----------
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor
  trains", SIAM J. Numer. Anal. 53(2):917-941, 2015, arXiv:1407.2042.
* C. Lubich, I. V. Oseledets, "A projector-splitting integrator for dynamical
  low-rank approximation", BIT 54(1):171-188, 2014, arXiv:1301.1058.
* Replaces ``tt/riemannian/riemannian.py`` of legacy ttpy.  The ``numba``
  fast path of that file is gone: it was a hand-unrolled six-fold loop that
  duplicated the mathematics of the plain path (two copies of one truth, one of
  which was only exercised when all the ``Z`` ranks happened to be equal), and
  the same contractions written as ``einsum`` are faster without a compiler.
  The ``debug=True`` assertions of the legacy file are gone as well; they are
  now tests against a dense projector built independently of this code.
"""

from __future__ import annotations

import numpy as np
from einops import einsum, rearrange

from .. import backend as bk
from ..core import _ops
from ..core.vector import vector
from . import _localops as lo

__all__ = ["project", "projector_splitting_add", "tt_qr",
           "cores_orthogonalization_step"]


def cores_orthogonalization_step(coresX, dim, left_to_right=True):
    """One QR step of a TT orthogonalization sweep, in place on the list.

    Args:
        coresX: List of cores; ``coresX[dim]`` and its neighbour are replaced.
        dim: Site to orthogonalize.
        left_to_right: ``True`` makes ``coresX[dim]`` left-orthogonal
            (``Q^H Q = I``) and pushes the triangular factor into ``dim + 1``;
            ``False`` makes it right-orthogonal (``Q Q^H = I``) and pushes into
            ``dim - 1``.

    Returns:
        The same list, mutated.  Mode sizes are preserved, ranks may shrink.
    """
    if left_to_right:
        if not 0 <= dim < len(coresX) - 1:
            raise IndexError(f"left-to-right step needs 0 <= dim < d-1, got {dim}")
        q, s = lo.left_orthogonalize(coresX[dim])
        coresX[dim] = q
        coresX[dim + 1] = einsum(s, coresX[dim + 1], "c a, a n b -> c n b")
    else:
        if not 0 < dim < len(coresX):
            raise IndexError(f"right-to-left step needs 0 < dim < d, got {dim}")
        s, q = lo.right_orthogonalize(coresX[dim])
        coresX[dim] = q
        coresX[dim - 1] = einsum(coresX[dim - 1], s, "a n b, b c -> a n c")
    return coresX


def _check_same_modes(x, others, what):
    n = np.asarray(x.n)
    for z in others:
        if not np.array_equal(np.asarray(z.n), n):
            raise ValueError(
                f"{what}: mode sizes differ, {list(n)} vs {list(np.asarray(z.n))}")
        if z.d != x.d:
            raise ValueError(f"{what}: dimension mismatch, {x.d} vs {z.d}")


def _check_boundary(cores, what):
    if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
        raise ValueError(
            f"{what} needs boundary ranks equal to 1, got r[0]="
            f"{cores[0].shape[0]}, r[d]={cores[-1].shape[2]}")


def project(X, Z):
    """Orthogonal projection of ``Z`` onto the tangent space of the manifold at ``X``.

    Args:
        X: A :class:`tt.vector`; the point of the fixed-rank manifold.  Its
            ranks are made minimal first (``X.round(0)``), because a redundant
            rank is a point where the manifold is not smooth and the formula
            below stops meaning anything.
        Z: A :class:`tt.vector`, or a list of them.  For a list the projection
            of the *sum* is returned, ``P_X(sum_i Z_i)``, computed without ever
            forming the sum (whose rank would be the sum of the ranks).

    Returns:
        tt.vector: ``P_X Z``, with TT ranks ``2 r(X)`` (the representation is
        not rank-minimal; call ``.round(0)`` if the minimal one is wanted).

    Note:
        ``P_X`` is a projector: ``P_X P_X = P_X``, ``P_X z = z`` for tangent
        ``z``, and ``<Z - P_X Z, t> = 0`` for every tangent ``t``.  All three
        are checked in the tests, against a dense projector assembled from the
        SVDs of the unfoldings of ``X`` -- i.e. from ``X`` alone, not from this
        code.
    """
    z_list = [Z] if isinstance(Z, vector) else list(Z)
    if not z_list:
        raise ValueError("project got an empty list of tensors")
    for z in z_list:
        if not isinstance(z, vector):
            raise TypeError(f"project expects tt.vectors, got {type(z)!r}")
    if not isinstance(X, vector):
        raise TypeError(f"project expects a tt.vector X, got {type(X)!r}")

    X = X.round(eps=0)
    _check_same_modes(X, z_list, "project")
    d, n = X.d, [int(v) for v in X.n]
    coresX = list(X.cores)
    _check_boundary(coresX, "project")
    for z in z_list:
        _check_boundary(list(z.cores), "project")
    coresZ = [list(z.cores) for z in z_list]

    if d == 1:
        # The manifold is the whole space (rank 1 boundary to boundary): the
        # tangent space is everything and the projection is the sum itself.
        out = coresZ[0][0]
        for cz in coresZ[1:]:
            out = out + cz[0]
        return vector.from_list([out])

    dtype = bk.result_dtype(bk.dtype_of(coresX[0]),
                            *[bk.dtype_of(c[0]) for c in coresZ])
    coresX = _ops.to_dtype(coresX, dtype)
    coresZ = [_ops.to_dtype(c, dtype) for c in coresZ]
    rx = _ops.ranks(coresX)

    # --- right-to-left: right-orthogonal frames V_k, and the right interfaces -
    coresX = _ops.orthogonalize(coresX, center=0)
    if _ops.ranks(coresX) != rx:
        raise RuntimeError(
            "orthogonalization changed the TT ranks of X after rounding to "
            f"minimal ranks ({rx} -> {_ops.ranks(coresX)}); X is rank deficient "
            "and the tangent space is not defined there")

    coresP = []
    for k in range(d):
        r1 = 1 if k == 0 else 2 * rx[k]
        r2 = 1 if k == d - 1 else 2 * rx[k + 1]
        coresP.append(bk.zeros((r1, n[k], r2), dtype=dtype, like=coresX[0]))
    for k in range(1, d):
        # top-left block of every core but the first: the right-orthogonal frame
        r2 = 1 if k == d - 1 else rx[k + 1]
        coresP[k][:rx[k], :, :r2] = coresX[k]

    # rhs[idx][k]: (rz_k, rx_k), the contraction of Z_k..Z_{d-1} against the
    # conjugated right-orthogonal frame V_k..V_{d-1}
    rhs = [[None] * (d + 1) for _ in z_list]
    for j, cz in enumerate(coresZ):
        rhs[j][d] = bk.eye(1, 1, dtype=dtype, like=coresX[0])
        for k in range(d - 1, 0, -1):
            tmp = einsum(coresX[k].conj(), rhs[j][k + 1], "a i b, c b -> a i c")
            rhs[j][k] = einsum(cz[k], tmp, "p i c, a i c -> p a")

    # --- left to right: left-orthogonal frames Q_k and the delta cores --------
    lhs = [bk.eye(1, 1, dtype=dtype, like=coresX[0]) for _ in z_list]
    for k in range(d):
        if k < d - 1:
            coresX = cores_orthogonalization_step(coresX, k, left_to_right=True)
            if coresX[k].shape[2] != rx[k + 1]:
                raise RuntimeError(
                    f"left orthogonalization dropped rank at site {k}: "
                    f"{coresX[k].shape[2]} != {rx[k + 1]}")
        q = coresX[k]
        r1, _, r2 = q.shape
        for j, cz in enumerate(coresZ):
            proj = einsum(lhs[j], cz[k], "a p, p i s -> a i s")
            if k < d - 1:
                lhs_new = einsum(q.conj(), proj, "a i b, a i s -> b s")
                delta = proj - einsum(q, lhs_new, "a i b, b s -> a i s")
                delta = einsum(delta, rhs[j][k + 1], "a i s, s b -> a i b")
                lhs[j] = lhs_new
            else:
                delta = proj                       # rz_d = 1, nothing to project out
            if k == 0:
                coresP[k][:, :, :r2] = coresP[k][:, :, :r2] + delta
            else:
                coresP[k][rx[k]:, :, :r2] = coresP[k][rx[k]:, :, :r2] + delta
        if k < d - 1:
            if k == 0:
                coresP[k][:, :, r2:] = q
            else:
                coresP[k][rx[k]:, :, r2:] = q

    return vector.from_list(coresP)


def projector_splitting_add(Y, delta):
    """``Y + delta``, retracted onto the manifold of TT tensors of rank ``r(Y)``.

    This is the Lie-Trotter projector splitting of LOV 2015, section 4.2: a
    single left-to-right sweep of alternating "K" steps (update a core with the
    part of ``delta`` seen through the current frames) and "S" steps (undo the
    part that the next K step will count twice).  It costs one sweep and no SVD,
    and unlike ``(Y + delta).round(...)`` it never leaves the fixed-rank
    manifold.

    Args:
        Y: A :class:`tt.vector`, the base point.
        delta: A :class:`tt.vector`, the increment; arbitrary rank.

    Returns:
        tt.vector: a tensor with exactly the ranks of ``Y``.

    Note:
        Exactness (LOV Thm 4.1): if ``Y + delta`` is itself of rank ``r(Y)``,
        the result *is* ``Y + delta``, in exact arithmetic and to roundoff in
        practice.  Otherwise it is a first-order accurate retraction:
        ``psa(Y, t Z) = Y + t P_Y Z + O(t^2)``.  Both are tested.
    """
    if not isinstance(Y, vector) or not isinstance(delta, vector):
        raise TypeError("projector_splitting_add expects two tt.vectors, got "
                        f"{type(Y)!r} and {type(delta)!r}")
    delta = delta.round(eps=0)
    _check_same_modes(Y, [delta], "projector_splitting_add")
    d = Y.d
    coresY = list(Y.cores)
    coresD = list(delta.cores)
    _check_boundary(coresY, "projector_splitting_add")
    _check_boundary(coresD, "projector_splitting_add")

    dtype = bk.result_dtype(bk.dtype_of(coresY[0]), bk.dtype_of(coresD[0]))
    coresY = list(_ops.to_dtype(coresY, dtype))
    coresD = list(_ops.to_dtype(coresD, dtype))
    if d == 1:
        return vector.from_list([coresY[0] + coresD[0]])

    ry = _ops.ranks(coresY)
    coresY = _ops.orthogonalize(coresY, center=0)
    if _ops.ranks(coresY) != ry:
        raise RuntimeError(
            f"orthogonalization changed the ranks of Y ({ry} -> "
            f"{_ops.ranks(coresY)}); Y is rank deficient and the retraction "
            "onto its rank is not defined")

    rhs = [None] * (d + 1)
    rhs[d] = bk.eye(1, 1, dtype=dtype, like=coresY[0])
    for k in range(d - 1, 0, -1):
        tmp = einsum(coresY[k].conj(), rhs[k + 1], "a i b, c b -> a i c")
        rhs[k] = einsum(coresD[k], tmp, "p i c, a i c -> p a")

    lhs = bk.eye(1, 1, dtype=dtype, like=coresY[0])          # (ry_k, rd_k)
    s = bk.eye(1, 1, dtype=dtype, like=coresY[0])            # (ry_k, ry_k)
    for k in range(d):
        # K step: the core absorbs delta seen through the current frames.
        kpart = einsum(lhs, coresD[k], "a p, p i q -> a i q")
        cc = einsum(kpart, rhs[k + 1], "a i q, q b -> a i b")
        cc = cc + einsum(s, coresY[k], "a c, c i b -> a i b")
        if k == d - 1:
            coresY[k] = cc
            break
        q, rr = lo.left_orthogonalize(cc)
        if q.shape[2] != ry[k + 1]:
            raise RuntimeError(
                f"the K step lost rank at site {k}: {q.shape[2]} != {ry[k + 1]}; "
                "the projector splitting is not defined at a rank-deficient point")
        coresY[k] = q
        # S step: subtract what the next K step would otherwise count twice.
        lhs = einsum(q.conj(), kpart, "a i b, a i q -> b q")
        s = rr - einsum(lhs, rhs[k + 1], "b q, q c -> b c")
    return vector.from_list(coresY)


def tt_qr(X, left_to_right=True):
    """QR factorization of a TT tensor.

    Args:
        X: A :class:`tt.vector`.
        left_to_right: ``True`` orthogonalizes every core from the left, so the
            unfoldings of the result have orthonormal columns and the remaining
            factor ``R`` multiplies the *last* rank index; ``False`` does the
            mirror image and ``R`` multiplies the first.

    Returns:
        tuple: ``(Q, R)`` -- a :class:`tt.vector` with orthonormal cores and a
        small matrix.  For the usual boundary ranks 1 both are trivial to use:
        ``R`` is ``1x1`` and ``X == R[0, 0] * Q``, so ``Q`` carries the shape
        and ``R`` the norm and the sign.  In general
        ``X = Q x_last R`` (or ``R x_first Q``).

    Note:
        Redundant ranks are removed first (``X.round(0)``); a QR of a
        rank-deficient representation would silently return cores whose
        columns are not independent.
    """
    if not isinstance(X, vector):
        raise TypeError(f"tt_qr expects a tt.vector, got {type(X)!r}")
    X = X.round(eps=0)
    d = X.d
    cores = list(X.cores)
    if d == 1:
        c = cores[0]
        n = c.shape[1]
        if left_to_right:
            q, rr = bk.qr(rearrange(c, "a n b -> (a n) b"))
            return vector.from_list([rearrange(q, "(a n) c -> a n c", n=n)]), rr
        rr, q = lo.right_orthogonalize(c)
        return vector.from_list([q]), rr

    if left_to_right:
        for k in range(d - 1):
            cores = cores_orthogonalization_step(cores, k, left_to_right=True)
        q, rr = lo.left_orthogonalize(cores[d - 1])
        cores[d - 1] = q
    else:
        for k in range(d - 1, 0, -1):
            cores = cores_orthogonalization_step(cores, k, left_to_right=False)
        rr, q = lo.right_orthogonalize(cores[0])
        cores[0] = q
    return vector.from_list(cores), rr
