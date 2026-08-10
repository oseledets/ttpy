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

from dataclasses import dataclass

import numpy as np
from einops import rearrange
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from ..core import _ops
from ..core.vector import vector
from . import _localops as lo

__all__ = ["project", "projector_splitting_add", "tt_qr",
           "cores_orthogonalization_step",
           "Frames", "frames", "project_delta", "tangent_to_tt",
           "tangent_inner", "tangent_gram", "retract", "transport"]


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


def _left_step_checked(cores, k, what):
    """The left-to-right QR step of :func:`project`, refusing a rank drop.

    The tangent space of the fixed-rank manifold is defined only at a point of
    *exactly* that rank.  At a rank-deficient representation the closed-form
    projector still returns a Hermitian idempotent of the right trace -- it
    just projects onto a strictly larger space than the caller asked for, so
    no invariant can see the mistake (``docs/NUMERICS.md``).

    The test is exact rather than heuristic and costs nothing beyond the QR the
    sweep performs anyway.  Once the cores right of ``k`` are right-orthogonal
    and the cores left of ``k`` have been made left-orthogonal, the triangular
    factor ``R_k`` of this step carries *exactly* the singular values of the
    ``(k+1)``-st unfolding of the tensor -- both surrounding frames are
    orthonormal and change nothing.  So a rank drop of that unfolding is a
    small singular value of ``R_k``, measured against its largest, at the
    ordinary LAPACK numerical-rank threshold.

    Args:
        cores: Core list, mutated at ``k`` and ``k + 1`` like
            :func:`cores_orthogonalization_step`.
        k: Site to orthogonalize; ``0 <= k < d - 1``.
        what: Name of the caller, used in the message.

    Returns:
        The same list.

    Raises:
        ValueError: unfolding ``k + 1`` is numerically rank deficient.  The
            message carries the stated rank, the numerical rank and the
            measured singular value ratio.
    """
    q, s = lo.left_orthogonalize(cores[k])
    sv = np.asarray(bk.to_numpy(bk.svd(s)[1]), dtype=np.float64)
    tol = max(s.shape) * bk.eps_of(bk.dtype_of(s))
    if sv.size and sv[-1] <= tol * sv[0]:
        raise ValueError(
            f"{what}: unfolding {k + 1} of X has TT rank {cores[k].shape[2]} "
            f"but numerical rank {int(np.sum(sv > tol * sv[0]))} "
            f"(smallest/largest singular value = {sv[-1] / sv[0]:.2e} <= "
            f"{tol:.2e}).  The tangent space of the fixed-rank manifold is not "
            "defined at a rank-deficient point; round X onto its true rank "
            "first -- X.round(1e-14), not X.round(0), which keeps every "
            "singular value by definition and removes nothing.")
    cores[k] = q
    cores[k + 1] = einsum(s, cores[k + 1], "c a, a n b -> c n b")
    return cores


def project(X, Z):
    """Orthogonal projection of ``Z`` onto the tangent space of the manifold at ``X``.

    Args:
        X: A :class:`tt.vector`; the point of the fixed-rank manifold.  Its
            representation must have exactly the rank it claims: a redundant
            rank is a corner of the manifold, where the tangent space is not
            defined and the formula below silently projects onto a larger
            space.  This is checked (:func:`_require_full_rank`) and refused,
            because the wrong answer is otherwise indistinguishable from the
            right one (``docs/NUMERICS.md``).
        Z: A :class:`tt.vector`, or a list of them.  For a list the projection
            of the *sum* is returned, ``P_X(sum_i Z_i)``, computed without ever
            forming the sum (whose rank would be the sum of the ranks).

    Returns:
        tt.vector: ``P_X Z``, with TT ranks ``2 r(X)`` (the representation is
        not rank-minimal; call ``.round(1e-14)`` if the minimal one is wanted).

    Raises:
        ValueError: ``X`` is rank deficient (see above), the mode sizes of
            ``X`` and ``Z`` differ, or a boundary rank is not 1.

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

    # round(0) only drops ranks that are structurally impossible (r_{k+1} above
    # r_k n_k); it cannot drop a numerically deficient one, because "discarded
    # tail below eps = 0" is false for every nonzero tail by definition.  The
    # numerical check is _require_full_rank below.
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
            # the same QR as cores_orthogonalization_step, plus the check that
            # the point really has the rank it claims (see _left_step_checked)
            coresX = _left_step_checked(coresX, k, "project")
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

    Note:
        Unlike :func:`project`, a numerically rank-deficient ``Y`` is *not*
        refused here: the splitting only ever needs the frames themselves, not
        the space they are supposed to span exactly, and exactness was measured
        to hold at such a point (``docs/NUMERICS.md``).  The retraction is then
        onto the manifold of the *stated* rank, which is what the caller asked
        for.
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
        Structurally impossible ranks are removed first (``X.round(0)``).  A
        *numerically* rank-deficient representation is left alone and is not an
        error here: ``bk.qr`` still returns cores with orthonormal columns and
        ``X = Q R`` still holds to roundoff -- only the columns of ``Q`` that
        correspond to a zero on the diagonal of ``R`` are arbitrary; verified at
        such a point (``docs/NUMERICS.md``).  (:func:`project` does refuse one,
        because there the deficiency changes the answer.)
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


# --- the tangent representation: one owner (plan riemannian-autodiff sec. 8) --

@dataclass
class Frames:
    """The two orthogonal frame families of a full-rank point ``X``.

    Attributes:
        U: ``U[k]`` left-orthogonal for ``k < d-1``; ``U[d-1]`` is the
            ``mu = d`` core ``S_d``.
        V: ``V[k]`` right-orthogonal for ``k > 0``; ``V[0]`` is the ``mu = 1``
            core ``S_1``.
        S: The ``mu``-orthogonal core that was requested from :func:`frames`.
        r: The TT ranks, as verified full.
    """

    U: list
    V: list
    S: object
    r: list


def frames(X, mu=1, *, check_rank=True):
    """Left- and right-orthogonal frames of ``X`` and its ``mu``-orthogonal core.

    This is the first half of :func:`project`, factored out as the single owner
    every tangent-space routine shares (``project_delta``, ``tangent_to_tt``,
    ``transport``, the Riemannian autodiff).

    Args:
        X: A :class:`tt.vector` of exactly the rank it claims.
        mu: 1-based site of the non-orthogonal core in ``Frames.S``.
        check_rank: Verify at every left step that the point really has full
            rank (:func:`_left_step_checked`); a rank-deficient point has no
            tangent space and is refused, exactly as in :func:`project`.

    Raises:
        ValueError: rank deficiency, a bad boundary rank, or ``mu`` outside
            ``1..d``.
    """
    if not isinstance(X, vector):
        raise TypeError(f"frames expects a tt.vector, got {type(X)!r}")
    X = X.round(eps=0)
    d = X.d
    if not 1 <= int(mu) <= d:
        raise ValueError(f"mu must be in 1..{d}, got {mu}")
    cores = list(X.cores)
    _check_boundary(cores, "frames")
    rx = _ops.ranks(cores)
    coresR = _ops.orthogonalize(cores, center=0)
    if _ops.ranks(coresR) != rx:
        raise ValueError(
            "orthogonalization changed the TT ranks of X "
            f"({rx} -> {_ops.ranks(coresR)}); X is rank deficient and the "
            "tangent space is not defined there")
    V = list(coresR)                       # V[0] = S_1, V[k>0] right-orthogonal
    cur = list(coresR)
    S = cur[0] if int(mu) == 1 else None
    U = []
    for k in range(d - 1):
        if check_rank:
            cur = _left_step_checked(cur, k, "frames")
        else:
            cores_orthogonalization_step(cur, k, left_to_right=True)
        U.append(cur[k])
        if int(mu) == k + 2:
            S = cur[k + 1]
    U.append(cur[d - 1])                   # the mu = d core S_d
    return Frames(U=U, V=V, S=S, r=list(rx))


def project_delta(X, Z, *, weights=None, frames_=None):
    """Gauge cores of ``P_{T_X M} Z``, instead of the assembled rank-2r tensor.

    Args:
        X: The point of the manifold (full rank, verified).
        Z: A :class:`tt.vector` or a list of them; for a list the deltas of
            ``P_X(sum_j w_j Z_j)`` are computed without forming the sum -- the
            form the rank-1-sum preconditioner needs.
        weights: Scalars ``w_j`` for the list form; default all ones.
        frames_: Precomputed :class:`Frames` of ``X``, to skip the two
            orthogonalization sweeps.

    Returns:
        ``(deltas, frames)``: ``deltas[k]`` of shape ``(r_{k-1}, n_k, r_k)``
        satisfying the gauge ``ML(deltas[k])^H ML(U_k) = 0`` for ``k < d``, and
        the :class:`Frames` of ``X`` so the caller need not rebuild them.
        ``tangent_to_tt(X, deltas)`` equals :func:`project` ``(X, Z)`` to
        roundoff, and :func:`tangent_inner` on the deltas is the tangent inner
        product of [RNO19] eq. (22); both are pinned by tests.
    """
    z_list = [Z] if isinstance(Z, vector) else list(Z)
    if not z_list:
        raise ValueError("project_delta got an empty list of tensors")
    for z in z_list:
        if not isinstance(z, vector):
            raise TypeError(f"project_delta expects tt.vectors, got {type(z)!r}")
    fr = frames_ if frames_ is not None else frames(X)
    d = len(fr.U)
    _check_same_modes(X, z_list, "project_delta")
    if weights is None:
        w = [1.0] * len(z_list)
    else:
        w = [complex(v) if np.iscomplexobj(np.asarray(v)) else float(v)
             for v in weights]
        if len(w) != len(z_list):
            raise ValueError(f"{len(w)} weights for {len(z_list)} tensors")
    dtype = bk.result_dtype(bk.dtype_of(fr.V[0]),
                            *[bk.dtype_of(z.cores[0]) for z in z_list])
    coresZ = [_ops.to_dtype(list(z.cores), dtype) for z in z_list]
    U = _ops.to_dtype(fr.U, dtype)
    V = _ops.to_dtype(fr.V, dtype)
    n = [int(v) for v in X.n]

    if d == 1:
        out = w[0] * coresZ[0][0]
        for wj, cz in zip(w[1:], coresZ[1:]):
            out = out + wj * cz[0]
        return [out], fr

    # right interfaces against the conjugated right-orthogonal frames
    rhs = [[None] * (d + 1) for _ in z_list]
    for j, cz in enumerate(coresZ):
        rhs[j][d] = bk.eye(1, 1, dtype=dtype, like=V[0])
        for k in range(d - 1, 0, -1):
            tmp = einsum(V[k].conj(), rhs[j][k + 1], "a i b, c b -> a i c")
            rhs[j][k] = einsum(cz[k], tmp, "p i c, a i c -> p a")

    deltas = []
    lhs = [bk.eye(1, 1, dtype=dtype, like=V[0]) for _ in z_list]
    for k in range(d):
        acc = None
        for j, cz in enumerate(coresZ):
            proj = einsum(lhs[j], cz[k], "a p, p i s -> a i s")
            if k < d - 1:
                q = U[k]
                lhs_new = einsum(q.conj(), proj, "a i b, a i s -> b s")
                delta = proj - einsum(q, lhs_new, "a i b, b s -> a i s")
                delta = einsum(delta, rhs[j][k + 1], "a i s, s b -> a i b")
                lhs[j] = lhs_new
            else:
                delta = proj
            acc = w[j] * delta if acc is None else acc + w[j] * delta
        deltas.append(acc)
    return deltas, fr


def tangent_to_tt(X, deltas, *, frames_=None):
    """Assemble the rank-2r tangent tensor from its gauge cores.

    The inverse of :func:`project_delta`: the block ``S_k`` stack of [RNO19]
    section 4.1, identical to what :func:`project` returns.  No gauge check --
    any delta cores of the right shapes assemble.
    """
    fr = frames_ if frames_ is not None else frames(X)
    d = len(fr.U)
    if len(deltas) != d:
        raise ValueError(f"{len(deltas)} delta cores for a {d}-core point")
    if d == 1:
        return vector.from_list([deltas[0]])
    dtype = bk.result_dtype(bk.dtype_of(fr.V[0]),
                            *[bk.dtype_of(dl) for dl in deltas])
    U = _ops.to_dtype(fr.U, dtype)
    V = _ops.to_dtype(fr.V, dtype)
    dl = _ops.to_dtype(list(deltas), dtype)
    rx = fr.r
    n = [int(v) for v in X.n]
    cores = []
    for k in range(d):
        r1 = 1 if k == 0 else 2 * rx[k]
        r2 = 1 if k == d - 1 else 2 * rx[k + 1]
        core = bk.zeros((r1, n[k], r2), dtype=dtype, like=V[0])
        if k == 0:
            core[:, :, :rx[1]] = dl[0]
            core[:, :, rx[1]:] = U[0]
        elif k < d - 1:
            core[:rx[k], :, :rx[k + 1]] = V[k]
            core[rx[k]:, :, :rx[k + 1]] = dl[k]
            core[rx[k]:, :, rx[k + 1]:] = U[k]
        else:
            core[:rx[k], :, :] = V[k]
            core[rx[k]:, :, :] = dl[k]
        cores.append(core)
    return vector.from_list(cores)


def tangent_inner(deltas_a, deltas_b):
    """``<xi, eta>`` of two tangent vectors AT THE SAME POINT.

    The gauge makes it ``sum_k <dG_k^a, dG_k^b>_F`` ([RNO19] eq. (22)):
    ``O(d n r^2)`` instead of the ``O(d n r^3)`` TT contraction, and without
    its cancellation.  Silently wrong if the two lists come from different
    points -- that is the contract, not a check this function can make.
    """
    if len(deltas_a) != len(deltas_b):
        raise ValueError("tangent vectors of different lengths")
    out = None
    for a, b in zip(deltas_a, deltas_b):
        term = einsum(a.conj(), b, "a i b, a i b ->")
        out = term if out is None else out + term
    return out


def tangent_gram(delta_lists):
    """``(b, b)`` Gram matrix of ``b`` tangent vectors at one point."""
    b = len(delta_lists)
    g = np.empty((b, b), dtype=complex)
    for i in range(b):
        for j in range(i, b):
            v = complex(tangent_inner(delta_lists[i], delta_lists[j]))
            g[i, j] = v
            g[j, i] = np.conj(v)
    if np.allclose(g.imag, 0.0):
        g = g.real
    return g


def retract(X, xi, *, method="svd", rmax=None, return_discarded=False):
    """A point of ``M_{r(X)}`` near ``X + xi``.

    Args:
        X: The base point.
        xi: The increment, a :class:`tt.vector` (assemble delta cores with
            :func:`tangent_to_tt` first).
        method: ``'svd'`` -- TT rounding of ``X + xi`` to the ranks of ``X``
            ([RNO19] section 4.4), quasi-optimal, the default.  ``'psa'`` --
            :func:`projector_splitting_add`: one sweep, no SVD, exact when
            ``X + xi`` already has rank ``r(X)``, first-order otherwise.
        rmax: Target ranks for ``'svd'``; default the ranks of ``X``.
        return_discarded: Also return ``||X + xi - result||`` -- the local
            retraction error, the quantity a rank-adaptive wrapper needs.
            Costs one rank-4r norm.

    Returns:
        The retracted :class:`tt.vector`, or ``(vector, discarded)``.
    """
    if method == "svd":
        cap = int(rmax) if rmax is not None else int(max(int(v) for v in X.r))
        y = (X + xi).round(eps=0.0, rmax=cap)
    elif method == "psa":
        y = projector_splitting_add(X, xi)
    else:
        raise ValueError(f"unknown retraction method {method!r}")
    if not return_discarded:
        return y
    discarded = float(((X + xi) - y).norm())
    return y, discarded


def transport(deltas, X_old, X_new, *, frames_new=None):
    """Vector transport by re-projection: the deltas of ``P_{T_new} xi``.

    The deltas at ``X_old`` mean nothing at ``X_new``, so there is no shortcut:
    assemble, then :func:`project_delta` at the new point.  Skipping this and
    adding old-point deltas at the new point is the single easiest way to get
    a Riemannian method wrong -- the direction's rank then grows without
    bound (a measured 60x cost; ``docs/plans/riemannian-autodiff.md`` 2.4).

    Returns:
        ``(deltas_new, frames_new)``.
    """
    xi = tangent_to_tt(X_old, deltas)
    return project_delta(X_new, xi, frames_=frames_new)
