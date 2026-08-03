"""QTT methods for elliptic problems: the BPX preconditioner of [BK20].

[BK20] M. Bachmayr, V. Kazeev, *Stability of Low-Rank Tensor Representations and
Structured Multilevel Preconditioning for Elliptic PDEs*, Found. Comput. Math.
20 (2020) 1175-1236.

The one thing to understand before reading the code
---------------------------------------------------
The preconditioner is a sum over levels,

    C_L = sum_{l=0..L} 2^{-l} P_{l,L} P_{l,L}^T,

and the whole point of [BK20] Theorem 3 is that this sum must **not** be
assembled by adding ``L + 1`` TT matrices and rounding.  Every summand is a
chain built from the *same two cores* -- ``U_b`` on the first ``l`` sites,
``X_b`` on the remaining ones -- differing only in *where it switches* and by a
scalar.  A sum of chains that differ in one bond position is a two-state
automaton: state 1 = "still on ``U_b``", state 2 = "already switched", with the
switch carrying weight ``2^{-l}``.  So the sum is *already* a TT matrix with a
bidiagonal transfer core,

    C_l = [ U_b   2^{-l} U_b ]
          [  0    2^{-D} X_b ]

of TT rank exactly ``2 * 4^D`` -- 8, 32, 128 for ``D = 1, 2, 3`` -- **independent
of the number of levels**.  Assembled this way there is no intermediate rank
growth and no rounding anywhere in the construction.  It is the same mechanism
that makes a triangular Toeplitz matrix ``sum_k S^k`` rank 2 in QTT.

Two conventions have to be kept straight
----------------------------------------
*Level order.* [BK20] numbers level 1 as the **coarsest** scale, the most
significant bit; ttpy2's mode 1 is the **fastest** index, the least significant
bit.  A [BK20] core list is therefore reversed and rank-transposed on its way
into a ``tt.matrix`` (:func:`_to_ttpy`).

*Index layout.* For ``D > 1`` the rank bound holds in **level-major** order (the
bits of one level of all dimensions adjacent, [BK20] eq. (49)), not in the
dimension-major order that :func:`tt.kron` and :func:`tt.qlaplace_dd` produce.
Operators built here are level-major; pair them with
``tt.qlaplace_dn(..., order='level')``.  Composing layouts that disagree gives
silent nonsense, which is why neither is a default that can be reached by
accident.

See ``docs/plans/qtt-elliptic-bpx.md`` for the measurements behind all of this.
"""

from __future__ import annotations

import numpy as np

from ..core.matrix import matrix
from ..core.tools import level_major_order, qdiff, qtri_ones

__all__ = ["bpx", "prolongation", "solve_direct_1d"]

_I2 = np.eye(2)
_J = np.array([[0.0, 1.0], [0.0, 0.0]])     # [BK20] (37): J upper, J.T lower


# --- the core algebra of [BK20] Sect. 3 --------------------------------------

def _core(blocks):
    """A ``(p, m, n, q)`` core from a ``p x q`` table of ``m x n`` blocks."""
    p, q = len(blocks), len(blocks[0])
    m, n = np.shape(blocks[0][0])
    c = np.zeros((p, m, n, q))
    for a in range(p):
        for b in range(q):
            c[a, :, :, b] = np.asarray(blocks[a][b], dtype=float).reshape(m, n)
    return c


def _skron(a, b):
    """Strong Kronecker product ([BK20] Def. 1): ranks contract, modes multiply.

    This is how *dimensions* are combined at one level, and how neighbouring
    cores are merged.
    """
    p, m1, n1, r = a.shape
    r2, m2, n2, q = b.shape
    if r != r2:
        raise ValueError(f"rank mismatch in strong Kronecker: {a.shape} {b.shape}")
    return np.einsum("aijg,gklb->aikjlb", a, b).reshape(p, m1 * m2, n1 * n2, q)


def _cdot(a, b):
    """Core product ([BK20] Sect. 3.6): ranks Kronecker-multiply, modes contract.

    This is what turns ``P`` into ``P P^T``: the product of two operators whose
    cores are known separately.
    """
    p, m, k, q = a.shape
    p2, k2, n, q2 = b.shape
    if k != k2:
        raise ValueError(f"mode mismatch in core product: {a.shape} {b.shape}")
    return np.einsum("amkq,bknc->abmnqc", a, b).reshape(p * p2, m, n, q * q2)


def _ctr(a):
    """Mode transposition of a core ([BK20] (38))."""
    return a.transpose(0, 2, 1, 3)


def _to_ttpy(cores):
    """[BK20] core list (level 1 = coarsest = MSB) -> ``tt.matrix`` (core 1 = LSB)."""
    return matrix.from_list([c.transpose(3, 1, 2, 0).copy() for c in cores[::-1]])


def _kron_power(c, ndim):
    """The ``D``-fold strong Kronecker power of a core: one level, all dimensions."""
    out = c
    for _ in range(ndim - 1):
        out = _skron(out, c)
    return out


# --- the elementary cores of [BK20] Sect. 5.1 --------------------------------

_U = _core([[_I2, _J.T], [np.zeros((2, 2)), _J]])                     # (67)
_X = 0.5 * _core([[[[1.], [2.]], [[0.], [1.]]],
                  [[[1.], [0.]], [[2.], [1.]]]])                      # (67), 2x1
_PHAT = _core([[[[1.]]], [[[0.]]]])                                   # (67), 2x1
_AHAT = _core([[[[1.]], [[0.]]]])                                     # (75), 1x2


def _bpx_blocks(ndim):
    """The ``_b`` cores of [BK20] (80): ``P P^T`` in ``D`` dimensions."""
    a = _kron_power(_cdot(_AHAT, _AHAT), ndim)
    u = _kron_power(_cdot(_U, _ctr(_U)), ndim)
    x = _kron_power(_cdot(_X, _ctr(_X)), ndim)
    p = _kron_power(_cdot(_PHAT, _PHAT), ndim)
    return a, u, x, p


def _block2(top_left, top_right, bottom_right):
    """The bidiagonal transfer core ``[[A, B], [0, C]]`` of the level automaton."""
    p, m, n, q = top_left.shape
    p2, m2, n2, q2 = bottom_right.shape
    if (m, n) != (m2, n2):
        raise ValueError(f"mode mismatch: {top_left.shape} {bottom_right.shape}")
    out = np.zeros((p + p2, m, n, q + q2))
    out[:p, :, :, :q] = top_left
    out[:p, :, :, q:] = top_right
    out[p:, :, :, q:] = bottom_right
    return out


# --- the preconditioner ------------------------------------------------------

def bpx(d, D=1, weight=1, scaled=True):
    """The BPX preconditioner as a ``tt.matrix``, in level-major order.

    Built from the explicit cores of [BK20] Theorem 3 as a two-state automaton
    over the levels (see the module docstring): no summation over levels, no
    rounding, and a TT rank of exactly ``2 * 4^D`` whatever ``d`` is.

    Args:
        d: number of levels, so ``2^d`` nodes per dimension.
        D: spatial dimensions.  The result acts on ``2^(D d)`` unknowns laid out
            **level-major**; use ``tt.qlaplace_dn(..., order='level')`` for the
            operator it multiplies.
        weight: 1 gives ``C_L = sum_l 2^{-l} P_l P_l^T``, the symmetric
            two-sided preconditioner of [BK20] Theorem 2 that a linear solve
            wants; 2 gives ``C_{2,L} = sum_l 2^{-2l} P_l P_l^T``, the classical
            left preconditioner of Theorem 1 that an eigensolver wants.  They
            are different operators and the choice is not cosmetic: minimizing
            the Rayleigh quotient of ``C A C`` returns the eigenvector of the
            *preconditioned* operator.
        scaled: multiply by ``2^(weight * d)`` so the result pairs with the
            **unscaled** operators of :func:`tt.qlaplace_dn` (which carry no
            ``h^-2``).

    Returns:
        A symmetric positive definite ``tt.matrix`` of size ``2^(D d)``.
    """
    d, D, weight = int(d), int(D), int(weight)
    if d < 1:
        raise ValueError(f"d must be at least 1, got {d}")
    if D < 1:
        raise ValueError(f"D must be at least 1, got {D}")
    if weight not in (1, 2):
        raise ValueError(
            f"weight must be 1 (two-sided, for solves) or 2 (left, for "
            f"eigenproblems), got {weight}")

    a_b, u_b, x_b, p_b = _bpx_blocks(D)
    # [BK20] Theorem 3, as the automaton described in the module docstring.
    # State 1 = "still on U_b", state 2 = "already switched to X_b"; the switch
    # itself is an X_b, so the transfer core is [[U_b, X_b], [0, X_b]].  Reading
    # the (1,1) path gives the l = d term (all U_b) and the (1,2) paths give one
    # term per switch position, so the chain sum is exactly
    # sum_{l=0..d} A_b U_b^l X_b^{d-l} P_b -- the level sum, with no summation
    # of TT matrices and no rounding.
    #
    # The level weight 2^{-l} of C_L is already carried by the scaling inside
    # X_b (d - l copies of it), which is why nothing here is scaled per level
    # for weight=1.  Weight 2 needs one extra 2^{-l}, and l is the number of
    # U_b factors, so it goes on U_b.  Both are checked against a naive
    # per-level sum in tests/test_qtt_ell.py rather than argued.
    u_w = u_b if weight == 1 else (2.0 ** (-(weight - 1))) * u_b
    left = np.concatenate([a_b, np.zeros_like(a_b)], axis=3)   # start in state 1
    right = np.concatenate([p_b, p_b], axis=0)                 # accept either
    cores = [left]
    for _ in range(d):
        cores.append(_block2(u_w, x_b, x_b))
    cores.append(right)

    # the boundary cores have a 1x1 mode; merge them into their neighbours so
    # every remaining core is a genuine 2x2 QTT site
    merged = [_skron(cores[0], cores[1])] + cores[2:-1]
    merged[-1] = _skron(merged[-1], cores[-1])

    out = _to_ttpy(merged)
    # The chain carries a fixed 2^d of its own, for either weight (verified
    # against a per-level sum for d = 2..5, w = 1, 2: the ratio is exactly
    # 2^d elementwise).  Make the normalization explicit rather than folded
    # into the cores, where it would be one more thing to rediscover.
    return out * (2.0 ** ((weight - 1) * d if scaled else -d))


def prolongation(l, d, D=1):
    """``P_{l,L}`` of [BK20] Lemma 4: level ``l`` to level ``d``, level-major.

    A rectangular ``tt.matrix`` of size ``2^(D d) x 2^(D l)`` and TT rank
    ``2^D``.  Provided for tests and for anyone building their own multilevel
    scheme; :func:`bpx` does not call it, because calling it per level is
    exactly the assembly this module exists to avoid.
    """
    l, d, D = int(l), int(d), int(D)
    if not 0 <= l <= d:
        raise ValueError(f"need 0 <= l <= d, got l={l}, d={d}")
    a = _kron_power(_AHAT, D)
    u = _kron_power(_U, D)
    x = _kron_power(_X, D)
    p = _kron_power(_PHAT, D)
    cores = [a] + [u] * l + [x] * (d - l) + [p]
    merged = [_skron(cores[0], cores[1])] + cores[2:-1]
    merged[-1] = _skron(merged[-1], cores[-1])
    return _to_ttpy(merged) * (2.0 ** (-(d - l) / 2.0))


# --- the 1D direct solve -----------------------------------------------------

def solve_direct_1d(f, d, inv_coeff=None):
    """Exact solve of ``(M^T diag(a) M) v = f`` in one dimension.

    With ``M = tt.qdiff(d)`` the inverse is ``T diag(1/a) T^T`` where
    ``T = tt.qtri_ones(d)`` is the triangular all-ones matrix, since
    ``T M = I`` exactly.  Two matvecs, no iteration, no assembled preconditioned
    operator.

    This routine has no approximate mode and must never grow one.  Assembling
    ``T^T A T`` and iterating on it instead -- which is the "obvious"
    generalization -- returns an answer 96.7 % wrong at ``d = 30`` while the
    linear solver honestly reports ``converged=True`` and a residual of 2.6e-09,
    because the residual is of the system it was handed and the damage is in the
    representation: that product *is* the identity and rounds to TT rank 13.
    See ``docs/plans/qtt-elliptic-bpx.md`` V12b.

    Args:
        f: right-hand side, a ``tt.vector`` on ``2^d`` nodes.
        d: number of levels.
        inv_coeff: ``None`` for ``a = 1``, otherwise a ``tt.vector`` holding
            **1/a**, not ``a``.  Named that way because the inverse is what the
            formula uses and inverting a QTT vector is not free -- it needs a
            cross approximation, whose accuracy would then be this routine's
            accuracy without being visible in its signature.  The caller who
            knows ``1/a`` in closed form pays nothing; the caller who does not
            should compute it with ``tt.multifuncrs`` and own the tolerance.

    Returns:
        The nodal solution as a ``tt.vector``.
    """
    from ..core.tools import matvec

    d = int(d)
    t = qtri_ones(d)
    v = matvec(t.T, f).round(1e-14)
    if inv_coeff is not None:
        v = (v * inv_coeff).round(1e-14)
    return matvec(t, v).round(1e-14)


def laplace_dn_operator(d, D=1):
    """``M^T M`` in level-major order: the operator :func:`bpx` preconditions.

    A convenience so the two always agree about the layout; identical to
    ``tt.qlaplace_dn([d]*D, 'DN', order='level')``.
    """
    from ..core.tools import qlaplace_dn

    return qlaplace_dn([int(d)] * int(D), "DN",
                       order="level" if int(D) > 1 else "dim")
