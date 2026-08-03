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
``X_b`` on the remaining ones -- differing only in *where it switches*.  A sum
of chains that differ in one bond position is a two-state automaton: state 1 =
"still on ``U_b``", state 2 = "already switched to ``X_b``".  So the sum is
*already* a TT matrix with a bidiagonal transfer core,

    C_l = [ 2^{D-w} U_b   X_b ]
          [      0        X_b ]

of TT rank exactly ``2 * 4^D`` -- 8, 32, 128 for ``D = 1, 2, 3`` -- **independent
of the number of levels**.  Assembled this way there is no intermediate rank
growth and no rounding anywhere in the construction.  It is the same mechanism
that makes a triangular Toeplitz matrix ``sum_k S^k`` rank 2 in QTT.

The ``2^{D-w}`` is the one place ``D`` enters, and it is easy to lose: the bare
chain for level ``l`` is not ``P_l P_l^T`` but ``2^{D(d-l)} P_l P_l^T``, since
each of the ``d - l`` copies of ``X_b`` carries a ``2^{-D}``.  The automaton's
built-in level weight is therefore ``2^{-D l}`` while ``C_{w,L}`` wants
``2^{-w l}``, and the correction ``2^{(D-w) l}`` rides on ``U_b`` because ``l``
is exactly the number of ``U_b`` factors.  At ``D = 1`` and ``w = 1`` the factor
is 1 and the whole issue is invisible -- which is how a ``D > 1`` that produced
the textbook ranks and preconditioned nothing survived a first round of tests.
The chain sum is then ``2^{D d} C_{w,L}``, whence the final normalization.

The second thing, which is what you actually solve with
-------------------------------------------------------
Having ``C_L`` is not enough.  The preconditioned operator ``C A C`` must never
be *assembled*: its entries cancel over ``4^d``, so rounding the triple product
represents a matrix with error growing like ``4^d eps``, and its rank grows too
(measured 96, 135, 185 at ``d = 10, 14, 18``).  :func:`bpx_theta` gives the
fused factor ``Theta`` of [BK20] Lemma 5 with ``B = Theta^T Theta``, the same
matrix in exact arithmetic, at TT rank 6 and ``B`` at rank 17, flat in ``d``.

The difference, end to end, on ``-u'' = 1`` with ``u(0) = 0, u'(1) = 0``, AMEn
at ``eps = 1e-10``, b300/numpy/float64, interleaved runs:

======  ==========================  ============================
``d``   unpreconditioned            ``B = Theta^T Theta``
======  ==========================  ============================
10      30 sweeps, 0.63 s, 4.1e-10  7 sweeps, 0.07 s, 2.3e-14
18      30 sweeps, 2.51 s, 8.6e-06  7 sweeps, 0.18 s, 8.1e-14
26      30 sweeps, 11.1 s, 4.1e-01  7 sweeps, 0.41 s, 1.7e-13
30      30 sweeps, 23.3 s, 1.03     7 sweeps, 0.51 s, 1.8e-13
======  ==========================  ============================

At ``d = 30`` that is 2^30 unknowns, 46x faster, and the unpreconditioned answer
is simply wrong (relative error 1.03) -- reported as such by ``amen_solve``,
which does not converge and says so.  The sweep count and the rank are flat.

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

__all__ = ["bpx", "bpx_operator", "bpx_theta", "invert", "prolongation",
           "solve_direct_1d", "sqrt", "stiffness"]

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


def _dkron(a, b):
    """Kronecker product of two cores in **both** the rank and the mode indices.

    This is how the ``D`` dimensions of one level are combined ([BK20] writes it
    ``(x)D``).  Note that it is *not* :func:`_skron`: the strong Kronecker
    product contracts the shared rank index, which is right for neighbouring
    sites of one chain and wrong here, where the dimensions are independent and
    their ranks multiply.  Getting these two confused is why ``D > 1`` used to
    raise instead of run.
    """
    p1, m1, n1, q1 = a.shape
    p2, m2, n2, q2 = b.shape
    return np.einsum("aijb,ckld->acikjlbd", a, b).reshape(
        p1 * p2, m1 * m2, n1 * n2, q1 * q2)


def _kron_power(c, ndim):
    """The ``D``-fold Kronecker power of a core: one level, all dimensions."""
    return _kron_list([c] * ndim)


def _kron_list(cores):
    """:func:`_dkron` folded over a list -- one level, the dimensions in order."""
    out = cores[0]
    for c in cores[1:]:
        out = _dkron(out, c)
    return out


# --- the elementary cores of [BK20] Sect. 5.1 --------------------------------

_U = _core([[_I2, _J.T], [np.zeros((2, 2)), _J]])                     # (67)
_X = 0.5 * _core([[[[1.], [2.]], [[0.], [1.]]],
                  [[[1.], [0.]], [[2.], [1.]]]])                      # (67), 2x1
_PHAT = _core([[[[1.]]], [[[0.]]]])                                   # (67), 2x1
_AHAT = _core([[[[1.]], [[0.]]]])                                     # (75), 1x2

# [BK20] (78): the cores of the difference operator M_{L,1}, used by the fused
# construction of Lemma 5 -- the one that never represents C A C.
_T1 = _core([[[[1.0]]], [[[-1.0]]]])                                  # rank 2x1
_IHAT = _core([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])               # rank 2x2
_Y1 = 0.5 * _core([[[[1.0], [1.0]]]])                                 # mode 2x1
_N1 = _core([[[[1.0]]]])


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
    # The bare chain for level l is NOT P_l P_l^T: measured, it is
    # 2^{D(d-l)} P_l P_l^T, because each of the d - l copies of X_b carries a
    # 2^{-D}.  So the automaton's built-in level weight is 2^{-D l}, while
    # C_{w,L} wants 2^{-w l}.  The correction is 2^{(D-w) l}, and l is exactly
    # the number of U_b factors, so it rides on U_b.
    #
    # At D = 1 this reduces to 2^{1-w} and the distinction is invisible --
    # which is why a D > 1 that produced the right ranks and preconditioned
    # nothing went unnoticed until the per-level chains were compared against
    # P_l P_l^T directly.
    u_w = u_b if D == weight else (2.0 ** (D - weight)) * u_b
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
    # With that correction the chain sum is 2^{D d} C_{w,L}, so `scaled` (which
    # means 2^{w d} C, pairing with the unscaled operators of qlaplace_dn) needs
    # 2^{(w-D) d} and the unscaled form needs 2^{-D d}.  Explicit rather than
    # folded into the cores: this is the factor that hid the D > 1 defect.
    return out * (2.0 ** ((weight - D) * d if scaled else -D * d))


def bpx_theta(d, D=1):
    """The fused factors ``Theta_k`` of [BK20] Lemma 5 / Theorem 4.

    Returns a **list of ``D`` matrices**, one per direction, with

        B = sum_k Theta_k^T Theta_k

    the preconditioned operator ``C A C`` -- the same matrix as
    ``bpx(d, D) @ qlaplace_dn(...) @ bpx(d, D)`` in exact arithmetic, and not at
    all the same object in floating point.

    That difference is the practical content of [BK20]. Forming the product of
    three QTT factors and rounding it represents a matrix whose entries cancel
    over ``4^d``, so the representation error grows like ``4^d * eps``: measured
    1.3e-10 at ``d = 10``, 6.0e-04 at ``d = 20``, 4.8e+14 at ``d = 50``, while
    this form stays at 1.4e-14. The rank tells the same story -- ``round(C A C)``
    was measured at 96, 135, 185 for ``d = 10, 14, 18``, growing with ``d``,
    while ``Theta_k`` has TT rank exactly ``2^(2D) + 2^(2D-1)`` (6, 24, 96 for
    ``D = 1, 2, 3``) whatever ``d`` is.

    A list even when ``D == 1``, because for ``D > 1`` the sum must be applied
    as ``sum_k Theta_k^T round(Theta_k v)`` rather than assembled -- the
    assembled ``B`` has rank up to ``2 D 4^{2D}`` -- and one contract that is
    always right beats a convenient one that stops being right at ``D = 2``.

    Args:
        d: number of levels.
        D: spatial dimensions. The factors act on ``2^(D d)`` unknowns laid out
            **level-major**; pair them with
            ``tt.qlaplace_dn([d]*D, 'DN', order='level')``.

    Returns:
        ``list`` of ``D`` ``tt.matrix``, scaled to pair with the *unscaled*
        operators of :func:`tt.qlaplace_dn`.
    """
    d, D = int(d), int(D)
    if d < 1:
        raise ValueError(f"d must be at least 1, got {d}")
    if D < 1:
        raise ValueError(f"D must be at least 1, got {D}")

    u1 = _cdot(_U, _ctr(_U))
    x1 = _cdot(_X, _ctr(_X))
    a1 = _cdot(_AHAT, _AHAT)
    p1 = _cdot(_PHAT, _PHAT)
    w1 = _cdot(_T1, _IHAT)                       # [BK20] (83)
    z1 = _cdot(_Y1, _ctr(_X))
    k1 = _cdot(_N1, _PHAT)
    wz1 = _skron(w1, z1)
    wk1 = _skron(w1, k1)

    # The second state carries 2^{1-D}: the analogue, for Theta, of the
    # 2^{(D-w)l} that bpx needs, and measured the same way -- by requiring
    # sum_k Theta_k^T Theta_k to equal a dense C A C.  Verified for
    # D = 1, 2, 3 (spectra to 1e-14) at the Theorem 4 ranks 6, 24, 96.
    delta = 2.0 ** (1 - D)

    out = []
    for k in range(D):
        u = _kron_list([u1] * D)
        sw = _kron_list([wz1 if j == k else x1 for j in range(D)]) * delta
        st2 = _kron_list([z1 if j == k else x1 for j in range(D)]) * delta
        a_b = _kron_list([a1] * D)
        r1 = _kron_list([wk1 if j == k else p1 for j in range(D)])
        r2 = _kron_list([k1 if j == k else p1 for j in range(D)])
        left = np.concatenate(
            [a_b, np.zeros(a_b.shape[:3] + (st2.shape[0],))], axis=3)
        right = np.concatenate([r1, r2], axis=0)
        cores = [left] + [_block2(u, sw, st2) for _ in range(d)] + [right]
        merged = [_skron(cores[0], cores[1])] + cores[2:-1]
        merged[-1] = _skron(merged[-1], cores[-1])
        out.append(_to_ttpy(merged))
    return out


def bpx_operator(d, D=1, eps=1e-14):
    """``B = sum_k Theta_k^T Theta_k`` assembled, for callers that need a matrix.

    Rank 17 for ``D = 1``, flat in ``d``.  For ``D > 1`` the assembled operator
    is large (bounded by ``2 D 4^{2D}``, i.e. 512 already at ``D = 2``) and
    :func:`bpx_theta` applied factor by factor is the intended route; this
    function warns rather than refuses, because forming it at small ``d`` is a
    legitimate thing to do in a test.
    """
    d, D = int(d), int(D)
    factors = bpx_theta(d, D)
    if D > 1:
        import warnings
        warnings.warn(
            f"assembling B for D={D}: the factors of bpx_theta are meant to be "
            "applied as sum_k Theta_k^T round(Theta_k v), and the assembled "
            "operator has rank up to 2*D*4^(2D). Fine for a small test, not "
            "for a solve.", RuntimeWarning, stacklevel=2)
    total = None
    for th in factors:
        term = (th.T @ th).round(eps)
        total = term if total is None else (total + term).round(eps)
    return total


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


# --- variable coefficients ---------------------------------------------------

def stiffness(coeff, d):
    """``A = M^T diag(a) M`` in one dimension, with ``M = tt.qdiff(d)``.

    ``coeff`` is the coefficient ``a`` as a ``tt.vector`` on ``2^d`` nodes, or a
    scalar.  The rank of ``A`` is at most ``4 r_a``: two from each difference
    factor, times the rank of the coefficient.

    Nothing here needs the coefficient in closed form.  ``a`` itself is usually
    the easy part -- ``tt.xfun``, ``tt.stepfun`` and friends build the common
    ones -- and anything else comes from a cross approximation with
    :func:`tt.multifuncrs`, which is also how :func:`invert` produces ``1/a``.
    """
    from ..core.tools import diag, matvec, ones

    d = int(d)
    m = qdiff(d)
    if not hasattr(coeff, "cores"):
        return ((m.T @ m) * float(coeff)).round(1e-14)
    return (m.T @ diag(coeff) @ m).round(1e-14)


def invert(a, eps=1e-10, **kwargs):
    """``1/a`` elementwise, by cross approximation.

    A one-line wrapper over :func:`tt.multifuncrs`, here because the elliptic
    routines need it often enough that every caller writing the lambda
    themselves is how the tolerance ends up undocumented.  It refuses nothing:
    if ``a`` has a zero the cross will happily return infinities, and that is
    the caller's problem to notice -- which is why ``eps`` is explicit.
    """
    from .multifuncrs import multifuncrs

    return multifuncrs([a], lambda v: 1.0 / v[:, 0], eps=eps,
                       verb=kwargs.pop("verb", 0), **kwargs)


def sqrt(a, eps=1e-10, **kwargs):
    """``sqrt(a)`` elementwise, by cross approximation.

    Needed by the coefficient-dependent fused factor of [BK20] Lemma 5, where
    ``Lambda^{1/2}`` appears.  Taking the square root core-wise instead is exact
    only for a rank-1 coefficient; this is exact for any of them, at the price
    of a tolerance the caller can see.
    """
    from .multifuncrs import multifuncrs

    return multifuncrs([a], lambda v: np.sqrt(v[:, 0]), eps=eps,
                       verb=kwargs.pop("verb", 0), **kwargs)


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
            formula uses and it is not free -- it comes from a cross
            approximation whose accuracy becomes this routine's accuracy, so it
            belongs in the caller's hands rather than hidden here.  Use
            :func:`invert` (one call to ``tt.multifuncrs``) when ``1/a`` is not
            known in closed form.

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
