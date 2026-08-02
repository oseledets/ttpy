"""Projected local operators for one-site ALS-type sweeps.

Every alternating algorithm on a TT-matrix ``A`` and a TT-vector ``y`` needs the
same three objects, and this module is their only owner:

* the **interfaces** (partial contractions of ``y^H A y``), ``phi_left`` /
  ``phi_right``;
* the **local operator** at site ``k``, either applied to a block
  (:func:`local_matvec`, :func:`local_matmat`) or built densely
  (:func:`local_matrix`);
* the **interface operator** between two sites, i.e. the same thing with the
  mode index removed (:func:`interface_matvec`, :func:`interface_matrix`) --
  this is the ``S``-step of the projector-splitting integrator.

Index convention
----------------
``y``-cores are ``(r_k, n_k, r_{k+1})``, ``A``-cores are
``(ra_k, n_k, m_k, ra_{k+1})`` with the *row* index first (the layout of
``tt.matrix.to_list``).  An interface carries three indices in the order
``(bra, alpha, ket)``: ``bra`` is the one contracted with the conjugated frame
and becomes the *row* index of the local operator, ``ket`` becomes the column
index.  With that convention

    L_0 = R_d = 1,
    L_{k+1}[p',a',q'] = sum L_k[p,a,q] conj(Y_k[p,i,p']) A_k[a,i,j,a'] Y_k[q,j,q'],
    R_k[p,a,q]        = sum R_{k+1}[p',a',q'] conj(Y_k[p,i,p']) A_k[a,i,j,a'] Y_k[q,j,q'],

and the local operator at site ``k`` is

    (B x)[p,i,P] = sum L_k[p,a,q] A_k[a,i,j,a'] R_{k+1}[P,a',Q] x[q,j,Q].

If ``A`` is Hermitian and the frames are orthonormal, ``B`` is Hermitian; the
callers rely on that and check it where it matters.

Everything here is a pair of binary contractions -- never a single ternary
``einsum``, which ``einops`` would hand to ``np.einsum`` without ``optimize``
and which then costs orders of magnitude more.
"""

from __future__ import annotations

from einops import einsum, rearrange

from .. import backend as bk

__all__ = [
    "ones_interface", "phi_left", "phi_right",
    "local_matvec", "local_matmat", "local_matrix",
    "interface_matvec", "interface_matrix",
    "left_orthogonalize", "right_orthogonalize",
]


def ones_interface(like, dtype=None):
    """The trivial ``(1, 1, 1)`` interface that closes a TT boundary."""
    return bk.eye(1, 1, dtype=dtype or bk.dtype_of(like), like=like).reshape((1, 1, 1))


def phi_left(phi, acore, bra, ket):
    """Grow the left interface by one site.

    Args:
        phi: ``(r_k, ra_k, r_k)`` interface, ``(bra, alpha, ket)``.
        acore: ``(ra_k, n_k, m_k, ra_{k+1})`` matrix core.
        bra: ``(r_k, n_k, r_{k+1})`` core of the conjugated (row) frame.
        ket: ``(r_k, m_k, r_{k+1})`` core of the (column) frame.

    Returns:
        The ``(r_{k+1}, ra_{k+1}, r_{k+1})`` interface.
    """
    t = einsum(phi, bra.conj(), "p a q, p i P -> a q P i")
    t = einsum(t, acore, "a q P i, a i j A -> q P A j")
    return einsum(t, ket, "q P A j, q j Q -> P A Q")


def phi_right(phi, acore, bra, ket):
    """Grow the right interface by one site (mirror image of :func:`phi_left`).

    Args:
        phi: ``(r_{k+1}, ra_{k+1}, r_{k+1})`` interface.
        acore: ``(ra_k, n_k, m_k, ra_{k+1})`` matrix core.
        bra: ``(r_k, n_k, r_{k+1})`` core of the conjugated (row) frame.
        ket: ``(r_k, m_k, r_{k+1})`` core of the (column) frame.

    Returns:
        The ``(r_k, ra_k, r_k)`` interface.
    """
    t = einsum(phi, bra.conj(), "P A Q, p i P -> A Q p i")
    t = einsum(t, acore, "A Q p i, a i j A -> Q p a j")
    return einsum(t, ket, "Q p a j, q j Q -> p a q")


def local_matvec(left, acore, right, x):
    """Apply the projected local operator to one block ``x`` of shape ``(r, m, r)``."""
    t = einsum(left, x, "p a q, q j Q -> p a j Q")
    t = einsum(t, acore, "p a j Q, a i j A -> p i Q A")
    return einsum(t, right, "p i Q A, P A Q -> p i P")


def local_matmat(left, acore, right, x):
    """Apply the local operator to a stack of blocks, ``x`` of shape ``(r, m, r, k)``."""
    t = einsum(left, x, "p a q, q j Q k -> p a j Q k")
    t = einsum(t, acore, "p a j Q k, a i j A -> p i Q A k")
    return einsum(t, right, "p i Q A k, P A Q -> p i P k")


def local_matrix(left, acore, right):
    """Dense local operator, shape ``(r n r', r m r')``.

    Costs ``O((r n r')^2)`` memory -- the callers gate this behind a
    ``max_full_size`` and use :func:`local_matvec` above it.
    """
    t = einsum(left, acore, "p a q, a i j A -> p q i j A")
    m = einsum(t, right, "p q i j A, P A Q -> p i P q j Q")
    return rearrange(m, "p i P q j Q -> (p i P) (q j Q)")


def interface_matvec(left, right, s):
    """Apply the interface (``S``-step) operator to ``s`` of shape ``(p, q)``.

    Same as :func:`local_matvec` with the mode index dropped, i.e. with an
    identity in place of the matrix core.
    """
    t = einsum(left, s, "p a q, q Q -> p a Q")
    return einsum(t, right, "p a Q, P a Q -> p P")


def interface_matrix(left, right):
    """Dense interface operator, shape ``(p P, q Q)``."""
    m = einsum(left, right, "p a q, P a Q -> p P q Q")
    return rearrange(m, "p P q Q -> (p P) (q Q)")


def left_orthogonalize(core):
    """``core = Q S`` with ``Q`` of shape ``(r, n, rnew)``, ``Q^H Q = I``.

    ``rnew = min(r n, r')``; a rank-deficient core silently loses the deficient
    directions here, which is the standard behaviour of a QR sweep.
    """
    n = core.shape[1]
    q, s = bk.qr(rearrange(core, "a n b -> (a n) b"))
    return rearrange(q, "(a n) c -> a n c", n=n), s


def right_orthogonalize(core):
    """``core = S Q`` with ``Q`` of shape ``(rnew, n, r')``, ``Q Q^H = I``.

    The core-shaped LQ factorization: ``bk.qr`` of the conjugate transpose,
    written with explicit ``einops`` patterns so the transposition is visible.
    """
    _, n, r1 = core.shape
    q, s = bk.qr(rearrange(core.conj(), "a n b -> (n b) a"))
    return (rearrange(s.conj(), "c a -> a c"),
            rearrange(q.conj(), "(n b) c -> c n b", n=n, b=r1))
