"""Bilinear finite elements on a quad mesh, assembled in QTT z-order.

A port of the algorithm of L. Markeeva's ``qtt-laplace``
(https://github.com/RerRayne/qtt-laplace), reimplemented on ttpy2's primitives
rather than copied: the construction is hers, the code is ours.

The idea
--------
On a mesh of ``2^d x 2^d`` quadrilaterals every element carries the same four
bilinear basis functions, so the global stiffness matrix is

    K = sum_{l1, l2 in {0,1}^2}  P_{l1}^T  diag(a_{l1 l2})  P_{l2}

where ``P_l`` maps an *element* index to the *node* index of that element's
corner ``l`` -- the identity in a direction where ``l = 0`` and a shift where
``l = 1`` -- and ``a_{l1 l2}`` is the vector, over elements, of the local
integral between those two corners.  There are sixteen terms and no assembly
loop over elements: each is a product of QTT operators of rank 2 with a diagonal
whose rank is the rank of the coefficient field.

The z-order is what keeps that cheap.  Interleaving the bits of the two element
indices (``zkron``, already in ttpy2 as :func:`tt.zkron` / :func:`tt.zkronv`)
means a bond of the train separates *scales* rather than *directions*, so a
locally refined or slowly varying coefficient keeps a small rank where a
direction-major layout would not.  Compare ``tt/algs/qtt_ell.py``, where the
same argument appears for BPX and the layouts are made an explicit argument
because mixing them is silent.

What it is not
--------------
Bilinear elements only, and a logically Cartesian mesh: the geometry enters
through the per-element Jacobian, not through connectivity.  For a curved domain
with a high-order basis see :mod:`tt.algs.iga`, which takes the other route --
a tensor-product spline basis and a coefficient field compressed by cross.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core import tools as _tools
from ..core.matrix import matrix
from ..core.vector import vector

__all__ = ["placement", "assemble", "dirichlet_mask", "apply_mask",
           "local_stiffness_uniform", "local_entries", "node_grid",
           "sew", "interface_blocks", "block_system"]


# --- element -> node ---------------------------------------------------------

def placement(d):
    """``P[(lx, ly)]``: the element index mapped to the node at corner ``l``.

    In one direction the corner ``l = 0`` of element ``e`` is node ``e`` and the
    corner ``l = 1`` is node ``e + 1``, so the two operators are the identity
    and the shift -- **both restricted to the real elements** ``e = 0..n-2``.
    The element slot ``e = n-1`` is fake (its far corner would be node ``n``),
    and both operators must drop it: the shift does so on its own, the identity
    must have its last row zeroed.

    That last row is not a nicety.  With a full identity the fake elements
    deposit their ``(0, .)``-corner contributions on the last row of nodes,
    which is invisible under an all-Dirichlet mask (the square tests) and
    corrupts exactly the interface nodes of a glued multi-patch problem, where
    that side is free.  Measured on Markeeva's triangle: with the full identity
    the coupled energy *falls* under refinement (0.2457, 0.1981, 0.1679 at
    ``d = 2, 3, 4``) instead of approaching 0.3404 from above.  Her own ``W0``
    (materialized densely from her repository) has the zero row.
    """
    d = int(d)
    n = 2 ** d
    keep = _tools.ones(2, d) - _tools.unit(2, d, j=n - 1)
    w = [_tools.diag(keep.round(1e-14)), _tools.qshift(d).T]
    return {(lx, ly): _tools.zkron(w[lx], w[ly])
            for lx, ly in itertools.product((0, 1), (0, 1))}


def node_grid(d):
    """The ``(x, y)`` node coordinates on ``[0,1]^2`` as z-ordered TT vectors.

    ``tt.zmeshgrid`` gives the integer indices; this scales them to the unit
    square with ``2^d`` nodes per direction.
    """
    d = int(d)
    xx, yy = _tools.zmeshgrid(d)
    h = 1.0 / (2 ** d - 1)
    return xx * h, yy * h


# --- the local integrals ------------------------------------------------------

_A1 = np.array([[1.0, -1.0], [-1.0, 1.0]])        # 1D stiffness, unit element
_M1 = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])   # 1D mass


def local_stiffness_uniform(hx, hy):
    """The ``4 x 4`` bilinear element stiffness of a ``hx by hy`` rectangle.

    ``K[l1, l2] = (hy/hx) A(l1x,l2x) M(l1y,l2y) + (hx/hy) M(l1x,l2x) A(l1y,l2y)``
    -- the textbook tensor-product form, kept explicit because it is also the
    oracle the general path is checked against.
    """
    out = {}
    for l1 in itertools.product((0, 1), (0, 1)):
        for l2 in itertools.product((0, 1), (0, 1)):
            out[(l1, l2)] = (hy / hx) * _A1[l1[0], l2[0]] * _M1[l1[1], l2[1]] \
                + (hx / hy) * _M1[l1[0], l2[0]] * _A1[l1[1], l2[1]]
    return out


def local_entries(d, jac=None, eps=1e-10, **cross_kw):
    """The sixteen coefficient vectors ``a_{l1 l2}``, one value per element.

    Args:
        d: ``2^d`` elements per direction.
        jac: ``jac(e) -> J`` of shape ``(batch, 2, 2)`` giving the Jacobian of
            element ``e`` (a ``(batch, 2)`` array of element indices).  ``None``
            means the uniform unit square, where every element is identical and
            every coefficient vector is rank 1 -- which is the case the tests
            use as an oracle.
        eps: cross accuracy when ``jac`` is given.

    Returns:
        dict keyed by ``(l1, l2)`` of :class:`tt.vector` in z-order.
    """
    d = int(d)
    n_el = 2 ** d
    if jac is None:
        h = 1.0 / n_el
        k = local_stiffness_uniform(h, h)
        one = _tools.zkronv(_tools.ones(2, d), _tools.ones(2, d))
        return {key: one * val for key, val in k.items()}

    from .cross import cross

    # z-order: index bit k of the flat element index alternates y, x
    def split(idx):
        e = np.zeros((len(idx), 2), dtype=np.int64)
        for k in range(d):
            digit = idx[:, k]
            e[:, 0] += (digit % 2) * (2 ** k)
            e[:, 1] += (digit // 2) * (2 ** k)
        return e

    def entry(l1, l2):
        def f(idx):
            J = jac(split(idx))
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            adj = np.empty_like(J)
            adj[:, 0, 0], adj[:, 1, 1] = J[:, 1, 1], J[:, 0, 0]
            adj[:, 0, 1], adj[:, 1, 0] = -J[:, 0, 1], -J[:, 1, 0]
            # sum over the reference-element quadrature of grad.adj . grad.adj
            xg, wg = np.polynomial.legendre.leggauss(2)
            acc = np.zeros(len(idx))
            for (xi, wx), (et, wy) in itertools.product(zip(xg, wg), zip(xg, wg)):
                g1 = _grad_ref(l1, xi, et)
                g2 = _grad_ref(l2, xi, et)
                a1 = np.einsum("i,bij->bj", g1, adj, optimize=True)
                a2 = np.einsum("i,bij->bj", g2, adj, optimize=True)
                acc += wx * wy * np.einsum("bj,bj->b", a1, a2, optimize=True) / det
            return acc
        return f

    out = {}
    for l1 in itertools.product((0, 1), (0, 1)):
        for l2 in itertools.product((0, 1), (0, 1)):
            out[(l1, l2)] = cross(entry(l1, l2), 4, d, eps=eps, **cross_kw)
    return out


def _grad_ref(l, xi, eta):
    """Gradient of the bilinear basis at corner ``l`` on ``[-1,1]^2``."""
    sx, sy = 2.0 * l[0] - 1.0, 2.0 * l[1] - 1.0
    return np.array([0.25 * sx * (1.0 + sy * eta), 0.25 * sy * (1.0 + sx * xi)])


# --- assembly and boundary conditions ----------------------------------------

def assemble(entries, d, eps=1e-12):
    """``K = sum_{l1,l2} P_{l1}^T diag(a_{l1 l2}) P_{l2}``, rounded as it goes."""
    P = placement(d)
    total = None
    for (l1, l2), a in entries.items():
        term = P[l1].T @ _tools.diag(a) @ P[l2]
        total = term if total is None else (total + term)
        total = total.round(eps)
    return total


def dirichlet_mask(bcx, bcy, d):
    """The diagonal 0/1 mask of the free nodes, as a z-ordered TT-matrix.

    ``bcx``/``bcy`` are two-character strings, ``'D'`` for Dirichlet at that end
    and anything else for natural.  Rank 1 in each direction, so the mask costs
    ``O(d)``.
    """
    d = int(d)
    n = 2 ** d
    mx, my = _tools.ones(2, d), _tools.ones(2, d)
    if bcx[0] == "D":
        mx = mx - _tools.unit(2, d, j=0)
    if bcx[1] == "D":
        mx = mx - _tools.unit(2, d, j=n - 1)
    if bcy[0] == "D":
        my = my - _tools.unit(2, d, j=0)
    if bcy[1] == "D":
        my = my - _tools.unit(2, d, j=n - 1)
    return _tools.diag(_tools.zkronv(mx, my).round(1e-14))


def apply_mask(A, f, mask, eps=1e-10):
    """Replace the constrained rows of ``A`` by the identity and zero ``f``.

    ``(M A + I - M, M f)``: a homogeneous Dirichlet condition written so that the
    system stays square and the constrained unknowns are solved trivially.
    Non-homogeneous data is handled by lifting before the call, as everywhere
    else in ttpy2.
    """
    d = A.tt.d
    eye = _tools.eye(int(A.n[0]), d)
    return ((mask @ A + eye - mask).round(eps),
            _tools.matvec(mask, f).round(eps))

# --- gluing patches together -------------------------------------------------

_SIDES = ("BOTTOM", "RIGHT", "LEFT", "TOP")
_CORNERS = ("LLC", "LRC", "ULC", "URC")


def sew(d, side, inversed=False):
    """The trace operator of one side of a z-ordered patch, TT rank 1.

    Maps the ``4^d`` nodes of a patch to the ``2^d`` nodes of one of its edges.
    In z-order a mode carries the pair of bits ``(i_x, i_y)`` of one level, so
    "stay on the bottom edge" is the statement ``i_y = 0`` at every level -- a
    rank-1 condition, and the free index rides along in ``i_x``.  That is why
    the trace costs ``O(d)`` here and a permutation matrix elsewhere.

    ``inversed`` reverses the direction along the edge, which is what two
    patches meeting with opposite orientations need.
    """
    d = int(d)
    side = str(side).upper()
    if side in _CORNERS:
        core = np.zeros((1, 1, 4, 1))
        core[0, 0, _CORNERS.index(side), 0] = 1.0
        return matrix.from_list([core] * d)
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES + _CORNERS}, got {side!r}")

    b = r = l = t = 0.0
    if side == "BOTTOM":
        b = 1.0
    elif side == "RIGHT":
        r = 1.0
    elif side == "LEFT":
        l = 1.0
    else:
        t = 1.0
    rows = np.array([[l, b, t, r], [b, r, l, t]]) if inversed \
        else np.array([[b, r, l, t], [l, b, t, r]])
    core = np.zeros((1, 2, 4, 1))
    core[0, :, :, 0] = rows
    return matrix.from_list([core] * d)


def interface_blocks(d, side_m, side_p):
    """``(Pmp, Ppm, Pmm, Ppp)`` -- the four blocks that couple two patches.

    With ``Y_m``, ``Y_p`` the traces of the shared edge seen from either side,
    these are ``Y_m^T Y_p``, ``Y_p^T Y_m``, ``-Y_m^T Y_m`` and ``-Y_p^T Y_p``:
    the blocks of the jump ``Y_m u_m - Y_p u_p``, so adding them to the block
    system penalises a discontinuity across the interface.
    """
    ym = sew(d, side_m, inversed=False)
    yp = sew(d, side_p, inversed=True)
    return (ym.T @ yp, yp.T @ ym, (-1.0) * (ym.T @ ym), (-1.0) * (yp.T @ yp))


def block_system(blocks, rhs, eps=1e-10):
    """Pack an ``m x m`` grid of TT operators into one, with a patch mode.

    The patch index becomes a final mode of size ``m``: ``S = sum_ij B_ij (x)
    E_ij``.  One tensor train for the whole multi-patch problem, which is what
    lets a single ``amen_solve`` see the coupling.
    """
    m = len(rhs)
    total = None
    for i in range(m):
        for j in range(m):
            if blocks[i][j] is None:
                continue
            e = np.zeros((1, m, m, 1))
            e[0, i, j, 0] = 1.0
            term = _tools.kron(blocks[i][j], matrix.from_list([e]))
            total = term if total is None else (total + term)
            total = total.round(eps)
    vec = None
    for i in range(m):
        e = np.zeros(m)
        e[i] = 1.0
        term = _tools.kron(rhs[i], vector.from_list([e.reshape(1, m, 1)]))
        vec = term if vec is None else (vec + term)
    return total, vec.round(eps)
