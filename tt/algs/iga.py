"""Isogeometric analysis in the TT format: B-splines, and assembly by cross.

The method of [TTRA26]. Isogeometric analysis puts the same tensor-product
B-spline basis on the geometry and on the solution, so the parametric domain is
always the unit cube and the physical domain enters only through the map
``x(xi)`` and its Jacobian ``J = dx/dxi``. Pulling ``int grad u . grad v`` back
to the cube gives

    sum_{i,j} int (dN/dxi_i)^T R_ij(xi) (dN/dxi_j) dxi,   R = J^-1 J^-T |J|

so the **only** things that ever need approximating are the six independent
scalar fields ``R_ij`` and, for a source term, ``G = f |J|``. Each is a function
on the quadrature grid, i.e. exactly what :func:`tt.cross` eats; the geometry
itself is evaluated exactly at the sampling points and never compressed.

The assembly is then free. If ``R_ij`` has TT cores ``G_k[a, q, b]`` over the
quadrature index ``q`` of direction ``k``, the TT-matrix core of ``K_ij`` is

    core_k[a, s, t, b] = sum_q G_k[a, q, b] w_q  D^{di} N_s(x_q)  D^{dj} N_t(x_q)

with the derivative orders ``di = [k == i]`` and ``dj = [k == j]``. One
contraction per core, no assembly loop over elements, and the TT rank of ``K_ij``
is the TT rank of ``R_ij``.

What this module is not
-----------------------
There are no NURBS here: the map is any callable, and the ring of
``examples/iga_ring.py`` is parameterised analytically. Rational weights are the
step that would make it isogeometric in the strict sense, and they change only
:func:`geometry_field`, not the assembly.

[TTRA26] Q. T. Tran, D. P. Truong, K. O. Rasmussen, B. S. Alexandrov, *A tensor
train-based isogeometric solver for large-scale 3D Poisson problems*, Comput.
Methods Appl. Mech. Engrg. 453 (2026) 118802.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.matrix import matrix
from ..core.vector import vector

__all__ = ["open_knots", "quadrature", "bspline_basis", "gram_blocks",
           "stiffness_tt", "load_tt", "restrict", "restrict_vector",
           "embed_vector",
           "geometry_field",
           "SMALL_FIELD_RATIO"]


# --- the univariate B-spline layer -------------------------------------------

def open_knots(p, n_el):
    """Open (clamped) knot vector on ``[0, 1]`` with ``n_el`` equal elements.

    ``p + 1`` repeats at each end, so the basis is interpolatory at 0 and 1 --
    which is what makes a Dirichlet condition a statement about one coefficient
    rather than about a combination of them.  Returns a vector of length
    ``n_el + 2p + 1``; the basis has ``n_el + p`` functions.
    """
    p, n_el = int(p), int(n_el)
    if p < 1 or n_el < 1:
        raise ValueError(f"need p >= 1 and n_el >= 1, got p={p}, n_el={n_el}")
    interior = np.linspace(0.0, 1.0, n_el + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def quadrature(p, n_el, per_element=None):
    """Gauss-Legendre points and weights on ``[0, 1]``, element by element.

    ``per_element`` defaults to ``p + 1``, which integrates the products of two
    degree-``p`` splines exactly on each element -- the integrand of the mass
    matrix is a polynomial of degree ``2p``, and ``p + 1`` Gauss points are exact
    to degree ``2p + 1``.  It is *not* exact once a non-polynomial ``R(xi)``
    multiplies it, which is why the coefficient field is sampled on this same
    grid rather than integrated in closed form.
    """
    p, n_el = int(p), int(n_el)
    m = int(p + 1 if per_element is None else per_element)
    xg, wg = np.polynomial.legendre.leggauss(m)
    edges = np.linspace(0.0, 1.0, n_el + 1)
    h = edges[1] - edges[0]
    mid = 0.5 * (edges[:-1] + edges[1:])
    x = (mid[:, None] + 0.5 * h * xg[None, :]).reshape(-1)
    w = np.tile(0.5 * h * wg, n_el)
    return x, w


def bspline_basis(p, knots, x, deriv=1):
    """Dense values and derivatives of every B-spline of degree ``p`` at ``x``.

    Cox-de Boor, evaluated span by span: at any point only ``p + 1`` functions
    are non-zero, so the cost is ``O(len(x) p^2)`` and the dense
    ``(len(x), n_basis)`` result is a convenience, not the work.

    Returns:
        ``[N]`` if ``deriv == 0``, else ``[N, dN]`` -- each ``(len(x), n_basis)``.
    """
    p = int(p)
    knots = np.asarray(knots, float)
    x = np.atleast_1d(np.asarray(x, float))
    nb = len(knots) - p - 1
    out = [np.zeros((len(x), nb)) for _ in range(deriv + 1)]

    for m, xm in enumerate(x):
        # the span: the last i with knots[i] <= xm, clamped into the interior
        i = int(np.searchsorted(knots, xm, side="right") - 1)
        i = min(max(i, p), nb - 1)

        # Cox-de Boor upward recursion on the p+1 non-zero functions
        vals = np.zeros((p + 1, p + 1))          # vals[q, j] = N_{i-q+j, q}
        vals[0, 0] = 1.0
        for q in range(1, p + 1):
            for j in range(q + 1):
                idx = i - q + j
                acc = 0.0
                if j > 0:
                    den = knots[idx + q] - knots[idx]
                    if den > 0:
                        acc += (xm - knots[idx]) / den * vals[q - 1, j - 1]
                if j < q:
                    den = knots[idx + q + 1] - knots[idx + 1]
                    if den > 0:
                        acc += (knots[idx + q + 1] - xm) / den * vals[q - 1, j]
                vals[q, j] = acc
        out[0][m, i - p:i + 1] = vals[p]

        if deriv >= 1:
            # N'_{a,p} = p [ N_{a,p-1}/(t_{a+p}-t_a) - N_{a+1,p-1}/(t_{a+p+1}-t_{a+1}) ]
            for j in range(p + 1):
                idx = i - p + j
                d = 0.0
                if j > 0:
                    den = knots[idx + p] - knots[idx]
                    if den > 0:
                        d += vals[p - 1, j - 1] / den
                if j < p:
                    den = knots[idx + p + 1] - knots[idx + 1]
                    if den > 0:
                        d -= vals[p - 1, j] / den
                out[1][m, idx] = p * d
    return out


def gram_blocks(p, n_el, per_element=None):
    """The four quadrature-weighted outer products, indexed by derivative order.

    ``B[(a, b)][q, s, t] = w_q  D^a N_s(x_q)  D^b N_t(x_q)``, so contracting it
    with a coefficient core over ``q`` gives the univariate Galerkin matrix that
    coefficient asks for -- ``(1,1)`` for the stiffness direction, ``(0,0)`` for
    a mass direction, ``(1,0)`` and ``(0,1)`` for the mixed ones.

    Returns ``(blocks, x, w, n_basis)``.
    """
    knots = open_knots(p, n_el)
    x, w = quadrature(p, n_el, per_element)
    N, dN = bspline_basis(p, knots, x, deriv=1)
    D = {0: N, 1: dN}
    blocks = {(a, b): w[:, None, None] * D[a][:, :, None] * D[b][:, None, :]
              for a in (0, 1) for b in (0, 1)}
    # the one-sided factors, for a load vector rather than a matrix
    blocks["single"] = {a: w[:, None] * D[a] for a in (0, 1)}
    return blocks, x, w, N.shape[1]


# --- the geometry ------------------------------------------------------------

#: A component of ``R`` smaller than this fraction of ``max|R|`` is taken to be
#: identically zero and never handed to the cross.
#:
#: Not a nicety.  On an untwisted ring three components of ``R`` vanish
#: identically, and a cross asked to approximate roundoff does not notice:
#: measured on a ``128^3`` grid, ``R01`` (largest entry 2.0e-16) came back at
#: TT rank 85 after 4 631 168 evaluations and 5.0 s with ``converged=False``,
#: which is 97 % of the evaluations and 98 % of the time of the whole assembly.
#: The screen is absolute -- relative to the *field*, not to each component --
#: because a component's own scale is exactly what is meaningless when it is
#: noise.
SMALL_FIELD_RATIO = 1e-12


def geometry_field(fun, grids, jac=None, eps=1e-8, screen=None, fd_step=1e-6,
                   **cross_kw):
    """The six independent ``R_ij = (J^-1 J^-T |J|)_ij`` as TT vectors.

    Args:
        fun: ``fun(xi) -> x`` mapping the unit cube to the physical domain,
            vectorized over a ``(batch, 3)`` array and returning ``(batch, 3)``.
        jac: ``jac(xi) -> J`` of shape ``(batch, 3, 3)`` with ``J[:, i, k] =
            dx_i/dxi_k``.  **Give this if you have it.**  Without it the
            Jacobian is taken by central differences, whose cancellation error
            is ``eps_machine / fd_step`` -- 1e-10 at the default step -- and a
            component of ``R`` that is *exactly* zero comes back at that level
            instead of at zero, so the screen keeps it and ``tt.cross`` spends
            rank on fitting the noise (``docs/NUMERICS.md``).
        grids: the quadrature grids, one array per direction.
        eps: cross accuracy for every component.
        screen: components below ``screen * max|R|`` are returned as ``None``.
            ``None`` (default) means :data:`SMALL_FIELD_RATIO` with an analytic
            ``jac`` and ``100 * eps_machine / fd_step`` without one -- i.e. the
            threshold follows the accuracy of the derivative it is screening.
        fd_step: step of the central differences when ``jac`` is None.

    Returns:
        ``(R, scale)`` -- a dict keyed by ``(i, j)`` with ``i <= j`` whose values
        are :class:`tt.vector` or ``None``, and the largest ``|R|`` seen while
        screening.
    """
    from .cross import cross

    n = [len(g) for g in grids]
    d = len(grids)

    def sample(idx):
        xi = np.stack([grids[k][idx[:, k]] for k in range(d)], axis=1)
        return xi

    if jac is None:
        def jac_fd(xi):
            out = np.empty((len(xi), d, d))
            for k in range(d):
                e = np.zeros(d)
                e[k] = fd_step
                out[:, :, k] = (fun(xi + e) - fun(xi - e)) / (2 * fd_step)
            return out
        jac_use = jac_fd
        if screen is None:
            screen = 100.0 * float(np.finfo(np.float64).eps) / float(fd_step)
    else:
        jac_use = jac
        if screen is None:
            screen = SMALL_FIELD_RATIO

    def rfield(i, j):
        def f(idx):
            J = jac_use(sample(idx))
            det = np.linalg.det(J)
            Jinv = np.linalg.inv(J)
            return np.einsum("bik,bjk->bij", Jinv, Jinv)[:, i, j] * det
        return f

    # one coarse probe to set the scale the screen is relative to
    rng = np.random.default_rng(0)
    probe = np.stack([rng.integers(0, n[k], 512) for k in range(d)], axis=1)
    scale = 0.0
    raw = {}
    for i in range(d):
        for j in range(i, d):
            raw[(i, j)] = rfield(i, j)
            scale = max(scale, float(np.abs(raw[(i, j)](probe)).max()))
    if scale == 0.0:
        raise ValueError("the geometry field is identically zero; check `fun`")

    out = {}
    for (i, j), f in raw.items():
        if float(np.abs(f(probe)).max()) <= screen * scale:
            out[(i, j)] = None
            continue
        out[(i, j)] = cross(f, n, d, eps=eps, **cross_kw)
    return out, scale


# --- assembly ----------------------------------------------------------------

def _direction_core(gcore, block):
    """``core[a, s, t, b] = sum_q gcore[a, q, b] block[q, s, t]``."""
    g = np.asarray(gcore)
    r0, nq, r1 = g.shape
    ns = block.shape[1]
    acc = np.einsum("aqb,qc->abc", g, block.reshape(nq, -1), optimize=True)
    return np.ascontiguousarray(
        acc.reshape(r0, r1, ns, ns).transpose(0, 2, 3, 1))


def stiffness_tt(R, blocks, eps=1e-10):
    """``K = sum_ij K_ij`` as a :class:`tt.matrix`, rounded after each summand.

    Args:
        R: the dict from :func:`geometry_field`; ``None`` entries are skipped
            and off-diagonal ones are counted twice, since ``R`` is symmetric.
        blocks: one ``blocks`` dict per direction, from :func:`gram_blocks`.
        eps: rounding accuracy of the accumulating sum.

    The rank of each summand is the rank of its coefficient field, so a geometry
    whose ``R`` is rank 1 gives a rank-1 stiffness matrix in that term -- the
    point of the whole method.
    """
    d = len(blocks)
    total = None
    for (i, j), field in sorted(R.items()):
        if field is None:
            continue
        cores = []
        for k in range(d):
            b = blocks[k][(1 if k == i else 0, 1 if k == j else 0)]
            cores.append(_direction_core(field.cores[k], b))
        term = matrix.from_list(cores)
        if i != j:                      # R is symmetric: the (j, i) twin
            twin = []
            for k in range(d):
                b = blocks[k][(1 if k == j else 0, 1 if k == i else 0)]
                twin.append(_direction_core(field.cores[k], b))
            term = term + matrix.from_list(twin)
        total = term if total is None else (total + term)
        total = total.round(eps)
    if total is None:
        raise ValueError("every component of R was screened out")
    return total


def load_tt(G, blocks):
    """The load vector ``int f |J| N dxi`` from the TT field ``G = f |J|``.

    Same contraction as the stiffness with one free index instead of two, so a
    rank-1 source on a rank-1 geometry gives a rank-1 right-hand side.
    """
    cores = []
    for k, blk in enumerate(blocks):
        g = np.asarray(G.cores[k])
        cores.append(np.ascontiguousarray(
            np.einsum("aqb,qs->asb", g, blk["single"][0], optimize=True)))
    return vector.from_list(cores)


# --- boundary conditions ------------------------------------------------------

def restrict(A, keep):
    """Drop basis functions from a TT-matrix, one boolean mask per direction.

    With an open knot vector the first and last function of a direction are the
    only ones alive on that face, so a homogeneous Dirichlet condition is
    ``keep[k][0] = keep[k][-1] = False`` -- a slice of the cores, not a
    projection.
    """
    cores = []
    for c, m in zip(matrix.to_list(A), keep):
        m = np.asarray(m, bool)
        cores.append(np.ascontiguousarray(np.asarray(c)[:, m][:, :, m]))
    return matrix.from_list(cores)


def embed_vector(x, keep):
    """Inverse of :func:`restrict_vector`: pad the dropped basis functions back.

    The correction solved for on the interior has to return to the full index
    set before the Dirichlet lift can be added to it.
    """
    cores = []
    for c, m in zip(x.cores, keep):
        m = np.asarray(m, bool)
        c = np.asarray(c)
        full = np.zeros((c.shape[0], m.size, c.shape[2]), dtype=c.dtype)
        full[:, m, :] = c
        cores.append(full)
    return vector.from_list(cores)


def restrict_vector(x, keep):
    """:func:`restrict` for a TT-vector."""
    cores = []
    for c, m in zip(x.cores, keep):
        cores.append(np.ascontiguousarray(np.asarray(c)[:, np.asarray(m, bool)]))
    return vector.from_list(cores)
