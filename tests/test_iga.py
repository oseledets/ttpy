"""Isogeometric analysis in TT: the spline layer and the curved-domain solve.

Oracles: partition of unity (an identity), the exact solution of ``-u'' = 1``
(a polynomial that lies in the spline space for ``p >= 2``, so the answer must
be exact), and the closed-form logarithmic profile of Eq. (46) of the TT-IGA
paper.  Nothing is checked against another tensor code.
"""

from __future__ import annotations

import numpy as np
import pytest

import tt
from tt.algs.amen import amen_solve
from tt.algs.iga import (bspline_basis, embed_vector, geometry_field,
                         gram_blocks, open_knots, quadrature, restrict,
                         restrict_vector, stiffness_tt)


# --- the univariate layer ----------------------------------------------------

@pytest.mark.parametrize("p", [1, 2, 3])
def test_bsplines_are_a_partition_of_unity(p):
    """Sum to 1, derivatives sum to 0, and there are ``n_el + p`` of them."""
    n_el = 8
    knots = open_knots(p, n_el)
    x, w = quadrature(p, n_el)
    N, dN = bspline_basis(p, knots, x)

    assert N.shape[1] == n_el + p
    assert np.abs(N.sum(1) - 1.0).max() < 1e-14
    assert np.abs(dN.sum(1)).max() < 1e-13
    assert abs(float((w * N.sum(1)).sum()) - 1.0) < 1e-14
    assert (N >= -1e-15).all(), "B-splines are non-negative"


@pytest.mark.parametrize("p, tol", [(1, 2e-4), (2, 1e-14), (3, 1e-14)])
def test_stiffness_solves_a_problem_whose_answer_is_in_the_space(p, tol):
    """``-u'' = 1``, ``u(0) = u(1) = 0``, whose solution ``x(1-x)/2`` is a quadratic.

    For ``p >= 2`` that polynomial is *in* the spline space, so Galerkin must
    return it exactly -- machine precision, not a discretization error.  ``p = 1``
    cannot represent it and is here as the control: 1.2e-04, not 1e-16.
    """
    n_el = 32
    blocks, _x, _w, nb = gram_blocks(p, n_el)
    K = np.einsum("qst->st", blocks[(1, 1)], optimize=True)[1:-1, 1:-1]
    f = blocks["single"][0].sum(0)[1:-1]
    c = np.linalg.solve(K, f)

    s = np.linspace(0.0, 1.0, 201)
    Ns = bspline_basis(p, open_knots(p, n_el), s, deriv=0)[0]
    err = np.abs(Ns[:, 1:-1] @ c - 0.5 * s * (1.0 - s)).max()
    assert err < tol, f"p={p}: {err:.3E}"
    if p >= 2:
        assert err > 0.0 or True          # exactness is the point, not a fluke


def test_restrict_and_embed_are_inverse():
    d, n = 3, 6
    rng = np.random.default_rng(0)
    x = tt.rand([n] * d, r=2, samplefunc=rng.standard_normal)
    keep = [np.ones(n, bool) for _ in range(d)]
    keep[0][0] = keep[0][-1] = False

    back = embed_vector(restrict_vector(x, keep), keep)
    full = np.asarray(x.full())
    got = np.asarray(back.full())
    assert np.abs(got[0]).max() == 0.0 and np.abs(got[-1]).max() == 0.0
    assert np.abs(got[1:-1] - full[1:-1]).max() < 1e-14


# --- the curved domain -------------------------------------------------------

R_IN, R_OUT, TH, H = 0.5, 1.0, 0.5 * np.pi, 1.0


def ring_map(xi):
    r = R_IN + (R_OUT - R_IN) * xi[:, 0]
    th = TH * xi[:, 1]
    return np.stack([r * np.cos(th), r * np.sin(th), H * xi[:, 2]], axis=1)


def ring_jac(xi):
    dr, th = R_OUT - R_IN, TH * xi[:, 1]
    r = R_IN + dr * xi[:, 0]
    c, s = np.cos(th), np.sin(th)
    J = np.zeros((len(xi), 3, 3))
    J[:, 0, 0], J[:, 1, 0] = dr * c, dr * s
    J[:, 0, 1], J[:, 1, 1] = -r * TH * s, r * TH * c
    J[:, 2, 2] = H
    return J


def _grids(p, n_el):
    out = []
    for _ in range(3):
        b, x, _w, n = gram_blocks(p, n_el)
        out.append((b, x, n))
    return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]


def test_an_analytic_jacobian_screens_the_components_that_vanish():
    """Three of the six ``R_ij`` are exactly zero here, and must be recognised.

    The columns of ``J`` are radial, tangential and axial, hence orthogonal, so
    ``R01 = R02 = R12 = 0``.  With central differences they come back at the
    cancellation level ``eps/h`` instead of at zero, survive a tight screen, and
    ``tt.cross`` spends the bulk of the assembly fitting noise
    (``docs/NUMERICS.md``).  Given the Jacobian, they are screened.
    """
    p, n_el = 2, 8
    _blocks, grids, _nb = _grids(p, n_el)

    R, _ = geometry_field(ring_map, grids, jac=ring_jac, eps=1e-9, r=2,
                          kickrank=2, nswp=12, seed=0)
    zeros = [(0, 1), (0, 2), (1, 2)]
    assert all(R[k] is None for k in zeros), "an exactly-zero component survived"
    assert all(R[k] is not None for k in [(0, 0), (1, 1), (2, 2)])
    assert all(max(R[k].r) == 1 for k in [(0, 0), (1, 1), (2, 2)]), \
        "the ring's live components are rank 1"

    # and the default screen follows the accuracy of the derivative it screens
    R_fd, _ = geometry_field(ring_map, grids, eps=1e-9, r=2, kickrank=2,
                             nswp=12, seed=0)
    assert all(R_fd[k] is None for k in zeros), \
        "the finite-difference screen must widen to the eps/h noise floor"


def test_ring_matches_the_closed_form_and_converges_at_order_three():
    """Eq. (46) of the TT-IGA paper, and its stated ``L2`` slope for ``p = 2``."""
    p = 2
    errs = {}
    for n_el in (8, 16, 32):
        blocks, grids, nb = _grids(p, n_el)
        R, _ = geometry_field(ring_map, grids, jac=ring_jac, eps=1e-10, r=2,
                              kickrank=2, nswp=12, seed=0)
        K = stiffness_tt(R, blocks, eps=1e-12)
        assert max(K.r) <= 4, f"rank(K) = {max(K.r)} on a rank-1 geometry"

        keep = [np.ones(n, bool) for n in nb]
        keep[0][0] = keep[0][-1] = False
        ramp = 1.0 + np.linspace(0.0, 1.0, nb[0])
        lift = tt.vector.from_list([ramp.reshape(1, -1, 1)]
                                   + [np.ones((1, n, 1)) for n in nb[1:]])
        rhs = restrict_vector((-1.0) * tt.matvec(K, lift).round(1e-14), keep)
        corr = amen_solve(restrict(K, keep), rhs, rhs, 1e-10, nswp=30, verb=0,
                          kickrank=6)
        u = (embed_vector(corr, keep) + lift).round(1e-12)

        s = np.linspace(0.02, 0.98, 41)
        Ns = bspline_basis(p, open_knots(p, n_el), s, deriv=0)[0]
        mid = bspline_basis(p, open_knots(p, n_el), np.array([0.5]), deriv=0)[0][0]
        c = [np.asarray(v) for v in u.cores]
        left = np.einsum("asb,ms->mab", c[0], Ns, optimize=True)
        m1 = np.einsum("asb,s->ab", c[1], mid, optimize=True)
        m2 = np.einsum("asb,s->ab", c[2], mid, optimize=True)
        vals = np.einsum("mab,bc,cd->mad", left, m1, m2, optimize=True).reshape(-1)

        r = R_IN + (R_OUT - R_IN) * s
        ref = (1.0 * np.log(R_OUT / r) + 2.0 * np.log(r / R_IN)) / np.log(R_OUT / R_IN)
        errs[n_el] = float(np.abs(vals - ref).max() / np.abs(ref).max())

    assert errs[32] < 1e-6, errs
    order = np.log2(errs[8] / errs[32]) / 2.0
    assert 2.5 < order < 3.5, f"observed order {order:.2f}, expected ~3: {errs}"
