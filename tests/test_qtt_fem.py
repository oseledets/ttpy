"""Bilinear FEM assembled in QTT z-order (:mod:`tt.algs.qtt_fem`).

Oracles: the algebraic identities a stiffness matrix must satisfy (symmetry, and
constants in its kernel away from the boundary), the textbook element matrices,
and the exact solution of a manufactured Poisson problem.  Nothing is compared
against another tensor code.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import tt
from tt.algs.amen import amen_solve
from tt.algs.multifuncrs import multifuncrs
from tt.algs.qtt_fem import (apply_mask, assemble, dirichlet_mask,
                             local_entries, local_stiffness_uniform, placement)
from tt.core import tools as T

_M1 = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])


def local_mass_uniform(hx, hy):
    return {(l1, l2): hx * hy * _M1[l1[0], l2[0]] * _M1[l1[1], l2[1]]
            for l1 in itertools.product((0, 1), (0, 1))
            for l2 in itertools.product((0, 1), (0, 1))}


def const_entries(d, table):
    one = T.zkronv(T.ones(2, d), T.ones(2, d))
    return {k: one * v for k, v in table.items()}


def zsplit(d):
    """z-ordered flat index -> the two element/node indices."""
    idx = np.arange(4 ** d)
    ix = np.zeros(4 ** d, dtype=int)
    iy = np.zeros(4 ** d, dtype=int)
    for k in range(d):
        dig = (idx // (4 ** k)) % 4
        ix += (dig % 2) * (2 ** k)
        iy += (dig // 2) * (2 ** k)
    return ix, iy


# --- the placement operators --------------------------------------------------

def test_placement_is_the_identity_and_the_shift():
    """Corner 0 of element ``e`` is node ``e``; corner 1 is node ``e + 1``."""
    d = 3
    n = 2 ** d
    P = placement(d)
    ix, iy = zsplit(d)

    for (lx, ly), op in P.items():
        dense = np.asarray(op.full())
        # the diag-mask construction leaves 1e-16 dust; an entry of a placement
        # operator is either 1 or 0, so anything tiny is roundoff, not signal
        assert (np.abs(dense) < 1e-12).sum() + (np.abs(dense - 1) < 1e-12).sum() \
            == dense.size, "an entry is neither 0 nor 1"
        rows, cols = np.nonzero(np.abs(dense) > 0.5)
        # every entry must be 1 and must move each index by exactly l
        assert np.allclose(dense[rows, cols], 1.0)
        assert np.array_equal(ix[cols], ix[rows] + lx)
        assert np.array_equal(iy[cols], iy[rows] + ly)
        # and the fake element row e = n-1 must be dropped in BOTH directions:
        # a full identity deposits the fake elements' contributions on the last
        # row of nodes, which an all-Dirichlet mask hides and a glued interface
        # exposes (measured: the coupled energy falls under refinement)
        assert not np.any(ix[rows] == n - 1)
        assert not np.any(iy[rows] == n - 1)
        assert len(rows) == (n - 1) ** 2


# --- the assembled operator ---------------------------------------------------

@pytest.mark.parametrize("d", [3, 4, 5])
def test_stiffness_is_symmetric_and_kills_constants_everywhere(d):
    """A pure-Neumann stiffness matrix annihilates constants, with no excuses.

    This used to hold only on interior rows: the placement identity carried the
    fake element slot ``e = n-1``, whose contributions landed on the last row of
    nodes.  An all-Dirichlet mask hid that; a glued interface exposed it (the
    coupled energy of Markeeva's triangle fell under refinement).  With the fake
    row dropped, ``K 1 = 0`` holds everywhere -- which is what a Neumann
    operator owes.
    """
    n = 2 ** d
    h = 1.0 / (n - 1)
    K = assemble(const_entries(d, local_stiffness_uniform(h, h)), d)
    dense = np.asarray(K.full())

    assert np.abs(dense - dense.T).max() < 1e-13 * np.abs(dense).max()
    row_sums = dense @ np.ones(dense.shape[0])
    assert np.abs(row_sums).max() < 1e-11


@pytest.mark.parametrize("d", [3, 6, 10])
def test_the_rank_does_not_grow_with_the_mesh(d):
    """Rank bounded by 25 whatever ``d`` is -- the point of the z-order.

    Sixteen corner pairs, each rank 1 on a uniform mesh, would give 16; the
    element restriction (the dropped fake row ``e = n-1``, a diagonal 0/1 mask
    of rank 2 per direction) lifts the bound to 25 and it saturates there.
    What matters is that it does not grow with ``d``.
    """
    h = 1.0 / (2 ** d - 1)
    K = assemble(const_entries(d, local_stiffness_uniform(h, h)), d)
    assert max(K.r) <= 25, list(K.r)
    if d >= 6:
        assert max(K.r) == 25   # saturated, not growing


def test_dirichlet_mask_selects_the_interior():
    d = 4
    n = 2 ** d
    m = np.diag(np.asarray(dirichlet_mask("DD", "DD", d).full()))
    ix, iy = zsplit(d)
    want = ((ix > 0) & (ix < n - 1) & (iy > 0) & (iy < n - 1)).astype(float)
    assert np.abs(m - want).max() < 1e-14


# --- end to end ---------------------------------------------------------------

def test_poisson_converges_at_second_order():
    """``-Lap u = 2 pi^2 sin(pi x) sin(pi y)``, ``u = 0`` on the boundary.

    Exact solution ``sin(pi x) sin(pi y)``.  Bilinear elements give ``O(h^2)``,
    and that slope -- not the size of the error -- is what pins the assembly:
    a wrong element matrix converges to the wrong thing at some other rate, or
    to the right thing at first order.
    """
    errs = {}
    for d in (3, 4, 5, 6):
        n = 2 ** d
        h = 1.0 / (n - 1)
        K = assemble(const_entries(d, local_stiffness_uniform(h, h)), d)
        M = assemble(const_entries(d, local_mass_uniform(h, h)), d)

        xx, yy = T.zmeshgrid(d)
        xx, yy = xx * h, yy * h
        f = multifuncrs([xx, yy],
                        lambda v: 2 * np.pi ** 2 * np.sin(np.pi * v[:, 0])
                        * np.sin(np.pi * v[:, 1]), eps=1e-12, verb=0)
        rhs = T.matvec(M, f).round(1e-12)

        Kb, rhsb = apply_mask(K, rhs, dirichlet_mask("DD", "DD", d))
        u = amen_solve(Kb, rhsb, rhsb, 1e-10, nswp=40, verb=0, kickrank=8)

        ix, iy = zsplit(d)
        got = np.asarray(u.full(asvector=True))
        exact = np.sin(np.pi * ix * h) * np.sin(np.pi * iy * h)
        errs[d] = float(np.abs(got - exact).max() / np.abs(exact).max())

    assert errs[6] < 3e-4, errs
    order = np.log2(errs[3] / errs[6]) / 3.0
    assert 1.8 < order < 2.3, f"observed order {order:.2f}, expected 2: {errs}"


def test_a_curved_element_map_goes_through_the_cross_path():
    """``local_entries`` with a Jacobian must reproduce the uniform case.

    A constant Jacobian ``h I`` is the uniform mesh, so the cross path and the
    closed form have to agree -- which is the check that the quadrature, the
    adjugate and the z-order index split all line up.
    """
    d = 3
    h = 1.0 / (2 ** d - 1)

    def jac(e):
        J = np.zeros((len(e), 2, 2))
        J[:, 0, 0] = J[:, 1, 1] = 0.5 * h      # d(physical)/d(reference [-1,1])
        return J

    got = local_entries(d, jac=jac, eps=1e-12, r=2, kickrank=2, seed=0)
    want = local_stiffness_uniform(h, h)
    for key, ref in want.items():
        v = np.asarray(got[key].full()).reshape(-1)
        assert np.abs(v - ref).max() < 1e-10, f"{key}: {v[:3]} vs {ref}"


def test_multipatch_system_reproduces_the_published_triangle_energy():
    """The public multipatch solver against the qtt-laplace energy column.

    The whole path -- three bilinearly-mapped patches, z-order assembly,
    boundary masks, interface gluing, one amen_solve -- must land on the
    energies published with the original implementation
    (``triangle_tt_energy.txt`` of github.com/RerRayne/qtt-laplace), which
    is an oracle produced by a different code on a different stack.
    """
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent / "examples"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qtt_fem_triangle import QTTLAPLACE_TT, solve

    for d in (2, 3):
        energy = solve(d, eps=1e-8, verbose=False)
        ref = QTTLAPLACE_TT[d]
        assert abs(energy - ref) / ref < 1e-8, (d, energy, ref)
