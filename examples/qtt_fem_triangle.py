#!/usr/bin/env python
"""Poisson on a triangle, three glued QTT patches -- Markeeva's benchmark.

    python examples/qtt_fem_triangle.py            # d = 2..6
    python examples/qtt_fem_triangle.py 7

``-Lap u = 1`` on the triangle with vertices ``(0,0)``, ``(2.8, 0.3)``,
``(2, e)`` and ``u = 0`` on the whole boundary.  The triangle is split through
the midpoints of its sides and its centroid into **three quadrilaterals**, each
meshed by a bilinear map of the unit square, each assembled in QTT z-order, and
the three glued along their shared edges.

This is the benchmark of L. Markeeva's ``qtt-laplace``
(https://github.com/RerRayne/qtt-laplace) -- her ``SolutionOnTriangle``
notebook, her geometry, her one-point element rule and her interface coupling.
The algorithm is hers; this is a reimplementation on ttpy2, not a copy.

The reference is the energy ``int |grad u|^2``, and it comes from outside the
tensor world: her repository ships a FEniCS convergence curve, which reaches

    0.34034426  at 11253 dofs,  0.34037443 at 15882,  0.34039282 at 21787

so the continuum value is ~0.3404.  Her own QTT column reads 0.34854914 at
3*4^2 nodes, 0.34180604 at 3*4^3 and 0.34073671 at 3*4^4, approaching it from
above -- and those three numbers are what the run below has to reproduce.

Why a triangle is the interesting case: a quadrilateral patch touching the
centroid is a genuine curved-index map, so ``det J`` varies over the mesh and
the coefficient fields are not constant.  The uniform square in
``tests/test_qtt_fem.py`` never exercises that.

Reproduced, to her own precision
--------------------------------
    d=2:  0.34854914  (hers 0.34854914, agreement 2.1e-14)
    d=3:  0.34180604  (hers 0.34180604, agreement 1.5e-09)
    d=4:  0.34073671  (hers 0.34073671, agreement 1.6e-09)
    d=5:  0.34051294  (vs FEniCS continuum 0.34039, rel 3.5e-04)

approaching the continuum from above, as a Galerkin energy must.

Getting there found a real defect in the port, worth remembering.  A mesh of
``2^d`` nodes has ``2^d - 1`` elements, but the z-ordered diagonal carries
``4^d`` element slots; the row ``e = n-1`` is fake.  The shift operator drops
it by construction, the identity does not -- and with a full identity the fake
elements deposit their ``(0, .)``-corner contributions on the last row of
nodes.  Under an all-Dirichlet mask (every test on the unit square) that is
invisible.  On a glued problem those nodes are interface nodes, they are free,
and the energy *falls* under refinement instead of rising.  Her ``W0``,
materialized densely from her repository, has the zero row; ``placement`` now
does too, and the module docstring records the measurement.
"""

import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
import tt
from tt.algs.amen import amen_solve
from tt.algs.cross import cross
from tt.algs.qtt_fem import (apply_mask, block_system, dirichlet_mask,
                             interface_blocks, placement)
from tt.core import tools as T

R1 = np.array([0.0, 0.0])
R2 = np.array([2.8, 0.3])
R3 = np.array([2.0, 2.71828])

#: her FEniCS curve, the finest three rows of examples/triangle_energy.txt
FENICS = {11253: 0.34034426241362858, 15882: 0.34037442708539084,
          21787: 0.34039281527751902}
#: her own QTT column, examples/triangle_tt_energy.txt
HER_TT = {2: 0.34854913898796147, 3: 0.34180604253413738, 4: 0.34073671256641080}


def subdomains():
    """The three quads: (vertex, midpoint, centroid, midpoint)."""
    r12, r13, r23 = 0.5 * (R1 + R2), 0.5 * (R1 + R3), 0.5 * (R2 + R3)
    rc = (R1 + R2 + R3) / 3.0
    return [np.array([R1, r12, rc, r13]),
            np.array([r12, R2, r23, rc]),
            np.array([rc, r23, R3, r13])]


def corner_map(quad, d):
    """``pts(i, j)`` and the element-centre Jacobian, both vectorized."""
    p1, p2, p3, p4 = quad
    n = 2 ** d

    def pts(i, j):
        x = np.asarray(i, float) / (n - 1.0)
        y = np.asarray(j, float) / (n - 1.0)
        return (p1[:, None] * ((1 - x) * (1 - y))[None, :]
                + p2[:, None] * (x * (1 - y))[None, :]
                + p3[:, None] * (x * y)[None, :]
                + p4[:, None] * ((1 - x) * y)[None, :])

    def jac(ex, ey):
        a, b = pts(ex, ey), pts(ex + 1, ey)
        c, e = pts(ex + 1, ey + 1), pts(ex, ey + 1)
        return 0.25 * np.array([[b[0] + c[0] - a[0] - e[0],
                                 c[0] + e[0] - a[0] - b[0]],
                                [b[1] + c[1] - a[1] - e[1],
                                 c[1] + e[1] - a[1] - b[1]]])
    return pts, jac


def zsplit_index(idx, d):
    """z-ordered digits -> the two element indices."""
    ex = np.zeros(len(idx), dtype=np.int64)
    ey = np.zeros(len(idx), dtype=np.int64)
    for k in range(d):
        digit = idx[:, k]
        ex += (digit % 2) * (2 ** k)
        ey += (digit // 2) * (2 ** k)
    return ex, ey


# the bilinear basis at the element centre: value 1/4, gradients +-1/4
_G = {(0, 0): (-0.25, -0.25), (1, 0): (0.25, -0.25),
      (1, 1): (0.25, 0.25), (0, 1): (-0.25, 0.25)}
_CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]
_QUAD_AREA = 4.0                     # of the reference square [-1,1]^2


def patch_system(quad, d, eps):
    """Her ``assemble_on_quad``: stiffness and load of one patch, in z-order."""
    _pts, jac = corner_map(quad, d)

    def field(fn):
        def f(idx):
            ex, ey = zsplit_index(idx, d)
            return fn(jac(ex, ey))
        return cross(f, 4, d, eps=eps, r=2, kickrank=2, nswp=15, seed=0)

    det = field(lambda J: J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    j11 = field(lambda J: J[0, 0])
    j12 = field(lambda J: J[0, 1])
    j21 = field(lambda J: J[1, 0])
    j22 = field(lambda J: J[1, 1])
    from tt.algs.multifuncrs import multifuncrs
    idet = multifuncrs([det], lambda v: 1.0 / v[:, 0], eps=eps, verb=0)

    tj11 = ((j22 * j22 + j12 * j12) * idet).round(eps)
    tj22 = ((j11 * j11 + j21 * j21) * idet).round(eps)
    tj12 = ((-1.0) * (j22 * j21 + j12 * j11) * idet).round(eps)

    P = placement(d)
    A = M = None
    for l1 in _CORNERS:
        for l2 in _CORNERS:
            g1, g2 = _G[l1], _G[l2]
            kl = (_QUAD_AREA * (g1[0] * g2[0] * tj11 + g1[1] * g2[1] * tj22
                                + (g1[0] * g2[1] + g1[1] * g2[0]) * tj12)).round(eps)
            gl = (_QUAD_AREA * 0.0625 * det).round(eps)
            a = (P[l1].T @ T.diag(kl) @ P[l2]).round(eps)
            m = (P[l1].T @ T.diag(gl) @ P[l2]).round(eps)
            A = a if A is None else (A + a).round(eps)
            M = m if M is None else (M + m).round(eps)
    f = T.matvec(M, T.ones(4, d)).round(eps)
    return A, f


def solve(d, eps=1e-8, verbose=True):
    t0 = time.time()
    quads = subdomains()
    sysm = [patch_system(q, d, eps) for q in quads]
    t_asm = time.time() - t0

    # her masks: the sides that lie on the true outer boundary
    masks = [dirichlet_mask("DN", "DN", d),
             dirichlet_mask("ND", "DN", d),
             dirichlet_mask("ND", "ND", d)]
    A, F = [], []
    for (a, f), m in zip(sysm, masks):
        aa, ff = apply_mask(a, f, m, eps)
        A.append(aa)
        F.append(ff)

    # the coupling, lambda = 1/2 as in her notebook
    lam = 0.5
    n = 3
    B = [[None] * n for _ in range(n)]
    for i in range(n):
        B[i][i] = A[i]
    G = [f.copy() for f in F]

    def glue(i, j, side_i, side_j):
        Pij, Pji, Pii, Pjj = interface_blocks(d, side_i, side_j)
        B[i][j] = (Pij @ A[j] - lam * Pij).round(eps)
        B[j][i] = (Pji @ A[i] - lam * Pji).round(eps)
        B[i][i] = (B[i][i] - lam * Pii).round(eps)
        B[j][j] = (B[j][j] - lam * Pjj).round(eps)
        G[i] = (G[i] + T.matvec(Pij, F[j])).round(eps)
        G[j] = (G[j] + T.matvec(Pji, F[i])).round(eps)

    glue(0, 1, "RIGHT", "LEFT")
    glue(1, 2, "TOP", "BOTTOM")
    glue(2, 0, "LEFT", "TOP")

    # The system is driven by the *coupled* right-hand side -- it carries the
    # interface terms -- while the energy pairs the solution with the original
    # load, because int |grad u|^2 = u^T f for the Galerkin solution.  Her
    # notebook does the same, through a pair of names that read the other way
    # round (its `F` is built from `ggg` and its `G` from `fff`).
    S, load = block_system(B, F, eps=eps * 0.01)
    _, coupled = block_system(B, G, eps=eps * 0.01)

    t0 = time.time()
    u = amen_solve(S, coupled, coupled, eps, nswp=40, verb=0, kickrank=8)
    t_solve = time.time() - t0
    energy = float(T.sum(u * load))

    if verbose:
        dofs = 3 * 4 ** d
        ref = FENICS[21787]
        line = (f"  d={d}  dofs {dofs:>9,d}  rank(S)={max(S.r):4d}  "
                f"asm {t_asm:6.2f}s  solve {t_solve:6.2f}s  "
                f"energy {energy:.8f}  vs FEniCS {abs(energy - ref) / ref:8.2e}")
        if d in HER_TT:
            line += f"  vs her TT {abs(energy - HER_TT[d]) / HER_TT[d]:.2e}"
        print(line, flush=True)
    return energy


def main(argv):
    ds = [int(v) for v in argv[1:]] or [2, 3, 4, 5, 6]
    print(__doc__.split("Why a triangle")[0].strip())
    print()
    print(f"  FEniCS reference (her repository): {FENICS[21787]:.8f} at 21787 dofs")
    print(f"  her QTT column: " + ", ".join(f"d={k}: {v:.8f}" for k, v in HER_TT.items()))
    print()
    for d in ds:
        solve(d)


if __name__ == "__main__":
    main(sys.argv)
