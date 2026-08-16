# Poisson on a triangle: three glued QTT patches

$-\Delta u = 1$ with $u=0$ on the boundary of a scalene triangle — a domain no single tensor-product grid fits. The triangle is split through the side midpoints and the centroid into **three quadrilaterals**, each mapped bilinearly from the unit square, each assembled as a QTT finite-element system in z-order, and the three glued along their shared edges into one block train that a single `amen_solve` sees whole. The problem, the discretization and the reference numbers are those of [`qtt-laplace`](https://github.com/RerRayne/qtt-laplace) and its paper (L. Markeeva, I. Tsybulin, I. Oseledets, [JCP 424:109835, 2021](https://doi.org/10.1016/j.jcp.2020.109835)); the z-order machinery it runs on — `tt.zkron`, `tt.zkronv`, `tt.zmeshgrid`, `tt.zaffine` — is Markeeva's.

<img src="../../docs/media/qtt_fem_triangle.png" width="100%">

## The problem

On the triangle with vertices $(0,0)$, $(2.8, 0.3)$, $(2, e)$, find $u$ with

$$-\Delta u = 1, \qquad u|_{\partial\Omega} = 0.$$

Each of the three quads is a logically Cartesian mesh of $2^d \times 2^d$ nodes, addressed in **z-order**: the bits of the two grid indices are interleaved, so one TT mode of size 4 carries one *level* of both directions and a bond of the train separates scales rather than directions. On such a mesh the global stiffness matrix needs no assembly loop:

$$K \quad =\quad  \sum_{l_1, l_2 \in \{0,1\}^2} P_{l_1}^{\top} \mathrm{diag}(a_{l_1 l_2}) P_{l_2},$$

where $P_l$ maps an element index to the node at its corner $l$ (identity or shift per direction, rank 2 in QTT) and $a_{l_1 l_2}$ is the vector, over elements, of the local integral between those corners.

Why the triangle is the interesting case: a quadrilateral patch touching the centroid is a genuine curved-index map, so $\det J$ varies over the mesh and the sixteen coefficient fields $a_{l_1 l_2}$ are not constant — they are compressed by TT-cross.

## The code, walked through

The geometry is the qtt-laplace splitting — vertex, two side midpoints, centroid per quad:

```python
def subdomains():
    """The three quads: (vertex, midpoint, centroid, midpoint)."""
    r12, r13, r23 = 0.5 * (R1 + R2), 0.5 * (R1 + R3), 0.5 * (R2 + R3)
    rc = (R1 + R2 + R3) / 3.0
    return [np.array([R1, r12, rc, r13]),
            np.array([r12, R2, r23, rc]),
            np.array([rc, r23, R3, r13])]
```

One patch system is assembled as follows: the Jacobian entries are sampled at element centres by TT-cross (in z-order, via `zsplit_index`), combined into the transformed coefficients, and the sixteen $P_{l_1}^{\top} \mathrm{diag} P_{l_2}$ terms are summed — `placement(d)` supplies the corner operators built from `tt.zkron`:

```python
    det = field(lambda J: J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    ...
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
```

The gluing penalises the jump across a shared edge with $\lambda = 1/2$: `sew` is the trace operator of one patch side — in z-order "stay on the bottom edge" is the rank-1 statement $i_y = 0$ at every level, so the trace costs $O(d)$ — and `interface_blocks` turns two traces into the four coupling blocks:

```python
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
```

`block_system` packs the $3\times 3$ grid of TT operators into one train with a final patch mode of size 3, and the whole coupled problem is one `amen_solve` call. The system is driven by the *coupled* right-hand side while the energy pairs the solution with the original load, because $\int|\nabla u|^2 = u^{\top} f$ for the Galerkin solution:

```python
    S, load = block_system(B, F, eps=eps * 0.01)
    _, coupled = block_system(B, G, eps=eps * 0.01)

    u = amen_solve(S, coupled, coupled, eps, nswp=40, verb=0, kickrank=8)
    energy = float(T.sum(u * load))
```

## What comes out

The observable is the energy $\int|\nabla u|^2$, checked against two external columns from the qtt-laplace repository: its QTT energies (`triangle_tt_energy.txt`) and the FEniCS convergence curve (`triangle_energy.txt`), whose finest row gives $0.34039282$ at 21787 dofs, continuum $\approx 0.3404$. The run behind the picture above:

| $d$ | dofs $= 3\cdot 4^d$ | energy | vs qtt-laplace TT | vs FEniCS 0.34039282 |
|---|---|---|---|---|
| 2 | 48 | 0.34854914 | **1.7e-14** | 2.4e-02 |
| 3 | 192 | 0.34180604 | **1.5e-09** | 4.2e-03 |
| 4 | 768 | 0.34073671 | **1.6e-09** | 1.0e-03 |
| 5 | 3,072 | 0.34051294 | — (the reference column ends) | 3.5e-04 |
| 6 | 12,288 | 0.34046131 | — | 2.0e-04 |

The three published values are reproduced to the reference's own precision, and beyond that table the energy keeps approaching the continuum from above, as a Galerkin energy must. The left panel of the picture is the solved field itself, the three z-ordered patches pushed through their bilinear maps back into physical coordinates (dashed: the interior interfaces).

## Why believe it

* The reference comes from **outside the tensor world**: the qtt-laplace repository ships a FEniCS convergence curve (0.34034426 at 11253 dofs, 0.34037443 at 15882, 0.34039282 at 21787), and the QTT energies converge to it from above with the expected first-order rate in the energy.
* The overlap with **the published QTT column** is at its printing precision: 1.7e-14, 1.5e-9, 1.6e-9 at $d = 2, 3, 4$ — two independent implementations of the same algorithm agreeing digit for digit.
* The building blocks are pinned separately in `tests/test_qtt_fem.py`: `placement` against dense identity-and-shift, the assembled stiffness against symmetry and constant-killing, second-order Poisson convergence on the square, and the curved-map cross path against direct sampling.

## Run it

```bash
python examples/qtt_fem_triangle.py            # d = 2..6, ~30 s total
python examples/qtt_fem_triangle.py 7          # one level further
```

The convergence table prints per level: dofs, block-system rank, assembly and solve times, energy, distance to FEniCS and (where the reference column reaches) to its TT values.


## The solver is in the package

The reusable half of this example lives in `tt.algs.qtt_fem`, not in the
example file: `placement`, `local_entries`, `assemble`, `dirichlet_mask`,
`apply_mask`, `sew`, `interface_blocks`, and the top-level
`multipatch_system`, which turns a list of per-patch systems and a list of
shared edges into one block train:

```python
from tt.algs.qtt_fem import multipatch_system

S, coupled, load = multipatch_system(
    list(zip(A, F)),
    [(0, 1, "RIGHT", "LEFT"), (1, 2, "TOP", "BOTTOM"), (2, 0, "LEFT", "TOP")],
    d, lam=0.5, eps=eps * 0.01)
u = tt.amen_solve(S, coupled, coupled, eps)
energy = float(tt.sum(u * load))     # pair with `load`, not with `coupled`
```

So a new domain costs a geometry description and an edge list; the example
keeps only the triangle's own geometry and its reference numbers.

## References

* L. Markeeva — [`qtt-laplace`](https://github.com/RerRayne/qtt-laplace): the original benchmark, the `SolutionOnTriangle` notebook, the FEniCS and TT energy tables this page reproduces.
* L. Markeeva, I. Tsybulin, I. Oseledets — QTT solvers on complicated domains via z-order curves, *QTT-isogeometric solver in two dimensions*, [J. Comput. Phys. 424:109835, 2021](https://doi.org/10.1016/j.jcp.2020.109835) ([arXiv:1802.02839](https://arxiv.org/abs/1802.02839)): the line of work behind the patch technique; `tt.zkron` / `tt.zkronv` / `tt.zmeshgrid` / `tt.zaffine` are Markeeva's contributions to ttpy.
* S. Dolgov, D. Savostyanov — AMEn, *SIAM J. Sci. Comput.* 36(5), 2014: the solver the coupled block system goes through.
