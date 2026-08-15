# Variable-coefficient central differences in QTT

`tt.qtt_divgrad` builds the ordinary conservative nodal discretization of

\[
-\nabla\!\cdot(k(x)\nabla u)=f,\qquad u|_{\partial(0,1)^D}=0,
\]

on `N_j = 2**d_j` interior nodes.  This is separate from the
Kazeev--Bachmayr multilevel constructions in `tt.algs.qtt_ell`.

For one direction let `M = I-S` be the unscaled backward difference.  Its rows
represent the low boundary face and all interior faces, but not the high
boundary face.  If `kminus` is `k` sampled on those faces and `bplus` is the
coefficient on the high face, embedded at the last node, the directional
matrix is

\[
A_j=h_j^{-2}\left(M_j^\top\operatorname{diag}(k_j^-)
M_j+\operatorname{diag}(b_j^+)\right).
\]

Thus the full operator is a sum of symmetric positive-definite QTT products;
there is no dense grid and no normal equation.  A callable coefficient is
sampled directly at face centres by TT-cross.  The high-boundary mask is
inserted exactly because a sparse boundary slice is precisely the sort of
object that sampling-based cross should not be asked to discover.

```python
import numpy as np
import tt

def k(points):
    x, y = points[:, 0], points[:, 1]
    return 1.0 + 0.5*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

A, assembly = tt.qtt_divgrad(
    [8, 8], k,
    coefficient_eps=1e-12,
    round_eps=1e-13,
    return_info=True,
)
f = tt.ones(2, 16)
x, solve = tt.amen_solve(
    A, f, None, 1e-8, kickrank=8, rmax=64,
    check_true_res=True, return_info=True, verb=0,
)
print(assembly.operator_ranks, solve.true_res)
```

`qtt_divgrad_from_faces` is the lower-level path when the face coefficients
already exist as QTT vectors.  It accepts one full-grid left-face tensor and
one exactly masked high-face tensor per physical direction.

## Solver check

`examples/qtt_divgrad_solvers.py` compares adaptive AMEN with the fixed-rank
recycled local LOBPCG solver on `f=1`.  A three-coefficient check on a local
Apple CPU used a `256 x 256` grid, one BLAS thread, tolerance `1e-8`, fixed rank
64 and populated JIT caches:

| coefficient | operator rank | AMEN: time / sweeps / rank / residual | fixed LOBPCG: time / sweeps / residual |
|---|---:|---:|---:|
| `k=1` | 4 | 0.13 s / 8 / 39 / `1.52e-9` | 0.24 s / 7 / `1.88e-9` |
| smooth, `0.5 <= k <= 1.5` | 8 | 0.30 s / 9 / 50 / `1.73e-9` | 0.43 s / 7 / `7.94e-10` |
| four-quadrant, `k in {1,10}` | 7 | 0.69 s / 11 / 63 / `2.33e-9` | 0.77 s / 13 / `3.52e-9` |

For the smooth case, coefficient cross used 25,204 values rather than the
65,536-node grid.  Timings are hardware- and cache-dependent; the residuals
and ranks are the meaningful comparison.  With fixed rank 40 instead of 64,
the same smooth problem still reached a true residual of `3.87e-9` in nine
sweeps.

For fixed-rank LOBPCG, convergence of the projected gradient only certifies a
stationary point on the prescribed TT manifold.  If the full residual remains
larger than required, increase the fixed rank.  AMEN instead adapts the rank.

## Accuracy and positivity

The scalar path `qtt_divgrad(d, coefficient=constant)` is exact up to TT
roundoff and agrees with the scaled standard Dirichlet QTT Laplacian.  For a
callable, `coefficient_eps` controls face interpolation and `round_eps`
controls operator compression.  Every coefficient value visited by cross is
checked to be positive, but no sampling algorithm can certify positivity at
points it never evaluates.
