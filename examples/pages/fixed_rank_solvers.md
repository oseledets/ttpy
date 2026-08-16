# Fixed-rank and rank-adaptive TT solvers

The same SPD system can be solved two ways in the tensor-train format, and the
difference is not a tuning knob but a choice of geometry. `amen_solve` is
**rank-adaptive**: it enriches each bond with an approximate residual and
re-truncates, so the ranks grow and shrink as the solve decides it needs them.
`lobpcg_solve` is **fixed-rank**: it minimizes the energy over a *prescribed*
rank profile and never changes a bond, visiting one core at a time. This page
runs both on one variable-coefficient PDE, and then isolates, on a manufactured
problem, exactly when the fixed-rank answer is the true answer.

![the two solvers on one system, and the exactness of fixed-rank](../../docs/media/fixed_rank_solvers.png)

## The problem

A variable-coefficient diffusion on the unit square, assembled in QTT by the
conservative central stencil:

$$-\nabla\cdot(k\nabla u) = 1 \text{ in } (0,1)^2, \qquad u = 0 \text{ on } \partial(0,1)^2, \qquad k = 1 + 0.5\sin(2\pi x)\sin(2\pi y).$$

Both solvers attack the same $Au = f$. The fixed-rank one minimizes the SPD
energy over the manifold $\mathcal{M}_r$ of tensor trains of a prescribed rank:

$$E(x) = \tfrac12\langle x, Ax\rangle - \langle f, x\rangle, \qquad x \in \mathcal{M}_r.$$

Its stationarity measure is not the plain residual but the residual projected
onto the tangent space of that manifold,

$$\frac{\lVert P_{T_x\mathcal{M}_r}(Ax - f)\rVert}{\lVert f\rVert},$$

which can reach zero while $\lVert Ax - f\rVert/\lVert f\rVert$ does not — precisely
when the true solution does not fit at rank $r$. That gap is the whole subject.

## The code, walked through

One assembly, two solves:

```python
operator, assembly = tt.qtt_divgrad([bits, bits], coefficient, return_info=True)
rhs = tt.ones(2, 2 * bits)

# rank-adaptive: enriches and truncates bonds
amen_result, amen = tt.amen_solve(operator, rhs, None, tol,
                                  kickrank=8, rmax=max(64, rank), nswp=100)

# fixed-rank: energy minimization over a prescribed profile, bonds never change
fixed_result, fixed = tt.lobpcg_solve(operator, rhs, initial_guess(profile), tol,
                                      nswp=150, local_steps=24, local_prec="c")
```

`lobpcg_solve` reuses, at each return to a core, the previous search direction
transported into the new frame — $\widetilde p_k = (V_k^{\mathrm{new}})^{*} V_k^{\mathrm{old}} p_k$ —
as an exact one-dimensional coarse correction. The companion script
`lobpcg_fixed_rank.py` manufactures $f = A x_{\mathrm{exact}}$ with
$x_{\mathrm{exact}}$ already at the prescribed rank, so that the true solution
*does* fit and both measures can converge together.

## What comes out

On the div-grad system ($256^2$ grid, 65,536 unknowns, tolerance $10^{-8}$):

| solver | time | sweeps | max rank | stopping quantity | true residual |
|---|---|---|---|---|---|
| AMEn, adaptive | 0.33 s | 9 | 50 (grew to 62) | — | 1.7e-09 |
| LOBPCG, fixed  | 0.15 s | 9 | 40 (fixed)      | proj. grad 3.0e-09 | 3.9e-09 |

Both reach the tolerance in nine sweeps; the mechanism differs. AMEn's rank
climbs from 10 to 62 as it enriches, then truncates back to 50; LOBPCG's rank
sits at 40 the entire time. That is the left panel of the figure — both stopping
quantities falling from $\sim 10^{1}$ through $10^{-8}$, over a twin axis showing
one rank rising and the other flat.

The right panel is the manufactured case, where the solution genuinely has the
prescribed rank: the projected gradient decays geometrically to $9\times10^{-10}$,
and a marker shows the true residual landing at the same $10^{-9}$ floor. When
the rank is right, the fixed-rank stationary point *is* the solution.

## Why believe it

The manufactured problem is the clean oracle: $f = A x_{\mathrm{exact}}$ makes the
exact solution known and rank-$r$ representable, so both the projected gradient
and the honest residual $\lVert Ax - f\rVert/\lVert f\rVert$ must fall to zero — and
they do, to $10^{-9}$ and a relative solution error of $2\times10^{-9}$. On the
div-grad system the check is the true residual computed outside the solver
(`check_true_res=True`), so the fixed-rank stopping quantity is never taken on
faith: the page reports both, and where the true residual is larger, the honest
reading is that the prescribed rank was too small.

## Run it

```bash
python examples/qtt_divgrad_solvers.py            # AMEn vs LOBPCG on -div(k grad u)=1
python examples/qtt_divgrad_solvers.py 10 64 1e-8 # bits, fixed rank, tolerance
python examples/lobpcg_fixed_rank.py              # the manufactured exact-rank case
```

## References

* A. V. Knyazev — *Toward the optimal preconditioned eigensolver: locally
  optimal block preconditioned conjugate gradient method*, [SIAM J. Sci.
  Comput. 23(2):517, 2001](https://doi.org/10.1137/S1064827500366124) — the
  LOBPCG idea behind the fixed-rank energy minimization.
* S. Dolgov, D. Savostyanov — *Alternating minimal energy methods for linear
  systems in higher dimensions*, [SIAM J. Sci. Comput. 36(5):A2248,
  2014](https://doi.org/10.1137/140953289) — the rank-adaptive AMEn solver.
