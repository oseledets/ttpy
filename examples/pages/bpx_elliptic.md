# BPX multilevel preconditioning in QTT

[`amen_laplace`](amen_laplace.md) ends at a wall: the condition number of the
QTT Laplacian grows like $4^d$, and past a few dozen levels an unpreconditioned
solve cannot reach its tolerance in float64. A multilevel preconditioner removes
that wall — if it is *built* the right way. This example is the BPX construction
of Bachmayr & Kazeev, and its point is as much a warning as a method: the
preconditioned operator must never be assembled as the triple product $CAC$.

![the 4^d conditioning tamed, and the multiscale field it buys](../../docs/media/bpx_conditioning.png)

## The problem

The one-dimensional model is $-u'' = 1$ with a Dirichlet and a Neumann end,
whose exact solution $u(x) = x - x^2/2$ makes the error measurable; the
$D$-dimensional analogue lives on the unit cube. Discretized in QTT on $2^d$
points, the SPD system $Au = f$ has

$$\kappa(A) \sim 4^{d}, \qquad h = 2^{-d}.$$

The BPX preconditioner $C$ of [BK20] is uniformly well-conditioned in $d$:
applied symmetrically it bounds the spectrum independently of the depth,

$$\kappa(CAC) \le \kappa_0 \quad \text{independent of } d.$$

## The one idea worth taking away

$CAC$ is a mathematically bounded operator, but it must not be *formed*. Its
entries cancel across the $4^d$ spread of scales, so rounding the triple product
loses accuracy like $4^d\varepsilon$ and its TT rank grows with $d$:

$$\mathrm{round}(CAC): \quad \text{error} \sim 4^{d}\varepsilon, \quad \text{rank grows with } d.$$

[BK20] gives fused factors $\Theta_k$ (Lemma 5) so the same operator is a
flat-rank sum that is stable to assemble:

$$B = \sum_k \Theta_k^{\top}\Theta_k, \qquad \kappa(B) = \kappa(CAC) \le \kappa_0.$$

`bpx_theta(d, D)` returns those $\Theta_k$; the preconditioned matrix $B$ holds
**TT rank 17 in 1D, flat in $d$**, where a rounded $CAC$ would inflate without
bound.

## The code, walked through

The 1D driver solves the same system twice — plainly, and preconditioned — and
measures both against $u(x) = x - x^2/2$:

```python
a   = tt.qlaplace_dn(d, "DN")
rhs = (tt.ones(2, d) - 0.5 * tt.unit(2, d, j=n - 1)) * (h * h)
exact = (x - 0.5 * (x * x)).round(1e-14)

plain, i1 = amen_solve(a, rhs, tt.ones(2, d), 1e-10, nswp=30)   # no preconditioner

th = bpx_theta(d, 1)[0]                                          # the fused factors
c  = bpx(d, 1, weight=1, scaled=True)
b  = (th.T @ th).round(1e-14)                                    # B = Theta^T Theta, never CAC
w, i2 = amen_solve(b, tt.matvec(c, rhs).round(1e-12), tt.ones(2, d), 1e-10, nswp=30)
u = tt.matvec(c, w).round(1e-12)
```

$B$ is built as $\Theta^{\top}\Theta$ and never as $CAC$; the solve happens in the
preconditioned variable $w$ and is pulled back by one matvec with $C$.

## What comes out

The condition number, from dense eigenvalues (1D, preconditioner rank 8):

| $d$ | $\kappa(A)$ | $\kappa(BA)$ |
|---|---|---|
| 4  | 4.4e+02 | 5.67 |
| 6  | 6.7e+03 | 7.93 |
| 8  | 1.1e+05 | 9.52 |
| 10 | 1.7e+06 | 10.66 |

$\kappa(A)$ multiplies by four per level; $\kappa(BA)$ only creeps, log-slowly,
and the preconditioner rank stays fixed regardless of depth. That is the left
panel of the figure — a straight $4^d$ line against an almost-flat one.

The solve that buys, $-u'' = 1$ to $\varepsilon = 10^{-10}$:

| $d$ | unknowns | unpreconditioned error | BPX error | sweeps | rank $B$ |
|---|---|---|---|---|---|
| 10 | 1,024         | 1.4e-09 | 1.9e-14 | 7 | 17 |
| 18 | 262,144       | 1.3e-05 | 7.9e-14 | 8 | 17 |
| 22 | 4,194,304     | 1.4e-04 | 1.1e-13 | 8 | 17 |
| 26 | 67,108,864    | 1.7e-01 | 1.5e-13 | 8 | 17 |
| 30 | 1,073,741,824 | 1.0e+00 | 2.0e-13 | 8 | 17 |

At $d = 30$ — over a billion unknowns — the unpreconditioned answer is not merely
inaccurate but *wrong* (relative error $\approx 1$, hitting the sweep cap), while
BPX reaches $2\times10^{-13}$ in eight sweeps at flat rank 17. The right panel of
the figure is the two-dimensional field this conditioning buys: three Gaussian
peaks at widths $2^{-9}$, $2^{-6}$, $2^{-4}$ on a $512^2$ grid, the sharpest
almost a point — the multiscale resolution that makes the $4^d$ problem
unavoidable without a multilevel preconditioner.

## Why believe it

The conditioning claim is checked against dense `eigvalsh`, not asserted: the
$\kappa(A)$ and $\kappa(BA)$ columns are exact eigenvalue ratios at every $d$ the
dense computation can reach. The solve is checked against the closed-form
$u(x) = x - x^2/2$, so the error columns are the solver's own. And the
unpreconditioned column is left in on purpose — it is the control that makes the
preconditioned column mean something.

One honest limitation, as a method note: the 2D solve assembles $B$ at TT rank
161 because `amen_solve` needs a matrix, rather than applying the factors
$\Theta_k$ one at a time; the sweep cost is linear in that rank and dominates.
Teaching the solver to take a *factored* operator is the open item — the local
solves themselves need only about four GMRES iterations per block, exactly what
$\kappa(B) \approx 7$ predicts.

## Run it

```bash
python examples/bpx_elliptic.py            # all three parts, ~4 min
python examples/bpx_elliptic.py 1d         # ~40 s, up to 2^30 unknowns
python examples/bpx_elliptic.py cond       # ~15 s, dense eigenvalues
python examples/bpx_elliptic.py 2d 10      # ~2.5 min
```

## References

* M. Bachmayr, V. Kazeev — *Stability of Low-Rank Tensor Representations and
  Structured Multilevel Preconditioning for Elliptic PDEs*, [Found. Comput.
  Math. 20(5):1175, 2020](https://doi.org/10.1007/s10208-020-09446-z)
  ([arXiv:1802.09062](https://arxiv.org/abs/1802.09062)) — the BPX construction
  and the fused factors $\Theta_k$.
* V. Kazeev, B. Khoromskij — *Low-rank explicit QTT representation of the
  Laplace operator and its inverse*, [SIAM J. Matrix Anal. Appl. 33(3):742,
  2012](https://doi.org/10.1137/100820479) — the QTT Laplacian being
  preconditioned.
