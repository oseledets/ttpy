# The QTT Laplacian, and the wall it runs into

The simplest elliptic problem — $-u'' = 1$ on $[0,1]$ with $u(0)=u(1)=0$ —
discretized on $2^d$ interior points and solved in the QTT format by
`amen_solve`. The operator is a rank-3 tensor train at every $d$, so the grid
can be made enormous at no storage cost. What cannot be made free is the
*conditioning*: the condition number of the second-difference matrix grows
like $4^d$, and an unpreconditioned iteration in float64 eventually cannot beat
it. This example is that wall, measured — and its companion,
[`bpx_elliptic`](bpx_elliptic.md), is what tears it down.

![the conditioning wall of an unpreconditioned QTT solve](../../docs/media/amen_laplace_wall.png)

## The problem

On the $N = 2^d$ interior points with spacing $h = 1/(N+1)$ the discretization
is the standard three-point stencil, and its solution is known in closed form —
so the error reported below is the *solver's*, measured against the exact
solution of the discrete system, not against the PDE:

$$A u = h^2 f, \qquad A = \mathrm{tridiag}(-1,\ 2,\ -1), \qquad f \equiv 1, \qquad u_i = \frac{i(N+1-i)}{2}, \quad i = 1,\dots,N.$$

The eigenvalues of $A$ are $\lambda_k = 4\sin^2\big(k\pi/2(N+1)\big)$, so its
condition number is

$$\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}} = \cot^2\Big(\frac{\pi}{2(N+1)}\Big) = O(N^2) = O(4^d).$$

A backward-stable solve of $Au = h^2 f$ can be no more accurate than
$\varepsilon_{\mathrm{mach}}\kappa(A) \approx 10^{-16}\cdot 4^d$ in relative
terms — a floor that rises with $d$ and, near $d \approx 12$, climbs past any
tolerance one would actually ask for.

## The code, walked through

There is almost nothing to it: build the QTT Laplacian, build the constant
right-hand side, call the solver, and compare against the analytic train.

```python
def run(d):
    n = 2 ** d
    a = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d) * (1.0 / (n + 1) ** 2)
    x, info = amen_solve(a, rhs, None, EPS, verb=0, seed=0, return_info=True)

    i = np.arange(1, n + 1, dtype=float)
    analytic = i * (n + 1 - i) / 2.0 / (n + 1) ** 2
    got = np.asarray(x.full(asvector=True))
    err = np.linalg.norm(got - analytic) / np.linalg.norm(analytic)
```

`tt.qlaplace_dd([d])` is the Kazeev–Khoromskij explicit QTT representation of
the second-difference operator: TT rank 3, independent of $d$. The solver is
handed `return_info=True` so the run can read back the sweep count, the final
residual and whether it declared convergence — the numbers that make the wall
visible.

## What comes out

At the requested tolerance $\varepsilon = 10^{-10}$:

| $d$ | unknowns $2^d$ | sweeps | solution rank | residual | error vs $u^\star$ | converged |
|---|---|---|---|---|---|---|
| 8  | 256          | 3  | 7  | 3.5e-12 | 6.6e-13 | yes |
| 12 | 4,096        | 20 | 15 | 7.2e-10 | 3.3e-12 | no |
| 16 | 65,536       | 20 | 14 | 3.5e-07 | 1.9e-08 | no |
| 20 | 1,048,576    | 20 | 46 | 2.9e-04 | 5.3e-06 | no |
| 24 | 16,777,216   | 20 | 82 | 2.2e-01 | 1.4e-04 | no |

The operator stays rank 3 the whole way; the grid at $d=24$ has 16.8 million
points and costs no more to *store* than the grid at $d=8$. But at $d=8$ the
solve reaches machine precision in three sweeps, and past $d=12$ it cannot: the
sweep count pins at its cap, the achievable residual rises floor-first with
$\kappa(A)$, and the error against the analytic solution follows. This is the
left panel of the figure — the error curve crossing the requested tolerance
around $d \approx 12$ — and the right panel — the effort saturating the moment
the problem stops being solvable to tolerance.

The one thing the solver does *not* do is return a plausible wrong answer
quietly: past the wall it warns that it did not reach $\varepsilon$ and stops,
so the failure is legible rather than silent.

## Why believe it

The oracle is the analytic solution of the *discrete* system,
$u_i = i(N+1-i)/2$, exact to machine precision at any $d$, so the last column is
the solver's own error with no discretization ambiguity. The conditioning
statement is not folklore either: $\kappa(A) = \cot^2(\pi/2(N+1))$ is the exact
eigenvalue ratio, and $\varepsilon_{\mathrm{mach}}\kappa(A)$ predicts the
measured floor to the right order of magnitude at every row of the table.

## Run it

```bash
python examples/amen_laplace.py              # d = 8, 12, 16, 20
python examples/amen_laplace.py 8 14 20 26   # choose the grids
```

## References

* V. Kazeev, B. Khoromskij — *Low-rank explicit QTT representation of the
  Laplace operator and its inverse*, [SIAM J. Matrix Anal. Appl. 33(3):742,
  2012](https://doi.org/10.1137/100820479) — the rank-3 QTT Laplacian.
* S. Dolgov, D. Savostyanov — *Alternating minimal energy methods for linear
  systems in higher dimensions*, [SIAM J. Sci. Comput. 36(5):A2248,
  2014](https://doi.org/10.1137/140953289) — the AMEn solver.
* [`bpx_elliptic`](bpx_elliptic.md) — the multilevel preconditioner that
  removes this floor and reaches $\sim 10^{-13}$ where the plain solve stalls.
