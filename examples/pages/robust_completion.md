# Robust tensor completion: the loss ALS cannot have

A rank-2 tensor is recovered from observations of which 2% are corrupted by outliers at 100× the tensor's scale: alternating least squares — structurally locked to the square loss — chases the corruption, while Riemannian gradient descent with a log-cosh loss (`tt.rgd`, torch autodiff through the tangent parametrization) recovers the tensor.

<img src="../../docs/media/robust_completion.png" width="100%">

## The problem

Given a rank-$r$ tensor train $T$ of shape $n^{\times d}$, observe $N$ entries at random multi-indices $i_1,\dots,i_N$ and let a small fraction of the values be corrupted:

$$v_j = T_{i_j} + \varepsilon_j, \qquad \varepsilon_j = \pm 100 \max|T| \ \text{ for } \sim 2\% \text{ of the } j\text{'s}, \quad \varepsilon_j = 0 \text{ otherwise.}$$

Both methods solve the same completion problem, differing only in the loss $\rho$:

$$\min_{\mathrm{rank} X = r}\ \sum_{j=1}^{N} \rho\big(X_{i_j} - v_j\big).$$

Alternating least squares owes its existence to one structural fact: freezing all cores but one turns a *quadratic* functional into a linear local problem. Change $\rho$ and that machinery is gone — there is no "alternating robust regression" with a closed-form core update. So ALS is stuck with $\rho(z) = z^2$, where an outlier of magnitude $100\max|T|$ enters with the *square* of its size and dominates the fit.

The robust run uses the scaled log-cosh loss with width $w = 0.1\max|T|$,

$$\rho(z) = w \log\cosh(z/w),$$

which is quadratic for $|z| \lt w$, asymptotically linear for $|z| \gt w$, and smooth everywhere — each outlier is paid only linearly. Riemannian descent does not care what $\rho$ is: `tt.rgd` needs only a differentiable functional, and the gradient comes from torch autodiff through the tangent-space parametrization of the fixed-rank manifold ([NRO22]), without ever forming the Euclidean gradient — whose TT rank would be $N$, the number of samples.

## The code, walked through

The experiment draws a random rank-$r$ target, observes 40 entries per degree of freedom, and flips a 2% subset by $\pm 100\times$ the tensor's scale. The oracle is fixed up front: 50 000 fresh random entries of the dense ground truth that neither method ever sees:

```python
def run(d=5, n=6, r=2, outlier_frac=0.02, per_dof=40, seed=0):
    rng = np.random.default_rng(seed)
    target = tt.rand([n] * d, r=r)
    dense = np.asarray(target.full())
    scale = float(np.abs(dense).max())

    dof = sum((1 if k == 0 else r) * n * (1 if k == d - 1 else r)
              for k in range(d))
    nobs = per_dof * dof
    idx = np.stack([rng.integers(0, n, nobs) for _ in range(d)], axis=1)
    vals = dense[tuple(idx.T)].copy()
    nbad = int(outlier_frac * nobs)
    bad = rng.choice(nobs, size=nbad, replace=False)
    vals[bad] += 100.0 * scale * rng.choice([-1.0, 1.0], size=nbad)

    held = np.stack([rng.integers(0, n, 50_000) for _ in range(d)], axis=1)
    truth_held = dense[tuple(held.T)]
```

The ALS baseline gets no loss argument because there is nothing to pass — the square loss is the only one for which its closed-form core updates exist. Both methods see identical `idx`/`vals`:

```python
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_als, info = ttSparseALS({"indices": idx, "values": vals},
                                  [n] * d, ttRank=r, tol=1e-12,
                                  maxnsweeps=100, verbose=False, seed=seed)
```

The robust functional is written as plain torch code over the TT cores. Note the form of log-cosh: the naive `log(cosh(z))` overflows once $|z|$ passes $\sim 710$ in float64 — and with outliers at 100× the scale divided by $w = 0.1\times$ the scale, $|z|$ reaches $\sim 1000$ on the corrupted entries — and an `inf` loss autodiffs to a NaN gradient. The identity $\log\cosh z = |z| + \log\big(1 + e^{-2|z|}\big) - \log 2$ is exact and safe at any magnitude:

```python
    x0 = tt.rand([n] * d, r=r).to("torch", "cpu", "float64")
    t = bk.backend_of(x0.cores[0]).torch
    tvals = t.as_tensor(vals)
    width = 0.1 * scale                    # quadratic below, linear above

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        z = (element(x, idx) - tvals) / width
        # log cosh in the overflow-proof form: cosh overflows past |z| ~ 710,
        # and an inf loss autodiffs to a NaN gradient
        az = t.abs(z)
        return (az + t.log1p(t.exp(-2.0 * az)) - np.log(2.0)).sum() * width
```

The optimizer call is one line. `rgd` differentiates `f` through the tangent parametrization of the rank-$r$ manifold — the gradient is computed *as a tangent vector* (rank at most $2r$), so the sample-rank Euclidean gradient never exists as an object — and takes Armijo-controlled Riemannian steps with retraction back to the manifold:

```python
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_rgd, h = rgd(f, x0, maxit=500, tol=1e-8)
```

Both results are scored the same way, on the entries nobody observed:

```python
    def held_out_error(x):
        got = np.asarray(bk.to_numpy(element(x, held))).reshape(-1)
        return float(np.linalg.norm(got - truth_held)
                     / np.linalg.norm(truth_held))
```

## What comes out

Defaults throughout: $d=5$, $n=6$, $r=2$, 2% outliers, seed 0 — which the script reports as "rank-2 target, d=5, n=6: 3840 observations (40 per dof), 76 of them outliers at 100x the scale". Held-out relative error, i.e. $\Vert\hat{X} - T\Vert / \Vert T\Vert$ over the 50 000 unobserved entries:

| method | loss | held-out rel. err. (figure) | held-out rel. err. (rerun, this machine) |
|---|---|---|---|
| `ttSparseALS` | square | 1.8e+04 | 3.25e+02 (337 ms) |
| `tt.rgd` | log-cosh | 2.5e-02 | 1.95e-02 (3341 ms, 500 iterations, `maxit`) |

The figure column is what the README plot above records; the rerun column is a fresh execution of `python examples/robust_completion.py` on one development laptop. The exact magnitude of the ALS blow-up is not stable across environments — it is fitting $100\times$-scale corruption, and where that lands depends on floating-point details of the run — but it is always orders of magnitude above 1, while the log-cosh recovery sits stably at $\sim 2\cdot 10^{-2}$. ALS also reports a final fit of `4.4e-01` on the *observed* data: it did its job on the loss it has — the loss, not the solver, is what fails.

## Why believe it

* Inside the example itself, the oracle is the dense ground truth on 50 000 entries neither method observed — not any self-reported fit.
* `tests/test_riemannian_autodiff.py::test_rgd_logcosh_completion_recovers_the_target` pins the same machinery in a regime with a known answer: log-cosh completion of an on-manifold rank-2 target, where the minimum is zero at the target by construction; the recovered tensor must match the dense truth to `1e-6`, the Armijo sequence must be monotone, and the history bookkeeping must be consistent.
* `tests/test_riemannian_autodiff.py::test_rgd_reports_a_budget_stop_honestly` checks that a budget stop is reported as `maxit` with `converged=False` — relevant here, since the run above ends exactly that way.
* The ALS baseline is pinned separately in `tests/test_ports.py` against exact recovery of clean low-rank data, so its failure on outliers here is a property of the square loss, not of the implementation.

## Run it

```bash
python examples/robust_completion.py            # d=5, 2% outliers, ~5 s
python examples/robust_completion.py 6 8 3      # d, n, rank
python examples/robust_completion.py 5 6 2 0.1  # ... and outlier fraction
```

Requires the torch backend (`pip install torch`); time measured on one development laptop, imports included.

## References

* A. Novikov, M. Rakhuba, I. Oseledets, "Automatic differentiation for Riemannian optimization on low-rank matrix and tensor-train manifolds", *SIAM J. Sci. Comput.* 44(2):A843–A869, 2022, arXiv:2103.14974.
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor trains", *SIAM J. Numer. Anal.* 53(2):917–941, 2015 — the tangent-space machinery in `tt.algs.riemannian`.
