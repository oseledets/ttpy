# Sample-DIRT

`tt.transport` contains a sample-only prototype of the deep inverse Rosenblatt
transport of Cui and Dolgov. Unlike density-driven DIRT, a layer never queries
an unnormalised density. It receives samples from the next bridge measure.

Let `T_k` be the current triangular map, `F_k = T_k^{-1}`, and let `mu` be the
uniform product measure on the unit cube. For `X ~ nu_{k+1}`, define

```text
V = F_k(X),       h*(u) = d Law(V) / d mu.
```

The residual density is the unique minimizer of

```text
L(h) = 1/2 ||h||^2_L2(mu) - E h(V),
L(h) - L(h*) = 1/2 ||h - h*||^2_L2(mu).
```

`SampleDIRT.fit_layer` estimates the linear expectation from the supplied
samples. The quadratic term is not sampled: it is an exact TT contraction.
The fitted positive density is

```text
h(u) = (gamma + g(u)^2) / Z,
```

where `g` is a `tt.vector`. Its inverse Rosenblatt map `S_{k+1}` is computed by
right-to-left Gram contractions and analytic inversion of cellwise conditional
CDFs. The composition is updated as

```text
T_{k+1} = T_k o S_{k+1}.
```

If the residual fit is exact, this update maps `mu` exactly to `nu_{k+1}` even
when `T_k#mu` did not exactly equal the preceding bridge. Thus it is the
sample counterpart of the exact-ratio DIRT construction.

## Current discretization

The first implementation uses equal-width, piecewise-constant cells on
`[0, 1]^d`. This makes three invariants directly testable:

1. The density normalization and L2 norm are exact TT contractions.
2. Positivity follows from `gamma + g^2`.
3. Forward and inverse Rosenblatt maps agree to floating-point precision.

The stored object is only the root TT cores and `gamma`; the rank-squared TT for
`g^2` is never stored. A layer with modes `n_i` and root ranks `r_i` stores

```text
sum_i r_{i-1} n_i r_i
```

floating-point values. Cached right Gram environments are rebuilt on load.

Piecewise-linear and Fourier functional bases are the next extension. They
will replace cell lookup and cellwise CDF inversion while preserving the
`SampleDIRT` interface and the sample loss.

## Optimizers and contractions

`fit_layer(..., optimizer=...)` supports three fixed-rank solvers:

- `"adam"` differentiates all TT cores as ordinary parameters.
- `"riemannian"` projects the autodiff gradient onto the fixed-rank TT
  tangent space and retracts with TT-SVD or the orthogonal projector-splitting
  sweep (`riemannian_retraction="psa"`).
- `"riemannian-sgd"` keeps the normalization and fourth-moment contractions
  exact but estimates the linear sample expectation from a minibatch. Its
  momentum is a tangent vector and is reprojected by vector transport after
  every retraction; the RMS accumulator is the gauge-invariant scalar tangent
  norm. With zero momentum, the redundant transport is skipped exactly.
- `"als"` performs mixed-canonical bidirectional sweeps. QR moves the
  orthogonality centre, and L-BFGS solves the resulting one-core nonlinear
  problem (`als_inner_steps` controls its budget).

The ALS problem is not linear least squares: squaring and normalizing `g`
makes it rational-quartic in the active core. Orthogonalization nevertheless
removes most repeated work. At centre `k`,

```text
integral g^2 dmu = ||G_k||_F^2 / product_i n_i,
```

while the sample interfaces and the left/right fourth-order environments for
`integral g^4 dmu` are contracted once and cached for the whole local solve.
The global Adam and Riemannian losses also contract two and four copies of the
cores directly, so the rank-squared TT `g * g` is never materialized. Repeated
sample cells are coalesced exactly before optimization.

For low-dimensional cellwise models, coalescing is already a strong full-batch
optimization: in 2D with 40 cells per coordinate there are at most 1600 unique
sample states, independent of the raw sample count. Stochastic Riemannian
optimization is therefore intended primarily for higher-dimensional fits where
almost every sample has a distinct multi-index.

## Minimal use

Training uses Torch autograd while the returned transport is a numpy-backed TT:

```python
import numpy as np
import tt

target_samples = np.random.default_rng(0).beta(2, 5, size=(10_000, 4))
model = tt.SampleDIRT(4)
history = model.fit_layer(
    target_samples,
    modes=16,
    rank=4,
    gamma=1e-4,
    epochs=400,
    optimizer="als",
    als_inner_steps=8,
)

generated = model.sample(10_000, seed=1)
reference = model.inverse(generated)
model.save("sample_dirt_checkpoint")
```

For a sequence of bridge samples, call `fit_layer` in bridge order. It maps the
new samples through the complete inverse of the current composition before
fitting the next residual.

## Paper examples

Six executable tests are provided:

```bash
python examples/sample_dirt_2d_gallery.py --png sample_dirt_2d.png
python examples/sample_dirt_optimizer_benchmark.py
python examples/sample_dirt_correlated_gaussian.py
python examples/sample_dirt_predator_prey.py
python examples/sample_dirt_lorenz96.py
python examples/sample_dirt_corner_mixture.py
```

The 2D gallery fits banana, two-moons and spiral targets from samples alone and
shows the accumulated transport after every diffusion bridge. The correlated
Gaussian is the motivating example. The latter two use
the models and parameter settings of Sections 6.1 and 6.2. A small Metropolis
chain is used only to prepare an oracle target sample: neither `SampleDIRT` nor
the TT optimizer receives the posterior log density.

The optional PNG export in the gallery requires Matplotlib; `--json` writes the
same stage samples and diagnostics without a plotting dependency.

The Lorenz example defaults to dimension ten for a quick local run. Use
`--dimension 40` for the dimension in the article. Every script exposes the
number of samples, cells, TT rank and optimizer epochs as command-line options.

The corner-mixture benchmark follows Section 5.2 of Novikov, Panov and
Oseledets (UAI 2021): 128 equal-weight Gaussians at randomly selected corners
of an eight-dimensional cube.  Since the paper does not publish the corner
seed or component variance, the script fixes both explicitly.  It also reports
nearest-corner total variation and recovered-mode coverage, which reveal
higher-order dependence that is invisible in two-dimensional projections.

## Limitations

- All coordinates must currently be scaled to `[0, 1]`.
- The square-root parametrization makes optimization non-convex even though the
  density-ratio objective is quadratic in `h`.
- Fixed TT ranks are used; rank enrichment and hold-out stopping are not yet
  implemented.
- The posterior examples are validation problems for density approximation
  from samples, not replacements for the oracle that produced those samples.
