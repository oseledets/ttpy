# Changelog

## Unreleased

### Changed
- Split the former monolithic Sample-DIRT implementation into focused density,
  contraction, fitting, transport, and refinement modules under
  ``tt.transport``.  Only research experiments, method notes, and generated
  results moved to the standalone `sample-dirt` project; the reusable kernel
  and its unit tests remain part of ttypy.

### Added
- ``tt.lobpcg_solve``: a fixed-rank SPD linear solver with one-site energy
  sweeps, central block-Jacobi PCG, and one transported recycled direction per
  core.  It performs no rank enrichment, truncation, or SVD and reports the
  horizontal projected gradient separately from the full linear residual.
- ``examples/lobpcg_fixed_rank.py`` and ``docs/LOBPCG.md``.
- ``tt.qtt_divgrad`` and ``tt.qtt_divgrad_from_faces``: conservative central
  finite differences for variable-coefficient Dirichlet problems, assembled
  from QTT face coefficients without the multilevel Kazeev representation.
- ``examples/qtt_divgrad_solvers.py`` and ``docs/QTT_FD.md``.
- ``examples/sample_dirt_banana.py`` and its gallery page: an animated,
  sample-only probit bridge fitted by orthogonal ALS, with held-out analytic
  KL/TV, sliced-Wasserstein, and exact round-trip checks.
- Sample-only TT density and inverse Rosenblatt models in ``tt.transport``:
  scalar squared, direct entrywise-nonnegative, and purified/Born linear TT
  densities, including local purification indices, with exact normalization
  and piecewise-polynomial conditionals.
- Global minibatch NLL fitting for the direct nonnegative and purified models,
  including stable normalization gauges, GPU evaluation, serialization, and
  composition as residual Sample-DIRT layers.
- End-to-end global-KL refinement through mixed scalar and locally purified
  exact Rosenblatt layers, with scalar warm starts and validation no-regression
  checkpoints.
- Nonuniform and learnable linear-hat TT densities with exact Gram
  normalization and Rosenblatt maps.  Positive interval logits, uniform and
  marginal-quantile initializations, and an exact uniform-grid validation
  fallback add adaptive one-dimensional resolution without dense grids.
- Exact block-orthogonal probit coordinate layers for scalar TT transports,
  including end-to-end TT-core refinement through chains of fixed rotations
  and stochastic tangent-skew refinement of all rotations with fixed cores.
  Their cube Jacobian is identically one, block rotations are stored compactly,
  and scalar identity checkpoints make coordinate learning no-regression.
- Exact identity probit enablement for already fitted scalar Sample-DIRT
  layers.  It preserves the complete checkpoint density and Rosenblatt map
  while exposing full or blockwise tangent-skew coordinate optimization.
- Nonlinear exact probit coordinate corrections: radial pair twists and
  conditional pair rotations with alternating target/conditioner partitions.
  They preserve standard Gaussian measure pointwise, have analytic inverses,
  need no log determinant, and can be fitted alone or end to end through a
  complete Sample-DIRT chain with every TT core frozen.
- Exact DMRG-style rank continuation for fitted scalar Sample-DIRT chains.
  One-sided random bond columns and matching zero rows preserve the complete
  density/transport at checkpoint zero while giving new Schmidt directions a
  nonzero first-order global-NLL gradient.
- Exact nested-grid continuation for fitted scalar linear Sample-DIRT layers.
  Finite-element prolongation preserves the complete density and Rosenblatt
  map while opening new univariate nodal degrees of freedom at fixed TT rank.
- Coordinate-local exact continuation for selected physical variables inside
  selected Sample-DIRT layers, including permutation-aware CLI support.
- Gauge-canonical cross-fitted mode-refinement scores.  They project complete-
  chain stochastic NLL gradients off the old nested finite-element space in
  exact hat-Gram and TT-environment metrics, and use disjoint minibatch cross
  products to avoid the positive variance bias of squared stochastic
  gradients.
- Exact hat-mass TT orthogonalization for nodal linear densities.  A Cholesky-
  weighted QR makes every suffix environment identity in the true functional
  Gram metric while preserving the represented root, replacing the former
  cellwise `I/n` gauge in linear Adam and Sample-DIRT refinement.
- Stochastic block-coordinate global-KL refinement for selected TT layers,
  with exact frozen transports and cached fixed-layer right environments.
- Exact scale-invariant Sobolev regularization for multilinear TT roots in
  global Sample-DIRT optimization.  Hat mass and stiffness matrices are
  contracted directly, while checkpoint selection remains unregularized NLL.
- Tabular method-validation indices now use a dedicated seeded RNG, so data
  augmentation and other fit-only random draws cannot change the architecture
  comparison split.
- Paired tabular model selection now has an explicit `1e-7` nat minimum
  effect size, preventing deterministic TT gauge/canonicalization roundoff
  from selecting a much larger but functionally identical checkpoint.
- Tabular Gaussian target smoothing can use multiple independent jitter
  replicas per training row, reducing Monte Carlo error in the convolved
  empirical measure without adding density parameters.
- TT-preconditioned residual continuous flows, including VP conditional flow
  matching, exact or Hutchinson divergence evaluation, minibatch-OT paths,
  exact affine latent corrections, and cross-fitted direct-NLL refinement.
- Reproducible MINIBOONE/FFJORD coordinate and checkpoint audits with exact
  matched-subset scoring, repeated Hutchinson traces, atom stratification, and
  controlled standardized-coordinate jitter.

## 2.0.0rc1 — 2026-08-08

First release of the pure-Python rewrite. Relative to ttpy 1.x this is a new
implementation with the same public API; the full compatibility contract, every
deliberate behaviour change and every fixed legacy defect are in
[docs/COMPAT.md](docs/COMPAT.md). Highlights:

### Changed
- **Pure Python.** No Fortran, f2py, `numpy.distutils`, compilers or git
  submodules; the wheel is `py3-none-any`. Required dependencies: numpy, scipy,
  einops. Optional: `[torch]` (GPU/autograd backends), `[fast]` (numba kernels
  for the AMEn inner loops; without it the numpy path gives identical results).
- **Backends.** One code base runs on numpy and torch (CUDA, and MPS in
  float32); `tt.set_backend(...)`, per-array dispatch, mixing backends inside
  one tensor is a loud error.
- **Every iterative solver reports on itself**: per-sweep histories recorded
  even at `verb=0`, non-convergence warns with the value actually reached, and
  quantities that used to be assumed (eigenresiduals, tangent defects, held-out
  cross errors) are measured. A silent wrong answer is treated as the worst
  failure mode; see [docs/NUMERICS.md](docs/NUMERICS.md) for the measured
  limits behind the defaults.
- `amen_solve` default `max_full_size` is 1000 (was 50 in 1.x; the old value
  assumed a compiled local solver).
- `write()`/`read()` use `.npz`; the tt-fort binary format is gone.

### Fixed (present in ttpy 1.x, found against dense truth)
- `Toeplitz`, `IpaS`, `qshift` returned transposed (or, for `IpaS`, simply
  wrong) matrices.
- TT-matrix `reshape` mixed row bits with column bits.
- The real branch of the Fortran KSL applied K/S steps in the wrong order in
  the backward pass, breaking the Strang palindrome.
- EXPOKIT and PRIMME are replaced (Arnoldi with adaptive substepping;
  dense `eigh` + LOBPCG) and verified against `scipy.linalg.expm` and analytic
  spectra — see the measured comparison in docs/COMPAT.md.

### Testing
- 823 tests, all against dense ground truth, closed forms or mathematical
  invariants — never against the legacy implementation.
- Adversarial `test_verify_*` suites with oracles that share no code with the
  algorithms under test, including a double-double residual oracle
  (`tests/extended.py`) that is bit-identical across platforms.
- Benchmark problems with external references only
  (`bench/bench_showcase.py`): Pfeuty's closed form, Bethe ansatz,
  Jahnke–Huisinga, Genz integrals, Khoromskij's quantics ranks, and one
  documented refusal (Anderson localization).
