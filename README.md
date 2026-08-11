# ttpy 2

The Tensor Train toolbox, rewritten in pure Python.

```bash
pip install ttpy          # or: uv pip install ttpy
```

No Fortran, no `f2py`, no `numpy.distutils`, no compiler, no git submodules.
The wheel is `py3-none-any`, 169 KB, and installs into a fresh environment in a
quarter of a second. For comparison, building ttpy 1.x on a current machine
needs six separate workarounds — they are written down in
[docs/LEGACY_BUILD.md](docs/LEGACY_BUILD.md).

```python
import tt

A = tt.qlaplace_dd([12])                    # Laplacian in QTT, 2^12 unknowns
b = tt.ones(2, 12)
x = tt.amen_solve(A, b, b, 1e-8, verb=0)    # AMEn linear solver
print((tt.matvec(A, x) - b).norm() / b.norm())
```

Three dimensions of `2^12` points each -- 6.9e10 unknowns -- is
`tt.qlaplace_dd([12, 12, 12])`, and costs the same call.

## What it is

A tensor in the TT (tensor train) format is stored as `d` cores of shape
`(r, n, r)`, which turns `n**d` numbers into `d n r**2` and makes linear algebra
in dimension 100 possible. This package implements the format and the algorithms
around it: TT-SVD and rounding, cross approximation, elementwise functions,
AMEn linear solvers and matvecs, block eigensolvers, the KSL integrator,
Riemannian tools, and completion -- plus a QTT toolkit for elliptic problems
with BPX multilevel preconditioning (`tt.algs.qtt_ell`), which is new in 2.0.

`tt.transport` also contains an experimental sample-only deep inverse
Rosenblatt transport. Its default root-free estimator stores a centered direct
TT correction; the quadratic density-ratio loss uses exact TT contractions and
only the linear term is estimated from samples. Orthogonal ALS and stochastic
Adam variants are included. The construction and the correlated Gaussian,
predator--prey and Lorenz--96 examples are documented in
[docs/SAMPLE_DIRT.md](docs/SAMPLE_DIRT.md).

## Backends

The numerics run on numpy by default and on torch (CPU or GPU) on request:

```python
tt.set_backend("torch", device="cuda", dtype="float64")
```

Both produce the same numbers, including the same truncation ranks; that is a
test, not a hope (`tests/test_backend_torch.py`).

Which one is faster depends on the operation, and not in the direction you might
expect — see [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Short version: TT
rounding is a chain of many *small* sequential factorizations, so it is bound by
LAPACK call latency, not by FLOPs. A GPU does not help there, and neither do 64
BLAS threads (they hurt). Contractions such as `dot` are GEMM-bound and the GPU
wins by an order of magnitude. For large ranks use the randomized rounding, which
turns the SVD chain into matrix multiplications:

```python
y = x.round(rmax=100, method="randomized")
```

## Elliptic problems in QTT

The condition number of a QTT-discretized elliptic operator grows like `4^d`, so
an unpreconditioned iteration stops working long before the format does.
`tt.algs.qtt_ell` implements the multilevel preconditioner of Bachmayr and
Kazeev (FoCM 20, 2020) with the ranks their theory predicts: `2^(2D+1)` for the
preconditioner and `2^(2D) + 2^(2D-1)` for the fused factors, both independent
of the number of levels.

```python
from tt.algs.qtt_ell import bpx, bpx_theta

C = bpx(d, D=1)                       # the preconditioner, TT rank 8
theta, = bpx_theta(d, D=1)            # the fused factor: B = theta^T theta
```

`-u'' = 1` with `u(0) = 0, u'(1) = 0`, AMEn at `eps = 1e-10`, `d = 30`
(2^30 unknowns), one host, interleaved runs:

| | sweeps | time | relative error |
|---|---|---|---|
| unpreconditioned | 30 | 18.4 s | **1.01** (i.e. wrong) |
| with BPX | 8 | 0.27 s | 1.9e-13 |

The catch worth knowing before you use it: the preconditioned operator must
never be *assembled* as `C A C`. Its entries cancel over `4^d`, so rounding that
product loses accuracy like `4^d * eps` -- 6.0e-04 at `d = 20`, 4.8e+14 at
`d = 50` -- and its rank grows with `d`. `bpx_theta` gives the fused factors
instead, and `B = sum_k theta_k^T theta_k` is the same matrix at rank 17, flat
in `d`. Run `examples/bpx_elliptic.py` for the whole story.

## Coming from ttpy 1.x

Your `import tt` scripts should keep working. Read
[docs/COMPAT.md](docs/COMPAT.md) for the exceptions — in particular three places
where the old package was quietly returning a transposed or wrong result, now
fixed and checked against dense ground truth.

## Development

```bash
uv venv && uv pip install -e ".[test]"
pytest -q                       # core + backend + algorithms
python bench/bench_core.py --backends numpy torch --out bench/results/core.json
```

Requirements live in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) and are the
source of truth: the code should be reproducible from them and the tests. The
measured limits behind the defaults — accuracy floors, what each knob is worth
in which regime, what a float32 or MPS backend can and cannot do — are in
[docs/NUMERICS.md](docs/NUMERICS.md), which the docstrings point at rather than
carry themselves.

## Where it stands

888 tests, all against dense ground truth or a mathematical invariant — never
against the old implementation. Measured against the Fortran ttpy on one host
(details and raw data in [docs/PERFORMANCE.md](docs/PERFORMANCE.md)): faster on
rounding (1.2-1.5x), on `dot` (2x), on `tt_svd` (11.9x) and on `amen_solve`
(2.8x, and 60x more accurate on the same problem).  The KSL integrator, once
9x slower, now runs its sweeps as compiled numba kernels (float64 and
complex128) and is within ~2-3x of the Fortran on the reference problem --
measured on contended machines, so read `docs/PERFORMANCE.md` 3b before
quoting the number.

Testing against dense truth also turned up four defects in the old package
(transposed `Toeplitz` and `qshift`, a plainly wrong `IpaS`, a broken K/S order
in the real branch of the KSL integrator) and several in this one, all fixed and
documented in [docs/COMPAT.md](docs/COMPAT.md). Four of them were *silent wrong
answers* -- a returned number with nothing to say it was meaningless -- and that
is the failure mode this package tries hardest to make impossible.

### Planned, not shipped

Designed and specified in [docs/plans/](docs/plans/), with the measurements
behind each decision, but **not implemented**: the BUG / robust rank-adaptive
integrator, a block AMEn eigensolver, the second-order/preconditioned half of
the Riemannian roadmap (geomCG, trust region, rank adaptation), and the
coefficient-dependent form of the BPX factors.

Shipped from the Riemannian roadmap: the tangent-space machinery
(`tt.algs.riemannian`: `project_delta`, `frames`, `retract`, `transport`, the
cheap tangent inner product) and `tt.rgd` -- Riemannian gradient descent whose
gradient comes from torch autodiff through the tangent parametrization
(Novikov-Rakhuba-Oseledets, SISC 2022) without ever forming the Euclidean
gradient.  Its niche is the loss ALS structurally cannot have: see
`examples/robust_completion.py`, where 2% outliers send the square-loss fit
five orders of magnitude off while log-cosh descent recovers the tensor.
[docs/plans/ROADMAP.md](docs/plans/ROADMAP.md) has the dependency graph, the
ordered milestones and 43 benchmark problems split by what is runnable today.

Shipped from that list already: `tt.dmrg_cross`, a from-scratch port of
Savostyanov's greedy DMRG cross (`ttcross`) — rank +1 per bond per sweep, rook
pivoting on the residual; at equal digits on smooth integrands it needs 3–7x
fewer function evaluations and ~10x less wall time than `rect_cross`
(medians over seeds).  Faster than the Fortran original: 8–14% per evaluation
on the numpy path, 30–42% end to end when `fun` is numba-jitted (the bond
visit then runs as one compiled kernel).  Measured parity and the race in
[docs/plans/cross-approximation.md](docs/plans/cross-approximation.md) §2.1b.

Quantum dynamics runs at paper scale: the KSL projector-splitting integrator
(compiled sweeps for float64 *and* complex128 -- a Schroedinger step
`tau = 1j h` stays on the numba path) reproduces Fig. 3 of Lubich, Oseledets
& Vandereycken, SINUM 53(2), 2015 end to end --
`examples/henon_heiles_ksl_paper.py`: the 10-D Henon-Heiles spectrum with a
sine-DVR discretization and a complex absorbing potential, 6000 steps at rank
18 in ~15 minutes on 8 CPU cores (the paper reports 4425 s for its own code
and 54354 s for MCTDH on 2015 hardware).  The pedagogical variant with an
`eigb` cross-check of every peak is `examples/henon_heiles_spectrum.py`; the
f=2 machinery is pinned against a dense `expm` propagator in
`tests/test_examples.py`.

Two more paper reproductions live in `examples/`, both propagated by
Crank-Nicolson with a warm-started `amen_solve` per step and both checked
against oracles the papers did not have: `fokker_planck_dumbbell.py` -- the
polymer dumbbell in shear flow of Dolgov-Khoromskij-Oseledets, SISC 34(6),
2012, reproducing its Table 3 viscometric functions (eta 1.03291 vs 1.03281)
with the analytic beta=0 stationary state and an independent sparse
propagator as cross-checks -- and `sir_network_cme.py` -- the SIR-epidemic
master equation on a network of Dolgov-Savostyanov, AMC 460:128290, 2024,
where the 3^N-state distribution stays in TT (rank 11 at N=32) and
rare-event tails down to ~1e-12 are one dot product with an explicit
indicator train, where SSA would need ~5e13 trajectories.  The planned
integrator that conserves probability to machine precision on such problems
is specified in [docs/plans/tamen.md](docs/plans/tamen.md).

One known limitation with a workaround: `amen_solve` takes a matrix, so using
`bpx_theta` with it means assembling `B` at rank 161 in 2D instead of applying
the rank-24 factors one at a time. The sweep algebra is linear in that rank and
dominates; teaching the solver to accept a factored operator is the next item.

## References

* I. V. Oseledets, *Tensor-train decomposition*, SIAM J. Sci. Comput. 33(5), 2011.
* S. Dolgov, D. Savostyanov, *Alternating minimal energy methods for linear
  systems in higher dimensions*, arXiv:1301.6068, arXiv:1304.1222.
* A. Mikhalev, I. Oseledets, *Rectangular maximum-volume submatrices and their
  applications*, arXiv:1502.07838.
* H. Al Daas et al., *Randomized algorithms for rounding in the Tensor-Train
  format*, SIAM J. Sci. Comput. 45(1), 2023, arXiv:2110.04393.
* C. Lubich, I. Oseledets, *A projector-splitting integrator for dynamical
  low-rank approximation*, BIT 54, 2014.

## License

MIT, as before.
