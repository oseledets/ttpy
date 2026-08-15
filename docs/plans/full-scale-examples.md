# EXAMPLES: full-scale applications of TT/QTT, and what it would take to run them in ttpy2

This document owns the **application layer**: complete, runnable programs that
solve a problem somebody outside this repository cares about, checked against a
number somebody outside this repository published. It owns no algorithm.
`docs/plans/ROADMAP.md` remains the single owner of every cross-spec decision
and of the milestone order; every example below names the milestone it waits on
instead of proposing its own.

The distinction this document turns on, in the user's words: not benchmarks but
**full-scale examples**. `bench/bench_showcase.py` already measures *problems*
rather than *operations* — eleven of them, each with a reference that does not
come from ttpy2 (`docs/BENCHMARKS.md`). What it does not contain is a program a
chemist, an engineer or a machine-learning practitioner would recognise as their
own workflow: a molecule with a real force field, a domain with a real CAD
description, a model trained on a real dataset. Every entry below is judged by
that test.

**Sources actually read, and how.**

| what | where | state |
|---|---|---|
| Tran, Truong, Rasmussen, Alexandrov — TT-IGA | the attached PDF, extracted to `b300:/tmp/survey.txt` (17 pages, 981 lines) | **read in full**, including both cost tables and the reference list |
| tamen (Dolgov) | `github.com/dolgov/tamen`, cloned to `b300:~/work/ttpy-modern/scratch-survey/tamen` | **read**: `tamen.m` (683 lines), `amenany_sweep.m` (647), `chebdiff.m`, `test_conv.m`, `test_lyap.m`, `test_heat_adap.m`, `ttdR.m`, `README.md`, `LICENSE.md`. MATLAB; **not run** — b300 has neither MATLAB nor Octave |
| qtt-laplace (Markeeva, Tsybulin) | `github.com/RerRayne/qtt-laplace`, cloned to the same directory | **read**: `zoperations.py`, `generate_mesh.py`, `basis/permutation_matrix.py`, `basis/local_stiffness.py`, `README.md`. Python 2; **does not compile under Python 3** (M4) |
| ttvibr (Rakhuba) | `bitbucket.org/rakhuba/ttvibr`, cloned to the same directory | **read**: `CH3CN_calculation.ipynb`, `oscillator.py`, `eigsolvers.py`, the MCTDH Fortran shims. **Not run** — the notebook references a data file that is not in the repository (§1.3) |
| exp-machines (Novikov) | `github.com/Bihaqo/exp-machines`, cloned to the same directory | **read**: `src/optimizers/riemannian_sgd.py`, `src/optimizers/core_sgd.py`, `src/models/all_subsets.py`, `src/TTRegression.py`, `src/objectives/*.py`, `src/TFExpMachine.py`, the committed notebook outputs. **Not run** — its `ttpy==1.2.0` pin cannot be built on any modern Python (§1.4) |
| t3f (Novikov) | `github.com/Bihaqo/t3f` @ `8096b38` | cloned; **installed and imported** on b300 in a throwaway venv (§1.4) |
| xfac (tensor4all) | `github.com/tensor4all/xfac` @ `1803a3a`, cloned to the same directory | **read** (10 headers, ~2400 lines, named in §1.10) and **built and run** on b300 after six environmental blockers (§1.10) |
| TensorCrossInterpolation.jl | `github.com/tensor4all/TensorCrossInterpolation.jl` @ `ae4329a`, v0.9.19 | **read** (`globalpivotfinder.jl`, `globalsearch.jl`, `cachedfunction.jl`, `integration.jl`, half of `tensorci2.jl`); **never run** — no Julia on b300 |
| Larsson 2025 arXiv ancillaries | `arXiv:2504.05382`, `anc/` | **downloaded and read**: `all_energies.dat`, `avila_carrington.dat` |
| ttpy2 itself | `tt/algs/`, `tt/core/tools.py`, `bench/bench_showcase.py`, `tests/hamiltonians.py` | read; run on b300 for the six measurements of §0.2 |
| the four sibling specs and `docs/plans/ROADMAP.md` | this repository | read in full before writing |

**Ground rule for numbers.** Every number here is either cited from a named
source with its section or table, or measured on b300 by a script in
`b300:~/work/ttpy-modern/scratch-survey/` and marked **[measured]** with its
regime. Nothing was run locally (the local box has 4 GB and two cores). Anything
not measured says so, and §9 lists what was not verified at all.

---

## 0. The short version

### 0.1 What to build, in order

1. **TT-IGA Poisson on a curved 3D domain** (§3.1). The attached paper's
   pipeline: geometry-derived coefficient fields by `tt.cross`, the stiffness
   matrix as a rounded sum of rank-1 Kronecker terms, `amen_solve`, checked
   against the ring's closed-form logarithmic solution. **Buildable today**; the
   only missing ingredient is a univariate B-spline quadrature layer of roughly
   200 lines. It would be the first non-Cartesian geometry in the repository.
2. **tAMEn's convection test as a time-integration example** (§3.2), first with
   the implicit-Euler/`amen_solve` loop ttpy2 already has and the conserved mass
   as the oracle, then with the global space-time TT that makes tamen tamen.
   Arm one today; arm two is a new ~250-line module and no new mathematics.
3. **Acetonitrile CH₃CN, twelve modes, against a sub-milli-cm⁻¹ published
   table** (§3.3). The only entry that turns `tests/hamiltonians.py` from a set
   of model problems into chemistry, and the one that closes `eigenvalues.md`
   §9.5 / `ROADMAP.md` c2, which say the reference "is not in hand". It now is
   (§1.3). Ground state today; the 84-state block wants M2 and probably M7.
4. **Exponential Machines on MovieLens-100K** (§3.4). Measured here to be
   **buildable today** — `tt.riemannian.project(X, list_of_Z)` already does the
   one thing the published optimizer needs, at 2.57 ms for a 32-sample batch
   (M6) — which contradicts `riemannian-autodiff.md` §11.8 and `ROADMAP.md` c4,
   both of which class it as needing the not-yet-existing batch container.

### 0.2 The measurements made for this document

Six, all on b300, `~/work/ttpy-modern/ttpy2/.venv/bin/python` (numpy 2.4.6,
scipy 1.18.0), numpy backend, float64, single run each.

| # | what | headline result |
|---|---|---|
| M1 | TT ranks and `tt.cross` cost of the TT-IGA coefficient field `R(ξ)=J⁻¹J⁻ᵀ\|J\|` for a twisted 3D map, `n=128³`, `eps=1e-7`, 1 thread | all six ranks ≤ 2; **160 896** evaluations, **0.13 s**. The paper's NURBS ring at 133×66×66 needed 176 641 evaluations and 5.95 s |
| M2 | the same for an *analytic* ring, whose `R₀₁, R₀₂, R₁₂` vanish identically | `tt.cross` returns rank **85** on the identically-zero `R₀₁` after **4 631 168** evaluations (97 % of the run) with `converged=False`, and rank **10** with `converged=True` on the exactly-zero `R₀₂` |
| M3 | QTT convection–diffusion `−νu''+u'=1`, `d≤18`, `ν∈{1e-2,1e-4}`, central vs upwind vs `β=0` | the exact solution has QTT rank **3** at `eps=1e-10`; the computed iterate reaches rank **92**. Central differences overshoot by **1.66×** at `Pe_h=4.9`, upwind by 0.3 % |
| M4 | does `qtt-laplace` run? | **no**: `SyntaxError` at `generate_mesh.py:111` (Python-2 `print`), plus `np.int` (removed in numpy 1.24) in three files |
| M5 | can ttpy2 enter tamen's `test_lyap.m` regime (2D Poisson, `n=10000` per mode)? | **no**: dense TT-matrix cores would need **2.98 GiB** where tamen's `{d,R}` sparse cells need ~1.8 MiB. At `n=1000` ttpy2 takes **53.2 s** for 8 sweeps to `6.7e-07` |
| M6 | is the Exponential Machines Riemannian step reachable today? | **yes**: `project(X, [Z₁…Z_B])` at `d=10, n=2, r=4` costs 2.57 ms at `B=32` and 107 ms at `B=512`, and agrees with the explicitly-summed projection to 2.3e-14 |
| M7 | does `xfacpy` install on b300? | **not from PyPI** (`No matching distribution found`); it builds from source in 50.4 s at `-j32` only after six separate fixes, listed in §1.10 |
| M8 | xfac's prrLU cross on the counterexample of `cross-approximation.md` §1.3 | **5.66e-13** relative Frobenius at 76 581 distinct evaluations, seed-independent, where ttpy2 gives **3.82e-04** on 4 of 6 seeds at 88 641–162 106. But xfac's own `isDone()` returned `True` at 5.48e-01 error (§1.10) |

M1–M6 were run directly for this document; M7–M8 were run on b300 by delegated
agents of this survey, in throwaway environments, and are reported with their
own regimes in §1.4 and §1.10. Nothing in either group was re-run.

---

## 1. The sources, one entry each

### 1.1 TT-IGA — the attached paper

> Q. T. Tran, D. P. Truong, K. Ø. Rasmussen, B. S. Alexandrov, *A tensor
> train-based isogeometric solver for large-scale 3D Poisson problems*, Comput.
> Methods Appl. Mech. Engrg. **453** (2026) 118802,
> doi:10.1016/j.cma.2026.118802, CC BY-NC-ND. Theoretical Division, Los Alamos
> National Laboratory (LA-UR-25-29276).

**Code: none published.** The Data Availability statement reads, in full, "Data
will be made available on request." There is no repository link and no
supplementary archive; the implementation is stated to run on the MATLAB
TT-Toolbox (their ref. [31]) via `amen_cross` and `amen_solve`. Everything below
is from the text.

**What the method is, in ttpy2's conventions.** Isogeometric analysis puts the
same tensor-product B-spline basis on the geometry and on the solution, so the
parametric domain is always the unit cube and the physical domain enters only
through the map `x(ξ)` and its Jacobian `J(ξ) = ∂x/∂ξ` (their Eq. (25)). Pulling
the bilinear form back to the cube turns `∫_Ω ∇u·∇v dΩ` into

    Σ_{i,j=1..3} ∫ (∂N/∂ξ_i)ᵀ R_ij(ξ) (∂N/∂ξ_j) dξ,   R(ξ) = J⁻¹ J⁻ᵀ |J|

(their Eqs. (28)–(31)), and the load vector picks up `𝒢(ξ) = f(ξ)|J(ξ)|` (Eq.
(29)). The paper's whole contribution is that **the nine scalar fields `R_ij`
and the one field `𝒢` are the only things ever approximated**: they are
compressed by TT-cross to a user tolerance `ε` (Eq. (37)), the NURBS map itself
is evaluated *exactly* at the sampling points, and each rank-1 TT term of `R_ij`
then contributes a rank-1 Kronecker term to the stiffness matrix (Eq. (38)),
with the univariate factor being one of three Gram matrices —
`d(ξ)=N'ᵀN'`, `m(ξ)=NᵀN`, `c(ξ)=N'ᵀN` — chosen by whether the direction equals
`i`, `j`, both or neither (Eqs. (39)–(40)). `K = Σ_ij K_ij` is TT-rounded after
each summand (§4.2.1.2) and `amen_solve` finishes.

Translated: **`R_ij` and `𝒢` come from `tt.cross`; `K_ij` is a
`tt.matrix.from_list` of a chain of 1D Galerkin matrices; `K` is a `.round(eps)`
of a sum; the solve is `tt.amen_solve`.** Nothing in that list is missing from
ttpy2 except the univariate B-spline quadrature that produces `d`, `m`, `c`.

**Reference values it publishes.**

| problem | reference | what it pins |
|---|---|---|
| L-shape, `f = sin πx sin πy`, `u│∂Ω = 0`, `p = 1` | `u = sin(πx)sin(πy)/(2π²)` (Eq. (45)) | `L²` slope ≈ 2 |
| 3D ring, `r_in = 0.5`, `r_out = 1`, `h = 1`, `f = 0`, `u_in = 1`, `u_out = 2`, `p = 2` | `u(r) = [u_in log(r_out/r) + u_out log(r/r_in)] / log(r_out/r_in)` (Eq. (46)) | `L²` slope ≈ 3 |
| Table 1 (L-shape) | NNZ 9.09e8 (full grid) vs **8.00e4** (TT) at 513×257×257; TT-cross 4.06e5 evaluations / **49.1 s** against TT-SVD 3.05e8 / 7260 s; at 1025×513×513, 7.26e9 vs 1.59e5 and 2.20e6 evaluations / 306 s | the cost model `NNZ_TT = R_tot(b₁+b₂+b₃)` |
| Table 2 (ring) | 5.39e8 vs 8.80e4 at 261×130×130; 2.92e5 evaluations / **21.6 s** against 3.97e7 / 2540 s; at 517×258×258, 4.25e9 vs 1.75e5 and 4.48e5 / 190 s against 3.10e8 / 26100 s | the same |

The **rank tables in the two captions are the most directly checkable thing in
the paper**, because they are integers: L-shape `p=1`,
`r(R) = [[1 1],[1 1],[3 1]; [1 1],[1 1],[3 1]; [3 1],[3 1],[1 1]]`; ring `p=2`,
`[[1 1],[3 2],[3 2]; [3 2],[1 1],[3 1]; [3 2],[3 1],[1 1]]`.

**Honest counterweight.** The paper reports **no iteration counts, no residuals,
no solver tolerances and no `L²` numbers in any table** — the errors live only
in figures, which the text extraction does not carry, and §5.4's "six geometries
up to 0.5 billion DOFs" is a figure too. So its *cost* claims are reproducible
from the tables and its *accuracy* claims are reproducible only through the two
analytic solutions.

**[measured] M1 — the coefficient field is as cheap as they say, on a map we
control.** b300, numpy 2.4.6, `OMP_NUM_THREADS=1`, float64, single run;
midpoint grid `128×128×128` on `(0,1)³`; `tt.cross(eps=1e-7, r=2, kickrank=2,
nswp=12, seed=0)`; map `x = ρ(w)cos(2πv+κw)`, `y = ρ(w)sin(2πv+κw)`, `z = Hw`,
`ρ(w) = (0.5+0.5u)(1+0.3(w−½)²)`, `κ = 0.7`, `H = 1` — a twisted
hyperboloid-like domain with no rank-1 parametric direction, i.e. the class the
paper's §5.3 calls "a more stringent test":

| component | TT ranks | evaluations | s | converged |
|---|---|---|---|---|
| R₀₀ | [2, 2] | 27 392 | 0.022 | True |
| R₀₁ | [1, 1] | 25 088 | 0.020 | True |
| R₀₂ | [1, 1] | 26 240 | 0.021 | True |
| R₁₁ | [2, 2] | 27 392 | 0.022 | True |
| R₁₂ | [1, 1] | 26 368 | 0.021 | True |
| R₂₂ | [1, 1] | 28 416 | 0.023 | True |
| **six components** | | **160 896** | **0.13** | |

Accuracy of `R₀₁` against the dense field at `32³`: relative Frobenius error
**1.288e-15** at rank [1,1]. The paper's ring at 133×66×66 needed 176 641
evaluations and 5.95 s for all nine. **ttpy2's existing `tt.cross` already
reproduces the assembly cost regime of Table 2 on one core**, so the missing
piece really is only the spline layer.

**[measured] M2 — and the pipeline has a trap the paper does not mention.** Same
regime, the *untwisted* analytic ring `x = r cos 2πv, y = r sin 2πv, z = Hw`.
There the columns of `J` are orthogonal, so `R₀₁ = R₀₂ = R₁₂ ≡ 0` exactly.
`tt.cross` does not notice:

| component | max\|R_ij\| | TT ranks | evaluations | s | converged |
|---|---|---|---|---|---|
| R₀₀ | 1.254e+01 | [1, 1] | 25 088 | 0.020 | True |
| R₀₁ | **1.957e-16** | **[85, 1]** | **4 631 168** | **4.999** | **False** |
| R₀₂ | **0.0** | [10, 10] | 25 088 | 0.023 | True |
| R₁₁ | 1.585e-01 | [1, 1] | 25 216 | 0.021 | True |
| R₁₂ | **0.0** | [10, 10] | 25 088 | 0.019 | True |
| R₂₂ | 3.135e+00 | [1, 1] | 25 216 | 0.020 | True |

97 % of the evaluations and 98 % of the time go into approximating roundoff. A
TT-IGA front end must screen each `R_ij` against `‖R‖` — an *absolute* threshold
would be a second owner of the tolerance — before crossing it. This is the same
family of failure `docs/plans/cross-approximation.md` §1.3 documents from the
other side: every indicator reads ~1e-15 and the answer is meaningless. The
quieter half is `R₀₂`: rank 10 with `converged=True` on an identically zero
tensor.

### 1.2 tamen — time integration and ODEs in the TT format

> S. V. Dolgov, *Alternating minimal energy approach to ODEs and conservation
> laws in tensor product formats*, arXiv:1403.8085; published as
> doi:10.1515/cmam-2018-0023 (Comput. Methods Appl. Math.). Code:
> **`github.com/dolgov/tamen`**, **MIT** licence, © 2014 Sergey Dolgov, MATLAB,
> 2359 lines over 17 `.m` files. Read, not run.

**What it is.** Not a time-stepper wrapped around a TT solver. `tamen` solves
`dx/dt = Ax [+ y]` by making **time one more TT mode** and solving a whole
space-time block at once:

* the unknown is a `d+1`-dimensional TT whose last core carries the `n_t` time
  points (`tamen.m:11-20`); the initial state is the *last snapshot* of the
  previous block, extracted as `X{d}·X{d+1}(:,end)` (`tamen.m:249-254`);
* the time derivative is a dense `n_t × n_t` **Chebyshev spectral
  differentiation matrix** on `(0,1]` with a Dirichlet condition at `t = 0`
  (`chebdiff.m`, after Trefethen), inserted as one extra TT-matrix summand
  `As{d+1, Ra+1} = S_t` (`tamen.m:281`). `time_scheme='cn'` substitutes
  Crank–Nicolson mass and difference matrices (`tamen.m:218-230`);
* the **spatial** cores are updated by an AMEn sweep (`amenany_sweep.m`); the
  **temporal** core is then solved *directly* as a dense `(r_d n_t)²` system
  `AT = I + (S_t⁻¹ ⊗ I)·A_reduced` (`tamen.m:371-391`) — the local iteration
  count for the time mode is hard-wired to zero (`tamen.m:145-147`);
* the **time-discretization error** is estimated by re-evaluating the residual
  on a *twice as fine* Chebyshev grid through the interpolant
  `cheb2_interpolant(t_coarse, t_fine)` (`tamen.m:393-401`). If
  `tol/(time_error_damp · time_resid) < 1` the step is **rejected** and `X` is
  restored (`tamen.m:446-453`); otherwise
  `τ ← τ·min(0.5·(tol/(damp·res))^{1/n_t}, 2)` (`tamen.m:479`), the exponent
  `1/n_t` being the spectral order of the Chebyshev scheme (for CN it is `1/2`,
  `tamen.m:481`);
* **linear invariants are enforced, not hoped for.** `opts.obs` takes TT vectors
  `c_m`; their projections are appended to the solution basis at every bond
  (`amenany_sweep.m:318-323`), and after the local solve the component of the
  initial state inside `span{c_m}` is preserved exactly while only its
  orthogonal complement is rescaled to the recorded 2-norm (`tamen.m:348-367`).
  `test_conv.m` uses this to conserve `Σu` over 100 periods of a 2D convection.

**Two things `amenany_sweep.m` has that ttpy2's `amen.py` does not.**

1. `assemble_local_matrix` detects that the TT-matrix cores are **sparse** and
   assembles a sparse local system, permuting the indices so that the large
   spatial mode is senior (`amenany_sweep.m:158-176, 489`). ttpy2's
   `amen._local_matrix` always builds a dense array.
2. the `{d,R}` cell storage (`ttdR.m`): a TT-matrix is a `d × R` cell array of
   matrices **meaning their sum**, each of which may be a sparse `n × n` matrix.
   ttpy2's `tt.matrix` stores dense `(r, n, n, r)` cores. `amen_solve`'s
   docstring does accept "a list of matrices meaning their sum" — the same idea
   at the level of the *operator* — but the cores are still dense.

Everything else in `amenany_sweep.m` is recognisably ttpy2's `amen_solve`: the
same `z`-residual enrichment, the same White prediction, the same choice between
a residual-driven and a Frobenius truncation, the same right-block-Jacobi
preconditioner for the local iteration.

**[measured] M5 — what the dense-core representation costs.** b300,
`OMP_NUM_THREADS=8`, numpy, float64, single run. `A = A₁⊗I + I⊗A₁` with
`A₁ = tridiag(−1,2,−1)` built as a rank-2 ttpy2 `tt.matrix`, `b = ones`,
`amen_solve(eps=1e-6, nswp=8, kickrank=2, max_full_size=1e8)`, random rank-2
start:

| n | core storage | build s | solve s | rel. residual | rank | sweeps |
|---|---|---|---|---|---|---|
| 250 | 1.9 MiB | 0.00 | 1.72 | 5.61e-08 | 14 | 7 |
| 500 | 7.6 MiB | 0.01 | 7.41 | 4.34e-07 | 16 | 8 |
| 1000 | 30.5 MiB | 0.03 | 53.16 | 6.69e-07 | 17 | 8 |
| 2000 | 122.1 MiB | 0.12 | — | — | — | — |
| 10000 (`test_lyap.m`) | **2.98 GiB** (tamen's `{d,R}`: ~1.8 MiB) | — | — | — | — | — |

Wall time grows as `n^{2.1}` then `n^{2.8}` over that range (1.72 → 7.41 →
53.16 s for `n` doubling twice), which is the dense local solve, not the sweep
count — the sweep count is flat at 7–8. `test_lyap.m` runs `n = 10000`, and
tamen's `README.md` invites the reader to "refactor `test_lyap.m` for the
TT-Toolbox and run it on a laptop. Feel the necessity of the `{d,R}` format when
the MATLAB exhausts the memory." ttpy2 is on the wrong side of that line by
three orders of magnitude in storage alone.

**What is worth porting**, in descending order of value per line:

| item | what it buys | needs | size |
|---|---|---|---|
| the global space-time TT with a Chebyshev time core | one solve per *interval* instead of one per step; a solution object interpolatable at any `t` (`extract_snapshot.m`) | nothing new: `amen_solve` plus a 12-line `chebdiff` | M |
| the doubled-grid time residual and step rejection | the only honest `τ` controller in any TT integrator we would have; KSL and planned BUG both control *rank*, not *time* | the above | S |
| `opts.obs` — linear-invariant enrichment | exact mass conservation in the CME (`bench_showcase.bench_cme` reports `total_mass` and has no way to hold it) and in any master-equation or Fokker–Planck example | ~30 lines inside `amen.py`'s enrichment plus projection bookkeeping | S |
| sparse local systems and sparse cores | `test_lyap.m`'s regime — few modes, huge `n` — which M5 shows ttpy2 cannot enter | a sparse branch in `_local_matrix` **and** a sparse-core `tt.matrix`; this is a core representation change, not an algorithm | L |

**Where it does not help.** tamen is for linear `dx/dt = Ax` with `A` non-positive
definite (its own docstring); the nonlinear case is a function handle `A(X,t)`
re-parsed every sweep, i.e. a Picard iteration, and the paper does not analyse
it. For Schrödinger problems, KSL's structural properties — norm and energy
conservation, time reversibility (`ROADMAP.md` M5's counterweight) — are worth
more than adaptivity. And a global space-time TT buys nothing when the
solution's rank in time approaches `n_t`.

### 1.3 Vibrational spectra

**The `eigb` paper, now identified.** S. V. Dolgov, B. N. Khoromskij, I. V.
Oseledets, D. V. Savostyanov, *Computation of extreme eigenvalues in higher
dimensions using block tensor train format*, Comput. Phys. Commun. **185**(4),
1207–1216 (2014), arXiv:1306.2269, doi:10.1016/j.cpc.2013.12.017. This is
[DKOS14], the paper `ROADMAP.md` **Q1** says was not read and on which M2's
fidelity claim rests. It is still not read here — only its abstract page — so
**Q1 stays open**; what changes is that the reference is now exact.

**The vibrational paper, with code.** M. Rakhuba, I. Oseledets, *Calculating
vibrational spectra of molecules using tensor train decomposition*, J. Chem.
Phys. **145**, 124101 (2016), arXiv:1605.08422, doi:10.1063/1.4962420. Its
headline: 84 vibrational states of CH₃CN in about an hour and 100 MB. Code:
**`bitbucket.org/rakhuba/ttvibr`**, 5 commits, **no licence file**, Python 2
plus Fortran lifted from MCTDH.

`tests/hamiltonians.py::coupled_oscillator` already implements this paper's
§V.1 and cites it. What `ttvibr` has beyond that, read file by file:

* `CH3CN_calculation.ipynb` — the driver: `f=12, L=7, B=84, eps=1e-6`; harmonic
  frequencies `[3065, 2297, 1413, 920, 3149, 1487, 1061, 361]` cm⁻¹ (four
  non-degenerate plus four doubly degenerate → 12 modes); grid `N = [9]*12` with
  `N[1]=N[6]=N[7]=7` and `N[10]=N[11]=27`; pipeline `tt_harmsol` → `lobpcg` with
  an AMEn manifold preconditioner → `inverse_clustered` at `rmax=25` then 40.
* `oscillator.py` — `tt_harmsol` (Hermite-DVR Laplacian and harmonic start),
  `tt_bcoupled_pot`, `tt_hh_pot`, `eigb_to_list`.
* `eigsolvers.py` (31 KB) — `lobpcg`, `lobpcg_deflation`, `eig_block_sd`,
  `cluster`, `inverse_clustered`, `inverse_iter`, and the block helpers
  `block_matvec_left/right`, `block_bilin`, `block_qr`, `increase_rank`. This is
  a **complete LOBPCG-in-TT with deflation and clustered inverse iteration** —
  i.e. the ancestor of `eigenvalues.md` §2.2's LRRAP and of M7.
* `init1.F`, `mmlib.f`, `op1lib.f` are MCTDH sources compiled by `compile.sh`
  through f2py; `dvrweights.f` and `quadgauss/hermite_rule.f90` give the DVR
  weights and Gauss–Hermite nodes.

**Two defects in `ttvibr`, both measured by the survey and both decisive for how
we should use it.**

1. The notebook loads `ch3cn_data/ttpot.npy`, and the only file present is
   `ch3cn_data/ttpot_N15_L7.npy`. **The notebook cannot run as committed.** The
   shipped file is a Python-2 pickle of 12 TT cores with mode size **15**
   throughout, incompatible with the notebook's own `N = [9,7,…,27,27]`. The
   potential the paper actually used is not in the repository.
2. `ch3cn_data/levels.dat` column 8 is the Avila–Carrington reference (ZPE
   9837.4073, fundamentals 360.991, 723.181, 723.827, 900.662 cm⁻¹). Compared
   against Larsson's converged AC column: median |Δ| = 0.014 cm⁻¹, 90th
   percentile 0.037, **maximum 0.266 cm⁻¹, and 57 of 84 states differ by more
   than 0.01 cm⁻¹**. So `ttvibr`'s "exact" column is good to ~0.01–0.3 cm⁻¹.

**"Dolgov and Marchuk": not found, and probably not a paper.** An arXiv author
query for `Marchuk` (50 most recent, HTTP 200) plus keyword searches over
`tensor train`, `vibrational`, `Skoltech` return only N. G. Marchuk (Steklov,
Clifford algebras — "tensor" in an unrelated sense) and a Marchuk in
extragalactic astronomy. **There is no Dolgov–Marchuk paper on vibrational
spectra that this survey could find.** Two plausible intended names, both
verified to exist: **Savostyanov** (co-author of [DKOS14] above) and
**Markeeva** (§1.6, whose name is phonetically close and who *is* a
Dolgov-adjacent Skoltech TT author, though on isogeometric analysis, not
spectra). Marked **not verified** — this needs one sentence from the user.

**The reference table to actually use.** H. R. Larsson, *Benchmarking
vibrational spectra: 5000 accurate eigenstates of acetonitrile using tree tensor
network states*, J. Phys. Chem. Lett. **16**, 3991–3997 (2025),
arXiv:2504.05382. Error estimates below **0.0007 cm⁻¹**, and the paper's own
point is that earlier work underestimated its errors by up to two orders of
magnitude. Its arXiv ancillary files are the deliverable:

* `all_energies.dat` — 5002 lines, `number | CSC E, err | USC E, err | AC E,
  err` in cm⁻¹; 1000 states carry all three surfaces. **ZPE(AC) =
  9837.406703549974 cm⁻¹.**
* `avila_carrington.dat` — **the operator itself**: 311 terms, each a
  coefficient times a product of `1 | q | q² | q³ | q⁴` over the twelve modes.
  This is the sum-of-products quartic force field in normal coordinates,
  machine-readable, with no reconstruction ambiguity.
  `sarka_poirier_{CSC,USC}.dat` are two alternative surfaces in the same format.
* the kinetic operator is `−½ Σ_κ ω_κ ∂²/∂q̂_κ²` — the normal-coordinate Watson
  Hamiltonian with the Coriolis terms dropped, which is what the whole
  benchmark chain uses.

**The version trap, stated once so nobody falls into it.** The CH₃CN quartic
force field originates with Bégué, Carbonnière, Pouchan, J. Phys. Chem. A
**109**, 4611 (2005), who published only the largest coefficients of an 8-D
subset; the 12-D potential is reconstructed by symmetry and the reconstruction
is ambiguous. **At least five mutually different "original" surfaces circulate**
(Avila–Carrington 2011; Halverson–Poirier 2015; Baiardi 2017;
Sarka–Poirier CSC and USC 2021). A reimplementation that does not say which
surface it used produces numbers that cannot be compared with anything. Using
`avila_carrington.dat` as a *file* removes the ambiguity entirely.

Two further recent references with the same molecule, for cross-checking rather
than as ground truth: arXiv:2512.15875 (Sun, Milbradt, Knecht, Kumar, Mendl —
tree-tensor-network LOBPCG plus inverse iteration, all 84 states below 1 cm⁻¹,
built on PyTreeNet) and arXiv:2605.00998 (Larsson et al., thousands of states).

### 1.4 Exponential Machines and TT-in-ML

> A. Novikov, M. Trofimov, I. Oseledets, *Exponential Machines*,
> arXiv:1605.03795 (v1 2016-05-12, v3 2017-12-08), **ICLR 2017 workshop track**.
> Code: **`github.com/Bihaqo/exp-machines`**, **MIT**, © 2016 Alexander Novikov,
> Python. HEAD `84aceeb`, a merged PR that ports it to Python 3 and adds a
> Dockerfile. The README still carries the old title, *Tensor Train polynomial
> models via Riemannian optimization*.

**What the method is.** A model that scores a sample by contracting a TT weight
tensor with the rank-1 tensor of all `2^d` subsets of one-hot features:
`f(x) = ⟨W, ⊗_k (1, x_k)⟩ + b`. `W` lives on a fixed-rank TT manifold, so the
model has `O(d n r²)` parameters and expresses every interaction order at once.

**The optimizer is genuinely Riemannian and hand-derived — there is no autodiff
anywhere in that path.** Files read:

* `src/optimizers/riemannian_sgd.py::riemannian_sgd` — per minibatch: the
  Euclidean gradient's rank-structured form via `project_h`, then
  `tt.riemannian.project(w, [direction, reg*w_reg])`, then the retraction
  `(w − step·direction).round(eps=0, rmax=max(w.r))`. The step is an **exact
  line search** (`scipy.optimize.minimize_scalar`, or `fmin_bfgs` over
  `(step_w, step_b)` when an intercept is fitted) followed by **Armijo
  backtracking** (`beta=0.5, rho=0.1`). Also `increase_rank()` — rank growth by
  adding the orthogonal complement of the gradient, citing Riemannian Pursuit —
  and `build_reg_tens()`, the order-dependent regulariser `exp_reg`.
* `src/optimizers/core_sgd.py::core_sgd` — the baseline: plain SGD in the cores.
* `src/models/all_subsets.py` — `subset_tensor`, `tensorize_linear_init`,
  `gradient_wrt_cores`, `project_all_subsets` (the batched projection),
  `vectorized_tt_dot` / `_vectorized_tt_dot_jit` (numba `@jit(nopython=True)`).
* `src/TTRegression.py::TTRegression(BaseEstimator, LinearClassifierMixin)` — an
  sklearn-shaped API with `solver ∈ {'sgd', 'riemannian-sgd'}`.
* `src/objectives/{logistic,hinge,mse}.py` — hand-written loss/gradient pairs.
* `src/TFExpMachine.py` — a *separate* TensorFlow-1.x implementation used only
  for the MovieLens experiment; that one uses autodiff and is **not** Riemannian.

**Reference values, read from the committed notebook outputs.** MovieLens-100K
as binary classification, TT rank 10, `reg=0.012`, `exp_reg=1.8`, 50 epochs,
batch 256: **logistic-regression baseline test AUC 0.782123**, ExM 0.7661 at
epoch 2 rising to **0.7835** at epochs 42–48. UCI Car Evaluation (1728 rows, 21
one-hot features, binarised target, 1382/346 split) is the convergence-per-
iteration comparison, rank 4, `reg=0`, batch ∈ {−1, 100, 500}; the results are
pickled in the repo. The synthetic high-order-interaction experiment of the
paper is **not** in the repository.

**[measured] Installability.** `ttpy==1.2.0` **cannot be built** on any modern
stack. Four successive failures on b300, each a real blocker, in throwaway venvs
(the ttpy2 venv was untouched): (1) with build isolation,
`ModuleNotFoundError: numpy` — numpy is not a declared build dependency; (2)
`--no-build-isolation` with numpy 1.26.4, `ModuleNotFoundError: setuptools`;
(3) with setuptools, `ModuleNotFoundError: numpy.distutils` on Python 3.12,
where `numpy.distutils` is unavailable by design; (4) on Python 3.11 with numpy
1.26.4, `numpy/distutils/mingw32ccompiler.py → ModuleNotFoundError:
distutils.msvccompiler`; pinning `setuptools<60` with
`SETUPTOOLS_USE_DISTUTILS=stdlib` gets further and then dies on the package's own
layout, `ValueError: 'tt/cross/rectcross/rect_maxvol' is not a directory` (a
missing submodule in the sdist). This is `docs/REQUIREMENTS.md` R0's list,
reproduced from the outside — **and it is the single clearest argument for
reimplementing exp-machines on ttpy2**: the paper's code is unreachable for
anyone today, and ttpy2 is exactly the thing that unblocks it.

`t3f` (`github.com/Bihaqo/t3f` @ `8096b38`, **MIT**) *does* install on Python
3.12 with numpy 2.5.1, but `setup.py` declares only `numpy` while every module
imports TensorFlow, so a bare install raises on import; after adding
`tensorflow-cpu`, `import t3f` works with TF 2.21.0. It has an `autodiff.py`
(Riemannian autodiff) that exp-machines does not.
`riemannian-autodiff.md` §7.2's audit of t3f stands; nothing here changes it.
`tensorly` was **not checked**.

**[measured] M6 — the step is reachable with today's ttpy2.** b300,
`OMP_NUM_THREADS=1`, numpy, float64, single run. `d = 10`, `n = 2`, manifold
ranks `[1,2,4,4,4,4,4,4,4,2,1]`; `B` random rank-1 TT vectors standing in for a
minibatch's Euclidean gradient; `tt.algs.riemannian.project(X, Z_list)`:

| B | project | per sample | rank(P) | vs the explicitly summed projection |
|---|---|---|---|---|
| 1 | 1.39 ms | 1392 µs | 8 | 6.97e-16 |
| 8 | 2.57 ms | 321 µs | 8 | 1.35e-15 |
| 32 | 8.09 ms | 253 µs | 8 | 2.30e-14 |
| 128 | 27.01 ms | 211 µs | 8 | not computed (the summed tensor's rank is `B`) |
| 512 | 106.64 ms | 208 µs | 8 | not computed |

ttpy2's `project(X, Z)` already accepts a **list** and projects the sum without
ever forming it — the exact call `riemannian_sgd.py` makes. Extrapolating the
211 µs/sample plateau: a MovieLens-100K epoch at batch 256 is ~390 steps × 53 ms
≈ **21 s of projection per epoch**, so the published 50-epoch run is about
**17 minutes on one core**. That is a full-scale example that fits inside a
coffee break.

**This contradicts two existing documents** and they should be amended when the
example lands: `riemannian-autodiff.md` §11.8 and `ROADMAP.md` §5.4 c4 both
class Exponential Machines as needing "a data loader and the batch container",
i.e. S11, which is unscheduled. It needs a data loader; it does not need S11 to
be *correct*, and at MovieLens size it does not need it to be *fast enough*
either. What M1's `project_delta` and M4's `rgd` would buy is a factor, not a
possibility.

### 1.5 Convection–diffusion in QTT

The BPX machinery in `tt/algs/qtt_ell.py` and the theory behind it ([BK20],
`docs/plans/qtt-elliptic-bpx.md`) is for a **symmetric coercive** operator: BPX
comes from an `H¹` norm equivalence, `bpx` declares `spd=True`, and
`ROADMAP.md` §3 F4 forbids passing a non-SPD preconditioner to any
Rayleigh–Ritz method. Convection breaks all three:

1. **Non-symmetry.** `A = (ν/h²)L + (β/2h)(Sᵀ−S)` is not symmetric, so `κ(A)` is
   no longer the whole story (the field of values is), `amen_solve`'s
   residual-norm truncation is no longer equivalent to an energy-norm one, and
   the two-sided change of variables `C A C` that `bpx_theta` delivers no longer
   produces a symmetric problem.
2. **Boundary layers.** For `ν ≪ 1` the solution has a layer of width `ν` at the
   outflow boundary — and this is where QTT is at its best:
   `qtt-elliptic-bpx.md` §1.2 measured `exp(−x/1e-4)` at QTT rank **1**.
3. **The mesh Péclet condition.** Central differences oscillate for
   `Pe_h = βh/(2ν) > 1`. In QTT `h = 2^{-d}`, so this is a condition on `d`, and
   it is exactly why the problem belongs here: the grid that resolves the layer
   is unaffordable in any other format.

**[measured] M3 — what convection costs ttpy2 today.** b300,
`OMP_NUM_THREADS=1`, numpy, float64, single run. `−ν u'' + β u' = 1` on `(0,1)`,
`u(0)=u(1)=0`, `2^d` interior nodes, `h = 1/(2^d+1)`;
`A = (ν/h²)·qlaplace_dd([d])` plus, for `β = 1`, either `(β/2h)(Sᵀ−S)` (central)
or `(β/h)(I−S)` (upwind), with `S = IpaS(d,1) − I`;
`amen_solve(eps=1e-10, nswp=40, kickrank=4)` from a random rank-4 start;
`err_∞` is the nodal ∞-norm error against
`u(x) = x − (1−e^{(x−1)/ν})/(1−e^{−1/ν})` (`β=1`) or `x(1−x)/2ν` (`β=0`):

| d | ν | scheme | Pe_h | swp | conv | rel. residual | err_∞ | max\|u\|/max\|uₑ\| | rank | s |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1e-2 | — | 0 | 3 | True | 7.42e-11 | 1.15e-10 | 1.000 | 11 | 0.22 |
| 10 | 1e-2 | central | 4.9e-2 | 2 | True | 8.90e-12 | 1.00e+00 | 0.945 | 7 | 0.01 |
| 10 | 1e-2 | upwind | 4.9e-2 | 2 | True | 9.52e-12 | 1.00e+00 | 0.943 | 8 | 0.01 |
| 14 | 1e-2 | — | 0 | 40 | False | 1.05e-08 | 1.12e-08 | 1.000 | 16 | 0.32 |
| 14 | 1e-2 | central | 3.1e-3 | 40 | False | 7.54e-10 | 1.00e+00 | 0.944 | 26 | 0.30 |
| 14 | 1e-2 | upwind | 3.1e-3 | 40 | False | 1.17e-09 | 1.00e+00 | 0.944 | 36 | 0.32 |
| 18 | 1e-2 | — | 0 | 40 | False | 1.07e-05 | 1.15e-05 | 1.000 | 74 | 3.32 |
| 18 | 1e-2 | central | 1.9e-4 | 40 | False | 1.80e-06 | 1.00e+00 | 0.944 | 92 | 2.08 |
| 18 | 1e-2 | upwind | 1.9e-4 | 40 | False | 2.79e-07 | 1.00e+00 | 0.944 | 16 | 1.15 |
| 10 | 1e-4 | central | **4.9e+0** | 2 | True | 3.33e-13 | 1.66e+00 | **1.660** | 7 | 0.01 |
| 10 | 1e-4 | upwind | 4.9e+0 | 2 | True | 3.13e-13 | 1.00e+00 | 0.997 | 7 | 0.01 |
| 14 | 1e-4 | central | 3.1e-1 | 2 | True | 1.18e-11 | 1.00e+00 | 0.999 | 7 | 0.01 |
| 18 | 1e-4 | central | 1.9e-2 | 40 | False | 3.22e-08 | 1.00e+00 | 0.999 | 96 | 2.82 |
| 18 | 1e-4 | upwind | 1.9e-2 | 40 | False | 1.54e-08 | 1.00e+00 | 0.999 | 66 | 2.33 |

Three things come out of that table, and together they are the case for the
example in §3.5.

* **The `err_∞ = 1.00e+00` column is not a solver failure — it is a warning
  about the metric.** The relative residual is 1e-8 or better in every row:
  `amen_solve` solves the system it is given. The ∞-norm nodal error is O(1)
  because the *discrete* solution differs from the *continuous* one by O(1) at
  the handful of nodes inside the layer. A convection example must therefore
  assert in a norm the scheme converges in — `L²` or `H¹`, or the error at fixed
  *physical* points — which is what [BK20] and the space-time collocation papers
  both do. Asserting the ∞ norm is a mistake this measurement caught before any
  code was written.
* **`Pe_h = 4.9` at `d = 10, ν = 1e-4` gives an overshoot of `1.66×`**, upwind
  at the same point 0.997. That contrast is a reference value in the strict
  sense: it comes from the mesh Péclet condition, not from ttpy2.
* **The rank story is both the payoff and the problem.** The exact solution has
  QTT rank **3** at `eps = 1e-6` *and* at `eps = 1e-10`, for both `ν = 1e-2` and
  `ν = 1e-4` (measured by `tt.vector(u_exact, eps=…)` at `d = 18`). The computed
  iterate reaches 92 (central) and 66 (upwind), and neither converges in 40
  sweeps at `d = 18`. That gap is the representation ill-conditioning of
  `qtt-elliptic-bpx.md` §2.1 — `4^d·eps` — and it says a QTT
  convection–diffusion example is a **preconditioner** example, like the elliptic
  one, and that the preconditioner it needs is not the one we have.

**Literature, from the attached paper's own bibliography.** Adak, Danis, Truong,
Rasmussen, Alexandrov, *Tensor network space-time spectral collocation method
for solving the nonlinear convection diffusion equation*, J. Sci. Comput.
**103**(2), 46 (2025) [their ref. 16]; and Dibyendu et al., *Space-time finite
element tensor network approach for the time-dependent
convection–diffusion–reaction equation with variable coefficients*, Mathematics
**13**(14), 2277 (2025) [their ref. 32]. Both replace the preconditioner problem
with a different discretization (global space-time spectral collocation) rather
than solving it. **Neither was read**; they are named because they are the
literature the user asked for and because they point the same way as tamen.

### 1.6 Markeeva — QTT isogeometric analysis in two dimensions

> L. Markeeva, I. Tsybulin, I. Oseledets, *QTT-isogeometric solver in two
> dimensions*, J. Comput. Phys. **424** (2021) 109835, arXiv:1802.02839.
> Companion: L. Markeeva, I. Oseledets, *Building Z-permuted matrices in the QTT
> format*, Comput. Math. Math. Phys. **60** (2020),
> doi:10.1134/S096554252012009X (Russian version in Trudy MFTI 12(3), 44–56).
> Code: **`github.com/RerRayne/qtt-laplace`**, **no licence file**, Python 2,
> 989 lines, dormant since 2018-05-12.

This is almost certainly what "Markeev PhD" refers to. **The dissertation itself
was not found**: no Skoltech defence page (the obvious URL 404s), nothing in
`search.rsl.ru`, dissercat or `site:mipt.ru`, no Google Scholar thesis entry.
Marked **not verified**. Independent confirmation that this is the right author:
the attached TT-IGA paper cites that work as its ref. [17], "Markeeva et al. further
demonstrated high efficiency in 2D using the QTT format on uniform parametric
grids."

**What the method is.** A polygonal 2D domain is cut into quadrangles; each
quadrangle carries an isogeometric bilinear map with Jacobian `J(ξ)`; the
stiffness and mass integrands are `J^A J^{Aᵀ}/det J` and `det J` evaluated per
element (`generate_mesh.py::get_space_transorm_jacobians`, which builds
`TJ11 = (j22² + j12²)/det J`, `TJ22 = (j11² + j21²)/det J`,
`TJ12 = −(j22 j21 + j12 j11)/det J`, with `1/det J` taken by `tt.multifuncrs`);
the four-by-four element-local blocks are assembled and mapped into the global
matrix by permutation matrices. The one idea that makes it work is **z-order**:
the global row and column numbering follows a Morton curve, so a matrix that is
a Kronecker product or sum in the `(x, y)` numbering stays low-rank in QTT.

**The primitive is already in ttpy2. The application is not.** `tt/core/tools.py`
lines 122–155 define `zkron`, `zkronv`, `zmeshgrid` and `zaffine`; they are
line-for-line the same functions as `qtt_laplace/zoperations.py`, having arrived
through ttpy 1.x and been carried into `docs/REQUIREMENTS.md` R2's compatibility
list. What is *not* in ttpy2, and is the whole of the application:

* `basis/permutation_matrix.py` — `getW0`/`getW1`, two QTT matrices of rank 2
  built from three fixed `2×2×2×r` cores each, and `P[lx,ly] = zkron(W[lx],
  W[ly])`, the four element-corner-to-global maps. 110 lines, entirely explicit,
  and the single most portable thing in the repository.
* `basis/integrate_basis.py` (118 lines) — Gauss–Legendre element integrals of
  `∇φ_i · adj(J) ∇φ_j / det J` and `φ_i φ_j det J`.
* `generate_mesh.py::assemble_on_quad` — the 16-term loop over corner pairs,
  building `A += P[lx,ly]ᵀ diag(K_l) P[lxs,lys]` and the load vector, rounding
  at every step.
* `generate_interfaces.py`, `basis/boundary_condition.py`,
  `basis/quad_mesh_container.py` — sewing across quadrangle interfaces and
  imposing boundary conditions.

**[measured] M4 — it does not run.** b300, `python 3.12` from the ttpy2 venv,
`python -m compileall qtt_laplace`: `SyntaxError` at `generate_mesh.py:111`
(`print` as a statement), and `np.int` — removed in numpy 1.24 — in
`basis/local_mass.py:22`, `basis/local_stiffness.py:39`,
`basis/plot_z_matrix.py:15`. It also imports ttpy 1.x, which §1.4 measured to be
unbuildable. **Read, ported by hand, never executed.** With no licence file the
code cannot be vendored in any case; it can only be used as a specification.

### 1.7 JMLR 25(248) — tensor trains for BSDEs and parabolic PDEs

> L. Richter, L. Sallandt, N. Nüsken, *From continuous-time formulations to
> discretization schemes: tensor trains and robust regression for BSDEs and
> parabolic PDEs*, J. Mach. Learn. Res. **25**(248), 1–40 (2024),
> arXiv:2307.15496. Code: **`github.com/lorenzrichter/PDE-backward-solver`**,
> Python, **no licence file** (`license: null`, no `LICENSE`) — all rights
> reserved, so it may be read but not vendored.

**What the method is.** A parabolic PDE is rewritten as a backward stochastic
differential equation; the value function is represented in TT over a polynomial
basis (`TT_experiments/orth_pol.py`, `valuefunction_TT.py`) and fitted by
**regression on sampled trajectories** — the TT cores are solved for by least
squares against Monte-Carlo samples of the backward process, so
dimensionality is fought by latent low-rank structure rather than by a grid.
`pol_it.py` is a policy-iteration outer loop; the paper's contribution is a
family of discretizations (explicit versus implicit/iterative, plus "robust"
regression variants) derived from a continuous-time viewpoint, with an
accuracy/cost analysis and a comparison against neural BSDE solvers
(`NN_experiments/`).

**Dependency risk.** The TT path needs **`xerus`, SALSA branch** — a C++ library
with Python bindings that supplies the rank-adaptive ALS regression engine. That
is a compiled dependency of exactly the kind `docs/REQUIREMENTS.md` R1 exists to
avoid, and it means the repository cannot be run as an oracle without building
it. **Not built, not run.**

**Relevance to ttpy2.** The regression-onto-a-TT-manifold-from-samples problem
is the same problem `tt/algs/completion.py` solves in the sparse-observation
case, and the same one Exponential Machines (§1.4) solves with a Riemannian
step. Whether ttpy2 should acquire a `tt.algs.regress` is a real question and it
is **not** in `ROADMAP.md`. Listed as (iii) research below.

### 1.8 Mathematics 12(20):3277 — TetraFEM

> E. Kornev, S. Dolgov, M. Perelshtein, A. Melnikov, *TetraFEM: numerical
> solution of partial differential equations using tensor train finite element
> method*, Mathematics **12**(20), 3277 (18 Oct 2024),
> doi:10.3390/math12203277, CC BY 4.0. Terra Quantum AG; Dolgov also University
> of Bath.

Independently confirmed as the intended source: it is **ref. [19] of the
attached TT-IGA paper**.

**Reachability, stated plainly.** `www.mdpi.com` returns **HTTP 403** to this
survey's fetcher on the article page, the PDF and every mirror tried. The
metadata above comes from CrossRef; the only verbatim sentence obtained is the
abstract's first: *"In this paper, we present a methodology for the numerical
solving of partial differential equations in 2D geometries with piecewise smooth
boundaries via finite element method (FEM) using a Quantized Tensor Train (QTT)
format."* **The rest of the abstract, the whole body, and the Data Availability
statement are not verified.**

**What the method appears to be**, from that sentence plus the predecessor
arXiv:2305.10784 (Kornev, Dolgov, Pinto, Pflitsch, Perelshtein, Melnikov,
*Numerical solution of the incompressible Navier–Stokes equations for chemical
mixers via quantum-inspired Tensor Train Finite Element Method*): 2D FEM on
curvilinear domains where a structured grid is mapped through a curvilinear
transform, so that mass and stiffness matrices and right-hand sides are
assembled **directly in QTT** with each spatial index binarised into ~log n
virtual modes; the systems are then solved by AMEn/DMRG-type alternating
solvers, giving cost roughly logarithmic in the number of nodes. Demonstrated on
Poisson/heat problems and on nonlinear incompressible Navier–Stokes.

**Code: not found.** A GitHub code search for `tetrafem` returns zero
repositories; the `terra-quantum-public` organisation has nine repositories and
none is a FEM or PDE solver. TetraFEM appears commercially as a product. Treat
as **not public**, with the caveat that the paper's own Data Availability
statement could not be read.

This is the closest published relative of §1.6 (Markeeva) and of §3.6, and it is
where a ttpy2 QTT-FEM example would be compared — if the paper can be obtained.

### 1.9 arXiv:2510.13386 — functional tensor train neural network

> Y. Feng, M. K. Ng, K. Tang, Z. Zhang, *Functional tensor train neural network
> for solving high-dimensional PDEs*, arXiv:2510.13386, 15 Oct 2025, math.NA.
> No journal reference. **Code: none** — no repository, no data-availability
> statement. Read from the arXiv HTML full text.

**What the method is.** The solution is written in *functional* TT format,
`u(x₁,…,x_d) = Σ_α u₁^{α₀α₁}(x₁;θ₁)···u_d^{α_{d−1}α_d}(x_d;θ_d)`, where **each
core function is a small one-variable neural network**. There is no grid, so
irregular domains are admissible. An FTT-rank `(1,r,…,r,1)` is an input
hyperparameter. Training minimises a physics-informed residual
`J(θ) = ∫_Ω(𝓛u−f)² + β∫_{∂Ω}(𝓑u−g)²`; the algorithmic point is that the
separable structure **factorises the `d`-dimensional integral into products of
1-D Gauss–Legendre quadratures**, replacing Monte-Carlo collocation and removing
its statistical error. Homogeneous Dirichlet conditions are hard-wired as
`u_i = (x_i−a_i)(b_i−x_i)û_i`. Optimisation is Adam (≤5000 epochs, lr 3e-3) then
L-BFGS (≤1000, lr 0.1). Experiments: singular-function approximation (`d=4,6`),
3-D Poisson on an L-shape, Poisson at `d=3,5,7`, high-dimensional Helmholtz, and
Schrödinger eigenvalue problems; the baseline is a PINN.

**Verdict for ttpy2, stated so it is not re-litigated.** This is **not a TT
linear-algebra method** — there is no rounding, no cross, no ALS sweep, and the
TT is a parametrisation for a neural optimiser. It does not belong in `tt/algs/`
and it would not exercise anything ttpy2 owns. What *is* worth taking is one
sentence of it: the quadrature factorisation, which is exactly the mechanism
that makes the TT-IGA load vector of §1.1 cheap, and which ttpy2 has no named
utility for. Everything else is out of scope, for the same reason
`ROADMAP.md` §7 excludes the Tucker format.

### 1.10 TCI — the tensor cross interpolation library family

`docs/plans/cross-approximation.md` already surveyed `ttcross` (Savostyanov,
Fortran, read not built) and `teneva` (Chertkov, Python, read and run) against
`tt/algs/cross.py`, and decided what to port (P1 the cache, P2 `kickrank2`, P3
the false-alarm warning, P4/P5 ttcross's indicators) and what not to. **This
entry says only what TCI adds beyond that document.** Its conclusions are not
repeated.

> J. Núñez Fernández, M. K. Ritter, M. Jeannin, J.-W. Li, T. Kloss, T. Louvet,
> S. Terasaki, O. Parcollet, J. von Delft, H. Shinaoka, X. Waintal, *Learning
> tensor networks with tensor cross interpolation: new algorithms and
> libraries*, arXiv:2407.02454, **SciPost Phys. 18, 104 (2025)**,
> doi:10.21468/SciPostPhys.18.3.104.
> M. K. Ritter, J. Núñez Fernández, M. Wallerberger, J. von Delft, H. Shinaoka,
> X. Waintal, *Quantics tensor cross interpolation for high-resolution,
> parsimonious representations of multivariate functions in physics and
> beyond*, **Phys. Rev. Lett. 132, 056501 (2024)**, arXiv:2303.11819.
> J. Núñez Fernández, M. Jeannin, P. T. Dumitrescu, T. Kloss, J. Kaye,
> O. Parcollet, X. Waintal, *Learning Feynman diagrams with tensor trains*,
> **Phys. Rev. X 12, 041018 (2022)**.

**Code, both read.** `github.com/tensor4all/xfac` @ `1803a3a`, **BSD-2-Clause**,
C++17 + CMake, python module `xfacpy` via pybind11 — headers read in full:
`matrix/mat_decomp.h` (432), `matrix/pivot_finder.h` (115),
`tensor/tensor_ci.h` (393), `tensor/tensor_ci_2.h` (451),
`tensor/tensor_train.h` (326), `tensor/tensor_function.h` (86),
`tensor/auto_mpo.h` (173), `grid.h` (133), `python/python_bindings.cpp` (301).
`github.com/tensor4all/TensorCrossInterpolation.jl` @ `ae4329a`, v0.9.19,
**MIT**, Julia, 5248 lines — `globalpivotfinder.jl`, `globalsearch.jl`,
`cachedfunction.jl`, `integration.jl` and the relevant halves of `tensorci2.jl`
read; the rest indexed by grep. Julia was **not installed and TCI.jl was never
run.**

**[measured] xfacpy builds and works on b300, but nothing about that was easy.**
Regime: b300, shared 256-core box, single attempt each. `xfacpy` is **not on
PyPI** (`ERROR: No matching distribution found for xfacpy`). `python3 -m venv`
itself fails on b300 (`ensurepip is not available`). `cmake` is not installed
(installed 4.4.2 via pip into a throwaway venv). cmake 4.4.2 then **rejects the
bundled armadillo**: `Compatibility with CMake < 3.5 has been removed` — needs
`-D CMAKE_POLICY_VERSION_MINIMUM=3.5`. The system Python 3.12 has no development
headers, so both pybind11 and carma fail their own `find_package(Python…)`;
pointing both at the pre-existing micromamba env `~/micromamba/envs/ttlegacy`
(Python 3.11.15, numpy 1.24.4) fixes it. Armadillo then builds **without
LAPACK** and `xfacpy` imports but dies at first use with
`RuntimeError: solve(): use of LAPACK must be enabled`; explicit
`-D BLAS_LIBRARIES=…/libopenblas.so -D LAPACK_LIBRARIES=…/liblapack.so` fixes
that. Build: **50.4 s at `-j32`**. Six blockers, all environmental, none of them
the library's fault, and every one of them an argument for
`docs/REQUIREMENTS.md` R1.

**What TCI adds beyond `cross-approximation.md`.** Eight things, of which the
first three matter to us.

1. **prrLU replaces "orthonormal basis + maxvol" entirely.**
   `RRLUDecomp::calculate` (`mat_decomp.h:122-162`) is Gaussian elimination with
   *complete pivoting* run in place on the two-site block. The pivot criterion is
   `argmax|Schur complement|` — ttcross's residual criterion, obtained free from
   the elimination instead of from a bordered inverse. Three consequences ttpy2
   has no analogue of: `pivotErrors()` returns `|diag(U)|` per rank and
   `max|A(npivot:,npivot:)|`, i.e. **a per-bond residual bound at zero extra
   function evaluations** — which is the indicator `cross-approximation.md` P4
   wants, structurally rather than bolted on; rank selection is the elimination's
   own stopping test (`|A(i0,j0)| < reltol·max_error`), so **rank is error-driven
   per bond and `kickrank`/`rf` become unnecessary**; and `P⁻¹` is never formed —
   interpolation is triangular solves on the stored LU, against our
   `Q @ pinv(Q[ind])`. That is what the paper's abstract means by "more stable".
2. **Accumulative pivot update — the ingredient §1.3 of the cross document is
   missing and never names.** `TensorCI2` keeps `I0[b+1]`, `J0[b]`: the *history
   of every pivot ever accepted* at that bond, unioned into the block's index
   sets at every visit (`tensor_ci_2.h:228-229, 240-241`; Julia
   `tci.Iset_history` → `extraIset`, `tensorci2.jl:526-527`, switched by
   `strictlynested=false`). The failure mode of `cross-approximation.md` §1.3 is
   a fixed point in which the index sets stop moving and interpolate themselves
   exactly; the accumulative update is built precisely so the algorithm can leave
   such a fixed point.
3. **Global pivots, which correct one of that document's own negative results.**
   `DefaultGlobalPivotFinder` (`globalpivotfinder.jl:143-195`) draws `nsearch=5`
   uniform random multi-indices and from each sweeps coordinate-wise keeping the
   argmax of `|f(x) − tt(x)|`, keeping winners whose error exceeds
   `abstol × 10`; `optimize!` runs it **after every 2-site sweep** and
   `convergencecriterion` declares convergence only when the last three sweeps
   *all* had error below tolerance, *all* found **zero** global pivots, and the
   rank stopped growing — ttcross's three-strike rule with the strikes on an
   independently sampled error rather than on the change between iterates.
   `cross-approximation.md` §5 prototyped exactly this alternating-maximisation
   probe **post hoc** and rejected it (0/50 detections at 120 evaluations against
   2/50 for uniform Monte Carlo). That rejection is correct for the post-hoc use
   and **does not transfer to the in-loop use**, where the residual field is
   still large and smooth. The document should record the distinction; as it
   stands §5 reads as a rejection of the whole idea.
4. `fullPiv` / `pivotsearch`: full search materialises the whole `(rn)×(nr)`
   block; **rook search** (`ARRLUDecomp`, `mat_decomp.h:269-327`) alternates
   prrLU on a row slice and a column slice, and each outer iteration first
   appends `|I0|` **uniformly random** unselected rows and columns
   (`take_n_random`). That is `cross-approximation.md`'s P2 (`kickrank2`) — but
   as *candidate-set* enrichment inside the pivot search, so it costs nothing in
   the final rank, where P2 inflates the rank and relies on the final rounding.
5. **Environment mode / weighted TCI** (`TensorCI1Param.weight`): the pivot
   search maximises `|residual| · weightRow[i] · weightCol[j]` with the weights
   the running left/right contractions of the weighted TT
   (`tensor_ci.h:244-253`, `pivot_finder.h:67-72`), so that when the target is
   `Σ_x w(x)f(x)` the error minimised is the error *in the integral*, not the
   entrywise max. TCI1 only — `TensorCI2Param.weight` carries a literal `TODO`.
   Adjacent: `cond`, a pivot mask threaded into the argmax, which is what makes
   constrained sums (fixed particle number) expressible.
6. **Caches, both better than teneva's** — and teneva's is what
   `cross-approximation.md` P1 proposes porting. xfac's `TensorFunction` collects
   the *misses* of a block request and evaluates them under
   `#pragma omp parallel for`, so the cache is also the thing that turns a block
   into a parallel batch of distinct points. Julia's `CachedFunction` keys on a
   **packed mixed-radix integer** with an **explicit overflow check at
   construction** and a loud failure when `UInt128` would not suffice — a
   pattern worth copying whatever key ttpy2 chooses. Also `BatchEvaluator`: the
   black box may return a whole `|I| × n^M × |J|` block in one call, which is the
   shape `cross.py::_evaluate` already produces and which neither ttcross nor
   teneva expose.
7. **Operations on the trains**: `compressSVD` / `compressLU` / `compressCI` on
   one object (we have only `round`, i.e. SVD); `sum(vector<TensorTrain>)` as a
   **binary-tree pairwise sum compressed after every merge**, defaulting to
   `compressCI` rather than SVD; `autompo::ProdOp`/`PolyOp` turning a sum of
   product operators into a TT in batches of 20 with `maxNTerm=100000` — which is
   *precisely* the CH₃CN construction of §1.3/P6, already solved; `contract` as a
   lazy `BatchEvaluator` so that MPO×MPS is done by cross rather than by zip-up;
   and `TensorTreeCI`, cross interpolation on a **tree**, which is outside both
   ttcross and teneva and outside ttpy2's format.
8. **Quantics helpers**: `grid::Quantics{a,b,nBit,dim,fused}` (`grid.h:62-128`)
   with `fused` packing the `dim` bits of one scale into a single index of size
   `2^dim`, against interleaved scale-major/variable-minor; `QTensorCI`,
   `QTensorTrain` with `integral() = tt.sum1()·grid.deltaVolume`, and 15-point
   Gauss–Kronrod nodes. **Which ordering is better is not verified** — xfac
   defaults to `fused=false`, QuanticsGrids.jl documents both with no preference,
   and no measured comparison was obtained.

**[measured] The decisive number: prrLU on `cross-approximation.md`'s own
counterexample.** `f(i) = 1/(10⁻² + |Σ_k i_k/9 − 5/2|)`, `d = 6`, `n = 10`, dense
oracle over all 10⁶ entries; xfac at `reltol=1e-10`, `bondDim=200`,
`useCachedFunction=true` (so `nEval` counts *distinct* multi-indices, comparable
to that document's `uniq` column); Python 3.11.15, numpy 1.24.4, float64, single
run, shared box:

| engine | mode | rel. Frobenius | rel. ∞ | distinct evals | max rank |
|---|---|---|---|---|---|
| xfac `TensorCI2` | `fullPiv=True` | **5.66e-13** | 3.20e-11 | 76 581 | 27 |
| xfac `TensorCI2` | rook, `nRookIter=3` | 8.19e-10 | 3.23e-08 | 47 998 | 26 |
| xfac `TensorCI1` | `fullPiv=True` | 5.66e-13 | 3.20e-11 | 93 901 | ~30 |
| xfac `TensorCI1` | rook | 8.19e-10 | 3.23e-08 | 23 805 | ~30 |
| **ttpy2** (`cross-approximation.md` §4.4, `eps=1e-8`) | — | **3.82e-04 on 4/6 seeds** | 9.24e-03 | 88 641–162 106 | 24–26 |
| teneva + cache (same source) | — | 3.82e-04 on 1/6 seeds | — | 91 943–125 238 | 26 |

Full-block prrLU reaches **5.66e-13 where ttpy2 reaches 3.82e-04**, at a
comparable or lower number of distinct evaluations, with **no randomness and no
seed dependence**, while being asked for a *tighter* tolerance. The worry in
`cross-approximation.md` §3.2 that a two-site block costs `n²r²` against our
`n r²` did not materialise: the shared index sets and the cache absorb it. That
makes **two-site prrLU with full search a stronger candidate for the §1.3 cure
than P2**, which buys the same correctness at 1.4–2.5× the evaluations and still
depends on a random draw.

**Counterweight, also measured, and it matters.** Driving
`TensorCI2::iterate(1)` in a loop, xfac's own `isDone()` returned `True` at
sweep 2 with a relative Frobenius error of **5.48e-01** (rook) and **5.69e-02**
(full), with `pivotError[-1]` reading 8.88e-16 at the first of those.
`isDone()` (`tensor_ci_2.h:210-222`) only asks whether every bond is
rank-saturated or has a small last pivot error — the *same* class of
self-referential indicator that `cross-approximation.md` §1.3 indicts.
`xfacpy` exposes **no high-level driver**: the safeguard (the global-pivot loop
and the three-strike criterion) exists only in TCI.jl's `optimize!`. So "TCI2 is
accurate here" must not be read as "TCI2 knows it is accurate", and a port that
took the prrLU and left the global-pivot loop behind would reproduce our own
defect with better constants.

---

## 2. What is portable, ranked

Ranked by value per line of new code, with what it depends on named. "Blocked"
means a ttpy2 object that does not exist today; **verified by reading**:
`tt/algs/` contains no `eig.py`, no `bug.py`, no `autodiff.py`, no `rieopt.py`,
and `tt/algs/riemannian.py` has no `frames` and no `project_delta`. What *does*
exist and is newer than `ROADMAP.md` §4's description of it: `tools.qdiff`,
`tools.qtri_ones`, `tools.qlaplace_dn`, `tools.level_major_order`,
`qtt_ell.bpx`, `qtt_ell.bpx_theta`, `qtt_ell.bpx_operator`,
`qtt_ell.prolongation`, `qtt_ell.stiffness`, `qtt_ell.solve_direct_1d` — i.e.
M3's K1–K3, A7 and A9 exist, and so do M6's K4 (half: `stiffness`, no
`load_vector`) and A8. What does not exist is A10, `qtt_ell.solve`, and there is
no `tests/elliptic.py`.

| # | item | source | what it buys | needs | not-yet-built dependency | rough size | where it does not help |
|---|---|---|---|---|---|---|---|
| P1 | univariate B-spline basis, derivatives and the three Gram matrices `d, m, c` on a Gauss grid | §1.1 Eqs. (2)–(4), (40) | the only missing ingredient of a full TT-IGA assembly; M1 shows everything else is already fast enough | numpy only | **none** | ~200 lines + tests | nothing else in ttpy2 uses splines; it is a leaf module |
| P2 | `iga.stiffness_tt(R_list, bases)` — the rank-1 Kronecker assembly of Eq. (38) and its `.round` | §1.1 §4.2.1 | turns the cross output into a `tt.matrix`; the operation is `tt.matrix.from_list` in a loop | P1, `tt.cross` | none | ~80 lines | when `R` has high rank the sum's rank is `Σ r₁r₂` before rounding; the paper never reports what that is |
| P3 | a magnitude screen on cross inputs | M2 | 97 % of an assembly's cost, and one silent rank-10 answer for the zero tensor | `tt.cross` | none | ~15 lines, plus a decision about *whose* norm | when all components are genuinely nonzero it is pure overhead |
| P4 | `chebdiff` + a global space-time TT step (`tamen`) | §1.2 `tamen.m`, `chebdiff.m` | a time integrator with a real error estimate and step rejection, which ttpy2 has nowhere | `amen_solve` | none | ~250 lines + tests | Schrödinger, where KSL's conservation laws matter more; and any problem whose time-rank approaches `n_t` |
| P5 | linear-invariant enrichment (`opts.obs`) | §1.2 `amenany_sweep.m:318-323` | exact conservation of `⟨c, x⟩` — mass in a CME, probability in a Fokker–Planck | `amen.py`'s enrichment | none | ~30 lines | problems with no linear invariant, i.e. most PDEs |
| P6 | a sum-of-products → MPO builder (311 terms of `q^p` over 12 modes) | §1.3 `avila_carrington.dat` | the first real molecule; generalises `tests/hamiltonians.py`'s hand-written MPOs to any published force field | `ho_operators`, `tt.matrix` arithmetic, `.round` | none for construction | ~150 lines | it produces one MPO; the *84 states* need an eigensolver, below |
| P7 | the Exponential Machines model, loss and Riemannian step | §1.4 `riemannian_sgd.py`, `all_subsets.py` | the first ML application; and it makes the existing `riemannian.project` earn its place | `riemannian.project` (**exists**, M6), `.round(rmax=)` | none for correctness; M1 (`project_delta`) and M4 (`rgd`) are speed and code-sharing | ~250 lines + a MovieLens loader | above a few thousand samples per step the per-sample plateau of 208 µs starts to matter (M6) |
| P8 | z-order QTT-FEM: `W0`/`W1`, `P[lx,ly]`, per-quad assembly | §1.6 `permutation_matrix.py`, `generate_mesh.py` | 2D FEM on a polygonal domain; the natural companion of P1–P2 in the other dimension | `zkron` family (**exists**) | none | ~400 lines, a re-derivation because the source is Python 2 and unlicensed | 3D, which the paper does not treat |
| P9 | LOBPCG-in-TT with deflation and clustered inverse iteration | §1.3 `ttvibr/eigsolvers.py` | 84 interior states; `eigb` at `B = 84` was measured to grow like `B^{2.2}` (`eigenvalues.md` §7.4) | `amen_solve`, block orthogonalisation | **M7** (`eig_lobpcg`, `rayleigh_ritz`, `block_orthogonalize`), which is gated on Q4/Q5 | L | small `B`, where `eigb` wins by a lot ([RNO19] Table 4: 26 s vs 251 s at `b = 5`) |
| P10 | sparse TT-matrix cores and a sparse local solve | §1.2 `ttdR.m`, `assemble_local_matrix` | tamen's `test_lyap.m` regime — `d = 2`, `n = 10⁴` — which M5 shows is out of reach by 3 orders of magnitude in storage | a core representation change | none, but it touches `R2` (the `core`/`ps` compatibility properties) | L, and it is the riskiest item here | QTT, where `n = 2` and sparsity is meaningless |
| P11 | regression of a TT onto sampled trajectories | §1.7 | a whole application class (BSDE, HJB, stochastic control) ttpy2 cannot enter | `completion.py`'s machinery, an orthogonal-polynomial basis | none strictly; the reference implementation needs `xerus` | L | — |
| P12 | two-site prrLU cross with an accumulative pivot set, and the in-loop global-pivot check | §1.10 (`mat_decomp.h`, `tensorci2.jl`) | 5.66e-13 where ttpy2 gives 3.82e-04 on the same function, seed-independently (M8); and a per-bond residual bound at zero extra evaluations | `cross.py` | none | L, and it belongs to `docs/plans/cross-approximation.md`, not here — this row exists to hand it that document | `n = 2` QTT, where a two-site block is `4r²` and the argument for one-site is strongest; and any regime where the black box is so cheap that evaluations do not dominate |
| P13 | a sum-of-product-operators → MPO builder with batched pairwise compression | §1.10 item 7 (`auto_mpo.h`) | exactly P6's problem, already solved once: batches of 20 summands, pairwise-summed, compressed at each merge, with an explicit `maxNTerm` guard | `tt.matrix` arithmetic, `.round` | none | S–M | operators with few terms, where a plain rounded sum is fine |

**Blocked on the unimplemented Riemannian/autodiff core (M1, M4):** only P9 in
the strict sense (through M7, which needs S3/S4), and the *efficiency* of P7.
Everything else in the table is buildable against `tt/` as it stands today. That
is the single most useful conclusion of this survey and it is not what the
existing plans assume.

---

## 3. The full-scale examples

Each entry: the problem and what it means, the published reference, the ttpy2
pieces, size and runtime on b300, what it demonstrates that the current examples
do not, and the readiness marker — **(i)** buildable today, **(ii)** after a
named milestone, **(iii)** needs research.

### 3.1 (i) TT-IGA: Poisson on a curved 3D domain — `examples/iga_ring.py`

**The problem.** Steady conduction in an annular duct: `−Δu = 0` on the 3D ring
`0.5 ≤ r ≤ 1`, `0 ≤ z ≤ 1`, with `u = 1` on the inner and `u = 2` on the outer
cylindrical surface. Physically the temperature (or potential) in a pipe wall;
mathematically the smallest problem that has a **curved boundary** and therefore
a non-trivial `R(ξ)`.

**Reference.** `u(r) = [u_in log(r_out/r) + u_out log(r/r_in)] / log(r_out/r_in)`
— §1.1 Eq. (46), closed form, independent of any discretization. Secondary
references: the `L²` convergence slope ≈ 3 for `p = 2` (their §5.2), and the
metric-tensor rank table in their Table 2 caption, which is a set of integers.

**ttpy2 pieces.** `tt.cross` (have; M1 measures it at 0.13 s for six components
at `128³`), `tt.matrix.from_list` (have), `.round` (have), `amen_solve` (have),
plus **P1** (the B-spline layer, ~200 lines), **P2** (the Kronecker assembly,
~80), **P3** (the magnitude screen, ~15) and a boundary-condition elimination
that subtracts the lifted Dirichlet data from the right-hand side — which the
paper describes in one sentence (§4.3) and defers to its own earlier work.

**Size and runtime.** ~350 new lines plus a `tests/iga.py` constructor module in
the shape `ROADMAP.md` §5.1 prescribes. Assembly at `261×130×130` should land
near the paper's 21.6 s (M1 suggests faster: ttpy2 did six components of a
harder map at `128³` in 0.13 s). The solve is `amen_solve` on an operator of TT
rank ≲ 30 with 4.4·10⁶ unknowns; minutes at worst. **Not measured** — no
assembly exists yet, so this is the one number in this section that is a guess.

**What it demonstrates that nothing in `bench/` does.** A **curved, non-Cartesian
geometry**. Every problem in `bench_showcase.py` lives on a box or a spin chain.
It also exercises `tt.cross` as an *assembly* engine rather than as a function
approximator, which is a use ttpy2 has never been tested for.

**Two arms, in order.** (a) the analytic ring map, which needs no NURBS at all
and is what M1/M2 already ran; (b) the same domain from a quadratic NURBS
description with weights, which is what makes it isogeometric. Arm (a) is the
example; arm (b) is §5.

### 3.2 (i)/(new module) Time integration: 2D convection with a conserved mass

**The problem.** `tamen/test_conv.m`: `∂u/∂t + ∂u/∂x + ∂u/∂y = 0` on
`[−10,10]²` with periodic central differences on a `2^10 × 2^10` QTT grid,
`u₀ = exp(−x²−y²)`, `τ = 0.2`, 100 steps — exactly one period.

**Reference.** Three, of different kinds and all external: (1) after one period
the solution returns to `u₀`, so `‖u−u₀‖/‖u₀‖` is an error whose zero is set by
the geometry and not by us; (2) `Σu` is conserved exactly because the periodic
shift is a permutation — this is what `opts.obs` enforces and what
`test_conv.m` plots; (3) `‖u‖₂` is conserved because the operator is
skew-symmetric.

**ttpy2 pieces.** Arm (a), today: `qshift`/`IpaS`/`Toeplitz` (have),
`amen_solve` (have), Crank–Nicolson stepping in a loop, and `P5` for the
invariant. Arm (b): `P4`, the global space-time TT with the Chebyshev core, the
doubled-grid residual and the step rejection.

**Size and runtime.** Arm (a) ~100 lines of example. Arm (b) ~250 lines of
`tt/algs/tamen.py` plus tests. Runtime: 100 steps of `amen_solve` on a rank-3
operator over `2^20` unknowns — the `bench_cme` row already does 40 such steps
at `32^10` in the existing suite, so seconds to a minute.

**What it demonstrates.** ttpy2 has exactly one time integrator, `ksl`, and
`ROADMAP.md` M5 documents it returning `‖y₁‖ = 1.0677e+108` for a contraction
semigroup. This is the first example in which time is a first-class object with
an error estimate attached, and the first in which a conservation law is
*enforced* rather than reported.

### 3.3 (ii) Acetonitrile CH₃CN: twelve modes against a published cm⁻¹ table

**The problem.** The vibrational Schrödinger equation for CH₃CN in normal
coordinates,

    H = −½ Σ_{κ=1..12} ω_κ ∂²/∂q_κ² + V(q),

with `V` the 311-term Avila–Carrington quartic force field, each term a
coefficient times a product of `q_κ^p`, `p ≤ 4`. This is a real molecule with a
real force field, and it is the standard 12-D benchmark of the field.

**Reference.** Larsson, J. Phys. Chem. Lett. **16**, 3991 (2025),
arXiv:2504.05382, ancillary `all_energies.dat`, AC column: **ZPE
9837.406703549974 cm⁻¹** and 5000 states with error estimates below **0.0007
cm⁻¹**. The operator is the ancillary `avila_carrington.dat` itself, so the
surface-version ambiguity of §1.3 does not arise. Secondary: Rakhuba–Oseledets
JCP 145:124101 (2016) report 84 states in ~1 h / 100 MB; arXiv:2512.15875
reports all 84 below 1 cm⁻¹ with tree tensor networks.

**ttpy2 pieces.** `ho_operators` generalised to `q⁴` (the Galerkin-vs-truncated
distinction in `tests/hamiltonians.py`'s docstring becomes load-bearing at
fourth order), **P6** (the sum-of-products → MPO builder), `eigb` for the ground
state today, **M2**'s `eig_amen` for rank adaptation, and **M7**/**P9** for the
84-state block.

**Size and runtime.** P6 ~150 lines; the driver ~80. Runtime for the ground
state: **not measured**, and this is the entry's main risk. `eigb` at `B = 2` on
a 12-mode operator with `n ≈ 9..27` is in the regime `bench_showcase`'s
`coupled_oscillator` already covers at `d = 32`, but the potential's MPO rank is
the unknown — Rakhuba's own run used LOBPCG with an AMEn manifold
preconditioner, not `eigb`, and `eigenvalues.md` §7.4 measured `eigb` growing
like `B^{2.2}`.

**Staging.** (ii-a) build the MPO, report its TT ranks, and check the **harmonic
limit** (drop every anharmonic term; the answer is `½ Σ ω_κ`, exact) — that is a
test with an analytic oracle and it lands today. (ii-b) the ground state and the
ZPE against 9837.4067 cm⁻¹, with `eigb(B=2)`. (ii-c) the 84 fundamentals and
overtones, after M2 and probably M7.

**What it demonstrates.** It closes `eigenvalues.md` §9.5 and `ROADMAP.md` c1/c2,
both of which say the reference is not in hand and the surfaces are not public.
Both statements are now false: the surface and the reference table are both
public arXiv ancillary files. It is also the first example whose output is in
**physical units a spectroscopist reads**, which is the difference between a
benchmark and an application.

### 3.4 (i) Exponential Machines on MovieLens-100K

**The problem.** Binary classification with all-subsets polynomial features: the
weight tensor `W` is a TT of rank 10 over `d` binary modes, one per one-hot
feature, and the score is `⟨W, ⊗_k(1, x_k)⟩ + b`. The point of the model is that
it carries every interaction order at once in `O(d n r²)` parameters.

**Reference.** The repository's own committed notebook outputs: logistic
regression test AUC **0.782123**, Exponential Machines **0.7835** at rank 10,
`reg = 0.012`, `exp_reg = 1.8`, 50 epochs, batch 256. These are numbers we did
not compute, from the authors of the method, on a public dataset.

**ttpy2 pieces.** `riemannian.project(X, list)` (**exists** — M6 measured it at
2.57 ms for `B = 32` and agreeing with the summed projection to 2.3e-14),
`.round(eps=0, rmax=)` (have), `tt.dot` (have); new: the all-subsets rank-1
encoder, the logistic loss and its gradient, the exact line search plus Armijo,
and `increase_rank`. **P7**, ~250 lines.

**Runtime, from M6.** ~53 ms of projection per step at batch 256, ~390 steps per
epoch, so ~21 s/epoch and **~17 minutes for the published 50-epoch run**, single
core, projection only. The loss and the encoder are cheap by comparison; the
number to watch is the per-sample plateau, 208 µs, which is what a batch
container (S11) would attack.

**What it demonstrates.** The first machine-learning application in the
repository, and — more useful internally — that **Riemannian optimization in
ttpy2 is usable today**. Every current statement about it is negative:
`riemannian-autodiff.md` §5.2 measured `ttSparseALS` beating Riemannian GD 4–5×
on completion, §6.1 measured the tangent projection buying nothing on QTT, and
§11.8 classes this very example as aspirational. A working ExM run would be the
first positive datum.

**Counterweight, at the point of recommendation.** The published gain over
logistic regression is **+0.0014 AUC**. That is not a compelling machine-learning
result and it must not be sold as one; the example's value is that it is a
complete, published, reproducible training loop on a TT manifold, not that the
model wins.

### 3.5 (i) + (iii) QTT convection–diffusion with a boundary layer

**The problem.** `−ν u'' + u' = 1` on `(0,1)`, `u(0) = u(1) = 0`, at
`ν = 10⁻⁴` and `2^d` nodes — a boundary layer of width `10⁻⁴` on a grid that
resolves it, which is affordable in no other format.

**Reference.** The exact solution `u(x) = x − (1−e^{(x−1)/ν})/(1−e^{−1/ν})`,
closed form; and the mesh Péclet condition, which predicts the **1.66×**
overshoot M3 measured for central differences at `Pe_h = 4.9` and the absence of
one for upwind (0.997).

**ttpy2 pieces.** All present: `qlaplace_dd`, `IpaS`, `Toeplitz`, `amen_solve`.
The *example* is buildable today (i). What is **(iii) research** is making it
converge: M3 measured rank 92 for a solution of true rank 3, and no
preconditioner in `qtt_ell.py` applies to a non-symmetric operator.

**What it demonstrates.** Two things, one positive and one negative, and the
negative one is the reason to build it. Positive: the layer is compressible —
the exact solution is QTT rank 3 at `1e-10` for both `ν`. Negative: the QTT
*representation* is where the difficulty lives, not the format, and a
non-symmetric problem has no BPX. In the spirit of `bench_showcase.bench_anderson`
— a documented refusal is a result — this row belongs in the suite precisely so
that "QTT handles convection–diffusion" is not asserted without the
preconditioner that would make it true.

**A note on the metric, from M3.** Assert in `L²` or at fixed physical points.
The ∞-norm nodal error is 1.0 in every converged row of that table and it is not
measuring the solver.

### 3.6 (ii) QTT-IGA in two dimensions on a polygonal domain

**The problem.** Markeeva's: `−∇·(a∇u) = f` on a polygon cut into quadrangles,
each with a bilinear isogeometric map, assembled in z-order so the QTT ranks stay
bounded. Her repository ships a triangle and a star.

**Reference.** The JCP paper's energy-error-versus-vertex-count and
memory-versus-vertex-count curves against FEniCS (the three plots in the
repository's `img/bench.jpg`), plus a manufactured solution of our choosing.
**The numbers behind those plots are not in the repository** and the paper was
not read here, so the strict reference for a first version has to be a
manufactured solution and the `O(log n)` rank claim.

**ttpy2 pieces.** The `zkron` family (**exists**, contributed by Markeeva); new: **P8**
— `W0`/`W1`, `P[lx,ly]`, `integrate_basis`, `assemble_on_quad`, interface sewing,
boundary conditions. ~400 lines, and it is a re-derivation, not a port: M4 shows
the source does not run, and it has no licence.

**Why (ii) and not (i).** Nothing in `ROADMAP.md` blocks it. It is marked (ii)
because it should come *after* §3.1 — the 3D single-patch pipeline is simpler,
has a live paper behind it, and P1/P2 subsume most of the quadrature work.

### 3.7 (ii) Chemical master equation with an enforced conservation law

**The problem.** The existing `bench_showcase.bench_cme` row: the monomolecular
chain `0 → S₁ → … → S_d → 0`, `d = 10`, `n = 32`, integrated by implicit Euler.

**Reference.** Unchanged — Jahnke & Huisinga (2007): the exact solution is a
product of Poissons with means from `m' = Am + b`.

**What changes.** `P5` makes `Σp = 1` an enforced invariant rather than a
reported diagnostic (the row currently prints `total_mass` and does nothing with
it), and `P4` replaces the fixed 40 steps by an adaptive `τ` with a rejection
test. The docstring of that benchmark says outright that two errors are mixed in
its number and separating them "is a separate job"; `P4`'s doubled-grid residual
is exactly the tool that separates them.

**Size.** ~40 lines of change plus P4/P5. This is the cheapest genuine upgrade in
this document.

### 3.8 (iii) TT regression for a high-dimensional HJB / BSDE

Listed for completeness with its blocker named. §1.7's method needs a
least-squares fit of a TT over an orthogonal-polynomial basis to sampled
trajectories — a `tt.algs.regress` that ttpy2 does not have and `ROADMAP.md`
does not plan. `tt/algs/completion.py` is the nearest relative. Reference values
would have to come from the paper's figures, since its repository needs `xerus`
and carries no licence. **Research**, and the first question is whether ttpy2
wants a regression entry point at all.

### 3.9 Summary table

| # | example | readiness | new lines | reference | exercises |
|---|---|---|---|---|---|
| 3.1 | TT-IGA 3D ring | (i) | ~350 | Eq. (46), closed form | `tt.cross` as an assembler; curved geometry |
| 3.2 | 2D convection, conserved mass | (i)a / new module b | 100 / 250 | periodicity, `Σu`, `‖u‖` | time integration with an error estimate |
| 3.3 | CH₃CN, 12 modes | (ii) M2, M7 | ~230 | Larsson 2025, ZPE 9837.4067 cm⁻¹ | a real force field; interior eigenvalues |
| 3.4 | Exponential Machines | (i) | ~250 | AUC 0.7835 vs 0.782123 | `riemannian.project` on real data |
| 3.5 | QTT convection–diffusion | (i) example / (iii) preconditioner | ~120 | exact solution; `Pe_h` overshoot 1.66× | non-symmetric QTT; a documented refusal |
| 3.6 | QTT-IGA 2D polygon | (ii) after 3.1 | ~400 | manufactured + `O(log n)` ranks | z-order assembly |
| 3.7 | CME with an invariant | (ii) P4/P5 | ~40 | Jahnke–Huisinga | enforced conservation |
| 3.8 | TT-BSDE | (iii) | — | the paper's figures | regression onto a TT |

---

## 4. The ordered plan, and where it plugs into `ROADMAP.md`

`ROADMAP.md` §4 orders **M1–M8** by what the algorithms need from each other.
This document adds an **E-track** of examples. The rule between the two tracks:
**an example never blocks a milestone, and a milestone never waits for an
example.** Where an example needs a milestone it is scheduled after it; where it
does not — and §2 shows most do not — it can be built in parallel, by someone
else, without touching `tt/`.

| stage | lands | needs | why here |
|---|---|---|---|
| **E0** — the spline layer | P1, P2, P3, `examples/iga_ring.py`, `tests/iga.py` | nothing | M1 measured the expensive half already done and fast; the cheap half is a leaf module that touches no existing algorithm. It is also the only entry with a *current* paper behind it |
| **E1** — the ML application | P7, `examples/exp_machines.py`, a MovieLens loader | nothing (M6) | independent of E0, and it converts the repository's only negative-result story about Riemannian optimization into a positive one. Must land **before** M4, so that `rgd` has a real consumer to be designed against rather than a hypothetical one — the same argument `ROADMAP.md` uses for writing the `prec=` contract against `bpx` (`A7 → S7`) |
| **E2** — the molecule, part one | P6, `tests/vibrational.py`, the harmonic-limit test, the MPO rank report | nothing | the operator is a construction, not a solve. Landing it early makes the eigensolver work of M2 testable on a real problem instead of on Heisenberg chains, which is what `eigenvalues.md` §10 asks for |
| **E3** — time | P4, P5, `tt/algs/tamen.py`, `examples/convection.py`, the CME upgrade | nothing | the only stage that adds an algorithm module. Ordered after E0–E2 because it is the largest, not because anything blocks it. Should land **before** M5 (BUG) for the same reason as E1: M5's counterweight is that KSL is silently wrong in a measurable band, and a tamen with step rejection is the honest comparison |
| **E4** — the molecule, part two | the CH₃CN ground state and ZPE | **M2** (`eig_amen`) | `eigb` at `B = 1` "structurally cannot" adapt ranks (`eigenvalues.md` §1.2); a 12-mode quartic potential is exactly where that bites |
| **E5** — QTT convection–diffusion, as a documented refusal | the fixture of measurement M3 (§1.5 — not milestone M3), promoted to `bench/` | nothing | it is 120 lines and it is already measured. It goes in whenever someone wants it; it is listed last among the unblocked because it produces no capability |
| **E6** — the 84 states | P9 | **M7** (`eig_lobpcg`), itself gated on Q4/Q5 | the only example in this document genuinely blocked on the Riemannian core |
| **E7** — 2D QTT-IGA | P8 | E0 (for the quadrature) | after 3D, for the reason given in §3.6 |
| **E8** — sparse cores | P10 | a representation decision `ROADMAP.md` has not taken | listed so it is not forgotten; it is the only route into tamen's `test_lyap.m` regime and it is a core change, so it needs its own spec |

**Dependencies named, in one sentence each.** E0 → nothing. E1 → nothing;
**M4 should depend on E1**, not the reverse. E2 → nothing. E3 → nothing;
**M5 benefits from E3**. E4 → M2. E5 → nothing. E6 → M7 → (Q4, Q5). E7 → E0.
E8 → a new spec.

**The one edge worth arguing about.** E1 before M4 inverts the natural reading
("build the optimizer, then the application"). The justification is measured:
M6 shows the application does not need the optimizer, and `ROADMAP.md` M4's own
counterweight says `rcg` was measured *worse* than `rgd` in one regime and that
completion's iteration count varied by a factor of 24 between two draws of the
same problem. A first-order Riemannian method designed without a real objective
in front of it is how that happens.

---

## 5. What TT-IGA needs specifically, and whether it is within reach

The honest split: **the tensor half is done, the spline half is a well-defined
library, and the CAD half is a project of its own.**

### 5.1 What is already there

M1 settles the part that looked hardest. `tt.cross` finds the geometry-derived
coefficient fields of a genuinely 3D map at rank ≤ 2 in 0.13 s for six
components at `128³`, and the paper's own cost table (176 641 evaluations, 5.95 s
at 133×66×66) is in the same regime. `tt.matrix.from_list`, `+`, `.round` and
`amen_solve` cover the rest of §4.2 of the paper without a single new line.

### 5.2 What has to be written, precisely

1. **A univariate B-spline evaluator.** Cox–de Boor (their Eq. (3)) and its
   derivative (Eq. (4)) on an open knot vector, evaluated only at the `p+1`
   nonzero basis functions per point — which is what makes the paper's exact
   geometry evaluation affordable. ~120 lines, no dependencies, and completely
   testable: partition of unity to machine precision, the derivative against
   central differences, and reproduction of a known NURBS circle (their Fig. 1).
2. **The three univariate Gram matrices** `d(ξ) = N'ᵀN'`, `m(ξ) = NᵀN`,
   `c(ξ) = N'ᵀN` (their Eq. (40)), assembled by Gauss–Legendre quadrature per
   knot span. Banded, `(2p+1)n − p(p+1)` nonzeros per factor — the number their
   Table 1 uses for its full-grid comparison. ~60 lines.
3. **The rank-1 Kronecker assembly** of Eq. (38): for each `(i,j)` and each
   `(α₁, α₂)` of the cross output, contract the TT core against the right Gram
   matrix and emit a chain of three banded matrices. ~80 lines. The one design
   decision is whether to keep the summands separate (`amen_solve` accepts a
   list of matrices meaning their sum — `ROADMAP.md` Q7 asks exactly this) or to
   round the sum; the paper rounds after every summand (§4.2.1.2), and Q7's
   experiment would settle it on a real operator instead of a hypothetical one.
4. **Boundary conditions.** The paper spends one sentence (§4.3): subtract the
   boundary term from the right-hand side and solve for the interior only, with
   the details deferred to their refs. [16] and [32]. That is a real gap in the
   source and it is the piece most likely to be got silently wrong, because a
   wrong lifting produces a plausible field. It must be tested against the
   analytic solution on a domain where the Dirichlet data is not constant — the
   ring, where `u_in ≠ u_out`, is exactly such a domain.

### 5.3 What is a project of its own

**NURBS geometry input.** Everything above works with an analytically
differentiable map. The paper's actual claim is stronger: the geometry comes
from CAD, as a control net plus knot vectors plus weights, and is evaluated
exactly (their Eqs. (9)–(13)). To reproduce their six geometries — closed and
open hemisphere, ring, L-shape, hyperboloid, quarter torus (their Fig. 9) — one
needs those control nets, and **the paper does not publish them** ("data will be
made available on request"). Building a NURBS solid modeller is not a tensor
project.

**The verdict.** A TT-IGA example on an **analytically parameterised** curved 3D
domain is within reach and is recommendation #1 of this document. A TT-IGA
example on a **CAD-supplied NURBS patch** is reachable only by (a) asking the
authors for the control nets, or (b) taking a NURBS reader from an existing
library, which contradicts `docs/REQUIREMENTS.md` R1's dependency policy unless
it is pure Python and small. The paper's own closing paragraph concedes the
deeper version of the problem: "the attainable TT ranks depend not only on the
geometry but also on its CAD parameterization… systematically optimizing the
parameterization to reduce tensor ranks… is left for future work."

**One thing that is not in the paper and should be in ours.** M2. The paper
reports the metric-tensor ranks for its geometries and never mentions that some
components of `R` can vanish identically, which for the ring — one of its own
test cases — they do in the analytic parameterisation. Whether their NURBS ring
also has near-zero components is not stated; their Table 2 caption gives
`R₀₁` rank `[3 2]`, which is not 1, so presumably not exactly. The screen of P3
costs fifteen lines and 97 % of an assembly.

---

## 6. Open questions

Each with the experiment or the source that settles it. Numbered from Q20 to
avoid colliding with `ROADMAP.md` §6's Q1–Q15.

**Q20 — Who is "Marchuk"?** **ANSWERED — see `docs/plans/functional-tt.md` §0.1.**
The name is **Marzouk** (Youssef Marzouk, MIT), and identifying him does not
rescue the reference: `au:Marzouk_Y AND au:Dolgov` returns zero on arXiv, Marzouk
has no paper on vibrational spectra, and his entire TT output is two
function-approximation papers. The original pairing conflated the vibrational
line (Dolgov–Khoromskij–Oseledets–Savostyanov; Rakhuba–Oseledets — both already
cited in §1.3) with the Marzouk *functional-TT* and Dolgov *sampling* lines,
which `functional-tt.md` surveys. Nothing in §1.3 needs a new citation; §9's
bullet needs amending.

**Q21 — Does the paper's NURBS ring really have a rank-`[3 2]` `R₀₁`, and if so
why?** *Blocks:* nothing, but it is the cleanest available check that a
reimplementation of §5 is faithful. For the exact polar parameterisation
`R₀₁ ≡ 0` (M2). Their Table 2 says `[3 2]`. Either the NURBS parameterisation of
a circular arc is genuinely non-orthogonal (likely — a quadratic NURBS circle is
not an arc-length parameterisation) or the rank is roundoff, which is what M2
measured happening. *Experiment:* implement the quadratic NURBS circle of their
Fig. 1, form `R`, and compare both the rank and `max|R₀₁|/max|R₀₀|`. Two hours
once P1 exists.

**Q22 — What is the TT rank of the CH₃CN quartic-potential MPO?** *Blocks:* the
whole runtime estimate of §3.3, which is currently a guess. 311 sum-of-product
terms over 12 modes; the naive bound is 311 and the true rank is whatever the
rounding finds. *Experiment:* build it from `avila_carrington.dat` and round at
`1e-8`, `1e-10`, `1e-12`; an hour, and it needs only P6.

**Q23 — Does `eigb` reach the CH₃CN ground state at all?** *Blocks:* whether
§3.3 stage (ii-b) is really pre-M2. `eigenvalues.md` §1.2 measured `eigb` at
`B = 1` returning a wrong eigenvalue with `converged=True` on a 10-site
Heisenberg chain; a 12-mode molecule is not obviously easier. *Experiment:*
after Q22, run `eigb(B=2)` against the harmonic limit first (where the answer is
`½Σω_κ` exactly) and then against 9837.4067 cm⁻¹.

**Q24 — Should `amen_solve` take a *list* of matrices meaning their sum for the
TT-IGA operator?** This is `ROADMAP.md` Q7 with a concrete instance attached.
The paper rounds `Σ_ij K_ij` after every summand; keeping the nine summands
separate avoids the rounding but multiplies the sweep cost by nine.
*Experiment:* assemble both ways at `65×33×33` and compare peak memory, sweep
count and final residual. Half a day once P2 exists.

**Q25 — Does the invariant enrichment of `opts.obs` survive ttpy2's truncation?**
*Blocks:* P5, and therefore §3.2 and §3.7. tamen appends the projected invariant
to the basis *before* the QR and then reconstructs the solution factor from the
`R` factor (`amenany_sweep.m:326-345`). ttpy2's `amen_solve` truncates with a
residual-driven criterion by default (`trunc_norm=1`), which tamen's code path
guards against by branching. *Experiment:* implement it, then check `⟨c,x⟩` over
100 steps of the CME at `eps ∈ {1e-6, 1e-10}` — the invariant must be conserved
to machine precision, not to `eps`.

**Q26 — Is a batch container the thing that makes Exponential Machines
practical?** This is `ROADMAP.md` Q5 with a real workload. M6 measured a
per-sample plateau of 208 µs from `B = 128` upward, so the answer at MovieLens
scale is "no, 17 minutes is fine". At a dataset with 10⁷ samples it would be
"yes". *Experiment:* after P7, profile one epoch and report the split between
projection, encoding and line search.

**Q27 — What does a non-symmetric QTT operator need?** *Blocks:* §3.5's second
half, and any convection example that claims to work. BPX rests on coercivity
(`ROADMAP.md` §3.4). Three candidates, none measured here: a streamline-diffusion
stabilisation absorbed into a symmetric part; the space-time collocation of the
attached paper's refs. [16]/[32], which changes the discretization rather than
preconditioning it; and normal equations, which square the condition number.
*Experiment:* the cheapest discriminator is to measure the rank of the iterate
in each formulation at `d = 18, ν = 1e-4` and compare against the true rank 3
that M3 established.

**Q28 — Should ttpy2's cross be rebuilt on prrLU?** *Blocks:* nothing in this
document, but it may change `cross-approximation.md` §5's ranked port list. M8
measured a nine-order-of-magnitude accuracy difference on that document's own
counterexample at a comparable evaluation budget — but across two interpreters,
two numpy versions and two tolerances, and on one function. *Experiment:* a
same-harness A/B, ttpy2's `cross` against `xfacpy.TensorCI2(fullPiv=True)` at
matched `eps` on the three functions of `cross-approximation.md` §4.2 (smooth
`d=5`, QTT `d=20`, kink `d=6`) plus the §1.3 reproducer, reporting distinct
evaluations at matched achieved error. Half a day, and `xfacpy` is now built on
b300 (M7). This question belongs to `cross-approximation.md`; it is recorded here
because this survey is where the measurement was made.

**Q29 — Which quantics ordering, fused or interleaved?** *Blocks:* nothing today,
but it is a decision ttpy2's QTT layer has already taken implicitly, and
`ROADMAP.md` §2 (`level_major_order`, `merge_levels`) shows how expensive a wrong
implicit ordering is. xfac defaults to interleaved; QuanticsGrids.jl documents
both and states no preference; **no measured comparison was found** (§1.10 item
8). *Experiment:* the ranks of the §1.1 coefficient fields and of the §3.5
boundary layer in both orderings, at fixed tolerance.

**Q30 — Is there a licence problem with any of this?** `tamen` is MIT, `t3f` is
MIT, `TensorCrossInterpolation.jl` is MIT, `xfac` is BSD-2-Clause — all four are
compatible with ttpy2's MIT. **`qtt-laplace`, `ttvibr` and
`PDE-backward-solver` have no licence file**, which means all rights reserved.
None of the latter three may be vendored or adapted; they may be read and their
published mathematics reimplemented. This is not a legal opinion, and the three
affected examples (§3.3, §3.6, §3.8) should be written from the *papers*, with
the repositories used only to disambiguate.

---

## 7. Explicitly not proposed

So the next reader does not re-litigate it.

| not doing | why |
|---|---|
| a NURBS geometry kernel | §5.3. The control nets of the paper's six geometries are not published, and a solid modeller is not a tensor project |
| the functional-TT neural network of §1.9 | it is a neural parametrisation, not a TT linear-algebra method; it exercises nothing ttpy2 owns (§1.9) |
| vendoring `qtt-laplace`, `ttvibr` or `PDE-backward-solver` | no licence file in any of the three (Q28) |
| reproducing TetraFEM's numbers | the article is behind a 403 for this survey and no code exists (§1.8). It can be cited, not reproduced |
| a second cross implementation for TT-IGA | `docs/plans/cross-approximation.md` already owns that decision; §1.1's pipeline uses `tt.cross` as it stands, and M1 shows that is enough |
| **rebuilding `cross.py` on prrLU as part of any example** | it is the right thing to consider and it is not this document's to schedule. M8 is handed to `cross-approximation.md` as P12 with Q28 as its experiment (§8 item 5) |
| depending on `xfacpy` at runtime | M7: not on PyPI, needs cmake, C++17, armadillo, a LAPACK, and Python development headers. `docs/REQUIREMENTS.md` R1 forbids exactly this. It is a *reference implementation* for A/B measurement on b300, never a dependency |
| `ttvibr`'s `levels.dat` as ground truth | measured to differ from Larsson's converged values by up to 0.266 cm⁻¹ on 57 of 84 states (§1.3) |
| MCTDH's Fortran DVR routines | `ttvibr` compiles them through f2py; ttpy2's whole premise (R0, R1) is that there is no Fortran. Hermite-DVR nodes and weights are `numpy.polynomial.hermite.hermgauss` |

---

## 8. Coherence: what this document requires of the others

Minimal, and listed so it can be checked. **Nothing under `tt/` was touched and
no sibling spec was edited by this pass.**

1. `docs/plans/riemannian-autodiff.md` §11.8 and `docs/plans/ROADMAP.md` §5.4
   **c4** class Exponential Machines as needing the batch container S11. M6
   measures that it does not, for correctness or at MovieLens scale. When §3.4
   lands, both should be amended to say "needs a data loader; S11 is a
   performance question, and Q26 is its experiment".
2. `docs/plans/eigenvalues.md` §9.4/§9.5 and `ROADMAP.md` §5.4 **c1/c2** say the
   published vibrational tables are "not in hand" and that the CH₃CN potential
   surfaces "are not public". §1.3 shows both the surface and a sub-milli-cm⁻¹
   reference table are public arXiv ancillary files. Those entries move from (iii)
   to (ii).
3. `ROADMAP.md` **Q1** names [DKOS14] as unread with no citation; §1.3 supplies
   the exact reference (CPC 185(4):1207–1216, arXiv:1306.2269). The question
   itself stays open — the paper is still unread.
4. `ROADMAP.md` §4 describes M3 and M6 as future work. Reading `tt/` shows
   `qdiff`, `qtri_ones`, `qlaplace_dn`, `level_major_order`, `bpx`, `bpx_theta`,
   `bpx_operator`, `prolongation`, `stiffness` and `solve_direct_1d` all exist —
   **M3's K1–K3, A7, A9 and M6's A8 have landed**, K4 by half — and
   `bench_showcase.py` already imports `solve_direct_1d`. Still missing:
   `qtt_ell.load_vector`, `qtt_ell.solve` (A10), and `tests/elliptic.py`. That is
   documentation drift, not a defect, but §4 of the roadmap now understates what
   is built, and §5.2's entry a18 ("the runnable-today arm has no reference
   number") may no longer be true.
5. `docs/plans/cross-approximation.md` §5 records the post-hoc rook/alternating-
   maximisation residual probe as **rejected** (0/50 detections at 120
   evaluations against 2/50 for uniform Monte Carlo). §1.10 item 3 shows TCI.jl
   runs the *same* search **inside** the sweep loop, where the residual is still
   large, and makes the three-strike convergence test depend on it. The
   rejection is correct for the post-hoc use and does not transfer; as written,
   §5 reads as a rejection of the whole idea and should say which use it
   rejects. **P12 and P13 of §2 belong to that document, not this one** — they
   are listed here only because this survey is where they were measured, and
   Q28 is the experiment that decides P12.
6. `docs/BENCHMARKS.md` is in Russian and `docs/plans/*` in English. This
   document follows the plans. If §3's examples produce benchmark rows, their
   reference provenance belongs in `BENCHMARKS.md` in its language.

---

## 9. What I did not verify

* **I read no paper in full except the attached one.** [DKOS14],
  Rakhuba–Oseledets 2016, Larsson 2025, Markeeva 2021, Kornev 2024, Richter
  2024, Feng 2025 and Dolgov's tamen paper (arXiv:1403.8085) are known to me
  through their abstracts, their code, their ancillary files and this survey's
  agents — **not through their text**. Every algorithmic description above that
  is not from the attached paper or from source code I read is at that level of
  confidence.
* **I ran no MATLAB.** b300 has neither MATLAB nor Octave, so every statement
  about `tamen` comes from reading `.m` files. I did not verify that
  `test_conv.m` produces the conservation behaviour its plots claim, and I did
  not verify the `1/n_t` step-size exponent against the paper.
* **`www.mdpi.com` returns 403** to this survey's fetcher on every URL tried, so
  §1.8 rests on CrossRef metadata, one verbatim abstract sentence, and the
  predecessor arXiv paper. Its Data Availability statement is unread, so "no
  code" is an inference from a GitHub code search, not a fact from the article.
* **Markeeva's PhD thesis was not found.** No defence page, no repository entry,
  no catalogue record. The JCP paper and the `qtt-laplace` repository are what
  §1.6 rests on; if the thesis contains more, this document does not know it.
* **"Marchuk" was not identified** (Q20). I searched and found nothing; I did
  not guess.
* **§3.1's runtime is a guess.** No TT-IGA assembly exists, so the "minutes at
  worst" for the solve is an extrapolation from M1's assembly time and from
  `bench_showcase`'s existing `amen_solve` rows, not a measurement. Likewise
  §3.3's runtime, which depends on Q22's unmeasured MPO rank.
* **The sizes in §2 and §3 are engineering estimates**, in the same sense and
  with the same reliability as `ROADMAP.md` §4's.
* **M2's zero-component finding is specific to the analytic parameterisation
  I chose.** I did not check whether a NURBS ring has the same property (Q21),
  and the paper's Table 2 suggests it does not.
* **I did not re-run any number from any sibling spec.** Every `ROADMAP.md`,
  `eigenvalues.md`, `qtt-elliptic-bpx.md` or `riemannian-autodiff.md` figure
  quoted here is quoted, with its section, and was not re-measured.
* **I did not run the test suite.** `ROADMAP.md` records 723 tests in 46.82 s
  and `cross-approximation.md` records 801 in 59.40 s; I confirmed neither, and
  this document adds no code, so nothing here can have broken either.
* **`tensorly` was not checked** at all, and `t3f` was imported but not
  exercised: `import t3f` succeeding with TensorFlow 2.21.0 is not the same as
  its Riemannian routines being correct there.
* **The exp-machines AUC numbers were read from committed notebook outputs**,
  not reproduced — its `ttpy==1.2.0` pin does not build (§1.4), which is the
  measurement, not an excuse.
* **The TCI papers' technical claims are section headings, not verbatim text.**
  `scipost.org` returns an "Access Denied" interstitial to this survey's fetcher
  and the arXiv HTML truncates before §4, so everything §1.10 says about the
  *algorithms* comes from source code that was read, and everything attributed to
  the *paper* is its abstract, its confirmed metadata and its section titles.
  §4.3.6 "Ergodicity" in particular can be named here but not summarised.
* **TensorCrossInterpolation.jl was never executed.** No Julia was installed.
  Every claim about `optimize!`, `DefaultGlobalPivotFinder`,
  `convergencecriterion` and the Julia `CachedFunction` is from source with line
  numbers, not from a run.
* **M8 is a single run of one function under a different interpreter than
  ttpy2's column.** xfac ran on Python 3.11.15 / numpy 1.24.4 at `reltol=1e-10`;
  the ttpy2 numbers are quoted from `cross-approximation.md` §4.4 (Python 3.12,
  numpy 2.5.1, `eps=1e-8`). The nine-order-of-magnitude gap is far larger than
  either difference could explain, but the two were **not run under one
  harness** — which is exactly what Q28 asks for. `nEval` counting *distinct*
  multi-indices is also an inference from reading `TensorFunction::evalCache`,
  not from instrumenting it.
* **M1–M6 are single runs on a shared 256-core machine**, with no repeats and no
  error bars, at the thread counts stated. `bench/bench_core.py`'s median-of-N
  contract was not used; nothing here is a performance claim that would need it,
  and the two numbers that come closest — M5's `n^{2.1..2.8}` and M6's per-sample
  plateau — should be re-measured under that contract before being quoted
  anywhere else.
* **M2's `R₀₂`, `R₁₂` rank-10 result was not investigated.** `tt.cross` returned
  TT rank 10 with `converged=True` on a tensor that is exactly zero in floating
  point. I recorded it and did not chase it; whether the returned object has norm
  zero, and whether `_left_basis`'s deliberate non-truncation is the mechanism,
  is unexamined and belongs to `cross-approximation.md`.
