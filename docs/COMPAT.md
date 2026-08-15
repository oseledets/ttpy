# Compatibility with the old ttpy

Existing `import tt` code must keep working. Below is everything that changed,
and why.

## What stayed exactly the same

`tt.vector` / `tt.matrix` and their attributes `d, n, r, core, ps, erank,
is_complex`, `from_list / to_list / full / round / norm / copy`, the arithmetic,
`__getitem__`, and the whole zoo of constructors (`ones, rand, eye, xfun,
linspace, sin, cos, delta, stepfun, unit, qshift, IpaS, Toeplitz, qlaplace_dd,
kron, mkron, zkron, zkronv, zmeshgrid, zaffine, concatenate, sum, reshape,
permute, matvec, col, diag, dot`).

The index order is preserved: mode 1 is the fastest (`flat = i1 + n1*i2 + ...`),
a TT-matrix core is `(r, i, j, r)` with the row index first, and the merge is
`s = i + n*j`. `x.core` and `x.ps` give exactly the same F-ordered layout as
before (`tests/test_core.py::test_core_and_ps_match_legacy_layout` checks it
byte for byte).

## What changed deliberately

### 1. Storage is a list of cores, not a flat buffer

The owner of the truth is now `x.cores` (a list of `(r,n,r)` arrays). `x.core`
and `x.ps` are **computed** properties. Assigning `x.core = buf` rebuilds the
cores, but requires `n` and `r` to be known already; the old idiom — create an
empty `tt.vector()`, set `d, n, r`, call `get_ps()`, drop in `core` — no longer
works. Use `tt.vector.from_list(cores)` or `tt.vector.from_flat(core, n, r)`
instead. The reason: two owners of the same numbers drift apart sooner or later.

### 2. Bugs found by comparison against dense truth are fixed

* **`Toeplitz`, `IpaS` and `qshift` returned the transposed matrix.**
  `tt.qshift(d)` now gives the lower shift (ones on the first subdiagonal),
  `tt.IpaS(d, a)` the lower bidiagonal with `a` below the diagonal, and
  `Toeplitz(x, kind='L')` the lower triangular one. Verified against dense
  references. If your code compensated for the old behaviour with a transpose,
  remove the compensation. `IpaS` and `qshift` were also rebuilt through an
  explicit carry construction (binary addition) rather than through `Toeplitz`.
* **`reshape` of a TT matrix** now cuts rows and columns in step. The flat
  reshape mixed row bits with column bits and produced a silently wrong result.
* **`tt.dot(a, b)` conjugates its first argument** (`sum(conj(a) * b)`). Nothing
  changes for real data.

### 3. `amen_solve` changed its `max_full_size` default: 50 -> 1000

Local systems smaller than this are solved densely, larger ones by GMRES. The
value 50 made sense where the local solver was Fortran. In Python the threshold
is a different number: on a $2^{12}$ QTT Laplacian it is 444 ms at 50 against 8 ms
at 1000, and the dense path is more accurate as well (1.1e-9 against 1.8e-7) at
ranks 7 instead of 12. If you relied on the old value, pass
`max_full_size=50` explicitly.

### 4. Smaller things

* `x.full()` for a block TT (`r[0] > 1` or `r[-1] > 1`) returns the shape
  `(r0,) + n + (rd,)` with unit boundaries dropped. The old version claimed the
  shape `[rd, r0, n...]`, which did not match the layout.
* `tt.tensor` is a deprecated alias of `tt.vector` and warns via
  `DeprecationWarning`.
* `write()`/`read()` write `.npz` instead of the binary tt-fort format.
* `rand(n, d, r, samplefunc=...)`: `samplefunc(size)` is called as before, but
  the default generator is numpy's `default_rng` rather than the global
  `np.random`.
* `six`, `np.float` and `np.complex` are gone — the package runs on
  numpy >= 1.24.
* Sample-DIRT's reusable kernel is available from ``tt.transport`` and lazy
  ``tt.SampleDIRT``-style imports remain supported.  Its research experiments,
  reports, and generated artifacts live in the separate `sample-dirt` project,
  which imports ttypy rather than carrying a second implementation.

## External libraries that are gone

The old package vendored two foreign projects. Both were replaced and checked
against dense truth:

| was | size | replaced by | verified against |
|---|---|---|---|
| **EXPOKIT** (`tt-fort/expm/`, `dexp_mv`, `normest`) — the local matrix exponential in KSL | 5803 lines of F77/F90 | `expmv_krylov` + `_norm_estimate` in `tt/algs/ksl.py` (Arnoldi with adaptive substepping), ~50 lines | `scipy.linalg.expm` |
| **PRIMME** (`tt-fort/primme/`) — the local eigenproblem in `eigb` | 89 C/Fortran files, ~1 MB | a dense `eigh` for small blocks, `scipy.sparse.linalg.lobpcg` for large ones | the analytic eigenvalues of the Laplacian |

Measured on one machine, with bit-identical input.

**KSL.** $d=6, n=2$, ranks `[1,2,4,8,4,2,1]` — that is the whole space, so there
is no projection error and what shows is exactly the accuracy of the local
exponential:

| $\tau$ | old (EXPOKIT) | ttpy 2 |
|---|---|---|
| 1e-3 | 1.33e-08 | **1.96e-15** |
| 1e-2 | 1.26e-05 | **2.09e-15** |
| 1e-1 | 6.71e-03 | **8.24e-15** |

The gap of 7–12 orders of magnitude has two causes: EXPOKIT is called there with
a fixed loose tolerance, and the order of the K- and S-steps in the backward
pass of the real branch of `tt_ksl` breaks the Strang palindrome (see above).
On the full manifold our version reproduces the dense exponential to machine
precision. Test:
`tests/test_examples.py::test_ksl_is_exact_when_the_manifold_is_the_whole_space`.

Separately: on an initial condition with **unreachable** ranks ($r=8$ at $n=2$
on the first bonds) the old KSL returns a vector of norm 3.2e-08 for an input of
norm 1491 — i.e. essentially zero, without a single warning. Ours handles such
input normally.

**eigb.** $d=8$ (n=256), the 4 smallest eigenvalues, accuracy against the
analytic formula $4 \sin^2(\pi k / 2(N+1))$:

| | largest error | time |
|---|---|---|
| old (PRIMME) | 9.0e-17 | 26.8 ms |
| ttpy 2 | 1.4e-16 | **14.0 ms** |

So the replacement for PRIMME holds the same machine precision and is twice as
fast on this problem. Test:
`tests/test_examples.py::test_eigb_matches_a_dense_symmetric_eigensolver`.

## What is gone entirely

* Fortran (`tt-fort`, `amen_f90`, `tt_f90`, `tt_eigb`, `tt_ksl`, `maxvol.f90`,
  `cross.f90`), `f2py`, `numpy.distutils`, the submodules, the `setup.py` build.
  Installation is an ordinary `py3-none-any` wheel.
* The binary tt-fort file format (`.tt`). Old files have to be converted with
  the old package (or write a converter — the format is simple).

## `tt.eigb` and `tt.ksl` (replacing the Fortran kernels `tt_eigb.f90` / `tt_ksl.f90`)

The signatures are unchanged: `tt.eigb.eigb(A, y0, eps, rmax=150, nswp=20,
max_full_size=1000, verb=1) -> (y, lam)` and `tt.ksl.ksl(A, y0, tau, verb=1,
scheme='symm', space=8, rmax=2000, use_normest=1) -> y`, `tt.ksl.diag_ksl(...)`.
Old scripts run unchanged. What changed substantively:

* **The order of the K- and S-steps in the backward KSL pass is fixed.** The
  real Fortran branch (`tt_ksl`) did K first and then S on every core, so the
  backward pass was not the exact reverse of the forward one and the Strang
  palindrome broke. The complex branch (`ztt_ksl`) had it right, and that is
  what is reproduced. The order of the scheme is verified numerically against an
  **independent** oracle:
  `tests/test_verify_eigb_ksl.py::test_ksl_order_against_the_dense_projected_flow`
  integrates the projected ODE $y' = P_{T_y M} A y$ with a dense DOP853 (the
  projector is built from scratch inside the test, numpy only) and gives 1.00
  for `scheme='first'` and 2.00 for `scheme='symm'`, with a modelling error 25x
  larger than the splitting error being measured. The comparison has to be
  against the projected flow, not against `expm(tau A) y0`: relative to the
  latter the two schemes are indistinguishable.
* **KSL measures and reports what a fixed rank cannot see.** `check_rank=True`
  (the default) computes the off-tangent part $(I - P_{T_y M}) A y$ and records
  `defect_rel` and `step_error_est` (= $\tau \|(I-P) A y\| / \|y\|$) in the
  history; when `step_error_est > defect_warn` it raises an explicit
  `RuntimeWarning`. Measured: `step_error_est` predicts the true step error
  against `scipy.linalg.expm` to within 3%. It costs one TT matvec plus one
  sweep; `check_rank=False` restores the old price.
* **The local exponential** is our own Arnoldi with EXPOKIT-style adaptive
  substepping instead of `dexp_mv`; `space` is the same Krylov dimension, and
  `use_normest` only affects the choice of the first substep (and cannot change
  the result: `test_ksl_knobs_do_not_change_the_answer`). An unreachable
  accuracy is a `RuntimeError`, not a quiet answer.
* **The local eigenproblem in `eigb`** at `size > max_full_size` is solved by
  `scipy.sparse.linalg.lobpcg` on an implicit operator (instead of PRIMME); the
  true local residuals are measured and land in `history.max_local_res`.
  Problems smaller than $5B + 10$ always take the dense path.
* **`eigb` measures its own residual and reports failure.** `ermax` (the
  movement of the Ritz values) cannot tell convergence from being stuck:
  one-site ALS cannot grow a rank, so with $B = 1$ and a rank-1 initial guess
  the iteration stands still, `ermax` drops to 1e-14, and what used to come back
  was `lam = 7.8e-3` where the minimum is `1.5e-4` — silently. Now
  `check_residual=True` (the default) computes $\|A y_i - \lambda_i y_i\|$ in the TT
  format (matvec + sum + QR sweep, never expanding into a dense vector) and puts
  it in `history.res` / `history.res_rel`; a relative residual above `res_warn`
  (1e-2) raises a `RuntimeWarning` with the numbers. It costs about one sweep.
  The cure for the underlying problem is a higher-rank initial guess or $B > 1$.
* **`sym_tol` defaults** to $\sqrt{\varepsilon}$ of the working dtype rather than a fixed
  `1e-8`: in float32 the projection of the local matrix is asymmetric at the
  1e-7 level from rounding alone, and a fixed threshold rejected every float32
  problem.
* **New optional arguments** (the default behaviour is unchanged):
  `return_history=True` returns a history object (`EigbHistory` / `KslHistory`)
  with per-sweep or per-step records — written even at `verb=0`; `eigb` also
  takes `lobpcg_maxiter`, `sym_tol`, `check_residual`, `res_warn`, and `ksl`
  takes `local_tol`, `check_rank`, `defect_warn`. A non-symmetric `A` in `eigb`
  is now an error rather than a silently symmetrized problem, and `nswp` without
  convergence raises a `RuntimeWarning` carrying the indicator reached.
* **The operator follows the vector's backend** (the `amen_mv` convention): a
  numpy `A` with a torch iterate used to die inside einops with a `TypeError`.

## `tt.optimize`, `tt.completion`, `tt.riemannian`, `tt.solvers`

Four modules that were pure Python in the old ttpy as well. The signatures are
unchanged and the old import paths work (`from tt.optimize import tt_min`,
`from tt.completion.als import ttSparseALS`, `from tt.riemannian import
riemannian`, `from tt.solvers import GMRES`, `tt.min_tens`, `tt.min_func`,
`tt.GMRES`); the implementation moved to `tt/algs/{optimize,completion,
riemannian,solvers}.py`. All four gained an optional `return_history=True`
(`ttSparseALS` always returned its history) — records are kept even at `verb=0`.

### `min_tens` / `min_func` (formerly `tt/optimize/tt_min.py`)

* The left and right index sets are stored separately. In the old code one array
  `Jy` meant the left set or the right one depending on the sweep direction;
  that worked, but it was never checked.
* The sweep truncates the smoothed block to `rmax` singular vectors **in both
  directions**. The old code went left through an SVD and right through a plain
  QR, so the index sets grew to about 4·`rmax` on every other half-sweep.
* `min_func` calls `fun` **only** on a `(P, d)` array — including the final
  re-evaluation at the record point (the old code passed a `(d,)` vector there,
  and a vectorized function died at the very end of a successful run).
* The returned value is always re-evaluated at the returned point;
  `history.consistency` is its discrepancy with what the sweep saw (nonzero only
  for a non-deterministic function).
* New keyword arguments: `rho` (the steepness of the default smoothing function
  $\pi/2 - \arctan((p - \lambda)/\rho)$; `0.5` for `min_func`, as in the signature, and
  `1.0` for `min_tens`, as in the old code), `seed`, `return_history`.
* `history.evaluations` counts **with repetitions**: adjacent sweeps look at
  overlapping blocks.

### `ttSparseALS` (formerly `tt/completion/als.py`)

* It no longer damages its input: the old code divided `cooP['values']` by the
  norm in place.
* The least-squares matrices are assembled for all samples at once by two
  interface passes instead of one Python `getRow` call per (sample, slice) pair:
  $O(P d r^2)$ in BLAS instead of $O(P d^2 r^2)$ in the interpreter.
* A slice that no sample touched keeps its previous value instead of being
  zeroed (zeroing changes `X` without changing the functional — it silently
  destroys rank).
* `alpha` is finally used (in the old code the call was commented out): it is
  the `rcond` of the local problem, and `alpha <= 0` means the exact solution,
  the only mode with a guarantee that the functional decreases monotonically.
* `converged` now means "the functional reached `tol`". Stopping at an ALS
  stationary point above `tol` is `stop_reason='stalled'` and
  `converged=False`. Measured: a rank-2 tensor `8x8x8x8` (96 parameters),
  recovered at the true rank with `alpha=0`: with ~820 distinct samples 4 of 6
  starts reach `fit ~ 1e-15` and 2 stall at `1e-1`; with ~1330 samples, 6 of 6.
* `time.clock` (removed in Python 3.8) is gone.

### `project` / `projector_splitting_add` / `tt_qr` (formerly `tt/riemannian/`)

* The `numba` branch is gone: it duplicated the same mathematics with unrolled
  sixfold loops and only engaged when every rank in the list `Z` was equal. The
  same contractions through `einsum` are faster and need no compiler.
* They work on complex tensors as well (the frames enter the contractions
  conjugated); on real data the formulas coincide with the old ones.
* The `debug=True` branch with its inline `assert`s is gone — replaced by tests
  against a dense projector assembled from the SVDs of the unfoldings of `X`,
  independently of this code.
* They work on the torch backend (verified on CUDA).

### `GMRES` (formerly `tt/solvers.py`)

* Restarts are a loop, not recursion (the old version called itself once per
  restart and ran into the stack at a large `maxit`).
* `u_0` is not damaged. The old version did `u_0 += ...` in place.
* The **true** relative residual $\|b - A x\| / \|b\|$ of the computed `x` is
  returned. The old version returned the residual of the first iteration of the
  last restart and declared convergence from it.
* The small least-squares problem on the Hessenberg matrix is solved densely
  (`lstsq`) instead of with hand-accumulated Givens rotations: the old rotations
  were real and corrupted the complex case, and the inner product was conjugated
  on the wrong side.
* The relaxation of the matvec accuracy is capped at one: a relative error of 1
  means "return anything".
* Non-convergence is a `RuntimeWarning` carrying the residual reached, not
  silence.
* The internal `_iteration` argument (the recursion counter) is gone from the
  signature.

### Changes after adversarial verification (`tests/test_verify_ports.py`)

* `project` now **refuses** to work at a rank-deficient point. Measured: a
  rank-1 tensor written with TT ranks `(1, 2, 2, 1)`, $d = 3$, $n = 4$, float64 —
  the formula returned a correct Hermitian idempotent projector (idempotence
  1.6e-16) differing from the tangent projector at that point by 31 % of its
  norm. The previous guard ("orthogonalization changed the ranks") never fired:
  a QR never drops rank. `X.round(0)` does not reduce the rank either — `chop`
  at `eps <= 0` returns the full size by definition; `X.round(1e-14)` does. The
  test is exact rather than heuristic: the singular values of the triangular
  factor $R_k$ of the left QR sweep, after the right orthogonalization, are
  exactly the singular values of the $(k+1)$-st unfolding of $X$.
* `projector_splitting_add` at such a point, by contrast, is **left working**:
  it measures 1.3e-15 on the same example, and refusing would be a regression.
  `tt_qr` there gives orthogonality and reconstruction at 1e-15.
* `ttSparseALS` no longer stays quiet about complex data: `cooP['values']` (or
  `x0`) with an imaginary part is a `TypeError`, not a cast to `float64` behind
  a `ComplexWarning` followed by "`fit ~ 1e-30`" for a fit to half the data.
* `ttSparseALS` counts and reports how many local systems the data fails to
  determine: `info.underdetermined_slices`, `info.empty_slices`,
  `info.determined`, plus a `RuntimeWarning`. Measured: a rank-4 tensor of shape
  `6x6x6` (144 parameters) from 38 samples gives `fit = 4.9e-31`,
  `converged = True`, and a relative error against the truth of 5.8. Now
  `determined = False`. `converged` without `determined` only means "reproduces
  the samples".
* `ttSparseALS` scales `x0` together with the data. Previously `maxnsweeps = 0`
  returned `||values|| * x0`, and "start from the exact solution" started from
  `||values||` times the solution. The factor goes into the zeroth core, which
  the very first local solve overwrites entirely — a run of one sweep or more is
  unaffected.
* `min_tens` / `min_func`: `nswp < 1` and `rmax < 1` are a `ValueError`
  (previously `nswp=0` died with `AttributeError: 'NoneType' object has no
  attribute 'reshape'`); `rmax=None` works as "no cap", as the docstring of
  `_search` always promised; `history.index_sizes` at `d = 1` is a list of pairs
  just as it is at `d > 1` (previously `[1, 1]`, which made `max_index_set` and
  `repr(history)` die with a `TypeError`); an all-NaN block is a
  `FloatingPointError`, not an `AttributeError`.
* `GMRES`: `eps < 0` is a `ValueError` (previously it silently meant "truncate
  nothing and never converge"); the division by an exact zero in the matvec
  accuracy relaxation, at `eps = 0` with an invariant Krylov subspace, is closed
  by an explicit branch.
* **The oracle in `tests/test_ports.py` was wrong for the complex case**, and it
  was the oracle that got fixed, not the code: the projector onto the row space
  of an unfolding was assembled from `vh[:r].conj().T`, i.e. onto the complex
  conjugate of the row space. Such a matrix is Hermitian, idempotent and has the
  right trace, so no invariant sees it, and at a point of maximal rank — which is
  where the complex test stood (`n = [3, 4, 3]`, rank 3, tangent space all 36
  dimensions) — it is simply the identity. On a non-degenerate example
  (`n = [3, 4, 5]`, rank 2, tangent space 24 inside 60) the discrepancy is 74 %.
  The correct oracle is `vh[:r].T`; it agrees with a basis of the tangent space
  built straight from the definition ($\mathrm{span}_k \tau(C_1, \ldots, dC_k, \ldots, C_d)$) to
  2.6e-15, and with `project` to 6.7e-16. The tests that compare against the
  dense projector now check that the case is non-degenerate.

## Default changes in 2.0 (after the first wave of porting)

Not a legacy-compatibility matter (these arguments did not exist in `ttpy` 1.x),
but it changes the behaviour of code written against early 2.0 builds.

* **`amen_solve(..., check_true_res=)` is now `False`.** The exact residual
  $\|A x - f\| / \|f\|$ used to be computed after suitable sweeps and served as
  the stopping criterion. The product $A x$ has cores
  $(r^A_k r^x_k, n_k, r^A_{k+1} r^x_{k+1})$ — the ranks multiply — and on a
  preconditioned 2D problem ($r_A = 161$, $r_x = 122$) that is rank 19642 and
  12.3 GB in a single core, with a measured peak of 91.8 GiB, for a number that
  is only printed. The same product rounded to 1e-12 has rank 256.

  Consequences for the caller: `info.true_res` is now `nan` (not a guess but
  "not measured"), and the stopping criterion is `max_res`, the local residual
  $\|B_k x_k - \mathrm{rhs}_k\|$ of every block **before** it is solved. The guarantee
  "never return an iterate worse than one already seen" now rests on an active
  measure. The exact residual is still available through the same argument and
  warns about its cost before allocating (`true_res_budget`).

* **`ksl` refuses a stiff step instead of returning a number.** The S-steps of
  the projector splitting run backwards in time, so for a dissipative `A` they
  amplify; the following K-step shrinks the data but not the rounding error that
  accumulated. Measured on $dy/dt = -(2^L+1)^2\,\mathrm{Laplace}\,y$: at $\tau\|A\| = 169$
  it returned $\|y\|$ = 3.3e+106 where the exact norm is 0.307. Now every local
  exponential records its growth factor, the history carries `max_growth` and
  `roundoff_floor`, exhausting the digits entirely is a `RuntimeError`, and
  exceeding the requested `local_tol` is a warning. The threshold is measured,
  not derived; the table of measurements sits next to `KSL_GROWTH_EXPONENT`.

* **`eigb` warns on the backward error, not on the relative residual.** The
  `res_warn` threshold is no longer a fixed `1e-2` but $\sqrt{\varepsilon}$, and it
  applies to $\|A y - \lambda y\| / \|A\|_2$ rather than $/ \|A y\|$. The latter
  demands *relative* accuracy of every eigenvalue, which is unreachable at the
  bottom of the spectrum: on `qlaplace_dd([10])` the values are correct to 1e-9
  absolute while `res/||Ay||` is 3.0e-04. `history.res_rel` is kept, and
  `history.res_back` and `history.anorm` were added.

* **`tt.permute` returns a compressed representation.** Bubble transpositions
  used to leave rank slack behind: on a three-peak separable function shuffled
  into Morton order at $d = 15$, rank 1024 for a tensor whose own rank is 102.
  The tensor was right, the representation was not.
