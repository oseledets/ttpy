# ttpy2 — requirements

This document is the source of truth. The code may be deleted and regenerated
from this file plus the tests.

## R0. Why

`ttpy` (2024) neither builds nor runs on a current machine:

* the core and half the algorithms are Fortran (`tt-fort`, `amen_f90`, `tt_f90`,
  `tt_eigb`, `tt_ksl`, `maxvol.f90`, `cross.f90`); building needs `gfortran` +
  `f2py` + numpy.distutils, which is gone in numpy >= 1.26;
* three submodules, two of them on bitbucket;
* the Python code uses `np.float`, `np.complex` (removed in numpy 1.24),
  `six.moves`, `d is 2` (comparing ints with `is`).

Goal: **the same mathematics, zero compilation, faster, `pip install` in
seconds.**

## R1. Installation

* Pure Python package. No Fortran/C/f2py/cmake. The wheel is `py3-none-any`.
* Required dependencies: `numpy`, `scipy` (both ship wheels).
* GPU is optional: `pip install ttpy[torch]`. A missing torch must not break the
  import.
* Acceptance: in an empty venv `uv pip install .` completes with no compiler,
  and `python -c "import tt; tt.rand(2,10,3).round(1e-8)"` works.

## R2. API compatibility

Existing `import tt` scripts must run unchanged. Preserved:

* `tt.vector` (`tt.tensor` is a deprecated alias), `tt.matrix`;
* the attributes `d, n, r, core, ps, erank, is_complex`;
* `from_list / to_list / full / round / norm / copy / write`, the arithmetic
  `+ - * kron dot diag`, `__getitem__` with slices;
* `tt.ones/rand/eye/xfun/linspace/sin/cos/delta/stepfun/unit/qshift/IpaS/
  Toeplitz/qlaplace_dd/kron/mkron/zkron/zkronv/zmeshgrid/zaffine/concatenate/
  sum/reshape/permute/matvec/col/diag/dot`;
* `tt.multifuncrs, tt.multifuncrs2, tt.GMRES`;
* `tt.amen.amen_solve, tt.amen.amen_mv, tt.eigb.eigb, tt.ksl.ksl,
  tt.ksl.diag_ksl, tt.cross.*, tt.maxvol.maxvol, tt.optimize.tt_min.*,
  tt.riemannian.*, tt.completion.*`.

**The internal representation changes.** The owner of the truth is
`cores: list[ndarray]` of shape `(r_k, n_k, r_{k+1})`. The flat `core` and the
pointer array `ps` are derived properties kept for compatibility, not storage.
Assigning `x.core = buf` rebuilds `cores` — a converter, not a second owner.

## R3. Backends

* One body of code for numpy and torch. The dispatcher is the single owner of
  the knowledge "how do I do svd/qr/matmul/dtype" (`tt/backend.py`).
* `tt.set_backend("numpy" | "torch", device=..., dtype=...)`.
* All cores of one tensor share a backend and a dtype; mixing is a loud error,
  not a silent conversion.

## R4. Numerics

* float64 by default (complex128 for complex data).
* Rounding accuracy: `round(eps)` gives a relative Frobenius error <= `eps` for
  exactly representable sums; checked against dense truth.
* Test tolerances: never tighter than `4 * eps(dtype) * scale`.
* An unexpected state — non-convergence, a rank above `rmax` at the requested
  accuracy, a singular local problem — is an explicit error or a warning
  carrying the number, never a plausible answer substituted quietly.

## R4a. QTT for elliptic problems (new in 2.0)

Not legacy compatibility but a new capability. The owner is
`tt/algs/qtt_ell.py` plus four constructors in `tt/core/tools.py`.

* Constructors: `qdiff` (the difference operator `I - S`, rank 2), `qtri_ones`
  (its **exact** inverse, rank 2), `qlaplace_dn(d, bc, order)` with mixed
  boundary conditions, `level_major_order`.
* `bc='DN'` is the only combination with exactly `2^l` degrees of freedom per
  level, so it is the only one fit for multilevel prolongations;
  `qlaplace_dd` cannot be used there. `bc='NN'` is singular and is **refused**.
* The index order is an explicit argument, not a default: the preconditioner's
  rank bounds hold only in the **level-major** layout, while `kron` and
  `qlaplace_dd` produce dimension-major. An operator and a preconditioner in
  different layouts compose into silent nonsense, so neither layout may be
  reached by accident. `merge_levels` is the single owner of the translation.
* `bpx(d, D, weight, scaled)` is the BPX preconditioner ([BK20], Theorem 3),
  assembled as a two-state automaton over the levels: **no summation** of TT
  matrices and no rounding anywhere. The rank is exactly `2*4^D`, independent
  of `d`.
* `bpx_theta(d, D)` gives the fused factors ([BK20] Lemma 5 / Theorem 4),
  `B = sum_k Theta_k^T Theta_k`, of rank `2^(2D) + 2^(2D-1)`. It returns a
  **list** even for `D = 1`: for `D > 1` the sum must not be assembled, it is
  applied factor by factor.
* Acceptance: `kappa(C A C)` stays bounded as `d` grows, while `kappa(A)` grows
  like `4^d`. Checked with dense eigenvalues.
* A variable coefficient needs no closed form: `1/a` and `sqrt(a)` are taken by
  cross approximation (`tt.multifuncrs`), wrapped as `qtt_ell.invert` /
  `qtt_ell.sqrt`. The cross tolerance becomes the scheme's tolerance, which is
  why it is in the signature rather than hidden inside.

## R5. The correctness oracle

The legacy code is **not** an oracle. The truth is:

1. a dense numpy computation for small `d, n` (the full tensor, the full linear
   system, the full eigenproblem, the full `expm`);
2. invariants (orthogonality after a sweep, `A x = b` with residual <= eps,
   monotone energy, ranks <= the theoretical ones);
3. algorithms compared against each other (`amen_solve` vs `GMRES` vs a dense
   solve).

Every algorithm is covered by a test against (1) or (2).

## R6. Speed

A measurement always comes with its regime: backend, device, dtype, thread
count, problem size, repetitions. Target scenarios:

* `round` / `full_to_tt` on large TT tensors (n=2, d=30..60, r=50..400);
* `amen_solve` on qlaplace_dd(d) and Toeplitz;
* `cross` / `multifuncrs2` on functions of QTT grids.

Comparison: numpy (CPU, 128 cores) vs torch (B300) vs the original ttpy wherever
it could be built.

## R7. Explainability

Every iterative algorithm returns or logs its history: sweeps, residuals, ranks,
time. `verb=0` is silent, but the history stays available programmatically.
