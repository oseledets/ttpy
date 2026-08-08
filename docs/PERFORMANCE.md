# Performance

A number without its regime is not a measurement. Every figure below was taken
on one machine with its regime stated next to it; the raw data is in
`bench/results/*.json` together with the environment (host, versions, thread
count, sizes, repetitions).

**Test bench.** 2 x Intel Xeon 6767P (128 physical cores, 256 threads), 2 TB
RAM, NVIDIA B300 SXM6 (275 GB), CUDA 13.0, Python 3.12, numpy 2.x (OpenBLAS),
torch 2.13.0+cu130. Median of 3 runs after a warm-up, with the GPU synchronized
before the timer stops. The machine was not entirely idle while the numbers were
taken (background jobs were running), so read them as "order of magnitude and
ratio", not as a record.

## 1. Rounding is LAPACK latency, not flops

`round` for a TT tensor is a chain of `d` sequential QR and SVD calls on
matrices of roughly `(n r) x r`. At `d = 60, n = 2, r = 200` that is 120 LAPACK
calls on 400x200 matrices. Such calls can neither be fused (core `k+1` depends
on `k`) nor parallelized internally.

Two consequences follow that surprise many people.

**Multithreaded BLAS hurts.** Regime: numpy, float64, `d=60, n=2, r=200`,
operation `round(1e-8)`:

| threads | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| median, ms | **496** | 995 | 1269 | 1353 | 2209 | 1681 |

One thread is 3.4x faster than 64. If your process is doing nothing else, set
`OMP_NUM_THREADS=1..4` for TT workloads with ranks in the hundreds.

**In fp64 the GPU loses to the CPU on the same operations.** Regime: float64,
numpy (4 threads) against torch/B300:

| operation | size | numpy CPU | torch B300 |
|---|---|---|---|
| `round` | d=30, n=2, r=50 | **15.7 ms** | 268 ms |
| `round` | d=60, n=2, r=200 | **1468 ms** | 3670 ms |
| `round` | d=10, n=8, r=400 | 1877 ms | **1277 ms** |
| `add+round` | d=30, n=2, r=100 | **272 ms** | 1279 ms |
| `matvec+round` | d=60, n=2, r=150 | **3608 ms** | 5511 ms |
| `dot` | d=30, n=2, r=100 | 21.2 ms | **2.9 ms** |
| `dot` | d=60, n=2, r=200 | 309 ms | **10.8 ms** |
| `dot` | d=20, n=4, r=300 | 804 ms | **9.9 ms** |

The picture is exactly the one the analysis predicts: where a contraction is
computed (`dot` — nothing but GEMMs, not a single factorization) the GPU is
7–80x faster; where a chain of small factorizations is computed the CPU wins,
and the gap narrows as the matrices grow (at `n=8, r=400` the GPU is already
ahead).

## 2. What to do about it: change the algorithm, not the flags

Since the bottleneck is the SVD chain itself, the chain has to go. Randomized
rounding (Al Daas et al., SIAM J. Sci. Comput. 45(1), 2023, arXiv:2110.04393)
replaces it with a sketch against a random TT plus one QR per core, i.e. with
matrix multiplications:

```python
y = x.round(rmax=100, method="randomized")          # the same rank
y, err = x.round(rmax=100, method="randomized", return_error=True)
```

`err` is an upper estimate of the error; it honestly saturates at
`||x|| * sqrt(eps)`, because it is computed through the difference
`||x||^2 - ||y||^2`, which loses every significant digit once the error is
small. Below that threshold the package says "I cannot tell you more precisely"
instead of producing a confident small number.

Accuracy: on random rank-20 TT tensors (d=12, n=2) the error of randomized
rounding never exceeded 3x the error of the deterministic one at the same target
rank (`tests/test_core.py::test_randomized_round_matches_svd_accuracy`).

### Measurement: deterministic versus randomized

Regime: `bench/bench_round.py`, median of 3 runs, a random TT whose cores are
normalized by 1/sqrt(r) (otherwise the norm of the product overflows float32),
`OMP_NUM_THREADS=4`, one B300 as the GPU. `err` is the relative truncation
error, measured identically for both methods.

| dtype / device | problem | SVD | randomized | speed-up | err SVD / rand |
|---|---|---|---|---|---|
| numpy fp64 | d=30, n=2, 100→50 | 82 ms | **43 ms** | 1.9x | 0.645 / 0.825 |
| numpy fp64 | d=60, n=2, 200→100 | **711 ms** | 876 ms | 0.8x | 0.877 / 0.985 |
| numpy fp64 | d=20, n=4, 300→150 | **793 ms** | 1396 ms | 0.6x | 0.897 / 0.981 |
| torch fp64 | d=60, n=2, 200→100 | 4427 ms | 1498 ms | 3.0x | same |
| torch fp64 | d=20, n=4, 300→150 | 1884 ms | 758 ms | 2.5x | same |
| torch fp32 | d=60, n=2, 200→100 | 319 ms | **204 ms** | 1.6x | same |
| torch fp32 | d=20, n=4, 300→150 | 153 ms | **96 ms** | 1.6x | same |

Read it like this:

1. **The fastest combination is fp32 on the GPU with randomized rounding.** On
   `d=20, n=4, 300→150` that is 96 ms against 793 ms for numpy/fp64, i.e.
   **8.3x**. On `d=60, n=2` it is 204 ms against 711 ms, **3.5x**.
2. **Randomization helps specifically on the GPU** (2.5–3.0x in fp64, 1.6x in
   fp32); on the CPU at large ranks it even loses (0.6x), because there the SVD
   chain is already up against optimized LAPACK and the sketch only adds passes
   over memory.
3. **fp64 on a B300 is useless for factorizations**: 4427 ms against 711 ms on
   the CPU for the same problem. That is a property of the hardware (Blackwell
   has double precision cut down), not of the library.
4. **The truncation error does not depend on the arithmetic precision** (fp32
   and fp64 gave identical `err`): it is set by the discarded spectrum, not by
   the arithmetic. But on random data with a flat spectrum the randomized method
   is noticeably worse (0.985 against 0.877): the sketch is designed for a
   decaying spectrum. For smooth functions and QTT sums the gap disappears; for
   white noise it does not.

Practical conclusion: turn `method="randomized"` on for the GPU and for large
ranks, and keep the default on the CPU. The library does not choose for you,
because the choice depends on the spectrum of your data, which it does not know.

## 3. Against the old ttpy (Fortran)

The old package could be built after all — the recipe and its six workarounds
are in [LEGACY_BUILD.md](LEGACY_BUILD.md). That gives a reference point.

Regime: one host (Xeon 6767P), both packages run **back to back, pinned to the
same cores** (`taskset -c 100-120`), `OMP_NUM_THREADS=4`, float64, identical
input tensors (cores from a fixed seed, handed over through `from_list`), median
of 3 runs. Fortran on the left (numpy 1.24, Python 3.11), ttpy 2 on the right
(numpy 2.5, Python 3.12).

The first version of this table was taken without core pinning and while other
jobs were loading the machine; it overstated ttpy 2's disadvantage by a factor
of 2–3. The numbers below were taken again.

| problem | old (Fortran) | ttpy 2 | ratio |
|---|---|---|---|
| `round` d=30, n=2, r=50 | 10.5 ms | 11.0 ms | 1.05x slower |
| `round` d=60, n=2, r=100 | 374 ms | **304 ms** | **1.23x faster** |
| `round` d=20, n=4, r=150 | 441 ms | **352 ms** | **1.25x faster** |
| `add+round` d=30, r=50 | 126 ms | **101 ms** | 1.24x faster |
| `add+round` d=60, r=100 | 1397 ms | **908 ms** | **1.54x faster** |
| `matvec+round` d=30, r=50 | 189 ms | **125 ms** | 1.51x faster |
| `matvec+round` d=40, r=100 | 980 ms | **629 ms** | 1.56x faster |
| `dot` d=60, r=100 | 10.1 ms | **5.1 ms** | **1.98x faster** |
| `tt_svd` d=10, n=4 (1M elements) | 8272 ms | **698 ms** | **11.9x faster** |
| `amen_solve` d=12, eps=1e-6 | 23.2 ms | **8.2 ms** | **2.8x faster** |
| `amen_solve` 2D, 1024x1024, eps=1e-6 | 198.5 ms | **193.1 ms** | **1.03x faster** |

How to read this:

* **On the core operations pure Python is not slower than Fortran but faster**
  (1.2–1.6x). The reason is not craftsmanship: all the heavy work goes to a
  modern LAPACK/BLAS, while Fortran from 2013 carries its own implementations.
* **`tt_svd` is 11.9x faster** — the same effect in its purest form: a blocked
  `gesdd` against the old code.
* **`dot` used to be 4.2x slower and is now 2x faster**, after the hot path was
  moved from `einops.einsum` to two explicit GEMMs: on contractions of size
  `r x n x r` parsing the pattern costs more than the arithmetic itself
  (42.2 ms -> 5.1 ms). The block case (`r0 > 1`) still goes through einsum, and
  both paths are covered by tests against a dense contraction.
* **`amen_solve` used to be 30x slower — because of an inherited default.**
  `max_full_size=50` came from ttpy 1.x, where the local solver is compiled
  Fortran and the size at which a dense solve stops paying off is low. In an
  interpreted implementation that threshold is a completely different number. On
  a 2^12 QTT Laplacian: 444 ms at 50 against 8 ms at 1000, and the dense path is
  **more accurate** as well (residual 1.1e-9 against 1.8e-7) at lower ranks
  (7 against 12). The default was changed to 1000 and the difference is
  documented in COMPAT.md. After that the solver is 2.8x faster than Fortran and
  60x more accurate on the same problem. This is exactly the case where a
  constant outlived the mechanism that justified it.

* **Both implementations fall apart at d >= 20, and only one of them says so.**
  The QTT Laplacian on 2^30 points has a condition number of order 1e17, i.e.
  the problem is unsolvable in double precision without preconditioning.
  Measured (eps=1e-8):

  | d | old (Fortran) | ttpy 2 |
  |---|---|---|
  | 12 | 23.2 ms, residual 2.0e-8 | 8.2 ms, residual 3.4e-10 |
  | 20 | 1126 ms, residual **1.2e+03**, silently | 1384 ms, residual 3.1e-05, with a warning |
  | 30 | 3439 ms, residual **6.4e+05**, silently | 7875 ms, residual 8.0, with a warning |

  The old package returns garbage without a word. The new one returns a result
  markedly closer to the solution and prints the residual it actually reached,
  which sweep was the best, and what exactly failed to converge.
* At eps=1e-10 **neither package reaches the requested accuracy** (7.8e-10 for
  the old one, 5.6e-10 for the new), but the old one stays quiet while the new
  one warns with the real residual and names the blocks that did not converge.

## 3a. AMEn: how the 2D gap was closed

The one problem where Fortran stayed ahead. Runs interleaved, both packages on
the same cores, 2D QTT Laplacian 1024x1024, eps=1e-6:

| time | what was done |
|---|---|
| 2102 ms | starting point |
| 972 ms | einsum routed through BLAS (einops does not forward `optimize`) |
| 719 ms | binary contractions compiled into a batched matmul |
| 645 ms | the wrapper removed from this path |
| 529 ms | GMRES: contiguous basis, Gram-Schmidt as two products, Givens |
| 445 ms | the local operator assembled once in BLAS layout (2x on the matvec) |
| 428 ms | inexact tolerance for the local solves (7482 iterations -> 4635) |
| 258 ms | the local solver entirely in a compiled kernel (numba) |
| 236 ms | interface contractions (`_project`/`_apply`/`_phi_next`) in kernels |
| 205 ms | the true residual computed from `max_res` rather than `max_dx` |
| **193 ms** | the double inversion of the preconditioner blocks removed |
| Fortran: **198 ms** | |

That is **10.9x** in total, from ten independent findings, each of them
measured. In 2D we are now faster than the original Fortran at the same accuracy
and the same ranks; in 1D at d=12 as well (11.0 against 11.5 ms, residual
1.3e-09 against 1.8e-07, rank 7 against 13). On the smallest problem (d=10) we
are slower (8.9 against 4.9 ms) because we overshoot: the true residual is
checked once the cheap indicator comes within an order of magnitude of the
threshold, and by that point it is already 1.3e-10 against the 1e-6 requested.

Five things were measured and **rejected**, so that nobody tries them again:

* **Cholesky for symmetric local systems** — 1.6x slower than `np.linalg.solve`
  at n=400 (scipy copies the matrix), and detecting symmetry per block cost more
  than the solve itself (dense solves 216 -> 430 ms).
* **Mixed precision with iterative refinement** — 1.09-1.39x, not worth the
  complexity.
* **Block Gauss-Seidel as a preconditioner** — its application is sequential in
  the blocks and costs 8-22x more than the Jacobi one.
* **Binary search for the rank when truncating in the residual norm** — the
  residual is only *almost* monotone in the rank, and bisection picked ranks
  that cost convergence (d=14 stopped at 1.3e-08 instead of 1e-10). A linear
  scan from the top exits in a few steps and therefore costs 30 ms out of 200,
  not more.
* **`sqrt(<x,x>)` instead of an orthogonalization for the norm** — 12x faster,
  but in the TT format that is a contraction whose intermediates cancel: 12
  tests failed, and on a convection problem it reported a residual of 1.25e-06
  where the true one was below 1e-06, i.e. it turned a converged run into a
  failed one.

## 3b. KSL: we are 9x slower than the legacy code (it was 23x)

**Careful with the framing.** The first version of this section claimed that the
legacy KSL "does not solve the problem": it gave a relative error of 1.10 at
every `tau`, including `tau -> 0`, where an integrator is obliged to return
`y0`. That claim was **wrong**, and the cause was my input. `y0` had ranks
`[1,4,4,4,4,4,1]` at `n = 2`, while a boundary bond of a TT tensor cannot exceed
`min(n^k, n^(d-k))`, i.e. 2. That is a rank-deficient, degenerate starting point
— the worst possible input for a fixed-rank integrator, not a working regime.

An honest measurement: `d = 6`, `n = 2`, a symmetric `A` with `||A||_2 = 1`,
ranks `y0 = [1,2,4,4,4,2,1]` (admissible), the same cores loaded from a file for
both implementations, b300/numpy/float64, minimum of 5 runs, a dense `expm` as
the reference:

| `tau` | legacy, time | legacy, error | ttpy2, time | ttpy2, error |
|---|---|---|---|---|
| 1e-8 | **0.23 ms** | 2.208e-09 | 4.26 ms | 2.208e-09 |
| 1e-4 | 0.22 ms | 2.208e-05 | 5.27 ms | 2.208e-05 |
| 0.05 | 0.23 ms | 1.111e-02 | 5.25 ms | 1.111e-02 |
| 0.20 | 0.23 ms | 4.507e-02 | 5.24 ms | 4.506e-02 |
| 0.80 | 0.22 ms | 1.854e-01 | 5.28 ms | 1.744e-01 |

The accuracy agrees to 3–4 digits (at `tau = 0.8` ours is slightly better) and
neither implementation changes the ranks. **Fortran is 23x faster.** That is an
honest gap and an open optimization target: 22 local exponentials (`2(2d-1)`)
with Krylov substepping on a problem of 64 numbers taking 5 ms is Python
overhead, not flops. The same gap was closed for `amen_solve` (see 3a) and was
not closed here at all.

Where we are better: on that same rank-deficient input the legacy code returns a
vector 1.0996 away from `y0` at `tau = 1e-8` and silently changes the rank from
4 to 2; ours gives 3.37e-09, exactly proportional to `tau`. So we are robust to
a degenerate starting point and it is not — but that is a narrow case and it
does not justify the gap in time.

Separately: the legacy code prints "Solving a complex-valued dynamical problem"
on real input, even though the Python wrapper selects the **real** branch
(`np.iscomplex(...).any()` is False, verified). The message lies about itself;
it does not affect the numbers.

The cost of the stiffness guard added to `_step_exp` (two norms per local
exponential) was measured against its own parent: **4.33 → 4.45 ms, +2.8 %**.

### How the gap was narrowed

A profile of one step (30 repetitions, `cProfile`) showed there is hardly any
arithmetic in it: the local problems are **4, 16 and 32 numbers**, one substep
each, Krylov 4–8. About 260k flops per step in total — Fortran does that in
0.22 ms, i.e. it runs at the speed of the arithmetic.

| change | time | what exactly |
|---|---|---|
| baseline | 4.30 ms | |
| a cache in `backend_of` | 3.92 ms | 508 norms per step, each constructing a backend object (18 270 `NumpyBackend.__init__` calls over 30 steps) |
| exact exponential for a small block | **2.00 ms** | at size ≤ 40 `expm` is cheaper than eight Arnoldi iterations: 10.6 / 21.8 / 46.0 µs at sizes 4 / 16 / 32 against ~180 µs for the Krylov path |

The accuracy did **not change in a single digit** (2.208e-09, 1.111e-02,
1.744e-01 before and after), and two approximations disappeared: an exact step
has neither a Krylov error nor substepping, so `err_est = 0` there is the truth
rather than a degeneracy. Such steps are marked `exact=True` in the history —
without that mark a zero estimate cannot be told apart from a collapsed Krylov
space, which reports the same thing and is lying.

**The gap is still 9x** (2.00 ms against 0.22), and there is no dominant item
left: `expm` 16 %, contractions through `einsum` 25 %, `QR` 15 %, `einops`
overhead 12 %. What remains is compiling the whole sweep (as `_fast.gmres_local`
does for AMEn), because every individual numpy call on a block of 32 numbers
costs about as much as all of its arithmetic.

## 3c. What changed after the first wave of measurements

The numbers in sections 1–3 were taken before these changes and describe the
earlier code.

* **The local Jacobi in `amen_solve` was quadratic in the operator rank.** The
  compiled kernel fuses all six loops (`r1 r2 n m R1 R2`), whereas a two-step
  contraction costs `r1 n m R1 R2 + r1 r2 n m R2`. At the ranks it was written
  for (`r=34, n=2, R_A=4`) the ratio is 3.6 and fusing wins on overhead; at
  `R_A = 161` the ratio is 84. The choice is now made by cost: **2708 ms → 46.7
  ms** per call at BPX shapes, with no change (0.09 ms) in the native regime.
  End to end on the 2D problem: **295 → 109 s**.
* **The exact residual in `amen_solve` is off by default.** Peak memory
  **33.85 GiB → 0.81 GiB** on the same problem, and the run went from 62.2 to
  47.6 s.
* **`tt.permute` recompresses its result.** Rank 1024 → 118 on a separable
  function in Morton order at `d = 15` (the tensor's own rank is 102).
* AMEn parity is intact: 1D `d = 12` at an attainable tolerance is 12.6 ms over
  3 sweeps, 2D `d = 7+7` is 93 ms.

## 4. Installation

| | |
|---|---|
| wheel | `ttpy-2.0.0.dev0-py3-none-any.whl`, 26 KB, `py3-none-any` |
| `uv pip install` into an empty venv | 0.3 s (warm uv cache) |
| compiled extensions | 0 |
| required dependencies | numpy, scipy, einops |

For comparison: the old ttpy required gfortran, f2py, `numpy.distutils` (removed
in numpy 1.26) and three git submodules, two of them on bitbucket.
