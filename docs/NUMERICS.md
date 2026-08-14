# Numerics: measured limits, regimes and costs

Docstrings say what a function does and what it promises. This file says what
was **measured** — the accuracy floors, the regimes in which a knob behaves the
way it does, the costs that make a default a default, and the failures that
were observed rather than feared.

The split exists because the two ages differently. An API contract is stable;
a measurement belongs to a machine, a BLAS, a version and a seed, and it wants a
regime line next to it, which is more than a docstring should carry.

Every entry states its regime: sizes, `eps`, dtype, seed, and the machine where
that matters. Numbers without a regime are not measurements and do not belong
here either.

Related: [PERFORMANCE.md](PERFORMANCE.md) for throughput, [COMPAT.md](COMPAT.md)
for legacy behaviour, [BENCHMARKS.md](BENCHMARKS.md) for the benchmark suite,
[plans/](plans/) for design notes.

---

## amen_solve

### Attainable residual (the accuracy floor)

The relative residual is bounded below by $\varepsilon_{\text{machine}} \|A\| \|x\| / \|f\|$.
For the QTT Laplacian on $2^{12}$ points with a constant right-hand side that
factor is 6.1e6, so the floor is ~1e-9 in float64.

Measured there, with the residual evaluated in double-double
(`tests/extended.py`; a float64 evaluation of `x.full()` has a 6e-10 noise floor
of its own and cannot see this):

| | relative residual left |
|---|---|
| LAPACK dense solve | 1.531e-10 |
| `amen_solve` | 4.7e-10 |

So `eps = 1e-10` at that size cannot succeed for anyone. The solver reports the
failure rather than claiming the accuracy.

### Is the reported residual honest?

`info.true_res` is computed in TT arithmetic. Against the double-double residual
of the returned cores, QTT Laplacian, float64, seed 0, at `d = 6, 8, 10, 12`
(ratio exact/reported):

| platform | d=6 | d=8 | d=10 | d=12 |
|---|---|---|---|---|
| x86 / OpenBLAS | 1.07 | 0.68 | 0.62 | 0.86 |
| arm64 / Accelerate | 0.93 | 0.78 | 0.50 | 0.87 |

The two platforms differ because the *solve* is BLAS-dependent, not the oracle:
the double-double residual of a d=12 float64 dense solve comes out 1.531e-10
against the 1.52e-10 the old x87 float128 oracle reported. The TT measurement
tracks the truth to within a few tens of percent either way and is never
optimistic by more than that — which is what
`test_reported_residual_is_not_optimistic` pins at a factor of 1.5.

### Why `check_true_res` is off by default

Forming $A x$ multiplies the ranks. Measured on a preconditioned 2D QTT system
(`r_A = 161`, `r_x = 122`): the product reaches rank 19642, one core is 12.3 GB,
peak 91.8 GB — for a number that is only reported. The same product rounded to
1e-12 has rank 256, so none of that size is needed; it exists only long enough
to form the product. `_residual_bytes` estimates this and the caller is told
what it costs rather than silently obeyed.

For an unpreconditioned operator the same product is harmless (rank 4 times
rank 20), which is why this is a warning and not a refusal.

### `max_full_size`: where the dense local solve stops paying

QTT Laplacian $2^{12}$ to `eps=1e-6`: 444 ms at `max_full_size=50` against 8 ms at
the default, and the dense route also reached a *better* residual (1.1e-9
against 1.8e-7) with lower ranks. Raise it when the local systems are hard for
GMRES, not for speed.

### A Cholesky route for the symmetric case was tried and dropped

Measured no benefit over the LU path on the local systems that arise here, at
the cost of a second code path that only applies when symmetry is known.

### Legacy sentinel values

`kickrank=-1` used to mean "no enrichment" and `rmax=0` used to mean "no cap" in
the Fortran-era interface. Both are now rejected: a negative rank and a zero cap
are far more often a bug in the caller than a request.

---

## amen_mv

### `tol` is a per-block threshold, not a certificate

The delivered error is usually at or below `tol` but can exceed it, because the
ALS frames are not the optimal ones and the run starts from a *random* `y0`.

Measured: float64, `d=6`, `n=m=4`, `r_A=r_x=4`, `tol=1e-1`, 15 seeds; the
optimal SVD truncation of the same product is 6.0e-2. Ten seeds land at 7.2e-2,
five at 1.8e-1..2.0e-1 — twice the request. Pin `seed` if a reproducible
accuracy is needed at a loose `tol`.

The spread closes as `tol` tightens: on a product with a decaying spectrum the
delivered error tracks the optimal truncation to three digits from `tol=1e-2`
down (`test_error_tracks_tol_where_truncation_is_active`).

### `kickrank` is cheaper than `nswp`

A sweep raises each TT rank by at most `kickrank + kickrank2`, so reaching rank
`r` from the default rank-2 guess needs at least `(r - 2)/kickrank` sweeps.
Measured on `d=16`, `n=m=8`, `r_A=r_x=12`, `tol=1e-8` (exact ranks 144):

| kickrank | sweeps | time | result |
|---|---|---|---|
| 4 | 36 | 27 s | rank 144 at 1.5e-14 |
| 16 | 9 | 6.8 s | rank 144 at 1.5e-14 |
| 40 | 4 | 6.1 s | rank 144 at 1.5e-14 |

With `nswp=20, kickrank=4` the same problem stops at rank 82 with a relative
error of 0.86 — and warns, loudly, rather than returning it as an answer.

### `renorm='direct'` versus `'gram'`

They are not equivalent: the Gram route squares the condition number of the
block. Measured end-to-end, float64 (reproduced by the two `test_gram_*` cases):

| problem | `direct` | `gram` |
|---|---|---|
| well-conditioned random operator (`d=6`, `n=m=8`, `r_A=r_x=3`, `tol=1e-10`) | 3.3e-15 | 3.7e-15 |
| ill-conditioned (`A = eye([8]*6)`, `x` with a block spectrum spanning 1e-10, `tol=1e-12`) | 6e-16 | stalls at 1e-8 |

In the second case `gram` is seven orders of magnitude worse and does not even
report convergence (`max_dx` plateaus at ~2e-10). Use `'gram'` only for tall
thin blocks of moderate condition number.

### `_gram_svd`: where the orthogonality goes

Orthogonality of `u` degrades like $\varepsilon_{\text{machine}} \mathrm{cond}(a)^2$ rather than
$\varepsilon_{\text{machine}}$. Measured on a dense `a` of size $4096 \times 64$ with a logarithmically
graded spectrum, float64, seed 90 (reproduced by
`test_gram_orthogonality_degrades_with_the_condition_number`):

| $\mathrm{cond}(a)$ | gram, dev. of $u^H u$ from $I$ | direct QR, dev. of $q^H q$ from $I$ |
|---|---|---|
| 1e0 | 9.6e-15 | 2.9e-15 |
| 1e3 | 6.0e-11 | 2.5e-15 |
| 1e7 | 2.3e-03 | 2.8e-15 |

The direct path is flat in $\mathrm{cond}(a)$; the Gram path is not, and at
$\mathrm{cond}(a) = 10^7$ it has no orthogonality left. The *reconstruction* `u diag(s) vh ~ a` stays
at ~1e-15 throughout — it is the orthogonality of the frame, not the
factorization, that is lost, and an ALS sweep depends on exactly that.

### The left-orthogonality check has to follow the precision

A genuinely orthogonal float32 core comes out of a QR at $\|Q^H Q - I\| \sim 5 \times 10^{-8}$,
so a fixed `1e-8` threshold rejects correct input in float32. The threshold is
$10^3 \varepsilon$: 1.2e-4 in float32, 1e-8 in float64, while a core that is not
orthogonal at all misses by O(1).

### What `z` is

`z` is an orthogonal projection of $r = (A x - y)/\|y\|$ onto a
rank-`kickrank` subspace, so $\langle z, r \rangle = \|z\|^2$ — measured to four digits. That
identity is the only property distinguishing a correct `z` from an arbitrary
enrichment subspace, since the accuracy of `y` is blind to it
(`test_z_is_the_projection_of_the_residual`).

---

## cross and multifuncrs

### What `eps` actually buys

`eps` enters `rect_cross` in two places, neither of them an error bound: the
threshold of the stopping rule (relative change between two sweeps) and the
accuracy of the final rounding. The sweeps do not truncate locally.

Measured on $1/(1+t)$, $t = (i+1)/2^d$ on a binary QTT grid, relative error on
2000 held-out points (`n_check`), float64, default `kickrank=5`, achieved
divided by requested:

| d | eps=1e-6 | eps=1e-10 |
|---|---|---|
| 10 | 0.43 | 0.27 |
| 20 | 0.19 | 0.35 |
| 40 | 0.19 | 0.11 |

All six runs reported `history.converged is True` and warned about nothing —
correctly, the stopping criterion *was* met. That the ratio stays below one is a
measurement on one smooth function, not a promise.

### The failure cross cannot see by itself

A feature carried by a few entries. For `funs` equal to 1 at a single point of a
$6^5$ grid and 1e-3 elsewhere, the run returns the constant 1e-3 (relative error
0.995) with `converged=True` and a relative change between sweeps of 1e-15.
Only `n_check` large enough to hit the feature sees it: 3000 points did, 20 did
not. Nothing else can.

`history.err_check` is a Monte Carlo measurement on points nobody looked at; it
resolves what a uniform sample of that size can hit and nothing finer. On the
QTT Coulomb kernel of the tests the *reported* error was 4.7e-4 against a true
2.0e-3 while the ranks were still free to grow.

### Multi-component runs share one budget

With `d2 > 1`, `eps` is a budget for the *stacked* tensor (the legacy
convention): a component carrying a fraction `w` of the joint norm is only
accurate to about `eps/w` relative to itself. Measured on five components
spanning two orders of magnitude, `eps=1e-9`: the smallest (0.7% of the joint
norm) came out at 1.2e-8. Call the method once per component if per-component
relative accuracy is what is wanted.

### Why the local basis is not truncated (`_left_basis`)

Truncating the local basis lets the greedy index sets reach a fixed point far
above `eps`. Measured on $1/(1 + i_1 + \cdots + i_4)$, `n = 8`, `eps = 1e-10`: the
ranks lock and the error stops there.

### `_select_rows`: the enrichment that moves the failure mode

`kickrank2` adds uniformly random extra rows on top of the greedy ones. They
look useless — a volume-maximising pivot is by construction better than a random
one — and they are the only thing measured to move the failure mode where the
greedy reaches a fixed point of its own index sets while a region it never
sampled still carries the error. Every internal indicator then reports 1e-15 and
the answer is wrong at 4e-04.

True relative error on the reproducer of
[plans/cross-approximation.md](plans/cross-approximation.md) (b300, numpy 2.4.6,
`d=6, n=10`):

| `kickrank2` | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0 | 3.82e-04 | 2.84e-10 | 3.82e-04 |
| 2 | 1.81e-06 | 2.84e-10 | 2.84e-10 |
| 4 | 2.84e-10 | 2.84e-10 | 2.84e-10 |

So `k2 = 2` is not always enough and `k2 = 4` was here; the price is 1.4-2.5x
more function evaluations. It is off by default because the mitigation is
partial and problem dependent — a knob to raise when the black box has localized
structure, not a fix that can be defaulted on.

### Non-finite values

The initial probe of `multifuncrs2` evaluates 8 points and refuses a non-finite
value there. The sweep guard in `cross.py` refuses any non-finite value produced
later and names the multi-index. The two are different messages and different
places, and `test_non_finite_only_in_the_sweep_names_the_index` exercises the
second one specifically — see [test tolerances](#test-tolerances-and-oracles).

---

## qtt_ell (BPX preconditioner)

Full derivation and measurements:
[plans/qtt-elliptic-bpx.md](plans/qtt-elliptic-bpx.md).

### Why $C A C$ must never be assembled

Its entries cancel over $4^d$, so rounding the triple product represents a
matrix with error growing like $4^d \varepsilon$: measured 1.3e-10 at `d = 10`,
6.0e-04 at `d = 20`, 4.8e+14 at `d = 50`, while the fused $\Theta$ form stays at
1.4e-14. The rank tells the same story: `round(C A C)` was measured at 96, 135,
185 for `d = 10, 14, 18`, growing with `d`, while $\Theta$ is rank 6 and $B$ rank
17, flat in `d`.

### End to end

$-u'' = 1$ with $u(0) = 0, u'(1) = 0$, AMEn at `eps = 1e-10`, b300 / numpy /
float64, interleaved runs:

| d | unpreconditioned | $B = \Theta^T \Theta$ |
|---|---|---|
| 10 | 30 sweeps, 0.63 s, 4.1e-10 | 7 sweeps, 0.07 s, 2.3e-14 |
| 18 | 30 sweeps, 2.51 s, 8.6e-06 | 7 sweeps, 0.18 s, 8.1e-14 |
| 26 | 30 sweeps, 11.1 s, 4.1e-01 | 7 sweeps, 0.41 s, 1.7e-13 |
| 30 | 30 sweeps, 23.3 s, 1.03 | 7 sweeps, 0.51 s, 1.8e-13 |

At `d = 30` that is $2^{30}$ unknowns, 46x faster, and the unpreconditioned answer
is simply wrong (relative error 1.03) — reported as such by `amen_solve`, which
does not converge and says so.

### `solve_direct_1d`

The direct QTT inversion returns an answer 96.7 % wrong at `d = 30` while
reporting `converged=True` and a residual of 2.6e-09. It solves the equation it
was given; the equation is not the one intended past moderate `d`.

---

## eigb

### Why the residual is measured and not assumed

`ermax`, the stopping indicator, says how much the iteration still *moves*, and
one-site ALS has a standard way of not moving while being wrong: it cannot grow
a rank. The block index is the only enrichment, so with `B = 1` and a rank-1
initial guess the iterate is trapped on the rank-1 manifold, the Ritz value
stops moving to 1e-14, and the run looks converged while being wrong by a factor
of 50. Hence `check_residual` and `block_residuals`.

### The denominator of the eigenresidual

$\|A y_i\|$ (that is, $\lambda_i$) asks every eigenvalue to be accurate
*relatively*, which no eigensolver at truncation accuracy `eps` can deliver near
the bottom of the spectrum: on `qlaplace_dd([10])` with `B = 4` the pairs are
right to 1e-9 absolute, yet $\mathrm{res}/\|A y\|$ is 3.0e-04 because $\lambda_1 = 9.4 \times 10^{-6}$.

$\|A\|_F$ overestimates in exactly the regime that matters — measured 78.4
against $\|A\|_2 = 4.0$ for the same operator, and 41.6 against 4.26 for a
10-site Heisenberg chain. A denominator 10-20x too large desensitizes the
warning by the same factor.

`spectral_norm_estimate` is used instead. It is a *lower* bound, so it can only
make the warning more eager. Measured against a dense $\|A\|_2$ at `its=12`:
0.94-0.97 of the truth on `qlaplace_dd`, 0.97 on Heisenberg, in 25-40 ms — and
both operators have a clustered top of the spectrum, the slow case for power
iteration, so this is close to the worst it does.

### Thresholds have to follow the precision

`sym_tol` defaults to $\sqrt{\varepsilon_{\text{machine}}}$: 1.5e-8 in float64, 3.5e-4 in float32.
A fixed 1e-8 would reject every float32 problem, whose projected local matrices
are asymmetric at the 1e-7 level from rounding alone.

`res_warn` defaults to `sqrt(eps)` floored at $8 \varepsilon_{\text{machine}}$ — the residual a
converged run actually reaches, since the eigenvalue error is quadratic in the
eigenvector error while the residual is linear. A *fixed* threshold is the wrong
shape: 1e-2 left six silent decades between an `eps=1e-8` request and the
warning, and that is exactly where a run whose rank never grows comes to rest.
See [plans/eigenvalues.md](plans/eigenvalues.md).

### `block_residuals` forms the residual vector

Expanding $\|z\|^2 - 2 \lambda \langle z, y \rangle + \lambda^2 \|y\|^2$ instead cancels down to
$\sqrt{\varepsilon} \|A y\|$ and would report 1e-9 where the truth is 1e-16.

---

## ksl

### The splitting order is only visible against the projected flow

The order is verified against the dense solution of the *projected* ODE
$y' = P_{T_y M} A y$ — the equation this integrator discretizes — where the
observed orders are 1.00 and 2.00. Measured against `expm(tau A) y0` instead,
the two schemes are indistinguishable: either the manifold contains the
trajectory and both are exact, or it does not and the tau-independent modelling
error hides the splitting error.

### `err_est` and the error part company when the flow grows

The Krylov error estimate is normalized by the norm of the *input*. On a
strongly non-normal operator with $\|\exp(A) x\| / \|x\| \sim 10^5$, asking for
`tol = 1e-10` delivers 5.5e-8 relative to the result and reports
`err_est = 6.4e-2`. Inside KSL, where $\tau \|B\|$ is small, the two agree.

### `step_error_est` is an indicator, not a bound

It is a first-order, one-point estimate — the tangent defect at the end point,
times `tau`. Measured on `diag_ksl` with a rank-3 guess against the exact
elementwise exponential:

| tau | predicted | actual |
|---|---|---|
| 0.01 | 4.73e-02 | 4.95e-02 |
| 0.1 | 1.13e-01 | 2.16e-01 |
| 0.3 | 6.64e-02 | 5.39e-01 |

Eight times optimistic at the largest step, and *non-monotone*, because the
defect is read at one point of a trajectory that has already left the manifold.
It warns in every one of those cases, so nothing is silent.

---

## completion

### A small `fit` is not a solved problem

With a nonzero `underdetermined_slices` count a run can reach `fit = 1e-31` and
still be 100 % wrong away from the samples. Measured on a rank-4 tensor of shape
$6 \times 6 \times 6$ fitted from 38 samples (144 parameters): `fit = 4.9e-31`,
`converged = True`, `determined = False`, relative error against the truth 5.8
(i.e. off by 580 % of the norm of the truth). The run warns.

`converged and determined` is the pair that means "solved"; `converged` alone
only means "reproduces the samples".

### ALS here is not globally convergent

Measured on a rank-2 tensor of shape $8 \times 8 \times 8 \times 8$ (96 parameters) recovered at rank
2 with `alpha = 0`:

| distinct samples | outcome over 6 random starts |
|---|---|
| ~820 | 4 reached `fit ~ 1e-15`, 2 stalled around 1e-1 |
| ~1330 | 6 of 6 reached `fit ~ 1e-15` |

`info.converged` and `info.stop_reason` report which of the two happened, so the
remedy (more samples, another seed, a better `x0`) is a decision the caller can
make. A stalled run is never returned as if it were a solution.

---

## riemannian

### Why `project` refuses a rank-deficient point

At a rank-deficient representation the closed-form projector still returns a
Hermitian idempotent of the right trace — it just projects onto a strictly
larger space than the caller asked for, so no invariant can see the mistake.
Measured on a rank-1 tensor written with TT ranks `(1, 2, 2, 1)`, `d = 3`,
`n = 4`, float64: 31 % relative error against the dense tangent projector, with
idempotence 1.6e-16.

### Why `projector_splitting_add` does not refuse one

The splitting only needs the frames themselves, not the space they are supposed
to span exactly, and exactness was measured to hold at such a point: 1.3e-15
relative, same rank-1 tensor and ranks as above.

### Nor does `tt_qr`

`bk.qr` still returns cores with orthonormal columns and $X = Q R$ still holds
to roundoff; only the columns of $Q$ corresponding to a zero on the diagonal of
$R$ are arbitrary. Verified on the same tensor: orthogonality and reconstruction
both hold to 1e-15.

---

## Backends and precision

### The MPS device is float32-real only

Measured on torch 2.13 / Apple silicon:

| capability | status |
|---|---|
| float64 | absent entirely — hence no complex128 either |
| complex64 storage | works |
| `linalg.qr`, `linalg.lu_factor` on complex64 | refused ("MPS currently supports float32 only") |
| `linalg.eigh`, `linalg.eig`, `linalg.lstsq` | not implemented |
| `linalg.qr`, `svd`, `solve`, `lu_factor` in float32 | work (`svd` silently falls back to the CPU) |

`tests/conftest.py` is the single place that asks what the device supports; the
capability tables there are this measurement.

### What a float32 backend can solve

`kappa(qlaplace_dd([d]))` is about $0.4 \cdot 4^d$ and the reachable residual is
$\kappa \varepsilon$. At `d=8` (kappa 1.3e4) float32 stalls at 1e-3 no matter how many
sweeps it is given — on the numpy backend as well as on MPS, so it is the
precision talking and not the device. Measured with `eps=1e-5` requested:

| d | kappa | float32 |
|---|---|---|
| 3 | 3.2e1 | converges, residual 1.3e-6, error 3.5e-7 |
| 4 | 1.2e2 | converges, residual 8.1e-6, error 6.3e-7 |
| 5 | 4.4e2 | marginal |
| 6 | 1.7e3 | does not converge |

### Host data adopts the backend's width

A backend's declared dtype is a statement about what lives on it, so incoming
host data takes that width while staying in its domain (real stays real). The
alternative — keeping whatever dtype numpy handed over — put float64 on a
float32 backend: twice the memory and traffic on CUDA, and on MPS `tt.ones`
could not place its own core on the device at all.

### `norm()` is a sweep, not $\sqrt{\langle x, x \rangle}$

The contraction is 12x faster on a rank-141 tensor (2.4 ms against 29.9 ms), but
in the TT format the inner product is a contraction whose intermediates cancel,
so it loses digits exactly where a norm matters — on a tensor built as a
difference. Tried, measured, reverted: 12 tests caught it, and on a convection
problem it reported a residual of 1.25e-06 where the true one was below 1e-06,
which turns a converged run into a failed one.

### `_binary_plan` caching

Re-deriving einsum subscripts on every call cost 0.42 s of a 1.75 s AMEn solve
over 18425 calls.

### `backend_of` on the hot path

Measured on one KSL step (`d=6`, `n=2`, rank 4): 508 norm evaluations go through
it, which is why it is a dictionary lookup and not an isinstance chain.

### `norm()` must not end in `.item()`

Returning a Python float detaches the value from the autograd tape: a functional
containing $\|x\|$ then differentiates to a silently wrong gradient — no error,
no NaN, just a missing term. Measured on a completion functional whose
$\|x - b\| = 29.4$ contributed exactly zero, `max |AD - finite differences| =
7.03e-01`. Reverting it was measured as well: 12 tests caught it, and on a
convection problem the solver reported a residual of 1.25e-06 where the true one
was below 1e-06.

---

## Discretizations

### `qlaplace_dn` and the `'DN'` case

`'DN'` — Dirichlet at 0, Neumann at 1 — is the only combination with exactly
$2^l$ degrees of freedom on every level, so it is the one the multilevel
prolongations of [BK20] are built for. Measured: $M^T M$ with `M = qdiff(d)`
equals $\mathrm{tridiag}(-1, 2, -1)$ with the last diagonal entry 1 to 2.0e-15 at
`d = 3`, and its smallest eigenvalue matches the analytic
$4 \sin^2(\pi / (2(2N+1)))$ to 1e-15. TT ranks 4 (`D = 1`) and 5 (`D > 1`,
measured at `D = 2, 3`).

### `qtt_fem.placement`: the last row of the identity

The fake element slot $e = n-1$ must be dropped by both corner operators. The
shift does so on its own; the identity needs its last row zeroed. With a full
identity the fake elements deposit their `(0, .)`-corner contributions on the
last row of nodes — invisible under an all-Dirichlet mask, and corrupting
exactly the interface nodes of a glued multi-patch problem. Measured on
Markeeva's triangle: with the full identity the coupled energy *falls* under
refinement (0.2457, 0.1981, 0.1679 at `d = 2, 3, 4`) instead of approaching
0.3404 from above.

### `iga.geometry_field`: give it an analytic Jacobian

Without `jac` the Jacobian is taken by central differences, whose cancellation
error is `eps_machine / fd_step` — 1e-10 at the default step — so a component of
`R` that is *exactly* zero comes back at that level instead of at zero. Measured
on the ring of `examples/iga_ring.py`: with differences, `R01` survives the
screen and `tt.cross` fits the noise at TT rank 46; with an analytic `jac` it is
screened and the assembly is three components instead of four.

---

## Test tolerances and oracles

### The extended-precision oracle

Several verification tests measure a residual smaller than the rounding error of
the obvious way to measure it: on the `d = 12` QTT Laplacian
$\|A\| \|x\| / \|f\|$ reaches 6e6, so a float64 evaluation of $A x - f$ carries a
relative noise floor near 6e-10 — the size of the residual being judged.

That accuracy came from `np.longdouble`, which is 80-bit x87 on Intel and an
alias for float64 on Apple silicon. On ARM the oracle therefore had no extra
precision at all, and two tests failed for a reason unrelated to the solver.
`tests/extended.py` replaces it with double-double (~106 bits of significand,
more than x87's 64) built out of float64 operations only, so it is identical on
every platform. Validation: the `d=12` float64 dense-solve floor comes out
1.531e-10 against the 1.52e-10 the x87 oracle used to report.

### GPU test calibration

`tests/conftest.py` carries one calibration row per working precision. The
float64 row is what the tests were originally written with; the float32 row is
measured on MPS with roughly a decade of headroom (worst observed parity 4e-7 on
a $4 \times 4 \times 4 \times 4$ round-trip). `QLAPLACE_D` comes from the float32 table
[above](#what-a-float32-backend-can-solve).

### Determinism of the non-finite sweep test

`test_non_finite_only_in_the_sweep_names_the_index` needs the initial probe to
miss a pole that the sweep then hits. Placing the pole at the global maximum of
the tensor makes that a question about maxvol's pivots, hence about LAPACK: on
Accelerate the sweep evaluated 7670 points and none of them was the pole.

Placing it on a whole hyperplane `i_2 == 37` of a 64-wide mode makes it
structural instead — the sweep enumerates the full range of every mode it
updates, so it cannot miss whatever the pivots are, while the probe draws 8
points from a seeded PCG64 and at 1/64 per point misses. Measured: 988
evaluations, 14 of them non-finite, the first reported at `[0, 0, 37, 5, 2]`.
`K = 17` would be hit by the probe — the probe is deterministic, not lucky.
