# RIEM: Riemannian optimization and automatic differentiation on TT manifolds

Implementation spec for `tt.algs.riemannian` and a new `tt.algs.autodiff`.

Sources actually read, and what each one is:

* **[NRO22]** A. Novikov, M. Rakhuba, I. Oseledets, *Automatic differentiation
  for Riemannian optimization on low-rank matrix and tensor-train manifolds*,
  arXiv:2103.14974v2, 25 pp. (published as SIAM J. Sci. Comput. 44(2):A843–A869,
  2022 — the version at hand is the arXiv preprint and carries no journal
  header, so the pagination below is the preprint's). **This is the automatic
  differentiation paper**, read in full (text extracted with `pypdf` on b300;
  figures were not rendered, but the paper's quantitative content is in tables,
  which extracted cleanly). Contents: the auxiliary function `g = f ∘ T_X` whose
  ordinary gradient with respect to the *delta cores* is the Riemannian gradient
  (§4.2 matrices, §5.2 tensors, Alg. 5.1/5.2); the complexity statement
  `O(F + d n r³)` (Prop. 5.2); the `stop_gradient` trick that turns
  `P_X B⁻¹(A X − F)` into a Riemannian gradient (§5.3); the approximate
  Riemannian Hessian-by-vector product (§6, Alg. 6.1/6.2); and CPU/GPU timings
  against a hand-written baseline in T3F (§7, Tables 2–3).
* **[RNO19]** M. Rakhuba, A. Novikov, I. Oseledets, *Low-rank Riemannian
  eigensolver for high-dimensional Hamiltonians*, J. Comput. Phys. **396**
  (2019) 718–737. Read here for the *manifold machinery only* — §4.1 (tangent
  representation and the `S_k` stack), §4.2 (projection and its cost), §4.3
  (the cheap tangent inner product, their eq. (22)), §4.4 (TT-SVD retraction),
  §4.5 and eq. (24)–(25) (the preconditioner as a sum of rank-1 TT-matrices).
  The *eigensolver* content of this paper is owned by
  `docs/plans/eigenvalues.md` §2.2 and is not duplicated here.
* **[BK20]** M. Bachmayr, V. Kazeev, *Stability of Low-Rank Tensor
  Representations and Structured Multilevel Preconditioning for Elliptic PDEs*,
  Found. Comput. Math. **20**:1175–1236, 2020. Identified, first page read only.
  This is the **BPX / representation-conditioning** paper and belongs to the
  separate planned BPX-elliptic spec, not to this one. It is referenced here
  once, in §6, as the source the preconditioner interface must be negotiated
  with.
* **[CL22]** G. Ceruti, C. Lubich, *An unconventional robust integrator for
  dynamical low-rank approximation*, BIT Numer. Math. **62**:23–44, 2022, and
  **[CKL23]** G. Ceruti, J. Kusch, C. Lubich, *A parallel rank-adaptive
  integrator for dynamical low-rank approximation*, arXiv:2304.05660v1.
  Identified, first page read only. Both belong to `docs/plans/bug-integrator.md`
  and are already fully digested there. They are **not** part of this spec.
* **[LOV15]** C. Lubich, I. V. Oseledets, B. Vandereycken, *Time integration of
  tensor trains*, SIAM J. Numer. Anal. 53(2):917–941, 2015 — not among the
  uploads; cited through the docstrings of `tt/algs/riemannian.py` and
  `tt/algs/ksl.py`, which implement it.
* **t3f** (https://github.com/Bihaqo/t3f), audited from a fresh clone of the
  default branch `develop`, HEAD `8096b38`, dated **2021-04-06**, v1.2.0.
  Network was reachable; the audit is **verified from source**, not from docs
  alone. §7 below.

Everything marked **measured** was produced by a throwaway prototype run on
**b300 (hostname Planck, 256 cores, 2 TB RAM, 8 GPUs)** before this document was
written, from `~/work/ttpy-modern/ttpy2` with `.venv/bin/python` (python 3.12,
numpy 2.4.6, scipy 1.18.0, torch 2.13.0+cu130). Prototypes are in
`~/work/ttpy-modern/scratch-riem/` on b300 (`p1_autograd.py`, `p2_tangent.py`,
`p3*_completion.py`, `p4_precond.py`, `p5_adverse.py`, `p6_gpu.py`,
`p7_timing.py`, `p8_deficient.py`, `p9_norm.py`, `p10_ssot.py`). Nothing below
is an estimate; where a number is missing it says "not measured". Every table
states its regime. Wall-clock numbers are medians of 3 or 5 repeats where the
table says so and single runs otherwise.

**Cross-spec decisions live in `docs/plans/ROADMAP.md`**, not here. Where this
spec and one of `bug-integrator.md`, `eigenvalues.md`, `qtt-elliptic-bpx.md` ask
for the same function, the reconciled signature and its owner are recorded there
(§2), together with the dependency graph (§1), the preconditioner contract (§3),
the milestone order (§4) and the consolidated open questions (§6). This spec's
§8(a) `project_delta` signature **won** the conflict with `eigenvalues.md`
§4(b), which now points here; its §8(d) `frames` won the conflict with
`bug-integrator.md` §4(b), at the price of a `check_rank=False` flag. §6.4
item 1 was amended: the "list of rank-1, callable is the slow path" framing does
not hold for BPX. §1.4's defect is **already fixed**, commit `670d84d`.

---

## 1. What we have now

### 1.1 What `tt/algs/riemannian.py` is

Three public functions, 432 lines, no `numba`, both backends.

```python
project(X, Z)                                  # P_{T_X M} Z, Z a vector or a LIST
projector_splitting_add(Y, delta)              # retraction Y <- Y + delta at fixed rank
tt_qr(X, left_to_right=True)                   # (Q, R), Q with orthonormal cores
cores_orthogonalization_step(cores, dim, left_to_right=True)   # one QR step, in place
```

`project(X, Z)` implements the closed form of [LOV15] Thm 3.1,

```
P_X = sum_{k=1..d} P_{<k} (x) I_k (x) P_{>k}  -  sum_{k=1..d-1} P_{<=k} (x) P_{>k},
```

as one sweep: right-to-left orthogonalization of `X` giving the right frames
`V_k` and the right interfaces `rhs[k] : (r_z(k), r_x(k))`; then left-to-right
QR giving the left frames `U_k`, and at each site the *delta core*

```
proj    = lhs @ Z_k                     # (r_{k-1}, n_k, rz_k)
lhs_new = U_k^H proj                    # gauge part
delta   = (proj - U_k lhs_new) @ rhs[k+1]      # (r_{k-1}, n_k, r_k)
```

which is exactly `δG_k` of [RNO19] eq. (20)–(21) and `Ṡ_k` of [NRO22] eq. (5.9).
The result is assembled into the rank-`2r` stack of [RNO19] §4.1 / [NRO22] (5.7)

```
S_1(i) = [δG_1(i)  U_1(i)],   S_k(i) = [[V_k(i), 0], [δG_k(i), U_k(i)]],
S_d(i) = [V_d(i); δG_d(i)]
```

and returned as a `tt.vector` — **the delta cores are computed and thrown away**.
That is the single most consequential gap in the module (§2.2, §8(a)).

Three things it does that are worth keeping verbatim:

* `Z` may be **a list of `tt.vector`s**, and then `P_X(Σ_i Z_i)` is computed
  without ever forming the sum. This is precisely the structure
  [RNO19] eq. (25) needs for a preconditioner given as a sum of rank-1
  TT-matrices, and §6 below uses it at `ρ_B = 61` terms without modification.
* It **refuses a rank-deficient `X`** (`_left_step_checked`), because at a corner
  of the manifold the closed form still returns a Hermitian idempotent — of the
  wrong space. The docstring records 31 % relative error against the dense
  projector on a rank-1 tensor written with ranks `(1,2,2,1)`.
* Everything is written for the Hermitian inner product, so complex input works.

`projector_splitting_add(Y, delta)` is the Lie–Trotter projector splitting
([LOV15] §4.2): one left-to-right sweep of K steps (absorb `delta` seen through
the current frames) and S steps (subtract what the next K step double-counts).
It keeps the ranks of `Y` exactly and is *exact* when `Y + delta` happens to
have rank `r(Y)`.

### 1.2 Measured: it works

Regime: b300, numpy, float64.

The 34 existing tests touching this module (`tests/test_ports.py`,
`tests/test_verify_ports.py`, selected by `-k "project or riemannian or tt_qr"`)
pass in **2.52 s** wall.

Re-verified independently (`p2_tangent.py`, §(a)) against a dense
`N x N` tangent projector assembled from `numpy.linalg.svd` of the dense
unfoldings of `X` — i.e. from `X` alone, not from this code:

| `d, n` | ranks of `X` | `r_z` | `‖P_tt z − P_dense z‖ / ‖P z‖` | idempotence of the oracle |
|---|---|---|---|---|
| 3, 4 | `1,2,2,1` | 3 | 1.52e-15 | 6.01e-16 |
| 4, 3 | `1,2,2,2,1` | 2 | 1.18e-15 | 9.68e-16 |
| 4, 3 | `1,3,3,3,1` | 5 | 8.24e-16 | 6.53e-16 |
| 5, 2 | `1,2,2,2,2,1` | 4 | 9.67e-16 | 4.61e-16 |
| 3, 5 | `1,4,4,1` | 6 | 7.34e-16 | 7.79e-16 |

The list form was exercised at `ρ_B = 41…61` summands inside a working
preconditioned eigensolver (§6, table 2) and produced eigenvalues correct to
2e-09; the loud refusal at a rank-deficient point was reproduced
(`p5_adverse.py` §(B)):

```
project: unfolding 1 of X has TT rank 2 but numerical rank 1
(smallest/largest singular value = 0.00e+00 <= 4.44e-16). ...
```

**Cost of `project`**, numpy backend, float64, median of 5, b300:

| `d` | `n` | `r` | `r_z` | ms | `d n r r_z²` |
|---|---|---|---|---|---|
| 20 | 2 | 10 | 10 | 2.72 | 4.0e+04 |
| 40 | 2 | 10 | 10 | 5.60 | 8.0e+04 |
| 20 | 2 | 20 | 20 | 3.93 | 3.2e+05 |
| 20 | 2 | 40 | 40 | 8.78 | 2.6e+06 |
| 20 | 4 | 20 | 20 | 4.86 | 6.4e+05 |
| 20 | 2 | 20 | 40 | 3.95 | 1.3e+06 |
| 20 | 2 | 20 | 80 | 4.26 | 5.1e+06 |

The theoretical cost is `O(d n r r_z²)` ([RNO19] §4.2). The last three rows hold
`d, n, r` fixed and multiply `r_z` by 2 and 4, i.e. the model predicts `x4` and
`x16` — and the measured time moves 3.93 → 3.95 → 4.26 ms. **At QTT sizes
`project` is python-dispatch-bound, not flop-bound**: the time is linear in `d`
(2.72 → 5.60 ms for `d = 20 → 40`) and almost independent of the rank, i.e. it
is `d` times a constant of about 135 µs per site, which is roughly six of the
~7 µs numpy calls that `docs/plans/bug-integrator.md` §6 measures, plus a QR.
The consequence for §5's algorithm ordering: on a QTT problem the cost of a
Riemannian step is dominated by the number of `project` calls, not by their
size, so batching (or the torch/GPU path of §4.4) matters more than any flop
count.

### 1.3 What it does not have

* No delta/gauge representation exposed (§2.2). Without it the cheap tangent
  inner product of [RNO19] eq. (22) is impossible and every Gram matrix costs
  `O(d n r³)` instead of `O(d n r²)`.
* No vector transport, no retraction other than the projector splitting, no
  fused `P_X (A Z)`, no Riemannian gradient of anything, no optimizer.
* No autodiff.
* `projector_splitting_add` is a *first-order* retraction only; there is no
  second-order retraction and no rank adaptation anywhere.

### 1.4 A defect (FIXED): `bk.norm` silently killed the autograd graph on torch

Fixed in the commit that landed this document; the paragraphs below describe the
code as it was, and the fix is recorded at the end of the section. Regression
tests: `tests/test_autograd.py`, which fails 3 of its 5 cases against the old
code (verified by restoring `.item()` on b300 and re-running).

`tt/backend.py:297`:

```python
class TorchBackend(Backend):
    def norm(self, a):
        return self.torch.linalg.norm(a).item()      # <-- python float
```

`.item()` detaches. `_ops.norm` (`tt/core/_ops.py:392`) returns `bk.norm(out[0])`
and `vector.norm` (`tt/core/vector.py:296`) returns `_ops.norm(self.cores)`, so
**`x.norm()` on the torch backend is a python float and contributes exactly zero
to any gradient**. When the norm is the whole objective the failure is loud
(`AttributeError: 'float' object has no attribute 'backward'`). When it is one
term of a sum it is silent and the answer is wrong.

Exact reproducer (`p9_norm.py`, b300, torch 2.13.0+cu130, CPU, float64,
`d=4, n=3, r=2`, seed 0):

```python
import numpy as np, torch, tt
from tt import backend as bk
from tt.core.vector import vector
bk.set_backend("torch", device="cpu", dtype="float64")
g = np.random.default_rng(0)
rk = [1, 2, 2, 2, 1]
cores = [torch.tensor(g.standard_normal((rk[k], 3, rk[k+1])),
                      dtype=torch.float64, requires_grad=True) for k in range(4)]
x = vector.from_list(cores)
b = vector.from_list([torch.tensor(g.standard_normal((rk[k], 3, rk[k+1])),
                                  dtype=torch.float64) for k in range(4)])
f = tt.dot(x, x) + (x - b).norm()      # a differentiable term PLUS a norm term
f.backward()                            # no error, no warning
```

Output on b300:

```
type(f) = Tensor  requires_grad = True
max |AD grad - FD grad| = 7.031e-01     (central differences, h = 1e-6)
||x-b|| = 29.39962312767259
```

`f.requires_grad` is `True` — the *other* term keeps the graph alive — so
nothing in the run says that a term worth 29.4 in the objective contributed
nothing to the gradient. This is the "hidden unknown" the project forbids, and
it was the one blocking defect for §4.

**The fix, and what it dragged with it.** `TorchBackend.norm` now returns the
0-d tensor. That also removed an asymmetry that was there all along — numpy's
`np.linalg.norm` returns a 0-d numpy scalar, so the two backends had been
disagreeing about the return type of a public method.

Two consequences had to be handled, and both are in the same commit:

* *A tensor threshold met numpy.* Truncation tolerances are built from norms, so
  `delta` arrived at `_ops.chop` as a 0-d cuda tensor and raised from inside
  `tail < eps**2` (17 tests). `chop` now coerces its threshold to a float on the
  way in, next to the `bk.to_numpy` it already applied to the singular values.
  Detaching there is correct rather than merely convenient: `chop` returns a
  *rank*, a discrete choice, and nothing differentiable passes through it.
* *Scalar multiplication stopped accepting the result.* `x * (1 / x.norm())` is
  what every algorithm writes, and `vector.__mul__` tested `isinstance(other,
  Number)`, which a 0-d tensor is not. Keeping the tape alive through `||x||`
  and then breaking it at the scaling would have been pointless, so
  `bk.is_scalar` / `bk.scalar_dtype` were added as the single owner of "may this
  thing scale a TT tensor", and `vector.__mul__/__rmul__/__truediv__/__add__`
  and `matrix.__mul__/__rmul__` now ask it. `_ops.scale` takes its dtype opinion
  from `scalar_dtype`, which returns `None` for a plain Python real so that a
  float32 tensor is not dragged up to float64 by a literal.

718 tests pass on b300 (numpy + torch, CUDA present).

**Fix, one owner.** `TorchBackend.norm` must return the backend scalar (a 0-d
`torch.Tensor`), and callers that want a python number must say so. Audited:
`bk.norm` is called 48 times inside `tt/`, of which 39 already wrap the result
in `float(...)`; the 9 that do not are
`_ops.py:106,168,258,405,407`, `core/tools.py:676,742,818`, `algs/eigb.py:263`,
and every one of them uses the value in arithmetic that works unchanged on a 0-d
tensor. `_ops.norm` and `vector.norm` then become differentiable for free.
The compatibility risk is that `x.norm()` stops being a `float` for torch users;
`float(x.norm())`, `f"{x.norm():.2e}"` and every comparison keep working, and
the numpy path is unchanged. The alternative — leaving `bk.norm` alone and
adding a second, differentiable norm — creates two owners of one truth and is
rejected on that ground.

Note that `_ops.norm`'s docstring explains why it is an orthogonalization sweep
rather than `sqrt(<x,x>)` (cancellation on a difference, 12 tests). That
argument is unaffected: the QR sweep is differentiable in torch (measured, §4.1).

### 1.5 A second, structural defect: two owners of the tangent projection

`tt/algs/ksl.py:347` `tangent_defect(A, y)` computes `‖(I − P_{T_y M}) A y‖` by
its own inlined sweep, using the same decomposition as `project` but never
calling it. That is a second implementation of the same mathematics.

Measured (`p10_ssot.py`, b300, numpy, float64): `tangent_defect`, the quantity
`sqrt(‖z‖² − Σ_k ‖δG_k‖²)` computed from the delta cores, and the direct
`‖z − project(y, z)‖` agree to **0.0, 0.0 and 1.54e-16** relative on three
problems (`d,n,r,R = 6,4,3,2`; `10,2,4,3`; `8,3,2,2`, `z = A y`). They are the
same object. Once `project_delta` exists (§8(a)), `tangent_defect` must be

```python
deltas, _, _ = project_delta(y, matvec(A, y))
proj2 = sum(inner(dG, dG) for dG in deltas)
```

which is also *cheaper*, because it drops the second sweep `tangent_defect`
currently pays for. Same for `cores_orthogonalization_step`, which duplicates
`_localops.left_orthogonalize` + a push.

There is a third, subtler duplication: `projector_splitting_add(Y, delta)` and
`ksl._sweep_forward` are the same Lie–Trotter sweep, one solving the local
equations exactly (the increment is constant) and one with
`expmv_krylov`. This one is *not* worth merging — the constant-increment case
has a closed form and paying for a Krylov exponential to get it would be
absurd — but it should be documented as a deliberate second implementation with
a shared test (`projector_splitting_add(Y, tau*A@Y)` vs `ksl(A, Y, tau)` agreeing
to `O(tau²)`), not left as an accident.

---

## 2. The manifold machinery, in our conventions

Conventions throughout: TT cores `G_k : (r_{k-1}, n_k, r_k)`; TT-matrix cores
`A_k : (R_{k-1}, n_k, m_k, R_k)` with the **row index first**; merged vector mode
`s = i + n*j`; mode 1 is the fastest index; `ML(G) : (r_{k-1} n_k, r_k)` is the
left matricization. Code is 0-based, the mathematics below is 1-based.

### 2.1 The tangent space and its parametrization

`M_r = { X : rank_TT(X) = r }` is a smooth embedded submanifold of
`R^{n_1 x ... x n_d}` of dimension `Σ_k r_{k-1} n_k r_k − Σ_{k<d} r_k²`. At
`X ∈ M_r` write the two orthogonal representations

```
X = U_1 U_2 ... U_{d-1} S_d          # left-orthogonal:  ML(U_k)^H ML(U_k) = I
X = S_1 V_2 ... V_d                  # right-orthogonal: MR(V_k) MR(V_k)^H = I
```

Both are produced by the sweeps already in the codebase:
`_ops.orthogonalize(cores, center=0)` gives `S_1, V_2..V_d`, and
`_localops.left_orthogonalize` applied left to right gives `U_1..U_{d-1}`.
`project` builds both in one pass and this is the state every routine in §8
should share.

A tangent vector is parametrized by **delta cores** `δG_k : (r_{k-1}, n_k, r_k)`,

```
ξ = δG_1 V_2 ... V_d + U_1 δG_2 V_3 ... V_d + ... + U_1 ... U_{d-1} δG_d
```

with the **gauge condition** ([RNO19] eq. (21), [NRO22] eq. (5.6))

```
ML(δG_k)^H ML(U_k) = 0,   k = 1, ..., d-1
```

(no condition on `δG_d`). The gauge makes the parametrization a bijection, and
it is what makes the tangent inner product cheap:

```
<ξ, η> = Σ_{k=1..d} <δG_k^ξ, δG_k^η>_F                    [RNO19] eq. (22)
```

Measured (`p2_tangent.py` §(b), b300, numpy, float64): the deltas produced by
the `project` sweep satisfy `max |ML(δG_k)^H ML(U_k)|` = **7.9e-16** (d=4,n=3,r=2),
**7.1e-15** (d=6,n=4,r=3), **4.3e-14** (d=10,n=2,r=4); rebuilding the rank-`2r`
tensor from `(U, V, δG)` reproduces `project`'s output to **2.9e-16 … 9.5e-16**
relative; and the cheap inner product agrees with the full TT contraction
`tt.dot(P Z_1, P Z_2)` to **3.6e-16, 6.4e-16, 8.1e-16** relative on the same
three problems. So eq. (22) holds in our conventions, exactly as written.

The rank-`2r` stack is the `S_k` block form quoted in §1.1.

### 2.2 Projection `P_{T_X M}` — the workhorse

Implemented (§1.1), verified (§1.2), cost `O(d n r r_z²)` in theory and
dispatch-bound in the QTT regime (§1.2, table). What is missing is the
**delta-returning** variant, `project_delta` (§8(a)). Everything else in this
document depends on it:

* the Gram matrix of `b` tangent vectors costs `O(b² d n r²)` from deltas and
  `O(b² d n r³)` from the assembled tensors ([RNO19] §4.3) — the factor that
  makes their LOBPCG affordable;
* summing tangent vectors at the same point is *free* in the deltas
  (`δG^{αξ+βη} = α δG^ξ + β δG^η`) and costs a rank-`4r` add plus a rounding in
  the assembled form. t3f has exactly this function (`add_n_projected`, §7);
* the Riemannian autodiff of §4.3 produces deltas natively and consumes deltas
  natively; going through the assembled tensor at every step is the difference
  between `O(F)` and `O(F + d n r³)`.

The `Z` argument must keep accepting a list. A fused `project_matvec(X, A, Z)`
(project the matvec without forming it) is [RNO19] §4.2's `O(d n r r_y R (r_y + n R))`
against the naive `O(d n r_y² R² (r + n))`; t3f has it (`riemannian.project_matmul`).
It is *not* required for the recommended first build (§5), because §4.4 measures
the autodiff route beating our current naive route by up to 17x on CPU already.

### 2.3 Retraction

Three candidates, in the order we should adopt them.

1. **TT-SVD retraction** `R_X(ξ) = round(X + ξ, rmax=r)` ([RNO19] §4.4;
   quasi-optimality and the retraction property are Absil–Oseledets and
   Steinlechner). Cost `O(d n r³)` with `r' = 3r` before truncation. It is
   available today as `(X + xi).round(0.0, rmax=r)` and it is what every
   measurement in §5 and §6 used. **Recommended default.**
2. **Projector splitting** `projector_splitting_add(X, ξ)` (exists). One sweep,
   no SVD, exact when `X + ξ` is already of rank `r`. It is a valid
   *first-order* retraction. Cheaper than (1) by the SVD, but it is not
   quasi-optimal and it does not tell you how much it discarded.
3. **Second-order retractions.** Not implemented, not measured, and I could not
   verify a formula for the TT case from the sources at hand ([NRO22] cites
   Absil–Oseledets for the survey but gives none). Deferred to §13 Q4.

**Honest counterweight.** The fixed-rank manifold is not closed: if the solution
has rank *below* `r`, the minimizer sits on the boundary. Measured
(`p8_deficient.py`, b300, torch CPU float64, `d=6`, `n=10`, completion,
`|Ω| = 100 × dof`, 400 Riemannian GD steps, TT-SVD retraction):

| true rank | manifold rank | final `f` | `min_k σ_min/σ_max` of the unfoldings, start → end |
|---|---|---|---|
| 1 | 3 | 1.74e-02 | 5.90e-01 → 2.89e-01 |
| 2 | 3 | 2.94e-02 | 6.14e-01 → 2.21e-01 |
| 3 | 3 | **3.71e-30** | 5.75e-01 → 3.81e-01 |

Two facts, and the second is the surprising one. (i) Overestimating the rank
costs 28 orders of magnitude of residual on this problem — the fixed-rank
manifold is *not* forgiving of a rank guess that is too large, which is the
opposite of the usual intuition. (ii) The iterate did **not** approach the
rank-deficient boundary in 400 steps (the smallest relative singular value stays
at 0.2–0.3), so `project`'s refusal did not fire and there was no crash. The
failure mode is a stalled residual, not an exception. A `rgd` that does not
report its residual would therefore be silent here, and §10 test 5 pins that.

### 2.4 Vector transport

The cheapest valid transport on an embedded submanifold is
`T_{X→Y}(ξ) = P_{T_Y M} ξ` — one `project` call, rank stays `2r`. It is what
[NRO22]'s reference implementation and geomCG use, and it is what §5's CG used.

In the delta representation there is no shortcut: the deltas at `X` mean nothing
at `Y`, so the transport is assemble → `project_delta` at `Y`. Cost
`O(d n r (2r)²) = O(d n r³)`.

Measured consequence of *forgetting* it: a first attempt at Riemannian CG added
`β · η_{k-1}` without re-projecting (a rounding to 1e-13 after the add does not
help — the sum of two tangent vectors at *different* points has no low-rank
structure to find). The direction's rank then grows by `2r` per iteration until
the mode sizes bound it, and 300 iterations of completion at `d=6, n=10, r=3`,
`|Ω| ≈ 4190` took **191 s** against **3.2 s** for the same 300 iterations with
the transport inserted — a 60x self-inflicted cost, and the answer was no
better. Recorded because it is the single easiest way to get this wrong.

### 2.5 Riemannian gradient from a Euclidean gradient

`grad f(X) = P_{T_X M} ∇f(X)` ([NRO22] eq. (3.2)). Three ways to get it, and the
choice is the whole content of §4:

* (i) form `∇f(X)` as a TT and call `project` — correct, and disastrous when
  `∇f` has large rank. Measured: for the completion functional with
  `|Ω| = 200` samples at `d=6, n=4`, the Euclidean gradient `2 P_Ω(X − A)` has
  TT rank **59** after `tt_svd` at 1e-13, against `r = 3` for `X`. In general
  its rank is `|Ω|`.
* (ii) hand-written adjoints per functional (what [RNO19] does, what t3f's
  "improved" baseline does);
* (iii) **[NRO22]'s autodiff**: differentiate `g = f ∘ T_X` with respect to the
  delta cores at `(S_1, 0, ..., 0)` and enforce the gauge. Never forms `∇f`.
  Measured to be correct to 1.9e-15 on exactly that completion functional
  (§4.3).

### 2.6 What is shared with `ksl.py` — one owner per truth

| truth | today | after |
|---|---|---|
| left/right orthogonalization of a core | `_localops.left_orthogonalize` / `right_orthogonalize` | unchanged, one owner |
| one QR step of a sweep, in place | `riemannian.cores_orthogonalization_step` **and** the inlined loops in `ksl`, `amen`, `project` | `_localops` gains `push_left`/`push_right`; `cores_orthogonalization_step` keeps its name and delegates (legacy API) |
| the frames `U_k, V_k, S_µ` at a point | rebuilt independently in `project`, `projector_splitting_add`, `tangent_defect`, `ksl` | **`riemannian.frames(X, mu=1)`**, one owner, returned as a small dataclass and passed around |
| projection onto `T_X M` | `riemannian.project` **and** `ksl.tangent_defect` (measured identical to 1.5e-16, §1.5) | `project_delta` is the owner; `project` and `tangent_defect` both call it |
| the Lie–Trotter tangent sweep | `riemannian.projector_splitting_add` **and** `ksl._sweep_forward` | two implementations kept on purpose (different local solves), joined by a shared test |
| retraction | nothing named | `riemannian.retract(X, xi, method=...)`, dispatching to `round` or `projector_splitting_add` |

The rule this table encodes: **`tt.algs.riemannian` owns the geometry of `M_r`;
`tt.algs.ksl` owns time integration on it and must not re-derive the geometry.**

---

## 3. Cost summary

Notation: `d` modes, mode size `n`, manifold rank `r`, rank of the object being
projected `r_z`, TT-matrix rank `R`, `b` tangent vectors, `F` = cost of one
evaluation of the objective.

| operation | cost | source |
|---|---|---|
| frames `U, V, S` at a point | `d n r³` | [RNO19] §4.1 |
| `P_X z`, `z` a TT of rank `r_z` | `d n r r_z²` | [RNO19] §4.2 |
| `P_X (A y)` naive (form `Ay`, then project) | `d n r_y² R² (r + n)` | [RNO19] §4.2 |
| `P_X (A y)` fused | `d n r r_y R (r_y + n R)` | [RNO19] §4.2 |
| `P_X (Σ_{q≤ρ} B_q A y)`, `B_q` rank-1 | `ρ · d n r² R (r + n R)` | [RNO19] eq. (25) |
| `<ξ, η>` from deltas | `d n r²` | [RNO19] eq. (22) |
| `<ξ, η>` from the rank-`2r` tensors | `d n r³` | [RNO19] §4.3 |
| Gram of `b` tangent vectors, from deltas | `b² d n r²` | [RNO19] §4.3 |
| TT-SVD retraction from rank `3r` | `d n r³` | [RNO19] §4.4 |
| Riemannian gradient by autodiff | `F + d n r³` | [NRO22] Prop. 5.2 |
| approximate Riemannian Hessian-vector by autodiff | `F + d n r³` | [NRO22] Prop. 6.2 |

The `O(F + d n r³) = O(F)` claim of [NRO22] rests on two things that must hold
in ttpy2 too: the program `p` must be evaluable on a rank-`2r` tensor at cost
`O(2^q F)` for a polynomial degree `q` (true for every op in `_ops.py`), and
reverse-mode AD must cost a constant times the forward pass (true for torch).
Both are structural; the measured ratio is in §4.4.

---

## 4. Automatic differentiation — the decisive measurements

### 4.1 Does torch autograd flow through our ops today? **Yes, and further than expected.**

Regime: b300, torch 2.13.0+cu130, backend `torch/cpu`, float64, `d=4, n=3, r=2`,
symmetric TT-matrix of rank 2, gradient of a scalar functional with respect to
all TT cores, checked against **central finite differences** (`h = 1e-6`) on
every core entry; the number reported is
`max_k ‖g_AD − g_FD‖_∞ / max_k ‖g_FD‖_∞`. Prototype `p1_autograd.py`.

| functional | status | rel. err vs FD | note |
|---|---|---|---|
| `_ops.dot(x, x)` | **OK** | 1.65e-10 | |
| `tt.dot(x, x)` | **OK** | 1.65e-10 | |
| `<A x, x>` via `tt.matvec` + `tt.dot` | **OK** | 2.29e-11 | |
| Rayleigh quotient `<Ax,x>/<x,x>` | **OK** | 2.89e-11 | |
| `‖x − b‖²` via `_ops.sub` + `dot` | **OK** | 6.36e-10 | goes through the zero-padded `add` |
| **`x.norm()**2`** | **NO GRAPH** | — | returns a python `float` (§1.4) |
| **`_ops.norm(cores)**2`** | **NO GRAPH** | — | same owner |
| `round_cores(x, 1e-12)` then `dot` | **OK** | 8.02e-10 | SVD backward |
| `round(x − b, 1e-10)` then `dot` | **OK** | 1.10e-08 | |
| `orthogonalize` then `dot` | **OK** | 6.17e-10 | QR backward |
| dense `full(x)²` summed | **OK** | 1.33e-10 | |
| completion `Σ_Ω (x_i − a_i)²` | **OK** | 4.88e-10 | entries gathered core by core |
| `riemannian.project(X, b)` squared norm | **OK** | 7.90e-10 | differentiates through the projector |
| `projector_splitting_add(X, b)` squared norm | **OK** | 1.83e-09 | differentiates through the retraction |
| `amen_solve(A, x, x)` squared norm | **OK** | 4.90e-09 | differentiates through the whole solver |

Why it works: `tt/backend.py`'s `einsum` compiles two-operand patterns into
`transpose` + `reshape` + `@`, all of which are torch autograd primitives;
`TorchBackend.asarray` returns the tensor unchanged when no dtype is forced, so
`vector.from_list` does not detach; `_ops.add` writes into a fresh `bk.zeros`
buffer with slice assignment, which torch differentiates through `index_put_`;
and `_ops.chop` calls `bk.to_numpy` only on the *singular values*, i.e. only the
discrete rank decision leaves the graph, while the retained `u, s, vh` stay in
it.

The last three rows are worth naming separately. Differentiating through
`amen_solve` unrolls the whole ALS iteration into the tape; it is correct (4.9e-09
against finite differences) but the memory grows with the sweep count, and the
gradient is that of *the iterate the solver returned*, not of the exact solution
map. It is a curiosity, not a recommendation.

**Only one thing breaks, and it is ours.** `bk.norm`, §1.4. Everything else in
the table is a torch primitive behaving normally.

### 4.2 Where it breaks anyway: degenerate singular values

Regime as above; prototype `p5_adverse.py` §(A).

| input to `round_cores` / `orthogonalize` | gradients |
|---|---|
| `ones` tensor written with TT ranks `(1,2,2,2,1)` (rank deficient) | OK |
| generic full-rank random TT | OK |
| `orthogonalize` on the rank-deficient `ones` | OK |
| **`d=2, n=2` "identity tensor", singular values exactly `(1, 1)`** | **NaN in every core** |

Exact reproducer (b300, torch 2.13.0+cu130, CPU, float64):

```python
import numpy as np, torch
from tt import backend as bk
from tt.core import _ops
bk.set_backend("torch", device="cpu", dtype="float64")
c0 = np.zeros((1, 2, 2)); c0[0, 0, 0] = 1.0; c0[0, 1, 1] = 1.0
c1 = np.zeros((2, 2, 1)); c1[0, 0, 0] = 1.0; c1[1, 1, 0] = 1.0
cores = [torch.tensor(c, dtype=torch.float64, requires_grad=True) for c in (c0, c1)]
f = _ops.dot(_ops.round_cores(cores, 1e-12), _ops.round_cores(cores, 1e-12))
f.backward()
# cores[0].grad and cores[1].grad are all NaN
```

This is `torch.linalg.svd`'s backward, whose formula contains `1/(σ_i² − σ_j²)`;
it is not a ttpy2 bug. But **ttpy2 inherits it silently**, and the trigger is
not exotic: any tensor with a symmetry has degenerate singular values, and a
QTT Laplacian eigenvector, a permutation-symmetric Hamiltonian ground state and
`tt.ones` all have them. Two consequences for the design:

* the recommended Riemannian AD of §4.3 **never differentiates through an SVD**
  — the frames `U, V, S` are computed *outside* the tape and the tape sees only
  `cat` and `einsum`. That is not an accident of the algorithm; it is one of its
  main practical virtues, and it is why [NRO22] Alg. 5.2 is the right shape.
* `tt.algs.autodiff` must **refuse** to differentiate a user function that calls
  `round`/`project`/`tt_qr` inside itself, or at least warn. t3f's docstrings
  carry exactly this warning ("may not work for some functions, e.g. ones that
  include QR or SVD decomposition ... In this case this function can silently
  return wrong results!"), and t3f is right to. Our version should be louder:
  §10 test 8 asserts a raise, not a warning.

Also verified: the same experiment on `cuda:0` (chosen after `nvidia-smi` showed
GPUs 0 and 1 at 0 % utilization; GPUs 2–7 were at 94–100 %) — `<x,x>` at
`d=20, n=2, r=20` gave finite gradients on device, dtype float64.

### 4.3 Riemannian autodiff ([NRO22] Alg. 5.2) — measured correct in our conventions

The algorithm, in ttpy2 terms:

```
frames:  U_1..U_{d-1} left-orthogonal, V_2..V_d right-orthogonal, S_1 (mu=1 core)
R_k  :=  S_1 if k == 1 else 0                        # the tangent parametrisation
g(R) :=  f( assemble(U, V, R) )                      # Alg. 5.1: the rank-2r stack
D_k  :=  d g / d R_k   at R                          # ordinary reverse-mode AD
delta_k := D_k - ML(U_k) ( ML(U_k)^H ML(D_k) ),  k < d;   delta_d := D_d
```

`assemble` is exactly the `S_k` stack of §2.1 and `T_X(S_1, 0, ..., 0) = X`, so
`g(R^0) = f(X)`.

Measured (`p2_tangent.py` §(c), b300, torch CPU float64), against
`2 · project(X, A X)` computed on the numpy backend — i.e. against the *other*
route, not against itself:

| problem | `‖AD grad − 2 P_X(AX)‖ / ‖·‖` |
|---|---|
| `d=4, n=3, r=2, R_A=2` | **1.48e-15** |
| `d=6, n=4, r=3, R_A=2` | **2.63e-15** |
| `d=8, n=2, r=3, R_A=3` | **2.48e-15** |

and for the completion functional `f(X) = Σ_{Ω} (X_i − a_i)²`, `d=6, n=4, r=3`,
`|Ω| = 200`, against `project(X, tt_svd(dense Euclidean gradient, 1e-13))`:

```
||AD grad - P_X(dense Euclidean grad)|| / ||.|| = 1.91e-15
rank of the Euclidean gradient = 59        (against r = 3 for X)
```

**One discrepancy with the paper, worth recording.** [NRO22] eq. (5.11) writes
the gauge step with a **minus**, `Ṡ_k = ∂g/∂R_k − U_k Σ_j U_k^H ∂g/∂R_j`, while
Alg. 5.2 line 9 writes `D_k := D_k + U_k^L((U_k^L)^T D_k)` with a **plus**. Only
the minus is the projection onto the gauge complement, and only the minus
reproduces `project` — measured 1.5e-15 with minus. Either the paper has a typo
in Alg. 5.2 or the text extractor lost a sign; t3f's `_enforce_gauge_conditions`
uses `proj_delta -= q @ (q.T @ proj_delta)`, i.e. the minus, which settles it.
Our implementation uses the minus and §10 test 3 pins it.

### 4.4 What it costs: AD against the explicit route, CPU and GPU

Regime: b300, float64, CPU restricted to 8 threads (`OMP_NUM_THREADS=8`,
`MKL_NUM_THREADS=8`, `torch.set_num_threads(8)`), GPU `cuda:0` (0 % utilization
at the time, checked with `nvidia-smi`), median of 3 runs, TT ranks capped to
structurally possible values. Objective `f(X) = <A X, X>`, `A` symmetric of
TT rank `R_A`. Prototype `p7_timing.py`.

"numpy explicit" is `2 · project(X, tt.matvec(A, X))` — i.e. what ttpy2 can do
today, and what [NRO22] calls the **naive** baseline. ttpy2 has no fused
`project_matmul`, so their "improved" column has no counterpart here and this
comparison must not be read as AD-versus-hand-written-best.

| `d` | `n` | `r` | `R_A` | numpy explicit (ms) | torch-CPU AD (ms) | torch-GPU AD (ms) | AD/explicit (CPU) | AD vs explicit, value |
|---|---|---|---|---|---|---|---|---|
| 20 | 2 | 10 | 3 | 7.1 | 7.8 | 19.3 | 1.09 | 1.02e-14 |
| 20 | 4 | 20 | 3 | 202.4 | 23.6 | 20.6 | **0.12** | 1.49e-14 |
| 40 | 2 | 20 | 3 | 51.1 | 28.7 | 35.3 | 0.56 | 2.54e-14 |
| 10 | 20 | 10 | 5 | 71.8 | 13.6 | 9.6 | **0.19** | 9.71e-15 |
| 20 | 8 | 40 | 4 | 2184.2 | 130.4 | 40.8 | **0.06** | 1.68e-14 |
| 40 | 2 | 40 | 3 | 843.3 | 60.0 | 54.3 | 0.07 | 2.22e-14 |

Read the last column first: every route returns the same gradient to 1e-14.
Then the shape of the result:

* On the **QTT-like** case `d=20, n=2, r=10` AD is a wash (1.09x) and the GPU is
  2.7x *slower* than numpy — the tensors are 10x2x10 and everything is launch
  latency. **AD is not a win in the small-QTT regime.**
* As soon as `n r` grows, AD wins on CPU by 2–17x, because the naive explicit
  route pays for materializing `A X` at rank `R_A · r` while AD never does.
* On the GPU the largest case goes from 2184 ms (numpy explicit) to 40.8 ms —
  **54x** — and 130 ms (torch CPU) to 40.8 ms.

This reproduces the qualitative content of [NRO22] Table 2a (their AD beats
naive by 2.4x and improved by 3.3x on `<AX,X>` at `d=40, n=20, r=20` on CPU) on
our own code, and it is the measurement that decides §5.

### 4.5 Beyond the gradient: what the paper gives us for free

* **`stop_gradient`** ([NRO22] §5.3). `P_X B⁻¹(A X − F)` is not the Riemannian
  gradient of any quadratic when `B A` is nonsymmetric, but it *is*
  `P_X ∇h` for `h(X) = <B A c(X), X> − <B F, X>` where `c` is the identity with
  zero derivative. In torch, `c(X)` is `X.detach()`, and every op in `_ops.py`
  handles a detached operand without complaint. This is what makes a
  *preconditioned* Riemannian eigensolver or linear solver expressible as an
  autodiff of a scalar, and it is the bridge to §6. Not prototyped.
* **Approximate Riemannian Hessian-vector product** ([NRO22] §6, Alg. 6.2):
  `w(X) = <P_{c(X)}∇f, Z> = Σ_k <δG_k, δG_k^Z>` (the cheap inner product of
  §2.1), then a second Riemannian gradient of `w`. Same `O(F + d n r³)`. The
  curvature term of the true Riemannian Hessian is omitted, which is what makes
  it stable — the exact term contains inverted singular values ([NRO22] §3,
  citing Absil–Mahony–Trumpf). Not prototyped.

---

## 5. The algorithms to build, ordered

### 5.1 Recommendation: build `project_delta` + `tt.algs.autodiff.riemannian_grad` first

The measurements that decide it:

1. **Torch autograd already flows through 13 of 15 TT functionals built from
   existing ttpy2 ops, correct to 1e-9 or better against finite differences**
   (§4.1) — including `project`, `projector_splitting_add` and `amen_solve`.
   The infrastructure is there; we are not building an AD system, we are
   wrapping one.
2. **The [NRO22] algorithm reproduces `project(X, ∇f)` to 1.5e-15 in our
   conventions** (§4.3), on a functional whose Euclidean gradient has rank 59
   against a manifold rank of 3.
3. **It is 8–17x faster than what ttpy2 can do today on CPU and up to 54x on a
   GPU** (§4.4, `d=20, n=8, r=40`: 2184 ms → 130 ms → 40.8 ms).

It is also small: `project_delta` is `project` with two lines changed (return
the deltas instead of assembling them), `assemble` is 12 lines, and
`riemannian_grad` is 20 lines on top. The blocking prerequisite is the `bk.norm`
fix of §1.4 — without it, every objective a user writes with `.norm()` is
silently wrong.

**Order (status 2026-08-09: items 0-4 shipped):** items 0-4 landed in
`tt/algs/riemannian.py` (Frames, `project_delta`, `tangent_to_tt`,
`tangent_inner`, `tangent_gram`, `retract`, `transport`) and the new
`tt/algs/autodiff.py` (`riemannian_grad`, `rgd`), verified by
`tests/test_riemannian_autodiff.py` (15 tests: rebuild/gauge/inner-product
contracts at the plan's measured levels, the AD-vs-dense-gradient pairing at
1e-12, the minus-sign pin, the invariance check, non-quadratic recovery).
The showcase is `examples/robust_completion.py` -- log-cosh completion under
outliers, the loss ALS cannot have.  Remaining debt from this list: items 5-8,
and the `ksl` migration onto `frames` (section 1.5's second owner still
stands until then).  One addition beyond the plan: `TorchBackend.svd` gained
the gesvd fallback numpy already had -- a retraction step hit gesdd
non-convergence on near-repeated singular values, and `riemannian_grad` now
refuses non-finite values loudly (an overflowing loss autodiffs to NaN
deltas, measured the hard way with a naive log-cosh).

| # | item | why here | measurement that supports it |
|---|---|---|---|
| 0 | fix `bk.norm` (§1.4) | a silent wrong gradient blocks everything | 7.0e-01 absolute gradient error, §1.4 |
| 1 | `project_delta`, `tangent_to_tt`, `tangent_inner`, `frames` (§8a–d) | the SSOT the rest stands on; also makes `tangent_defect` cheaper | gauge 4e-14, rebuild 1e-15, inner product 8e-16 (§2.1) |
| 2 | `tt.algs.autodiff.riemannian_grad` (§8h) | §5.1 above | 1.5e-15 correctness, 0.06–1.09x cost |
| 3 | `retract`, `transport` (§8e–f) | the two remaining primitives of any first-order method | transport omission costs 60x (§2.4) |
| 4 | `rgd` — Riemannian gradient descent with exact/Armijo line search | the first solver; the honest baseline for everything after | §5.2 |
| 5 | preconditioner interface `prec=` as a **list of rank-1 TT-matrices** (§6) | without it the method is not competitive on elliptic problems | 475 → 10 iterations, and `>3000` → 12 at κ=1e5 (§6) |
| 6 | `rcg` (Riemannian CG) | standard next step — but see the counterweight in §5.2 | measured *worse* than `rgd` in one regime |
| 7 | `riemannian_hvp` + `rtr`/Gauss-Newton | [NRO22] §6; second-order without the curvature term | not measured |
| 8 | rank adaptation (§5.3) | the central practical difficulty | 1.7e-02 vs 3.7e-30 for a rank guess off by 2 (§2.3) |

`eig_lobpcg` / LRRAP is **owned by `docs/plans/eigenvalues.md` §2.2 and §5**, not
by this spec. What this spec owes it: `project_delta`, `tangent_inner`,
`tangent_gram`, `retract` and the rank-1-sum preconditioner interface — items
1, 3 and 5 above. `docs/plans/eigenvalues.md` §4(b) asks for exactly
`project_delta`, `tangent_to_tt` and `tangent_gram`; those signatures are
reproduced verbatim in §8 so that there is one owner and not two proposals.

### 5.2 Measured: Riemannian GD and CG on tensor completion, against `ttSparseALS`

Problem: recover `A ∈ R^{10×...×10}` (`d = 6`, 10⁶ entries) of TT rank 3 from
`|Ω|` uniformly random entries. The target is a **random point of `M_3` with
left-orthogonal cores** (QR of Gaussian blocks), `dof = 420`. Method: Riemannian
gradient by autodiff (§4.3), exact line search (the functional is quadratic
along a tangent direction), TT-SVD retraction. Baseline:
`tt.algs.completion.ttSparseALS` with `alpha=0.0`, `tol=1e-14`. Test error is
measured on 50 000 held-out entries. Regime: b300, torch CPU float64 (8 threads)
for the Riemannian runs, numpy float64 for ALS. `p3d_final.py`.

| `|Ω|/dof` | method | iterations / sweeps to train rel < 1e-6 | wall | train rel | test rel |
|---|---|---|---|---|---|
| 10 | Riemannian GD | >3000 | 27.6 s | 5.89e-01 | 2.36e+00 |
| 10 | `ttSparseALS` | >300 | 2.0 s | 5.93e-01 | 1.17e+01 |
| 30 | Riemannian GD | >3000 | 69.9 s | 7.65e-01 | 2.27e+00 |
| 30 | `ttSparseALS` | >300 | 5.0 s | 7.61e-01 | 4.97e+01 |
| 100 | Riemannian GD | **59** | 5.4 s | 1.59e-13 | 2.00e-13 |
| 100 | `ttSparseALS` | **20** | 1.2 s | 6.58e-08 | 7.49e-08 |
| 300 | Riemannian GD | **20** | 6.8 s | 5.90e-14 | 6.57e-14 |
| 300 | `ttSparseALS` | **8** | 1.4 s | 1.58e-08 | 1.66e-08 |

Four readings, three of them counterweights:

* **Below a sampling threshold neither method works.** At 10x and 30x the dof
  both stall at a train error of 0.6–0.77 with a *test* error above 1, i.e. the
  fit is worse than returning zero. `ttSparseALS` already warns about
  underdetermined slices; the Riemannian version must report the same thing, and
  §10 test 5 requires it.
* **Above the threshold ALS is 4–5x faster in wall clock**, and Riemannian GD is
  5 orders of magnitude more accurate — because ALS stops at its own `tol` on
  the fit while GD keeps descending. Neither is "better"; they answer different
  questions. Do not sell Riemannian completion as faster than our ALS.
* **The iteration count is wildly sample-set dependent.** The same problem, the
  same `|Ω| = 100 × dof`, a *different draw* of Ω: 59 iterations in one run and
  **1420** in another (`p3c_dbg.py`; monotone throughout, on a plateau at
  train rel ≈ 0.86 for 1400 iterations and then a drop to 3.4e-13 in a few
  steps). A `maxit` chosen from one run is worthless.
* **A naive Riemannian CG was measured to be *worse* than GD.** Fletcher–Reeves
  `β = ‖g_k‖²/‖g_{k-1}‖²`, direction transported by `project`, exact line
  search, restart on a non-descent direction: 2000 iterations without reaching
  1e-6 on the run where GD reached 3.4e-13 at iteration 1420. This is a
  statement about *that* CG, not about geomCG (Steinlechner), which uses a
  different `β` and a different transport and is reported to work; geomCG was
  not implemented and not measured. Recorded so that item 6 of §5.1 is not
  assumed to be free.

The 1420-vs-59 spread and the plateau are the honest summary of first-order
Riemannian completion: it is a non-convex landscape with long flat regions, and
nothing in the method detects them.

### 5.3 Rank adaptation — the central difficulty, and what each option costs

`M_r` is a fixed-rank manifold. Every method in this spec needs `r` as an input,
and §2.3 measures that guessing it 2 too high costs 28 orders of magnitude of
residual on a completion problem. The options, honestly:

1. **Ask the user.** What `eig_lobpcg` does in [RNO19] ("in its current version,
   the proposed algorithm lacks the rank adaptivity", their §8). Cost: zero code,
   and a solver whose accuracy is decided by a guess. This is the same defect
   `docs/plans/eigenvalues.md` §1.2 documents for `eigb` at `B = 1`.
2. **Rank continuation** (fit at rank 1, pad with a small random block, refit at
   rank 2, ...). Standard in the completion literature. **Measured and it did
   not help here**: at `|Ω| = 10 × dof` continuation gave train 6.68e-01 against
   6.49e-01 for a random rank-3 start, and at 30x 7.93e-01 against 8.08e-01 —
   i.e. inside the noise, and both far from the solution (`p3b_completion.py`).
   It costs one full solve per rank level.
3. **Grow into the normal space.** Compute the component of `−∇f` orthogonal to
   `T_X M`, truncate it to `kickrank` directions and add it to the iterate —
   the manifold analogue of AMEn's residual enrichment, and the same mechanism
   `docs/plans/eigenvalues.md` §2.1 recommends for the eigensolver. We already
   measure the normal component: `‖(I − P)Z‖` is `ksl.tangent_defect`, and after
   §1.5 it is `sqrt(‖Z‖² − Σ‖δG_k‖²)`, free once the deltas are computed. The
   *directions* are not free: extracting them needs `Z − P_X Z` as a TT of rank
   `r_z + 2r`, one rounding, and then a rank increase at every bond. Cost per
   step `O(d n r r_z² + d n (r + kickrank)³)`. **Not prototyped.**
4. **BUG-style augment-and-truncate.** `docs/plans/bug-integrator.md` §2.2 has
   the machinery (augment the frames to `2r`, run the Galerkin step, truncate by
   a tolerance `θ` and use the discarded singular values as the error estimate).
   For optimization rather than integration the augmentation would be
   `orth([U_k, ML(δG_k)])`, which is exactly the rank-`2r` structure `project`
   already returns. This is the option with the strongest theory behind it
   (robustness to small singular values) and it shares an owner with the BUG
   spec. **Not prototyped.**
5. **Do not use a manifold.** For a rank-adaptive answer, ALS/AMEn/DMRG solve
   the same problems and adapt ranks natively, and §5.2 measures ALS beating
   Riemannian GD 4–5x in wall clock on completion. The Riemannian family earns
   its place where the *rank is genuinely fixed by the budget* (a GPU memory
   limit, a `b`-eigenvector block whose block-TT rank would otherwise be `B r`,
   §[RNO19]) — not where the rank is a free parameter.

**Recommendation:** ship (1) with a loud residual report, implement (4) once
`docs/plans/bug-integrator.md` lands, and treat (3) as the experiment of §13 Q3.
Do not ship (2) as a default: it is measured not to help on our one test case.

---

## 6. Preconditioning

### 6.1 Why the unpreconditioned method is hopeless on QTT elliptic problems

`docs/plans/eigenvalues.md` §3.2 measured truncated steepest descent on the
Rayleigh quotient of `tt.qlaplace_dd([d])` on the local 2-core host and found
1413 iterations for 3 digits at `κ = 1.7e3` and no 3 digits in 4000 iterations
at `κ ≥ 2.7e4`. **I re-measured it on b300** rather than reusing it, because
this spec needs the comparison against a *genuine* Riemannian step (tangent
projection + retraction), which that prototype did not have.

Regime: b300, numpy, float64, `tt.qlaplace_dd([d])`, oracle
`λ_1 = 4 sin²(π/(2(N+1)))`, seed 0, rank cap 4, step from the exact 2x2
Rayleigh–Ritz in `span{x, z}`. `p4_precond.py` §(1).

| `d` | `N` | `κ` | method | iterations to rel. 1e-3 | rel. err at the cap | wall |
|---|---|---|---|---|---|---|
| 6 | 64 | 1.71e+03 | truncated SD (`z = r`) | **1413** | 9.96e-04 | 2.6 s |
| 6 | 64 | 1.71e+03 | Riemannian GD (`z = P_x r`) | **1398** | 9.96e-04 | 3.5 s |
| 8 | 256 | 2.68e+04 | truncated SD | >4000 | 1.54e-01 | 8.8 s |
| 8 | 256 | 2.68e+04 | Riemannian GD | >4000 | 2.48e-01 | 12.6 s |
| 10 | 1024 | 4.26e+05 | truncated SD | >1500 | 1.12e+02 | 4.0 s |
| 10 | 1024 | 4.26e+05 | Riemannian GD | >1500 | 9.24e+01 | 5.7 s |

The `d = 6` row reproduces the earlier 1413 **exactly**, on different hardware,
which is a useful check on both. And the second row is the point of the table:
**the tangent projection changes nothing.** 1398 against 1413 iterations, and
35 % more wall time for the projection. The manifold machinery is not what fixes
conditioning; a preconditioner is. Any spec that recommends Riemannian methods
for QTT-discretized elliptic operators without one is wrong, and this table is
why.

### 6.2 What the preconditioner is: [RNO19] eq. (24)–(25)

[RNO19] §4.5 assumes

```
B^{-1} = B_1 + ... + B_{rho_B},     each B_i a TT-matrix of TT rank 1,
```

because multiplying a TT-matrix of rank `R` by a rank-1 TT-matrix leaves the
rank at `R`. Then

```
P_x B^{-1} H x  =  P_x B_1 H x + ... + P_x B_{rho_B} H x
```

is assembled term by term at cost `O(b d n r² R (r + nR) ρ_B)`, and no
intermediate of rank `ρ_B R r` is ever formed. `riemannian.project` **already
accepts a list and sums inside** (§1.1), so the ttpy2 realisation of eq. (25) is
one call.

For a Kronecker-sum operator `A = Σ_i I ⊗ ... ⊗ L ⊗ ... ⊗ I` such a `B^{-1}`
comes from an exponential sum for `1/λ`: with `1/λ = ∫_0^∞ e^{-λ t} dt` and
`t = e^s`, the trapezoidal (sinc) rule gives

```
1/lambda  ~  h * sum_{q=-M..M} e^{s_q} exp(-lambda e^{s_q}),   s_q = q h + shift,
B^{-1}    =  h * sum_q e^{s_q}  exp(-e^{s_q} L) (x) ... (x) exp(-e^{s_q} L),
```

each summand a rank-1 TT-matrix. This is the construction [RNO19] cites
(Khoromskij, Constr. Approx. 30:599–620, 2009) and the one measured below.
`shift = -0.5 log(λ_min λ_max)` centres the quadrature on the spectrum.

### 6.3 Measured: preconditioning makes the iteration count independent of κ

Problem: `A = Σ_{i=1..D} I ⊗...⊗ L ⊗...⊗ I` with `L` the `n x n` Dirichlet
Laplacian `tridiag(−1, 2, −1)`, physical modes (not QTT), TT-matrix rank 2.
Oracle: `λ_1 = D · 4 sin²(π/(2(n+1)))`, exact; the ground state is the product
of sines, TT rank 1, so the manifold `M_1` **contains the exact answer** and the
only thing being measured is the convergence rate. Preconditioned iteration:
`z = P_x(Σ_q c_q B_q (A x − λ x))` through the list form of `project`; step from
the exact 2x2 Rayleigh–Ritz; retraction `round(·, rmax=1)`.
`γ = max_{λ∈[λ_min,λ_max]} |1 − λ Σ_q c_q e^{-λ t_q}|` measured on a 4000-point
log grid. Regime: b300, numpy, float64, single runs. `p4_precond.py` §(2).

| `D` | `n` | `κ` | `ρ_B` | `γ` | method | it → 1e-3 | it → 1e-8 | wall | final rel. err |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 32 | 4.41e+02 | 41 | 1.59e-02 | unpreconditioned | 475 | 875 | 1.4 s | 9.88e-09 |
| 4 | 32 | 4.41e+02 | 41 | 1.59e-02 | **preconditioned** | **10** | **16** | 0.1 s | 2.01e-09 |
| 4 | 128 | 6.74e+03 | 51 | 2.31e-02 | unpreconditioned | >3000 | >3000 | 8.5 s | 2.47e-02 |
| 4 | 128 | 6.74e+03 | 51 | 2.31e-02 | **preconditioned** | **11** | **17** | 0.2 s | 2.91e-09 |
| 4 | 512 | 1.07e+05 | 61 | 3.40e-02 | unpreconditioned | >3000 | >3000 | 166.4 s | 4.42e+00 |
| 4 | 512 | 1.07e+05 | 61 | 3.40e-02 | **preconditioned** | **12** | **18** | 2.4 s | 2.87e-09 |

The iteration count of the preconditioned method is **10, 11, 12** across a
242-fold increase of `κ` — the textbook signature of spectral equivalence, and
the reason [RNO19] can run Riemannian LOBPCG on vibrational Hamiltonians at all.
At `κ = 1.07e5` the unpreconditioned Rayleigh quotient after 3000 iterations is
5.4 times too large; the preconditioned one is correct to 2.9e-09 in 2.4 s.

Honest counterweights on the same table:

* The **rank-1-per-term structure is what makes it cheap**, and it exists only
  because the operator is a Kronecker sum over *physical* modes. For a **1D QTT**
  Laplacian, `exp(−tL)` is a single `2^d × 2^d` matrix whose QTT representation
  is not rank-1 across the QTT bonds, so eq. (24) does not apply and this
  construction gives nothing. That is exactly the gap §6.4 hands to BPX.
  **Since measured, and it is worse than "not rank-1":**
  `docs/plans/qtt-elliptic-bpx.md` §2.3 took the TT-SVD of the dense
  `expm(−t A_DN)` at `L = 6, 8, 10` and four values of `t` and found QTT ranks
  **8–21** (`L`-independent, but not 1), so `ρ_B ≈ 40..60` terms of rank ≈ 15
  would cost more than one AMEn sweep on the preconditioned operator. The
  exponential-sum route does not transfer to the by-scale setting at all.
* `ρ_B = 41..61` means 41–61 TT matvecs per iteration. Even so the
  preconditioned run at `n = 512` converged to 2.9e-09 in **2.4 s**, where the
  unpreconditioned run spent **166 s** on 3000 iterations to arrive at a
  Rayleigh quotient 441 % too large. But the per-iteration cost is `ρ_B` times
  higher, so on a problem where the unpreconditioned method converges in ~20
  iterations the preconditioner loses. The `n = 32` row shows the discount
  already: 875 → 16 iterations to 1e-8 is a factor 55, but 1.4 s → 0.1 s is only
  a factor 14.
* `γ ≈ 2e-2..3e-2` is the *measured* quadrature quality, not a tuned optimum; a
  Braess–Hackbusch best approximation would need fewer terms. Not measured.

### 6.4 What the eigensolver and the optimizer need from the planned BPX spec

Identical to the interface `docs/plans/eigenvalues.md` §3.3 states, and it must
stay one interface. Restated from this side:

1. `B⁻¹` must be exposable **as a list of rank-1 TT-matrices**, not only as a
   black-box `apply`. `project(X, [B_1 @ z, ..., B_ρ @ z])` is then the whole
   preconditioned projected residual, and no intermediate of rank `ρ R r` exists.
   A callable `prec(z) -> tt.vector` must also be accepted (it is what a BPX
   with a multilevel structure will naturally be), but it is the slow path and
   the API should say so.

   **Corrected by measurement, from the BPX side.** `docs/plans/qtt-elliptic-bpx.md`
   §4.2 item 1 answers this and rejects both halves: BPX is a **single
   `tt.matrix`** of TT rank `2^{2D+1}` — measured exactly 8 / 32 / 128 for
   `D = 1, 2, 3`, independent of `L` up to `L = 50` (their §1.5) — it cannot be
   made into a sum of rank-1 terms, and it is **not** a slow path (one matvec
   plus one rounding, the cost of one extra matvec by `A`, whose rank is 3–4).
   The rank-1-sum form stays correct for the Kronecker-sum-over-physical-modes
   case measured in §6.3 below, and only for it. The reconciled contract — three
   accepted forms, each declaring its side (`'left'` vs `'two-sided'`) and its
   rank, plus what it forbids — is `docs/plans/ROADMAP.md` §3.
2. SPD, or the 2x2 / 3b x 3b Rayleigh–Ritz that chooses the step loses its
   variational characterization.
3. Spectral equivalence with `d`- and mesh-independent constants — §6.3 measures
   what that buys (10 → 12 iterations across `κ = 4e2 → 1e5`).
4. [BK20] is the source for the QTT case, and it says something this spec cannot
   ignore: applying a BPX preconditioner to a low-rank representation cures the
   *matrix* conditioning but introduces **representation ill-conditioning**, and
   the redundancy has to be eliminated explicitly to get a decomposition free of
   both. A Riemannian method works entirely in the representation, so a
   preconditioner that is spectrally perfect and representationally ill-conditioned
   would still fail here. **Not verified**: I read only the first page of [BK20].
   This is §13 Q6.

---

## 7. What t3f has that we do not

Audited from source, fresh clone, branch `develop`, HEAD `8096b38` (2021-04-06),
v1.2.0. The premise that t3f is TF1-era is **wrong**: v1.2.0 is "full TF 2
support", CI pins `TF_VERSION=2.4.0`, `setup.py` requires only `numpy`. The repo
is dead since April 2021, which is a different problem.

### 7.1 API shape

`t3f/riemannian.py` (887 lines), seven public functions:

```python
project_sum(what, where, weights=None)          # P_x(sum_j w_j what[j]), never forms the sum
project(what, where)                            # element-wise over a batch
project_matmul(what, where, matrix)             # P_where(matrix @ what), fused
pairwise_flat_inner_projected(A, B)             # Gram of tangent vectors, O(b^2 d n r^2)
add_n_projected(tt_objects, coef=None)          # sum tangent vectors WITHOUT rank growth
tangent_space_to_deltas(tt)                     # block cores -> deltas
deltas_to_tangent_space(deltas, tt, left, right)   # deltas -> block cores
```

The "projected" representation is **not** a separate container: it is an
ordinary rank-`2r` `TensorTrain` with a monkey-patched python attribute
`tt.projection_on = where` holding a *reference* to the base point, and
`tangent_space_to_deltas` recovers the deltas by *slicing the block cores*
(`slice(r/2, None)` on the left rank axis, `slice(0, r/2)` on the right). Three
functions require the attribute and raise without it; everything else in the
library treats a projected TT as a plain rank-`2r` TT and silently destroys the
structure. `deltas_to_tangent_space` is deliberately **not** exported, with the
docstring "This function is hard to use correctly because deltas should abey the
so called gauge conditions. If the don't, the function will silently return
incorrect result."

`t3f/autodiff.py` (196 lines), the [NRO22] implementation:

```python
gradients(func, x, name='t3f_gradients', runtime_check=True)
hessian_vector_product(func, x, vector, name=..., runtime_check=True)
_enforce_gauge_conditions(deltas, left)          # proj -= q @ (q.T @ proj)
_is_invariant_to_input_transforms(f1, f2)        # the runtime check
```

`runtime_check=True` compares `func(x)` against `func` evaluated on the
delta-reconstruction of the same tensor and asserts a relative difference below
1e-5 — i.e. it checks that the user's function depends only on the *tensor*, not
on the particular cores. That is the exact condition under which a Riemannian
gradient of a program is defined, and it is the single best idea in the file.
Test coverage is two tests.

### 7.2 What is worth porting

| t3f thing | port? | why |
|---|---|---|
| `tangent_space_to_deltas` / `deltas_to_tangent_space` | **yes**, as `project_delta` / `tangent_to_tt` | §8(a)(b); this spec's item 1 |
| `_enforce_gauge_conditions` | **yes**, inside `riemannian_grad` | settles the sign question of §4.3 |
| `_is_invariant_to_input_transforms` runtime check | **yes**, and make it an error not a print | §10 test 8 |
| `project_sum(what, where, weights)` | **already have it** — our `project(X, [Z...])` is `project_sum`; add `weights` | §8(a) |
| `pairwise_flat_inner_projected` | **yes**, as `tangent_gram` | `O(b²dnr²)` vs `O(b²dnr³)`, [RNO19] eq. (22) |
| `add_n_projected` | **yes**, as delta arithmetic | rank-preserving sum of tangent vectors |
| `project_matmul` (fused `P_x (M @ z)`) | later | §4.4 shows AD already beats our naive route; revisit if a hand-written path is wanted |
| `hessian_vector_product` | later | [NRO22] §6; §5.1 item 7 |
| `TensorTrainBatch` + `batch_ops.gram_matrix` / `pairwise_flat_inner` | **the interesting one** | `docs/plans/eigenvalues.md` §4(a) defers a batch container; t3f shows what it buys and how to lay it out (`[batch, r, n, r]`, `left_tt_rank_dim` as the mechanism that keeps every other module batch-agnostic). Their measured batch-vs-single per-object time on a V100: matvec 0.744 → 0.14 ms, gram 0.973 → 0.001 ms |
| `approximate.add_n` / `reduce_sum_batch` | **yes, independently of Riemannian work** | binary-tree rounding: `round(a+b+c+d) ≈ round(round(a+b)+round(c+d))`, `log N` sequential rounds instead of `N`; and `reduce_sum_batch(coef)` computes `N` different weighted sums of one TT dictionary in a single batched pass. Nothing in ttpy2 does this |
| `kronecker` module (`det`, `slogdet`, `inv`, `cholesky` for all-rank-1 TT-matrices) | maybe | tiny, self-contained, and exactly the object §6.2 needs — a rank-1 TT-matrix. Our `B_q` in §6.3 is one |
| `ops.gather_nd` | **yes** | batched entry extraction without `full()`; §5.2's inner loop is a hand-rolled version of it |
| `ops.renormalize_tt_cores`, `frobenius_norm(differentiable=True)` | **yes, small** | the second one is literally the fix of §1.4, and t3f having a flag for it confirms the problem is real |
| `variables.py` (`get_variable`, `assign`) | **no** | the only `tensorflow.compat.v1` file in the library: variable scopes, `tf.get_collection`, `GraphKeys` |
| `nn.KerasDense` | **no** | depends on `variables.py` |
| `name='t3f_*'` on every function + `tf.name_scope` | **no** | pure TF idiom |
| `shapes.lazy_*` static/dynamic duality | **no** | an artefact of graph mode |
| `tensor_train_base.graph` / `op` / `eval(session=...)` | **no** | graph-mode leftovers, unused by the rest of the library |

Two implementation bugs read in t3f's source, worth not copying:
`add_n_projected` uses a leaked loop variable (`tt` from an earlier `for`) where
it means `first_obj_core`, and `deltas_to_tangent_space` reads
`lazy_tt_ranks(left)` where it means `right`. Both are harmless only because the
quantities coincide.

### 7.3 t3f's published benchmark numbers

From `docs/benchmark.rst`, DGX-1, Tesla V100, Xeon E5-2698 v4, float64,
`d = 10`, `n = 10`, TT rank 10 (rank 100 for the `project_rank100` row),
batch 100 (per-object time), read from the repo:

| operation | ttpy, CPU | t3f, CPU | t3f, GPU | t3f batch, CPU | t3f batch, GPU |
|---|---|---|---|---|---|
| matvec | 11.142 ms | 1.19 | 0.744 | 1.885 | 0.14 |
| matmul | 86.191 | 9.849 | 0.95 | 17.483 | 1.461 |
| norm | 3.79 | 2.136 | 1.019 | 0.253 | 0.044 |
| round | 73.027 | 86.04 | **165.969** | 8.234 | **161.102** |
| gram | 0.145 | 0.606 | 0.973 | 0.021 | 0.001 |
| `project_rank100` | 116.868 | 3.001 | 13.239 | 1.645 | 0.226 |

Two things to take from it. `project` at rank 100 is 39x faster in t3f than in
old ttpy on CPU — that is the number our §1.2 dispatch-bound measurement should
be compared against once we have a batched path. And **rounding is slower on
their GPU than on their CPU** (166 vs 86 ms), which they attribute to TF's
SVD-on-GPU; our own §4.4 shows the same shape at small QTT sizes (GPU 19.3 ms
against numpy 7.1 ms at `d=20,n=2,r=10`). A GPU path must be measured per
operation, not assumed.

---

## 8. Missing core functions

Exact signatures, one-line contract each, and the closest existing function.

**(a) The tangent representation.** *Closest: `riemannian.project`, which
computes these cores internally and then assembles and discards them.*

`docs/plans/eigenvalues.md` §4(b) asks for the same three functions and this is
the same request — but the two documents do **not** propose identical
signatures, and the difference has to be settled before either is implemented,
because two specs proposing two contracts for one function is exactly the
duplicated authority the project forbids. The differences: eigenvalues.md has
`project_delta(X, Z) -> list[array]`; this spec has
`project_delta(X, Z, *, weights=None, frames=None) -> (deltas, Frames)`. **This
spec's signature wins**, for two measured reasons: the caller almost always
needs the frames immediately afterwards (`tangent_to_tt`, `transport`,
`riemannian_grad` all do, and rebuilding them is the `O(d n r³)` term of §3),
and the `weights`/list form is what §6.3 measured at `ρ_B = 61`. When this lands,
`docs/plans/eigenvalues.md` §4(b) must be amended to point here rather than
restate it.

```python
def project_delta(X, Z, *, weights=None, frames=None):
    """Gauge cores of ``P_{T_X M} Z``, instead of the rank-2r tensor.

    Returns:
        (deltas, frames): ``deltas[k]`` of shape ``(r_{k-1}, n_k, r_k)`` satisfying
        the gauge condition ``ML(deltas[k])^H ML(U_k) = 0`` for ``k < d``, and the
        :class:`Frames` of ``X`` (see (d)) so that the caller need not rebuild them.
        ``Z`` may be a list, and then the deltas are those of ``P_X(sum_j w_j Z_j)``,
        computed without forming the sum -- the form [RNO19] eq. (25) needs.
    Contract: ``tangent_to_tt(X, deltas) == project(X, Z)`` to roundoff
        (measured 2.9e-16 .. 9.5e-16); ``tangent_inner`` on the result is
        [RNO19] eq. (22) (measured 3.6e-16 .. 8.1e-16).
    Raises: ValueError -- ``X`` is rank deficient (same check as ``project``).
    """
```

**(b)**
```python
def tangent_to_tt(X, deltas, *, frames=None):
    """Inverse of (a): the rank-2r S_k stack of [RNO19] Sec. 4.1. No gauge check."""
```

**(c)**
```python
def tangent_inner(deltas_a, deltas_b):
    """<xi, eta> for two tangent vectors AT THE SAME POINT: sum_k <dG_k^a, dG_k^b>_F.

    O(d n r^2) instead of the O(d n r^3) of a TT contraction, and without its
    cancellation.  Wrong (silently) if the two lists come from different points.
    """

def tangent_gram(delta_lists):
    """(b, b) Gram matrix of b tangent vectors at one point, O(b^2 d n r^2)."""
```

**(d) The frames, as one owner.** *Closest: the first half of `project`,
duplicated in `projector_splitting_add`, `ksl.tangent_defect` and `ksl`.*

```python
@dataclass
class Frames:
    U: list      # U[k] left-orthogonal,  k = 0..d-2 (U[d-1] is the mu=d core)
    V: list      # V[k] right-orthogonal, k = 1..d-1 (V[0] is the mu=1 core S_1)
    S: object    # the mu-orthogonal core, mu as requested
    r: list      # ranks, as verified full

def frames(X, mu=1, *, check_rank=True):
    """Left- and right-orthogonal frames of X and its mu-orthogonal core."""
```

**(e) Retraction, named.** *Closest: `(X + xi).round(0.0, rmax)` written by hand
at every call site, and `projector_splitting_add`.*

```python
def retract(X, xi, *, method="svd", rmax=None, return_discarded=False):
    """A point of M_{r(X)} near X + xi.

    ``method='svd'``   TT-rounding to ``rmax = r(X)`` ([RNO19] Sec. 4.4), quasi-optimal,
                       O(d n r^3); the default.
    ``method='psa'``   :func:`projector_splitting_add`, one sweep, no SVD, exact when
                       ``X + xi`` already has rank ``r(X)``.
    ``return_discarded`` reports the Frobenius norm of what the truncation threw
    away -- the only local estimate of the retraction error, and the quantity a
    rank-adaptive wrapper needs.
    """
```

**(f)**
```python
def transport(deltas, X_old, X_new, *, frames_new=None):
    """Vector transport by re-projection: deltas of P_{T_{X_new} M} xi.  O(d n r^3)."""
```

**(g) Fused projected matvec.** *Closest: `project(X, tt.matvec(A, Z))`, which
forms the rank-`R r_z` product first.* Deferred (§7.2), signature fixed now so
that callers do not have to change later:

```python
def project_matvec(X, A, Z, *, weights=None):
    """Deltas of P_X (A Z) without forming A Z.  [RNO19] Sec. 4.2,
    O(d n r r_z R (r_z + n R)) against O(d n r_z^2 R^2 (r + n)) for the naive route."""
```

**(h) Riemannian autodiff.** New module `tt/algs/autodiff.py`. *Closest: nothing
in ttpy2; t3f's `autodiff.gradients` / `hessian_vector_product`.*

```python
def riemannian_grad(f, X, *, runtime_check=True, frames=None):
    """Deltas of grad f(X) = P_{T_X M} grad_E f(X), by reverse-mode AD on f o T_X.

    ``f`` takes a LIST OF CORES and returns a backend scalar; it must depend only
    on the tensor those cores represent, which ``runtime_check`` verifies by
    evaluating ``f`` on two different core representations of X and comparing
    (t3f's ``_is_invariant_to_input_transforms``; we RAISE rather than warn).
    Requires the torch backend: on numpy it raises, naming the backend.
    Cost O(F + d n r^3) ([NRO22] Prop. 5.2); measured 0.06-1.09x the cost of
    ``project(X, grad_E f)`` for f = <A X, X> (Sec. 4.4).
    """

def riemannian_hvp(f, X, Z_deltas, *, runtime_check=True):
    """Deltas of P_X grad^2 f(X) Z for Z in T_X M -- the APPROXIMATE Riemannian
    Hessian, curvature term omitted ([NRO22] Sec. 3, eq. (3.4)); the exact term
    contains inverted singular values and is unstable at a small one."""

def euclidean_grad(f, X):
    """P_{c(X)}-free ordinary gradient w.r.t. the cores. Diagnostic only:
    it depends on the representation, so it is not a tensor-level object."""
```

**(i) The optimizers.** New module `tt/algs/rieopt.py`.

```python
def rgd(f, X0, *, rank=None, prec=None, maxit=200, eps=1e-8, line_search="armijo",
        callback=None, verb=1, return_history=False): ...
def rcg(f, X0, *, beta="pr+", restart=20, **kw): ...
def rlbfgs(f, X0, *, memory=5, **kw): ...
```

**(j) The `bk.norm` fix** (§1.4) — not a new function, a changed contract for an
existing one, and the prerequisite for everything above.

**(k) `round_cores` with `return_discarded`.** Identical to the request in
`docs/plans/bug-integrator.md` §4(a) and `docs/plans/eigenvalues.md` §4(h) —
the same owner, extended once, for three specs. `retract` needs it for (e).

---

## 9. Public API

Backward compatibility is absolute: `tt/riemannian/__init__.py` re-exports
`project`, `projector_splitting_add`, `tt_qr` and the `tt.riemannian.riemannian`
module alias, and `tests/test_ports.py` asserts them. **Nothing listed below
changes any of those four names or their signatures.**

New names in `tt/algs/riemannian.py` (additive):

```python
__all__ = ["project", "projector_splitting_add", "tt_qr",
           "cores_orthogonalization_step",                      # unchanged
           "Frames", "frames", "project_delta", "tangent_to_tt",
           "tangent_inner", "tangent_gram", "transport", "retract",
           "project_matvec"]
```

New modules, resolved through the existing lazy table in `tt/__init__.py`:

```python
_FUNCTIONS = {..., "riemannian_grad": ("tt.algs.autodiff", "riemannian_grad"),
                   "riemannian_hvp":  ("tt.algs.autodiff", "riemannian_hvp"),
                   "rgd":  ("tt.algs.rieopt", "rgd"),
                   "rcg":  ("tt.algs.rieopt", "rcg"),
                   "rlbfgs": ("tt.algs.rieopt", "rlbfgs")}
```

```python
def rgd(f, X0, *, rank=None, prec=None, maxit=200, eps=1e-8,
        line_search="armijo", retraction="svd", callback=None,
        verb=1, return_history=False):
    """Minimize ``f`` over the manifold of TT tensors of rank ``r(X0)``.

    Args:
        f: callable taking a LIST OF CORES and returning a backend scalar.  It must
            depend only on the tensor (checked once, see ``riemannian_grad``).
        X0: tt.vector; ITS RANKS ARE THE MANIFOLD and are never changed.  ``rank``
            may only be given to assert agreement with ``X0.r`` -- passing a
            different one raises, rather than silently choosing one of two owners
            of the same number (same rule as ``eig_amen``'s ``nblock``).
        prec: None, a callable ``tt.vector -> tt.vector``, or -- the preferred form
            -- a list of rank-1 ``tt.matrix`` summing to ``B^{-1}`` ([RNO19] eq. (24)).
            With the list form the preconditioned projected gradient is ONE
            ``project_delta(X, [B_q @ g for B_q in prec])`` and no intermediate of
            rank ``rho_B r`` is formed.
        line_search: 'armijo' (backtracking on the Riemannian gradient norm),
            'exact' (only legal when ``f`` is quadratic along a tangent direction --
            completion and the Rayleigh quotient are; asserted by a one-point
            check, not assumed), or a callable.
        eps: stopping tolerance on ``||grad f(X_k)|| / ||grad f(X_0)||``.
    Returns:
        ``X`` or ``(X, RiemHistory)``.
    Warns:
        RuntimeWarning: ``maxit`` reached (carries the achieved gradient norm and
            the achieved ``f``); or the objective stalled while the gradient is
            still large -- the plateau of Sec. 5.2, measured to last 1400
            iterations on a completion problem that then converged.
    Raises:
        ValueError: ``X0`` is rank deficient (the tangent space is undefined);
            numpy backend with an ``f`` that needs autodiff.
    """
```

```python
@dataclass
class RiemHistory:
    f: list                 # objective per iteration, MEASURED
    gradnorm: list          # ||grad f(X_k)||, from tangent_inner -- cheap, exact
    step: list              # accepted step length
    ls_evals: list          # objective evaluations spent in the line search
    normal_defect: list     # ||(I - P) grad_E f||, or NaN if not requested -- the
                            # rank-adaptation signal, free once the deltas exist (Sec. 1.5)
    discarded: list         # Frobenius norm thrown away by each retraction
    ranks: list
    converged: bool
    stop_reason: str
    nit: int
    time: float
```

`normal_defect` is the field this API exists to expose: it is the quantity that
says "the fixed rank cannot represent the descent direction", it is exactly what
`ksl.tangent_defect` already measures for the integrator, and after §1.5 it
costs nothing.

---

## 10. Validation tests, with named oracles

Every test names its oracle. Nothing is compared against the module's own output
except where the test is explicitly about two code paths agreeing.

1. **`project` against a dense tangent projector.** Oracle: `P` assembled from
   `numpy.linalg.svd` of the dense unfoldings of `X`. Exists
   (`test_project_matches_dense_projector`); measured 7.3e-16 … 1.5e-15 (§1.2).
   Keep unchanged.

2. **The gauge condition and the rebuild.** `project_delta` then `tangent_to_tt`
   must reproduce `project` to `4 * eps(dtype) * scale`, and
   `max_k |ML(δG_k)^H ML(U_k)|` must be below `8 * eps * d`. Oracle: the
   definition, [RNO19] eq. (21). Measured today: rebuild 2.9e-16 … 9.5e-16,
   gauge 7.9e-16 … 4.3e-14 (the 4.3e-14 at `d = 10`, i.e. it grows with `d` and
   the threshold must too).

3. **The cheap inner product equals the TT contraction.** `tangent_inner(a, b)`
   against `tt.dot(tangent_to_tt(a), tangent_to_tt(b))`, relative 1e-14.
   Oracle: [RNO19] eq. (22). Measured 3.6e-16 … 8.1e-16. This is also the test
   that pins the **sign** of the gauge projection (§4.3): with the paper's
   Alg. 5.2 "+" instead of the "−" of its eq. (5.11), the deltas are not in the
   gauge complement and this test fails.

4. **Riemannian autodiff against `project(X, ∇f)`.** `f(X) = <A X, X>` with
   symmetric `A`, whose Euclidean gradient `2 A X` we can form: assert
   `‖riemannian_grad(f, X) − 2 project(X, A X)‖ / ‖·‖ <= 1e-13`. Measured
   1.5e-15, 2.6e-15, 2.5e-15 (§4.3). The one place where comparing against our
   own code is the right test: it pins that the two routes to the same
   mathematical object agree.

5. **THE FAIL-LOUD TEST: an underdetermined completion.** `d=6, n=10`, target of
   TT rank 3 with orthogonal cores, `|Ω| = 10 × dof = 4200`. Measured (§5.2):
   both Riemannian GD (3000 iterations, 27.6 s) and `ttSparseALS` (300 sweeps,
   2.0 s) stop with a *training* error of 0.59 and a *test* error of 2.4 and
   11.7 respectively — a fit worse than returning zero. Assert that
   `rgd` **warns**, that `history.converged` is False, and that the message
   carries the achieved gradient norm. `ttSparseALS` already raises the
   corresponding `RuntimeWarning` about underdetermined slices; the Riemannian
   solver must not be quieter than the ALS one.

6. **THE SECOND FAIL-LOUD TEST: autodiff of a function that is not a function of
   the tensor.** Give `riemannian_grad` an `f` that reads the cores directly
   (e.g. `sum(c.sum() for c in cores)`, which changes under a gauge
   transformation that leaves the tensor fixed). `runtime_check` must **raise**,
   naming the relative discrepancy. t3f asserts the same condition; it prints a
   warning and uses `tf.Assert`. We raise, because a Riemannian gradient of such
   a function does not exist and returning a plausible array is the exact
   failure mode this project forbids.

7. **THE THIRD FAIL-LOUD TEST: differentiating through a degenerate SVD.**
   The two-core reproducer of §4.2 (singular values exactly `(1,1)`) currently
   returns **NaN** gradients from `round_cores`. Assert that `riemannian_grad`
   raises when the returned deltas contain a non-finite entry, with a message
   naming the SVD. This test is expected to *stay* — it pins a torch property we
   do not control, and it is why `riemannian_grad` never puts an SVD on the tape.

8. **THE FOURTH FAIL-LOUD TEST: `x.norm()` in an objective.** Until §1.4 is
   fixed this is a wrong answer with no signal (measured absolute gradient error
   7.0e-01). After the fix, assert `riemannian_grad` on
   `f = lambda cs: _ops.norm(_ops.sub(cs, b.cores))**2` matches central finite
   differences to 1e-6. Regression guard, forever.

9. **`project` refuses a rank-deficient point.** Exists
   (`test_project_refuses_a_rank_deficient_point`). Add the `rgd` companion:
   starting `rgd` from a rank-deficient `X0` raises with the same message, not
   at iteration 40.

10. **The preconditioned iteration count is `κ`-independent.** `D=4` Laplacian,
    physical modes, `n = 32` and `n = 128`, manifold rank 1, sinc exponential
    sum with `ρ_B = 41 / 51`. Oracle: `λ_1 = D 4 sin²(π/(2(n+1)))`, analytic.
    Assert the preconditioned run reaches relative 1e-8 in `<= 25` iterations at
    both `n` (measured 16 and 17, §6.3) **and** that the unpreconditioned run at
    `n = 128` does not reach 1e-3 in 200 (measured: not in 3000). The second
    assertion is what makes the test about the preconditioner and not about the
    problem.

11. **`retract(X, 0) == X`, and first-order accuracy.**
    `‖retract(X, t ξ) − (X + t ξ)‖ = O(t²)` for tangent `ξ`, checked at
    `t = 1e-1, 1e-2, 1e-3` with an observed order in `[1.8, 2.2]`. Oracle: the
    definition of a retraction. `test_projector_splitting_add_is_a_first_order_retraction`
    exists and does this for `method='psa'`; extend to `'svd'`.

12. **`tangent_defect` after the refactor is bit-comparable.** `ksl.tangent_defect`
    rewritten on top of `project_delta` must reproduce today's value to 1e-14 on
    the three problems of §1.5 (measured agreement today: 0.0, 0.0, 1.54e-16).
    This is the SSOT regression guard.

13. **Backend parity.** The same `rgd` run on numpy (with a hand-written
    gradient) and torch CPU (with `riemannian_grad`) must agree to 1e-12, and the
    torch-CPU and torch-GPU `riemannian_grad` to 1e-13 (measured 5.0e-15 …
    1.1e-14 in `p6_gpu.py`). Mirrors `tests/test_algs_torch.py`.

---

## 11. Hard test problems

Marked (i) runnable today, (ii) runnable after this spec lands, (iii)
aspirational. "Reference" says where the number comes from and whether it was
verified here.

### 11.1 (i) Tensor completion at known rank

`d = 6`, `n = 10`, target a random point of `M_3` with left-orthogonal cores
(QR of Gaussian blocks), `dof = 420`, `|Ω| ∈ {10, 30, 100, 300} × dof` uniform
without replacement, seed 5, test set 50 000 held-out entries.
Reference: **measured here** (§5.2). The regime boundary is the useful number:
at 10x and 30x nothing recovers the tensor (train 0.59/0.76, test 2.4/2.3); at
100x Riemannian GD reaches train 1.6e-13 / test 2.0e-13 in 59 iterations and
5.4 s, `ttSparseALS` reaches 6.6e-08 in 20 sweeps and 1.2 s.
Difficulty: the plateau. The same `|Ω| = 100 × dof` with a different draw of Ω
took **1420** iterations instead of 59 (measured).

**A hypothesis that was measured and did not hold, recorded so it is not
re-tried.** The first target was built from raw Gaussian cores rather than
orthogonal ones and nothing converged at `|Ω| ∈ {4, 10} × dof` (train 0.40–0.69,
test 1.7–10.8, for Riemannian GD, Riemannian CG and `ttSparseALS` alike). The
natural explanation — a Gaussian-core TT is entrywise spiky, so a uniform sample
set sees none of the mass — was tested by switching the target to
left-orthogonal cores. **It changed nothing at that sampling ratio**: on the
orthogonal target with `|Ω| = 4200 ≈ 10 × dof`, `ttSparseALS` from three
different random starts reached train 0.586 / 0.595 / 0.597 and a *full-tensor*
relative error of 6.4 / 8.1 / 6.3, while the same ALS started **from the true
tensor** stayed at a full-tensor error of 3.2e-15 with a fit of 4.3e-30
(`p3dbg.py`). So the instance is well posed and the solvers are not broken —
the binding constraint is the **sampling ratio**, not the target construction,
and the threshold on this problem lies between 30x and 100x the dof. (The
orthogonal target is still the better benchmark: measured
`max|entry| / median|entry| = 133` even there, and worse for the Gaussian one,
which is not measured.)

### 11.2 (i) Rayleigh-quotient minimization, 1D QTT Laplacian

`A = tt.qlaplace_dd([d])`, `d = 6, 8, 10`, manifold rank 4.
Reference: `λ_1 = 4 sin²(π/(2(N+1)))`, analytic, `N = 2^d`.
**Measured here** (§6.1): 1413 iterations (truncated SD) and 1398 (Riemannian
GD) to 1e-3 at `d = 6`; neither reaches 1e-3 at `d ≥ 8`. This is the *negative*
benchmark: it is in the suite to keep anyone from claiming a Riemannian
eigensolver is usable on QTT elliptic problems without a preconditioner.

### 11.3 (i) Kronecker-sum Laplacian with an exponential-sum preconditioner

`A = Σ_{i=1..D} I ⊗ ... ⊗ L ⊗ ... ⊗ I`, `L = tridiag(−1,2,−1)` of size `n × n`,
`D = 4`, `n ∈ {32, 128, 512}`, manifold rank 1, `B⁻¹` a sinc exponential sum with
`ρ_B = 41/51/61` rank-1 TT-matrix terms.
Reference: `λ_1 = D · 4 sin²(π/(2(n+1)))`, analytic. **Measured here** (§6.3):
preconditioned 10/11/12 iterations to 1e-3 and 16/17/18 to 1e-8 across
`κ = 4.4e2 … 1.1e5`; unpreconditioned 475 at `κ=4.4e2` and never at `κ ≥ 6.7e3`.
This is the benchmark that makes the preconditioner interface (§6.4) testable
without waiting for BPX.

### 11.4 (ii) Riemannian linear solve `A X = F` with `stop_gradient`

`A = tt.qlaplace_dd([d]*D)`, `F` a rank-1 right-hand side, objective
`h(X) = <B A c(X), X> − <B F, X>` ([NRO22] §5.3), preconditioner as in 11.3 for
the physical-mode case. Reference: `amen_solve` at `eps = 1e-10`, plus the
residual `‖A X − F‖ / ‖F‖` as a self-contained oracle. Difficulty: this is the
first test of `stop_gradient` in ttpy2 (`X.detach()` on the torch backend) and
of a *nonsymmetric* `B A`. **Not measured.**

### 11.5 (ii) Rayleigh quotient with a fixed block of `b` eigenvectors

The LRRAP setting of [RNO19] §3.2. Owned by `docs/plans/eigenvalues.md` §2.2;
this spec supplies `tangent_gram` (the `O(b² d n r²)` Gram) and the `prec=` list
interface. Reference and parameters: `docs/plans/eigenvalues.md` §9.3.

### 11.6 (ii) Riemannian Hessian-vector product against finite differences of the gradient

`f = <A X, X>`, `Z` a tangent vector: `H_X[Z] = 2 P_X A Z` exactly, so the
oracle is `2 project(X, tt.matvec(A, tangent_to_tt(X, Z)))`. Assert 1e-13.
The one problem where the approximate Hessian and the true one coincide (the
curvature term vanishes for a quadratic at a stationary point only — so the test
must be read as "the implemented object is `P_X ∇² f Z`", not "the Riemannian
Hessian"). **Not measured.**

### 11.7 (iii) Hénon–Heiles / molecular vibrational spectra

`docs/plans/eigenvalues.md` §9.4 and §9.5 own these. What this spec adds: they
are the *only* problems for which [RNO19]'s preconditioner (24) is stated to
hold, so they are the acceptance test for §6.4 item 1. Aspirational for the same
reason as there: the reference tables and the potential energy surfaces are not
in hand.

### 11.8 (iii) Exponential machines / a TT-parametrized model trained by Riemannian optimization

[NRO22] §7.1's fifth benchmark: `f(X) = Σ_i h(<X, W^{(i)}>, y^{(i)})` with rank-1
`W^{(i)}`, `d = 10`, `n = 500`, minibatch 32. Reference: t3f's published timings
(§7.3) and the accuracy of Novikov–Trofimov–Oseledets. Aspirational because it
needs a data loader and a batch container, not because of the mathematics. It is
the benchmark that would justify the batch TT container of §7.2.

---

## 12. What I did not verify

* **[BK20] was read only to page 1.** Everything §6.4 item 4 says about
  representation ill-conditioning is a paraphrase of its abstract. The
  interaction between a BPX preconditioner and a *Riemannian* iteration (which
  lives entirely in the representation) is not analysed anywhere I read.
* **[CL22] and [CKL23] were read only to page 1 here** — but they are read in
  full in `docs/plans/bug-integrator.md`, and §5.3 option 4 leans on that
  reading, not on mine.
* **[RNO19]'s figures were not rendered**; §6.2's account of eq. (24)–(25) is
  from the text, which is unambiguous there.
* **No prototype of `riemannian_hvp`.** Everything in §4.5 and §11.6 is
  structural. In particular the claim that the approximate Hessian is stable
  where the exact one is not is [NRO22]'s, not measured here.
* **No prototype of `stop_gradient` / §11.4.** `X.detach()` was not tried inside
  a ttpy2 objective. The claim that every op in `_ops.py` accepts a detached
  operand is read from the code, not run.
* **No prototype of `project_matvec`** (the fused projected matvec). The §4.4
  comparison is AD against our *naive* route only; [NRO22] Table 2a shows their
  "improved" route beating AD in some cells, and we have no equivalent, so I
  cannot say whether a hand-written fused path would beat AD in ttpy2.
* **geomCG was not implemented.** §5.2's negative CG result is about a naive
  Fletcher–Reeves CG that I wrote, and must not be read as a statement about
  Steinlechner's method.
* **No rank-adaptive Riemannian method was prototyped.** §5.3 options 3 and 4
  are proposals.
* **The GPU numbers are from `cuda:0` on a shared machine.** Utilization was 0 %
  and memory 29 GB of 275 GB at the time (`nvidia-smi` checked before each run,
  GPUs 2–7 were at 94–100 %), but no exclusivity was enforced; a neighbour's job
  starting mid-measurement would show up as noise. Medians of 3.
* **The `p5_adverse.py` §(C) timing table is discarded** and does not appear
  here: it generated TT ranks that were structurally impossible at the boundary
  bonds (`r = 20` at a bond with `n = 4` on one side), so the numpy path silently
  rounded them down and the torch path did not, and the two columns measured
  different problems. §4.4 (`p7_timing.py`) repeats it with capped ranks and
  fixed thread counts, and its numbers differ by up to 10x. Recorded because the
  first table looked entirely plausible.
* **Every completion number is a single run at a single seed**, and §5.2 measures
  a 24x spread in iteration count between two draws of Ω at the same `|Ω|`. Treat
  those iteration counts as orders of magnitude.
* **No measurement of `tangent_gram` at large `b`.** The `O(b² d n r²)` versus
  `O(b² d n r³)` claim is [RNO19] eq. (22) plus our 8e-16 agreement check at
  `b = 2`, not a timing.
* **t3f was audited from source but not executed** (no TensorFlow on either
  host). In particular `utils.in_eager_mode()` calls `context.in_eager_mode()`,
  which was the TF1 spelling; whether it still resolves under TF 2.4 — and hence
  whether t3f's eager autodiff path runs at all today — is unverified.
* **No numpy-backend autodiff exists and none was attempted.** §8(h) raises on
  numpy. Whether a numpy adjoint layer is worth writing is §13 Q5.

---

## 13. Open questions

**Q1 — Should `bk.norm` return a 0-d backend scalar, or should there be a
separate differentiable norm?** §1.4 recommends the former (one owner) and
audits the 9 call sites that would change. The experiment that settles the
compatibility half: run the full suite on b300 with the change in place and
count failures; the ones that matter are formatting and comparisons in
`tests/test_backend_torch.py` and `tests/test_algs_torch.py`. Cheap, not yet run.

**Q2 — In the QTT regime, is a Riemannian step ever worth it?** §1.2 measures
`project` as dispatch-bound at `n = 2` (flat in `r_z` from 20 to 80), §4.4
measures AD as a wash at `d=20, n=2, r=10` and the GPU as 2.7x *slower* there,
and §6.1 measures the tangent projection buying nothing over plain truncation.
The experiment: take the one QTT problem where the rank is genuinely fixed by a
memory budget (a `b`-eigenvector block, `docs/plans/eigenvalues.md` §7.4 measures
`eigb` at `B=16` needing rank 200) and compare LRRAP against block AMEn at equal
wall time. Until then, this spec's QTT claims are all negative ones.

**Q3 — Does growing into the normal space work?** §5.3 option 3. The signal is
free (`‖(I−P)∇f‖`, §1.5); the directions are not. The experiment: completion at
`|Ω| = 30 × dof`, where §5.2 measures *both* fixed-rank methods stalling at 0.76,
with `kickrank ∈ {0, 1, 2}` added to every bond every 10 iterations, comparing
the held-out error. If it does not beat 0.76 there, the option is dead.

**Q4 — Is a second-order retraction worth building?** [NRO22] cites
Absil–Oseledets for the survey and uses none. The experiment that settles it:
on the preconditioned problem of §6.3, where the iteration count is already 10–12,
a second-order retraction can save at most those; on the completion plateau of
§5.2 it would have to shorten a 1400-iteration flat region, which is a
landscape property and not a retraction property. My expectation is "no", and
the cheap version of the experiment is to measure the *retraction error*
(`return_discarded` of §8(e)) along the completion run and see whether it is
ever the dominant term.

**Q5 — Do we want autodiff on the numpy backend?** Today §8(h) raises there, so
a numpy user gets no Riemannian gradient at all and must hand-write one.
Options: (a) require torch for `tt.algs.autodiff` and document it — R1 says GPU
is optional and that absence of torch must not break `import tt`, which this
respects; (b) write adjoints by hand for the handful of expensive contractions
(`matvec_cores`, `dot`, `add`, `round_cores`), which is 200 lines and a
maintenance burden with two owners of every contraction; (c) route numpy through
torch-CPU internally, which contradicts R3's "cores inside one tensor are always
one backend". The experiment: count how much of §11's benchmark set a numpy-only
user could still run under (a). I expect "all of it via hand-written gradients
for the three quadratic functionals", which makes (a) the answer.

**Q6 — Does a BPX preconditioner survive contact with a Riemannian iterate?**
[BK20]'s central claim is that a preconditioner which fixes the matrix
conditioning can leave the *representation* ill-conditioned, and a Riemannian
method never leaves the representation. The experiment: build the [BK20]
preconditioner for `tt.qlaplace_dd([d])`, `d = 10..20`, and measure (i) the
eigenvalue convergence rate and (ii) the condition number of the delta
parametrization (the smallest singular value of the unfoldings, which
`p8_deficient.py` already tracks) along the iteration. If (ii) degrades while
(i) is fine, the BPX spec owes this one a redundancy-elimination step.

**Q7 — Batch TT: is it the thing that makes any of this fast?** t3f's own
benchmark (§7.3) shows batching buying 5x on CPU and 5–1000x on GPU per object,
and [RNO19]'s whole cost argument for `b = 84` eigenvectors rests on it
(`docs/plans/eigenvalues.md` §11 Q6 asks the same question from the other side).
The experiment that settles it for us: implement `tangent_gram` twice — once as
`b²` calls to `tangent_inner`, once as one batched `einsum` over a
`(b, r, n, r)` delta stack — and time both on b300 CPU and GPU at
`b = 4, 16, 64`. This is a half-day experiment and it decides whether the batch
container of §7.2 goes in before or after the optimizers.
