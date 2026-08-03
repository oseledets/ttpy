# BUG: the unconventional robust integrator, for `tt.algs.bug`

Implementation spec. Sources actually read:

* **[CL22]** G. Ceruti, C. Lubich, *An unconventional robust integrator for dynamical
  low-rank approximation*, BIT Numer. Math. **62** (2022) 23–44. Fixed-rank BUG,
  matrix (§3) and Tucker (§5).
* **[CKL23]** G. Ceruti, J. Kusch, C. Lubich, *A parallel rank-adaptive integrator
  for dynamical low-rank approximation*, arXiv:2304.05660v1, 12 Apr 2023, 22 pp.
  This is the **parallel** variant. It recaps the **rank-adaptive** BUG of
  Ceruti–Kusch–Lubich (BIT 62:1149–1174, 2022) in §2.1 — which is the algorithm we
  actually want — and adds three things of its own: (i) a fully parallel variant in
  which K, L *and* S are solved simultaneously and the coupling is recovered by
  assembling a 2r×2r block matrix instead of integrating one (§3.1–3.2); (ii) a
  **step-rejection strategy** (§3.3) that applies to plain rank-adaptive BUG as
  well; (iii) the observation that the parallel variant loses the exactness
  property and the energy/dissipation preservation that rank-adaptive BUG has.
  §1: the TT / tree-tensor-network extension of the parallel variant is *not* in
  the paper — "it is expected that its concepts extend to tensor trains … This will
  be explored in detail in future work."

Everything below marked **[derived]** is our reconstruction, not a quotation. It is
numerically validated (a working prototype was run against `scipy.linalg.expm`
before this document was written; every number quoted as "measured" comes from
that prototype, on this host, `.venv-tmp`, numpy 2.5.1, float64, single thread).

**Cross-spec decisions live in `docs/plans/ROADMAP.md`**, not here. Where this
spec and one of `eigenvalues.md`, `riemannian-autodiff.md`, `qtt-elliptic-bpx.md`
ask for the same function, the reconciled signature and its owner are recorded
there (§2), together with the dependency graph (§1), the preconditioner contract
(§3), the milestone order (§4) and the consolidated open questions (§6). Two
things below were amended by it: §4(b) (the mixed canonical form is
`riemannian.frames`, not a new `_ops` function) and §4(c) (the `expmv_krylov`
underflow guard, **already fixed**, commit `231ce52`).

---

## 1. What it is

BUG = **B**asis **U**pdate & **G**alerkin. Like KSL it integrates
`dy/dt = F(t, y)` — for us `F(t,y) = A y` with `A` a TT-matrix — keeping `y` on a
low-rank manifold. Unlike KSL it is **not** a splitting of the tangent-space
projector. It is two phases:

1. **Basis update.** Evolve the *frames* (in TT: the interface bases at each bond)
   with the old environment held fixed. Each frame equation is linear if `F` is.
   Then set the new basis to an orthonormal basis of `range([old basis, evolved
   basis])` — the **augmented** basis, of at most doubled rank.
2. **Galerkin.** Evolve the small connecting tensor forward in time inside the
   space the augmented bases span, starting from the exact projection of `y0` onto
   that space (which, because `Ran(old) ⊆ Ran(augmented)`, loses nothing —
   [CKL23] §2.2).
3. **Truncate** the augmented result back down with a tolerance `θ`, by SVD.

The four things it does that our `tt.algs.ksl` cannot:

* **It changes rank.** KSL is structurally fixed-rank: `ksl()` even raises if
  `max(y0.r) > rmax` because "KSL keeps the rank fixed, it cannot truncate for
  you". `tangent_defect` measures the off-manifold part `‖(I−P)Ay‖` and the
  history reports it, but nothing can act on it. BUG grows and shrinks the rank
  every step from one tolerance, and the discarded singular values are the error
  estimate — the same quantity, but now *actionable*.
  Measured (d=6, n=2, symmetric random TT-matrix of rank 4 normalised to
  `‖A‖₂=1`, `T=1`, 64 steps, starting from ranks `[1,2,2,2,2,2,1]` which are
  deliberately too small):

  | integrator | final ranks | rel. error vs `expm(T·A)y₀` |
  |---|---|---|
  | KSL, fixed rank 2 | `[1,2,2,2,2,2,1]` | **4.245e-01** |
  | BUG, `θ=1e-2`     | `[1,2,3,3,3,2,1]` | 1.172e-01 |
  | BUG, `θ=1e-4`     | `[1,2,4,8,4,2,1]` | **3.195e-03** |
  | BUG, `θ≤1e-6`     | `[1,2,4,8,4,2,1]` | 3.180e-03 |

* **No backward-in-time substep.** KSL's `S` step is `exp(−τ M)`. For a
  dissipative `A` that substep *grows*. Section 3 gives the measured blow-up.
* **It is nonexpansive when `A` is.** If `A ⪯ 0`, every BUG substep is
  `exp(τ Q^H A Q)` with `Q` orthonormal, hence a contraction, and augmentation and
  truncation are orthogonal projections. So `‖y₁‖ ≤ ‖y₀‖` **for every `τ`**, with
  no step-size condition. This is a conservation law we can assert in a test.
  KSL has no such property (it has the *other* one: it preserves norm and energy
  for Schrödinger, which BUG does not — [CL22] §1).
* **Parallelism** — but see the caveat in §6: for a **tensor train** there is
  essentially none. The parallelism in [CL22]/[CKL23] is over the *children of a
  node*; a TT is a caterpillar tree with one non-leaf child per node, so its BUG
  sweep is strictly sequential. Balanced tree tensor networks are where the
  parallel claim pays. Do not sell TT-BUG as a parallel method.

What we give up relative to KSL: time-reversibility ([CL22] §3, "apparently no
efficient way"); norm and energy conservation for Schrödinger ([CL22] §1); and —
measured — exactness when the manifold is the whole space. `ksl` reproduces
`expm(τA)y₀` to 1.0e-15 on d=6, n=2, `r=[1,2,4,8,4,2,1]`; BUG on the same input
gives 1.99e-05 at `τ=0.1`. That is not a bug in BUG, it is the method: BUG's
Galerkin space at node `k` is `Y⁰_{<k} ⊗ R^{n_k} ⊗ Û_{≥k+1}`, which is a proper
subspace of the full space even when the manifold is not.

---

## 2. The algorithm

### 2.1 Matrix case, fixed rank — [CL22] §3.1, verbatim structure

`Y₀ = U₀S₀V₀ᵀ`, `U₀ ∈ R^{m×r}`, `V₀ ∈ R^{n×r}` orthonormal columns, `S₀ ∈ R^{r×r}`.
Step `t₀ → t₁ = t₀+h`:

```
K-step  (parallel with L)   K(t)  ∈ R^{m×r},  K(t0) = U0 S0
        K' = F(t, K Vo^T) V0
        QR:  K(t1) = U1 R1                     # U1 in R^{m x r}
        M  = U1^H U0                           # r x r
L-step  (parallel with K)   L(t)  ∈ R^{n×r},  L(t0) = V0 S0^H
        L' = F(t, U0 L^H)^H U0
        QR:  L(t1) = V1 R1'                    # V1 in R^{n x r}
        N  = V1^H V0                           # r x r
S-step  (after both)        S(t)  ∈ R^{r×r},  S(t0) = M S0 N^H
        S' = + U1^H F(t, U1 S V1^H) V1         # NOTE: plus, and forward in time
Y1 = U1 S(t1) V1^H
```

Three differences from projector-splitting worth naming: the triangular factors
`R1` are thrown away (KSL reuses them); there is no minus sign; and `U1` here is
identical to KSL's `U1`, while `V1` is in general different ([CL22] §3.1).

### 2.2 Matrix case, rank-adaptive — [CKL23] §2.1

Same K and L steps, but the bases are **augmented** instead of replaced:

```
Û = orth([K(t1), U0])   in R^{m x r̂},  r̂ <= 2r     (QR of the m x 2r stack)
V̂ = orth([L(t1), V0])   in R^{n x r̂}
M̂ = Û^H U0  (r̂ x r),   N̂ = V̂^H V0  (r̂ x r)
Ŝ' = Û^H F(t, Û Ŝ V̂^H) V̂,   Ŝ(t0) = M̂ S0 N̂^H       # r̂ x r̂, forward
```

**Rank selection rule** ([CKL23] eq. (3.5) / §2.1 step 3): SVD `Ŝ(t₁) = P̂ Σ̂ Q̂ᵀ`,
`Σ̂ = diag(σ_j)` sorted non-increasing; choose the **minimal** `r₁ ≤ r̂` with

```
( sum_{j = r1+1}^{r̂}  sigma_j^2 )^{1/2}  <=  theta
```

then `S₁ = diag(σ₁..σ_{r₁})`, `U₁ = Û P₁`, `V₁ = V̂ Q₁` with `P₁, Q₁` the first `r₁`
columns. `θ` is an **absolute** tolerance in the papers; [CKL23] §5.1 uses
`θ = θ̄·‖Σ̂‖` with `θ̄ ∈ [1.5e-3, 5e-2]` in practice, i.e. relative in the
experiments. Note the stacking uses `U0`, **not** `U0 S0`: if `S₀` is numerically
singular those have different ranges, and the whole point is robustness to exactly
that.

**Step rejection** ([CKL23] §3.3) — applies to rank-adaptive BUG, not only to the
parallel variant:

1. If `r₁ = r̂` (i.e. **nothing** was truncated), repeat the step with `Û, V̂` in
   the role of `U₀, V₀`. Rationale: the rank cannot more than double in one step,
   which [CKL23] says (citing a third party) "does not work satisfactorily in
   cases where a steep increase of the rank is necessary within a step, for
   example when starting from rank 1". Repeating gives an arbitrary rank increase
   per step.
2. If `r₁ < r̂`, compute `η = ‖Ũ₁ᵀ F(t₀,Y₀) Ṽ₁‖` where `Ũ₁, Ṽ₁` are the *new*
   halves of the augmented bases, and repeat if `h·η > c·θ` (`c = 10` suggested;
   [CKL23] uses `c = 1` and `c = 5` in the experiments). The identity that makes
   this meaningful is [CKL23] (3.11):
   `‖P⊥_r(Y₀)[Û Ûᵀ F₀ V̂ V̂ᵀ]‖ = ‖Ũ₁ᵀ F₀ Ṽ₁‖ = η` — `η` is the normal component of
   `F₀` **as seen through the augmented subspace**, i.e. a computable surrogate for
   the inaccessible `‖P⊥_r(Y₀)F₀‖`.

### 2.3 Parallel variant — [CKL23] §3.1

K and L as in §2.2, plus a third *independent* r×r equation
`S̄' = U₀ᵀ F(t, U₀ S̄ V₀ᵀ) V₀`, `S̄(t₀) = S₀`, all three solved simultaneously.
The augmented coefficient matrix is then **assembled, not integrated**:

```
Û = (U0, Ũ1),  V̂ = (V0, Ṽ1)         # note the ordering: old block first
S̃K = Ũ1^T K(t1),   S̃L = L(t1)^T Ṽ1
Ŝ1 = [[ S̄(t1) ,  S̃L ],
      [ S̃K    ,   0  ]]              # 2r x 2r
```
then the same truncation rule (3.5). The zero block is justified in §3.2: the true
`(1,1)` block is `∫ Ũ₁ᵀF̂ Ṽ₁ dt` which is `O(hε)` because tangential matrices
`T = KV₀ᵀ + U₀Lᵀ` satisfy `Ũ₁ᵀTṼ₁ = 0`. Its norm is `hη + O(h²)` — the same `η`
as the rejection criterion, which is why the criterion matters *more* here.
Cost saving over rank-adaptive BUG: no 2r×2r ODE. Measured in [CKL23] §5:
26.8s vs 47.6s (planesource), 186/308/506s vs 256/462/782s (linesource),
180/330s vs 286/524s (lattice) — 1.4–1.8× wall clock, from omitting the 2r×2r
update alone, with no actual parallel execution. Cost paid: no exactness property,
no energy conservation, no gradient-flow dissipation.

### 2.4 Tucker — [CL22] §5.1, verbatim structure

`Y₀ = C₀ ⨉ᵢ U⁰ᵢ`, multilinear rank `(r₁..r_d)`.

```
for i = 1..d  in parallel:
    QR of the transposed i-mode matricization of the core:  Mat_i(C0)^T = Q_i S_i^{0,T}
    V_i^{0,T} = Q_i^T  (x)_{j != i} U_j^{0,T}          # r_i x n_{not i}
    K_i' = F_i(t, K_i V_i^{0,T}) V_i^0,   K_i(t0) = U_i^0 S_i^0     # n_i x r_i
    QR:  K_i(t1) = U_i^1 R_i^1
    M_i = U_i^{1,T} U_i^0                                            # r_i x r_i
core Galerkin (serial, after all i):
    C' = F(t, C (x)_i U_i^1) (x)_i U_i^{1,T},   C(t0) = C0 (x)_i M_i
Y1 = C(t1) (x)_i U_i^1
```

Note what is absent versus the Tucker projector splitting: **no r_i×r_i equations
at all** and no backward steps ([CL22] §5, item 2). That is the "reduced serial
computational cost" claim.

### 2.5 Tensor train — **[derived]**, in our conventions

Neither paper writes this down. What follows is derived by treating a TT as a
caterpillar tree tensor network — node `k` has two children, leaf `k` (basis
`I_{n_k}`, *uncompressed*, so nothing to augment) and the subtree `k+1..d` (basis
`U_{≥k+1}`, rank `r_k`, augmentable) — and applying §2.2 at each node. It was
implemented and validated before this spec was written; see §7. Fidelity to
Ceruti–Kusch–Lubich, *Rank-adaptive time integration of tree tensor networks*,
SIAM J. Numer. Anal. **61**(1):194–222 (2023) must be checked (§8, Q1).

Shapes, our conventions throughout: TT cores `G_k : (r_{k-1}, n_k, r_k)`,
`r_0 = r_d = 1`; TT-matrix cores `A_k : (R_{k-1}, n_k, n_k, R_k)` row index first;
interfaces `(bra, alpha, ket)` so `L_k : (r_{k-1}, R_{k-1}, r_{k-1})` and
`Rt_k : (r_k, R_k, r_k)`, exactly as `_localops.phi_left` / `phi_right` build them.
Mode 1 is the fastest index. Code is 0-based (`k = 0..d-1`); the maths below is
1-based.

**Definitions.**
`Y⁰_{<k}` = left frame over modes `1..k-1`, orthonormal columns, `r_{k-1}` of them.
`U⁰_{≥k}` = right/subtree frame over modes `k..d`, orthonormal columns, `r_{k-1}`
of them, represented by the right-orthogonal core `Q⁰_k` sitting on `U⁰_{≥k+1}`.
`S_{k-1} : (r_{k-1}, r_{k-1})` = the bond matrix of the mixed canonical form
`Y⁰ = Y⁰_{<k} · S_{k-1} · (U⁰_{≥k})ᵀ`.

```
BUG_TT_STEP(A, y0, tau, adaptive, theta, rmax):

  # ---- phase 0: mixed canonical form of the OLD tensor -------------------
  Q0[1..d] = orthogonalize(y0.cores, center=0)     # cores 2..d right-orthogonal
  L[0] = ones_interface                            # (1,1,1)
  X[k] : (r_{k-1}, n_k, r_k)                       # the centre core at site k
  cur = Q0[1]
  for k = 1..d:
      X[k] = cur                                   # == S_{k-1} Q0[k]
      if k < d:
          q, s   = left_orthogonalize(cur)         # q : (r_{k-1}, n_k, r_k)
          L[k+1] = phi_left(L[k], A[k], q, q)      # (r_k, R_k, r_k)
          cur    = s @ Q0[k+1]                     # einsum "a b, b n c -> a n c"

  # ---- phase 1+2 fused: ascent, Galerkin then augment --------------------
  Rt[d+1] = ones_interface
  Mhat    = None                                   # (r̂_k, r_k)
  for k = d down to 1:
      # -- initial value: y0 expressed in (Y0_{<k}, leaf k, Û_{>=k+1})
      if Mhat is None:  G0 = X[k];  OLD = Q0[k]                  # k == d
      else:             G0 = X[k]  x_3 Mhat^T                    # (r_{k-1}, n_k, r̂_k)
                        OLD = Q0[k] x_3 Mhat^T                   # same shape
      #   einsum: "a n c, b c -> a n b"
      # -- Galerkin / K equation at node k; local operator = (L[k], A[k], Rt[k+1])
      G1 = expmv( x -> local_matvec(L[k], A[k], Rt[k+1], x),  G0,  +tau )
      if k == 1:  out[1] = G1;  break
      # -- augment the parent leg
      Z    = concat([OLD, G1], axis=0)   if adaptive else G1     # (2 r_{k-1}, n_k, r̂_k)
      _, Ĉ = right_orthogonalize(Z)                              # Ĉ : (r̂_{k-1}, n_k, r̂_k)
      Mhat = <Ĉ*, OLD> over (n_k, r̂_k)                           # (r̂_{k-1}, r_{k-1})
      #   einsum: "b n c, a n c -> b a"
      Rt[k] = phi_right(Rt[k+1], A[k], Ĉ, Ĉ)                     # (r̂_{k-1}, R_{k-1}, r̂_{k-1})
      out[k] = Ĉ

  # ---- phase 3: truncate; out[2..d] are already right-orthogonal ---------
  #   one left-to-right SVD sweep with per-bond budget theta/sqrt(d-1)
  return round_absolute(out, theta, rmax)
```

Key facts that make this correct and cheap:

* `X[k] = S_{k-1} Q⁰_k` *is* what a left-orthogonalisation sweep produces at site
  `k`; the descent is byte-for-byte the entry sweep `ksl()` already does, plus one
  extra right-orthogonalisation up front. `L[k]` is built from the **old** frames
  only — the K equation must use the old co-range (`V₀`, not `V̂`).
* The Galerkin equation at node `k` uses **old left, new right**. That is what the
  recursion gives: the parent hands node `k` the environment `Y⁰_{<k}`, node `k`
  first updates its only compressible child (getting `Û_{≥k+1}`), then does its
  Galerkin inside `Y⁰_{<k} ⊗ R^{n_k} ⊗ Û_{≥k+1}`. At `k=1` the left environment is
  trivial, matching the root Galerkin of §2.2.
* The augmentation **never stacks TTs**. `U⁰_{≥k}` and `K_{≥k}(t₁)` are both
  expressed over the *same* child basis `Û_{≥k+1}` (legitimate because
  `Ran U⁰_{≥k+1} ⊆ Ran Û_{≥k+1}` by construction, and `Mhat` performs the change of
  basis). So augmentation is a concatenation of two connecting cores along axis 0
  and one `right_orthogonalize`. Internal ranks stay `≤ 2r`; there is no
  `3^depth` blow-up.
* Rank caps: `r̂_{k-1} = min(2 r_{k-1}, n_k · r̂_k)`. Measured on d=6, n=2,
  `r=[1,2,4,8,4,2,1]`: `r̂ = [1,4,8,8,4,2,1]` (the right half cannot grow — it is
  already maximal for a 2⁶ tensor), Galerkin block sizes `r_{k-1}·n_k·r̂_k` =
  `[8, 32, 64, 64, 16, 4]` for `k = 1..6`. Exactly `d = 6` local exponentials.

**Rank selection in TT.** The augmented train is truncated by one ordinary
left-to-right SVD sweep. Per-bond budget `δ = θ/√(d−1)`, and at bond `k` keep the
minimal `r_k` with `(Σ_{j>r_k} σ_j²)^{1/2} ≤ δ` — i.e. `_ops.chop(s, δ)`, which is
already the right function, but needs an **absolute** `δ` (§4). Report the
discarded 2-norm per bond; `√(Σ_k discarded_k²)` is the step's truncation error and
is the primary rank indicator.

**Fixed-rank BUG** = the same sweep with `Z = G1` (no concatenation) and no
truncation. Ranks are preserved exactly. Use this as the like-for-like comparison
against `ksl`.

---

## 3. Why it is robust to small singular values

**What goes wrong without either integrator.** The factor ODEs of dynamical
low-rank approximation (Koch–Lubich 2007, Prop. 2.1, cited as [12] in [CL22]) carry
a factor `S(t)⁻¹` on the right-hand side. A Runge–Kutta method applied to them has
a step-size restriction proportional to `σ_min`. And `σ_min` is *always* small:
"the smallest singular value retained in the approximation cannot be expected to be
much larger than the largest discarded singular value of the solution, which is
required to be small for good accuracy" ([CL22] §2, and [CKL23] §1 makes the sharper
point that the manifold curvature is proportional to `1/σ_min`). [CL22] Fig. 1
shows RK4 on those equations failing exactly this way where BUG does not, on
`A(t) = e^{tW₁} e^{tD} e^{tW₂ᵀ}` with singular values `e^t 2^{-j}`, N=100, ranks
4…32.

**Why BUG escapes it.** `S⁻¹` never appears. The K/L equations are posed for
`K = U S` and `L = V Sᴴ` — the *products* — not for `U`, `V`, `S` separately, so the
inversion that produces `S⁻¹` is never performed. Every step is a QR (unconditionally
stable) or a small ODE whose right-hand side is a compression `Qᴴ A Q` with `Q`
orthonormal, hence bounded by `‖A‖` with no dependence on the spectrum of `S`.
Formally, [CL22] Thm 4 = [CL22] Thm 2 verbatim: `‖Yₙ − A(tₙ)‖ ≤ c₀δ + c₁ε + c₂h`,
`cᵢ` depending only on `L`, `B`, `T`, **independent of singular values**. The proof
(§3.4) is a perturbation argument that *borrows* the projector-splitting local error
bound `ϑ` — BUG is robust because PSI is, plus Lemmas 2–4. So both are robust;
robustness is not the discriminator.

**What projector-splitting has that BUG avoids.** Exactly one thing: the sign.
KSL's `S` substep is `Ṡ = −U₁ᴴF(·)V₀`, integrated **forward** over `[t₀,t₁]` with a
minus sign, i.e. the flow `exp(−τM)`. [CL22] §3: "for strongly dissipative
problems, [this] is an unstable substep of the algorithm. This does not appear in
the new algorithm." Our `ksl.py` implements it as `_step_exp(..., -tau0, ...)` in
both `_sweep_forward` and `_sweep_backward`.

**Is it a real problem? Yes — measured.** Neither paper demonstrates it
numerically; both only assert it ([CL22] §1 and §3; [CKL23] §1 and §3). We
measured it. Test problem: the QTT heat equation `dy/dt = Δ_h y` on `2^L` points,
`A = −(2^L+1)² · tt.qlaplace_dd([L])`, `‖A‖₂ ≈ 4(2^L+1)²`, `y₀` a random TT of
rank 4 normalised to 1, `local_tol=1e-8`, `scheme='symm'`, `check_rank=False`.
Fixed rank for both integrators, so the modelling error is identical.

| L | τ | τ‖A‖ | `‖exp(τA)y₀‖` | KSL `‖y₁‖` | BUG `‖y₁‖` |
|---|---|---|---|---|---|
| 6 | 1e-5 | 0.2 | 9.133e-01 | 9.133e-01 | 9.138e-01 |
| 6 | 1e-4 | 1.7 | 4.853e-01 | 4.850e-01 | 4.951e-01 |
| 6 | 1e-3 | 16.9 | 1.336e-01 | 1.296e-01 | 1.807e-01 |
| 6 | 1e-2 | 169.0 | 9.043e-02 | **1.0677e+108** | **8.903e-02** |
| 10 | 1e-5 | 42.0 | ≤ 1 | **27.9** (rel. err 263) | 9.86e-02 |
| 10 | ≥1e-4 | ≥420 | — | `RuntimeError` from `expmv_krylov` | runs |

Read the two boldface rows. At `τ‖A‖ = 169` KSL **returns a number**: a tensor of
norm 1.07e+108 for a dissipative flow whose exact solution has norm 0.090. It does
not raise, and with `check_rank=False` nothing in the history flags it. At L=10,
`τ‖A‖ = 42`, KSL returns `‖y₁‖ = 27.9 > 1 = ‖y₀‖` — a *growing* solution of a
contraction semigroup. Beyond `τ‖A‖ ≈ 400` the Krylov substep control finally gives
up and raises, which is the honest failure, but the interval
`40 ≲ τ‖A‖ ≲ 300` is a silent-wrong-answer zone.

The mechanism is not mysterious: with `A ⪯ 0`, the S-step operator `M = L·Rt` is a
compression of `A`, so `−M ⪰ 0` and `exp(−τM)` amplifies by up to `exp(τ|λ_min(M)|)`.
Measured spectra of `interface_matrix(L[k], Rt[k])` at the entry state, L=6,
normalised Laplacian: `max λ(−M) = 3.79, 3.96, 3.36, 3.25, 2.52` at bonds 1..5,
i.e. **94–99 % of `‖A‖`** — the retained bases do contain unresolved-scale
components, so the S-step really does see the full stiffness. In exact arithmetic
the following K-step decays by the reciprocal and the product is `O(1)`; in floating
point the answer carries a relative error of order `eps · exp(2τ‖A‖)`, which passes
1 at `τ‖A‖ ≈ 18` — matching where the table goes wrong. BUG never forms either
factor: every substep has `‖exp(τ Qᴴ A Q)‖ ≤ 1`, so `‖y₁‖ ≤ ‖y₀‖` unconditionally.

**Honest caveat.** BUG is *not* stiff-stable in the sense of removing the
resolution requirement. At L=6, `τ = 1e-1` (`τ‖A‖ = 1690`) the BUG prototype's own
Krylov substeps underflow and it raises too. Forward-only removes the *cancellation*
catastrophe, not the *stiffness*. For genuinely parabolic problems the substep
solver should be implicit or rational, not Arnoldi (§8, Q5).

---

## 4. What the core is missing

Everything the sweep needs from `_localops` already exists and is directly
reusable: `operator_cores`, `ones_interface`, `phi_left`, `phi_right`,
`local_matvec`, `local_matrix`, `left_orthogonalize`, `right_orthogonalize`.
`interface_matvec` / `interface_matrix` are the KSL S-step and BUG never calls
them. `bk.concatenate(arrays, axis=0)` exists at module level and dispatches
correctly. `expmv_krylov` works unchanged for `t > 0`. Genuinely missing:

**(a) Absolute-tolerance rounding that reports what it discarded.** `round_cores`
currently hard-codes `delta = eps * ‖Y‖ / sqrt(d-1)` and throws the singular values
away. BUG needs an absolute `θ` and needs the tail. One owner, extended — not a
near-duplicate function:

```python
def round_cores(cores, eps=1e-14, rmax=None, abs_tol=None, return_discarded=False):
    """...
    abs_tol: absolute Frobenius budget for the whole truncation, split as
        abs_tol/sqrt(d-1) per bond.  Mutually exclusive with a nonzero eps.
    return_discarded: also return ``[float]`` of length ``d-1``: the 2-norm of the
        singular values dropped at each bond.  ``sqrt(sum(x**2))`` is the total
        truncation error and is exact (it is a norm of a discarded orthogonal
        complement, not a cancelling difference).
    """
```

**(b) The mixed canonical form.** `orthogonalize(cores, center)` gives one side.
BUG needs the left frames, the right-orthogonal cores, and the bond matrices
simultaneously:

```
Y = qL[0]..qL[k-1] @ s[k] @ qR[k+1]..qR[d-1],   for every k at once,
centre[k] = s[k-1] @ qR[k] : (r_{k-1}, n_k, r_k)
```

Cost: one right-to-left and one left-to-right QR sweep, `O(d n r^3)`. (If we
want to keep the surface small, `centre` alone plus `qR` is what the sweep uses;
`qL` is only needed to build `L[k]`, which the sweep does inline.)

**This spec originally proposed a new owner `_ops.mixed_canonical(cores)`. That
proposal is withdrawn.** `docs/plans/riemannian-autodiff.md` §8(d) independently
asks for the same object as `riemannian.frames(X, mu=1)`, and two owners of one
mathematical object is exactly what the project forbids.
`docs/plans/ROADMAP.md` §2.3 resolves it: the owner is
`riemannian.frames(X, mu='all', check_rank=False)`, accepting a bare core list,
with `mu='all'` returning every bond matrix and centre core. The
`check_rank=False` flag exists **for this sweep**: `frames` refuses a
rank-deficient `X` by default because the tangent projector is then the
projector of the wrong space (a 31 % relative error is recorded there), but BUG
must run at exactly those points — robustness to arbitrarily small singular
values is the whole method (§3), and BUG never builds a tangent projector. The
cost of the decision is that `tt/algs/bug.py` imports `tt/algs/riemannian.py`.

**(c) `expmv_krylov` mid-loop underflow guard — a latent defect, reproduced.**
`expmv_krylov` guards `beta0 == 0` on entry but not inside the substep loop. When
the local operator is strongly negative-definite the iterate underflows to exactly
zero between substeps; the next `_arnoldi` divides by `beta = 0`, the error estimate
becomes `NaN`, the shrink loop exhausts, and the call raises
`"the substep could not be reduced enough (estimated local error NAN vs target
0.000E+00); the local operator is likely not what the caller thinks"` — which
blames the caller for an underflow. Reproduced from `t5.py` above (L=6 heat,
`τ=0.1`) and from `B = diag(linspace(-4000,-5000,20))`. The fix is one branch: a
zero iterate has zero flow, so return zero and say so in `info` (`underflow=True`),
rather than manufacturing a `NaN` and a misleading message. This is not
BUG-specific; KSL can hit it too.

**(d) A truncation-aware history dataclass.** `KslHistory` has no field for
per-bond discarded mass, augmented ranks, or retries. `BugHistory` is new (§5).

**(e) Nothing needed for Tucker.** ttpy2 has no Tucker format at all. §2.4 is
documented here for completeness; implementing it is a separate decision.

---

## 5. Proposed API

`tt/algs/bug.py`, exported as `tt.algs.bug`, with a thin `tt/bug/__init__.py` if we
want to match the `tt.ksl` shim style.

```python
def bug_step(A, y0, tau, *,
             adaptive=True,          # augment bases and truncate; False = fixed rank
             tol=1e-10,              # truncation budget, RELATIVE to ||y0||
             abs_tol=None,           # absolute budget; overrides tol if given
             rmax=2000,              # hard cap on the rank after truncation
             space=8,                # Krylov dimension of the local exponentials
             local_tol=1e-8,         # relative accuracy of every local exponential
             reject=True,            # [CKL23] 3.3 criterion 1: retry when nothing
                                     # was truncated (the doubling cap bit)
             max_retries=2,
             verb=1,
             return_history=False):
    """One BUG step: y(tau) for dy/dt = A y, y(0) = y0, on the TT manifold.

    Returns the TT-vector y(tau), or (y, BugHistory) if return_history.

    Raises:
        ValueError  on shape mismatch, or if rmax makes `tol` unreachable and
                    the caller asked for `strict=True`.
        RuntimeError if a local exponential cannot reach `local_tol`.
    """
```

Arguments deliberately **not** offered: `scheme`. BUG has no palindromic
composition; `bug_step(A, y, tau/2)` twice is not a second-order method. If we want
order 2, that is the midpoint variant (Ceruti–Kusch–Lubich–Schrammer, BIT 2024), a
different algorithm and a separate entry point. Offering `scheme='symm'` here would
be a lie by API.

`adaptive=True` and `tol` relative by default, because that is the convention every
other ttpy2 entry point uses (`round`, `amen_solve`); `abs_tol` exists because the
papers' `θ` is absolute and someone reproducing Fig. 6 of [CKL23] needs it.

```python
@dataclass
class BugHistory:
    tau: complex
    adaptive: bool
    ranks: list              # TT ranks of the result
    ranks_augmented: list    # r̂ before truncation -- shows where the doubling
                             # cap bound the growth
    rank_capped: bool        # any bond where r1 == r̂: nothing was truncated there,
                             # so the rank wanted to grow further than 2r allowed
    discarded: list          # per-bond 2-norm of the dropped singular values
    trunc_error: float       # sqrt(sum(discarded^2)) -- the step's truncation error
    retries: int             # how many times the step was repeated (reject=True)
    steps: list              # one dict per node: node, size, substeps, krylov,
                             # err_est, time
    max_local_err: float
    total_substeps: int
    time: float
```

The two numbers a caller must be able to act on:

* `trunc_error` — how much of the step was thrown away. Compare with `tol·‖y‖`.
* `rank_capped` — the tolerance was met only because the rank could not grow
  further. **This is the case where a plausible answer hides a real failure**, and
  it is why `reject=True` is the default. Measured in §7, test 6.

There is deliberately no `check_rank` / `defect_warn` pair as in `ksl`: in BUG the
discarded singular values already *are* the rank indicator, computed for free, so a
separate `tangent_defect` call (one TT matvec at rank `R·r` plus a sweep, measured
at 1.6 ms of KSL's 9.8 ms on the reference problem) is redundant. `tangent_defect`
stays where it is, in `ksl.py`, and BUG does not call it.

`bug` (plural steps, matching `ksl`'s single-step contract) is **not** proposed:
`ksl` is one step and BUG should be too, so a time loop looks the same for both.

---

## 6. Cost

Notation: `d` modes, mode size `n`, TT rank `r`, TT-matrix rank `R`, Krylov
dimension `m` (= `space` × substeps).

**One `local_matvec(L, A_k, Rt, x)`** with `x : (p, n, q)`:
`n·p²·q·R + n²·p·q·R² + n·p·q²·R`. With `p = q = r` that is
`2 n r³ R + n² r² R²`. With `q = 2r` (BUG's augmented right leg) it is exactly
**2×** that.

| | local exponentials per step | block size | dominant cost |
|---|---|---|---|
| KSL `scheme='first'` | `d` K + `(d−1)` S = `2d−1` | `r n r` / `r²` | `d·m·(2nr³R + n²r²R²)` |
| KSL `scheme='symm'` | `2(2d−1)` | same | **2×** the above |
| BUG fixed rank | `d` | `r n r` | `d·m·(2nr³R + n²r²R²)` |
| BUG rank-adaptive | `d` | `r n 2r` | `2 d·m·(2nr³R + n²r²R²)` |

So asymptotically: **fixed-rank BUG = half of KSL-symm's dominant term**;
**rank-adaptive BUG = the same as KSL-symm's K-part**, and KSL additionally pays
its `2(d−1)` S-steps. BUG is *not* an order-of-magnitude win in flops. Its extra
serial costs are: one additional orthogonalisation sweep up front
(`O(d n r³)`), `d−1` augmentation QRs of `(2r × n·2r)` (`O(d n r³)` with a
constant of 8), `d−1` right-interface builds on doubled ranks (`4×` a normal
`phi_right`), and the closing truncation sweep, `d−1` SVDs of `(2r·n × 2r)`,
`O(8 d n r³)`. Its saving: no `tangent_defect`.

**Parallelism, honestly.** [CL22] and [CKL23] both lead with parallelism. For a
**Tucker** tensor the `d` K-equations are independent; for a balanced tree the
sibling subtrees are independent. For a **tensor train** the caterpillar has one
non-leaf child per node, so the ascent in §2.5 is strictly sequential: node `k`'s
Galerkin needs `Û_{≥k+1}`, which is node `k+1`'s output. **TT-BUG has no
intra-step parallelism.** Every parallel claim in the papers is about a different
topology.

**Prediction for the reference problem, then measurement.** Reference:
d=6, n=2, `r=[1,2,4,8,4,2,1]`, `A` symmetric of TT rank `[1,3,4,4,4,3,1]` and
`‖A‖₂=1`, `τ=1e-2`, `local_tol=1e-8` (defaults), the run quoted at 4.7 ms.

*Predicted, before running:* KSL does 22 local exponentials of block sizes
`{4,16,64,64,16,4}` (K) and `{4,16,64,16,4}` (S) per sweep; BUG does **6**, of
sizes `{8,32,64,64,16,4}` adaptive / `{4,16,64,64,16,4}` fixed. The step is
dispatch-bound, not flop-bound — 22 exponentials × ~53 numpy calls ≈ 1170 calls at
~7 µs is the whole 8 ms, against roughly 1.1 Mflop of actual arithmetic
(0.13 Gflop/s). So the right currency is *call count*, and BUG's is `6/22 = 0.27`
of KSL's, offset by larger blocks and the extra sweeps. Predicted **0.45–0.65×
KSL**, i.e. **2.1–3.1 ms** where KSL is 4.7 ms.

*Measured* (this host, 40 repetitions, prototype of §2.5 with no history
bookkeeping):

| | min | median | ratio to KSL(min) |
|---|---|---|---|
| KSL symm, `check_rank=False` | 8.21 ms | 13.50 ms | 1.00 |
| KSL symm, `check_rank=True` (the default) | 9.81 ms | 16.63 ms | 1.20 |
| BUG rank-adaptive, `θ=1e-10` | **4.90 ms** | 7.15 ms | **0.60** |
| BUG fixed rank | **3.92 ms** | 5.53 ms | **0.48** |

`nexp = 6` in both BUG modes, `substeps = 6` (one per exponential), augmented ranks
`[1,4,8,8,4,2,1]`, Galerkin sizes as predicted. Scaled to the caller's 4.7 ms
baseline: **BUG ≈ 2.8 ms rank-adaptive, ≈ 2.3 ms fixed rank**, and against KSL's
actual default (`check_rank=True`) the ratios are 0.50 and 0.40. Add ~10 % for the
history bookkeeping a real implementation will carry.

**Falsifiable sub-predictions** for the implementation to reproduce: exactly `d`
local exponentials per step; augmented ranks exactly `[1,4,8,8,4,2,1]`;
`ratio_to_ksl ∈ [0.45, 0.70]` on this problem.

---

## 7. Validation tests

Every test names its oracle. Nothing is compared against BUG's own output except
where the test is explicitly about two code paths agreeing. Numbers in
parentheses are what the prototype produced and are the assertion targets.

1. **d=1 and d=2 against `scipy.linalg.expm`.** Oracle: dense
   `sla.expm(τ A) @ y0.full(asvector=True)`. `d=1`: no sweep, BUG must be *exact*
   to `local_tol` (there is nothing to augment or truncate). `d=2` with maximal
   ranks: the augmented bases span everything on the right leg, so BUG is exact
   there too. Asserts the plumbing (index order, `phi_*` conventions, `full`'s
   F-ordered flattening) before anything else. Mirrors
   `test_ksl_small_d_against_dense_expm`.

2. **Numerically measured convergence order.** Oracle: dense `expm`, and the
   *order* is the assertion. d=6, n=2, maximal ranks `[1,2,4,8,4,2,1]` so `ε = 0`.
   Global error over `T=1` with `N ∈ {4,8,16,32,64}`:
   `8.369e-04, 2.140e-04, 5.409e-05, 1.360e-05, 3.408e-06`, observed orders
   `1.97, 1.98, 1.99, 2.00`. Assert order `≥ 0.9` (the theory guarantees 1, [CL22]
   Thm 4) and **record** the observed 2 without asserting it — see §8 Q3. Single
   step, same setup: `1.988e-05, 2.512e-06, 3.152e-07, 3.947e-08` for
   `τ = 0.1, 0.05, 0.025, 0.0125` — local ratio 7.9 ≈ 2³.

3. **Exactness property, [CL22] Thm 3.** Oracle: an analytic solution. Build
   `A(t)` of exact TT rank `r` with a known closed form (e.g. `y(t) = ⊗_k v_k(t)`
   with `v_k(t) = cos(ω_k t) a_k + sin(ω_k t) b_k`, rank 1, or a rank-`r` sum of
   such) and drive `F(t,·) = ẏ(t)` — the solution-independent case. BUG must
   reproduce `y(t₁)` to round-off provided `Uᵢ(t₁)ᴴUᵢ(t₀)` is invertible. This is
   the property the *parallel* variant of [CKL23] §3 explicitly loses, so if we
   ever implement that variant, the same test must be marked `xfail` for it — which
   is the cheapest possible check that we implemented the right one.

4. **Robustness to small singular values.** Oracle: dense `expm`, plus the
   *absence* of step-size restriction. Construct `y₀` with bond spectrum
   `{1, 2.8e-4, 4.7e-8, 1.5e-11}` (d=4, n=4). Assert the relative error is flat in
   the smallest singular value and scales with `τ` alone. Measured: BUG
   `8.593e-03` at `τ=0.1` and `8.626e-04` at `τ=0.01` (linear in τ, no `σ_min`
   dependence); KSL `1.626e-02` / `1.666e-03`. Extend with the [CL22] §6.1 setup
   (`D = diag(2^{-j})`, `N=100`) if we ever add a matrix path.

5. **Conservation law: nonexpansivity on a dissipative operator.** Oracle: the
   contraction property `‖y₁‖ ≤ ‖y₀‖`, which holds for BUG *unconditionally in τ*
   (proof in §3) and is a mathematical invariant, not a reference solution.
   `A = −(2^L+1)²·tt.qlaplace_dd([L])`. Assert `‖y₁‖ ≤ ‖y₀‖·(1+8ε_mach)` for
   `τ‖A‖ ∈ {0.2, 1.7, 16.9, 169}`. The same assertion applied to `ksl` **fails at
   `τ‖A‖ = 169`** (`‖y₁‖ = 1.0677e+108`), and the test should say so explicitly:
   `pytest.mark.xfail` on the KSL arm with the measured value in the reason, so the
   regression is visible rather than folklore. Second arm, L=10, `τ‖A‖ = 42`:
   KSL returns `‖y₁‖ = 27.9 > 1`; BUG returns `9.86e-02`.

6. **THE FAIL-LOUD TEST: the rank cannot more than double in one step.** This is
   the case where BUG returns a perfectly plausible tensor that is wrong by three
   orders of magnitude, and it must say so. Setup: d=6, n=2, start from
   `y₀` rounded to ranks `[1,2,2,2,2,2,1]`, dynamics that needs `[1,2,4,8,4,2,1]`,
   `T=1`, `N=64`, `θ = 1e-10` — a tolerance so tight it can only mean "give me
   everything". Measured outcome: the ranks do reach `[1,2,4,8,4,2,1]` and the
   final error is **3.180e-03** — against **3.408e-06** for the same run started
   from full ranks. A factor of 933, caused entirely by the doubling cap during the
   first two steps, and *invisible* in the final `trunc_error`, which by then is
   below `θ`. Assertions:
   * with `reject=False`: `history.rank_capped is True` on the first steps, and a
     `RuntimeWarning` naming the cap is raised. It must **not** return
     `trunc_error < θ` and no other signal.
   * with `reject=True` (the default, [CKL23] §3.3 criterion 1): `retries > 0` and
     the final error drops materially below 3.18e-03.
   * a second arm sets `rmax` below what `θ` demands (`θ=1e-6`, `rmax=4`, needed
     rank 8) and asserts `trunc_error > θ` is reported and warned about — the
     tolerance was *not* met and the answer must not pretend otherwise.

7. **Stiffness beyond the substep solver — must raise, not underflow silently.**
   L=6 heat, `τ = 1e-1`, `τ‖A‖ = 1690`. The Arnoldi substepping cannot cover this;
   the iterate underflows to exactly zero mid-loop. Assert a `RuntimeError` whose
   message names *stiffness / underflow* — not the current
   `"the local operator is likely not what the caller thinks"`, which blames the
   caller (see §4(c); fixing that guard is a prerequisite for this test). Assert
   also that the single-substep path, which currently returns `0.0` with
   `substeps=1` and no complaint (measured on
   `B = diag(linspace(-4000,-5000,20))`, `τ=0.1`), is either right or loud.

8. **Fixed-rank BUG preserves ranks exactly**, and **rank-adaptive BUG with
   `θ = 0` never truncates** — two invariants, no oracle needed, but they catch
   the off-by-one in `chop`'s absolute/relative switch that §4(a) introduces.

9. **Backend parity.** Same step on numpy and torch CPU agrees to `1e-12`, and a
   numpy `A` with a torch `y` runs (the mixed-backend path `operator_cores` already
   handles, exercised by `test_algs_torch.py`).

---

## 8. Open questions

Stated as questions, not as risks-with-mitigations. Each one is something I could
not settle from the two PDFs.

**Q1 — Is §2.5 the published TT-BUG?** The TT/TTN algorithm is in neither uploaded
paper. [CKL23] §1 explicitly defers it: "it is expected that its concepts extend to
tensor trains and general tree tensor networks … left for future work." §2.5 is our
derivation from the Tucker algorithm plus the caterpillar-tree reading; it is
self-consistent, reduces correctly at `d=1,2`, and validated numerically, but I
have not checked it against Ceruti–Kusch–Lubich, SINUM 61(1):194–222 (2023). Two
specific points to verify there: **(a)** whether truncation happens once at the end
(as we do) or at every node during the ascent — these give different ranks and
different error constants; **(b)** whether the tree is rooted at mode 1 (making the
method asymmetric in the modes, as ours is) or at the middle, or whether the paper
uses a balanced binary tree over the modes instead of a caterpillar, in which case
the parallelism claim of §6 changes completely.

**Q2 — Does the `d=2` TT case coincide with matrix BUG? It does not, and I do not
know if that matters.** A rank-`r` matrix as a `d=2` TT gives a caterpillar with an
*uncompressed* leaf at mode 1, so §2.5 does an L-step, augments `V̂`, then a
Galerkin over the whole `R^{n₁} ⊗ V̂` — an `n₁ × 2r` problem, not the `2r × 2r`
problem of §2.2. That is a *larger*, presumably more accurate Galerkin space, but
it is a different integrator, so [CL22] Thm 4 does not literally apply to it. The
Tucker algorithm §2.4 is the one that reduces to matrix BUG at `d=2`. Whether the
right TT algorithm compresses the mode legs too (making it Tucker-like and
`d`-parallel) is exactly Q1(b).

**Q3 — Why do we observe global order 2?** Theory ([CL22] Thm 4) gives
`c₀δ + c₁ε + c₂h`, i.e. order 1. Measured orders on the `ε = 0` full-rank problem
are `1.97, 1.98, 1.99, 2.00`, and the single-step error ratio is 8, i.e. local
order 3. Plain BUG is documented as first order — Ceruti–Kusch–Lubich–Schrammer
built a whole midpoint variant (BIT 2024) to get order 2 — so either this is an
artefact of `ε = 0` with a linear autonomous `A`, or our sweep has an accidental
symmetry. **Do not advertise order 2 and do not assert it in a test** until we have
measured the order on a problem with `ε ≠ 0` against the *projected* flow oracle
(the same oracle
`test_ksl_order_against_the_dense_projected_flow` already builds — but the
projected flow for BUG is not `y' = P_{T_yM}Ay`; identifying which ODE BUG
discretizes is itself open, since [CL22] §1 says it "can apparently not be
interpreted as a splitting integrator or be included in another familiar class").

**Q4 — What is `η` in TT?** [CKL23] §3.3 criterion 2 needs
`η = ‖Ũ₁ᵀ F₀ Ṽ₁‖`: the component of `F₀` in the "new directions only" block. In
TT there are `d−1` bonds and hence `d−1` candidate `η_k`, and the identity (3.11)
that makes `η` equal to the normal component of the *augmented-space* model of `F₀`
was proved for the matrix tangent space using `P⊥_r(Y₀)Z = (I−U₀U₀ᵀ)Z(I−V₀V₀ᵀ)`,
which has no one-line TT analogue. Options: (i) use only the truncation mass
`trunc_error` and the `rank_capped` flag as the indicator, which is what §5
proposes; (ii) reuse `tangent_defect` and pay for it; (iii) derive a per-bond `η_k`
and take the max. I do not know which is right, and the spec deliberately ships
(i) with the honest note that criterion 2 is therefore **not** implemented.

**Q5 — Which substep solver for parabolic problems?** §3 shows BUG removes the
cancellation catastrophe but not the stiffness: the Arnoldi substepping still needs
`τ‖B_k‖` manageable and dies at `τ‖A‖ ≈ 1700`. Both papers say only "solved
approximately using a standard integrator, such as a Runge–Kutta method or an
exponential integrator when `F` is predominantly linear" ([CKL23] §2.1) — which is
no help for a stiff `A`. Whether to add a rational/implicit local solver, and
whether its inexactness stays inside the "inexact substeps" perturbation result
([CL22] §3.2, citing Kieri–Lubich–Walach §2.6.3), is unresolved.

**Q6 — The tolerance's units.** [CKL23] uses `θ` absolute in the theory (3.5) and
`θ = θ̄‖Σ̂‖` in the experiments, with `θ̄` differing by 3–7× between the BUG and
parallel integrators *for the same resulting ranks* (§5.2: `θ̄ ∈ {0.05, 0.025,
0.01}` for BUG against `{0.005, 0.003, 0.0015}` for parallel). So `θ` is not
comparable across variants and the global bound `C₃θ/h` (Thm 4.1) says the error
degrades as `h → 0` at fixed `θ`. A default `tol=1e-10` may therefore be the wrong
shape of default entirely; the honest coupling is `θ ∝ h`. I have not worked out
what that constant should be, and §5 ships a fixed `tol` with this noted.

**Q7 — Rank-deficient augmentation.** [CKL23] §3.1 says `Ũ₁` is "filled with zero
columns if `(U₀, K(t₁))` has rank less than 2r". Our `right_orthogonalize` calls
`bk.qr` and returns `min(2r, n·r̂)` columns regardless of numerical rank, so a
rank-deficient stack silently contributes noise directions that the truncation
then has to remove. Whether to rank-reveal the augmentation (column-pivoted QR, or
an SVD with a threshold) is a real decision with a cost; I do not know how much it
matters, and it should be measured on a problem where `K(t₁)` nearly duplicates
`U₀` (small `τ`, which is the *common* case).
