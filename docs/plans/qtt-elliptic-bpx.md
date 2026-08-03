# QTT-ELL: elliptic problems in the QTT format, and multilevel (BPX) preconditioning

> **Status: the D=1 core of this spec is implemented and tested**, not merely
> proposed. `tt.algs.qtt_ell` ships `bpx`, `bpx_theta`, `prolongation` and
> `solve_direct_1d`; `tt/core/tools.py` ships `qdiff`, `qtri_ones`,
> `qlaplace_dn` and `level_major_order`, and `permute` now accepts a TT-matrix.
> Tests in `tests/test_qtt_ell.py` (48 of them) check against the closed form of
> [BK20] Lemma 4, against the per-level sum, and against dense linear algebra.
>
> Two things were learned in the implementation that this document had not
> settled, and both are recorded where they belong (§1.4, §2.2):
>
> * The level automaton's switching core is `X_b`, not a scaled `U_b`; the level
>   weight `2^{-l}` rides on `X_b`'s own scaling, so `weight=2` puts its extra
>   `2^{-l}` on `U_b` instead. The chain then carries a fixed `2^d` for either
>   weight. Rank comes out exactly 8 for `D = 1` at `d = 3 .. 40`.
> * `Theta` is rank 6 and `Theta^T Theta` rank 17, both flat in `d` — and the
>   end-to-end margin over the assembled triple product is larger than §2.2
>   estimated, because the triple product's *rank* grows too (96, 135, 185 at
>   `d = 10, 14, 18`), not only its error. Measured head-to-head below.
>
> | `d` | unpreconditioned | `B = Theta^T Theta` |
> |---|---|---|
> | 10 | 30 sweeps, 0.63 s, 4.1e-10 | 7 sweeps, 0.07 s, 2.3e-14 |
> | 18 | 30 sweeps, 2.51 s, 8.6e-06 | 7 sweeps, 0.18 s, 8.1e-14 |
> | 26 | 30 sweeps, 11.1 s, 4.1e-01 | 7 sweeps, 0.41 s, 1.7e-13 |
> | 30 | 30 sweeps, 23.3 s, **1.03** | 7 sweeps, 0.51 s, 1.8e-13 |
>
> `-u'' = 1`, `u(0) = 0`, `u'(1) = 0`, AMEn at `eps = 1e-10`, interleaved runs,
> b300/numpy/float64. `kappa(C A C)` reproduced independently of the way it was
> first measured: 5.6674 at `d = 4`, 10.6617 at `d = 10`, `lam_min -> 2`.
>
> **Not implemented:** everything for `D > 1` (`bpx` builds the cores but only
> `D = 1` is verified end to end; `bpx_theta` refuses `D > 1` loudly), variable
> coefficients (`Lambda^{1/2}`, §3.2 K4/K6), `stiffness`, `load_vector`, and the
> `solve` front end. See §7 for what that leaves open.

Implementation spec for a new module `tt.algs.qtt_ell` (+ additions to
`tt/core/tools.py`). Sources actually read, and what each one is:

* **[BK20]** M. Bachmayr, V. Kazeev, *Stability of Low-Rank Tensor
  Representations and Structured Multilevel Preconditioning for Elliptic PDEs*,
  Found. Comput. Math. **20** (2020) 1175–1236 (received 5 Mar 2018, accepted
  5 Nov 2019, open access). Uploaded file
  `1f51e746-s1020802009446z.pdf`, 62 pp., **read in full** (text extracted with
  `pypdf` on b300; the four figures were *not* rendered, so every claim that
  rests on a figure is marked as such). This is **the** paper the user meant.
  Contents: FEM discretization of a second-order elliptic operator on
  `(0,1)^D` with mixed Dirichlet/Neumann data, written as a product of
  "differentiation" factors `M_{L,alpha}` and coefficient factors
  `Lambda_{L,alpha,alpha'}` (§2.3); a *new* symmetric BPX variant `C_L`
  (§2.4, Theorem 2); the notion of **representation condition number** of a TT
  decomposition (§4, Definition 3/4, Propositions 1–5); explicit QTT cores for
  `C_L` and for the preconditioned operator `B_L = C_L A_L C_L` (§5, Lemmas
  1–5, Theorems 3–5); complexity of a soft-thresholding solver (§6); and
  numerical results up to `L = 50` levels, i.e. `2^50` nodes per dimension
  (§7), with a **direct comparison against AMEn wrapped from the Python TT
  Toolbox** (§7.2–7.4, Remark 10).
* **[KK12]** V. A. Kazeev, B. N. Khoromskij, *Low-rank explicit QTT
  representation of the Laplace operator and its inverse*, SIAM J. Matrix Anal.
  Appl. **33**(3):742–758, 2012 — **not read** (not uploaded). It is [BK20]'s
  reference [35] and the source of the QTT cores of `A_L` that ttpy2's
  `tt.qlaplace_dd` already implements. Everything §2.4 below says about the
  *inverse* is **measured here**, not taken from that paper.
* **[KO11]** B. N. Khoromskij, I. V. Oseledets, *QTT approximation of elliptic
  solution operators in higher dimensions*, Russ. J. Numer. Anal. Math.
  Modelling **26**(3):303–322, 2011 — **not read**. [BK20]'s reference [39],
  cited there as "a different class of preconditioners based on approximate
  matrix exponentials … for QTT decompositions". This is the exponential-sum
  family that `docs/plans/riemannian-autodiff.md` §6.2 measured; §2.3 below
  measures whether it transfers to a *1D QTT* operator.
* **[COR16]** A. V. Chertkov, I. V. Oseledets, M. V. Rakhuba, *Robust
  discretization in quantized tensor train format for elliptic problems in two
  dimensions*, arXiv:1612.01166 — **not read**. [BK20]'s reference [11]; the
  Volterra-integral reformulation. [BK20] §1.4 says of it: demonstrated to
  `L ≈ 20` in `D = 2`, "the matrix condition number still grows exponentially
  with respect to `L`, and numerical stability is still observed to be lacking
  for larger values of `L`". §2.4 below measures the `D = 1` special case,
  where it is not merely competitive but *optimal*.

The other three uploaded PDFs are **not** about elliptic problems and are not
owned by this spec:

* `c772d22e-main_1.pdf` = Rakhuba–Novikov–Oseledets, *Low-rank Riemannian
  eigensolver for high-dimensional Hamiltonians*, JCP 396 (2019) 718–737 —
  owned by `docs/plans/eigenvalues.md`.
* `ff3f5ac0-2103.14974v2.pdf` = Novikov–Rakhuba–Oseledets, *Automatic
  differentiation for Riemannian optimization on low-rank matrix and
  tensor-train manifolds* — owned by `docs/plans/riemannian-autodiff.md`.
* `5334b735-s10543021008730.pdf` = Ceruti–Lubich, *An unconventional robust
  integrator for dynamical low-rank approximation*, BIT 62 (2022) 23–44, and
  `2847be00-2304.05660v1.pdf` = Ceruti–Kusch–Lubich, *A parallel rank-adaptive
  integrator for dynamical low-rank approximation* — both dynamical low-rank
  **time integrators**, owned by `docs/plans/bug-integrator.md`. Neither
  mentions preconditioning; forcing them into this spec would be wrong.

Everything marked **measured** was produced by throwaway prototypes run on
**b300** (hostname Planck, 256 cores, 2 TB RAM), interpreter
`~/work/ttpy-modern/ttpy2/.venv/bin/python`, numpy backend, float64, CPU only
(no GPU was used: nothing here is large enough to need one), single runs unless
stated. Prototypes live in `~/work/ttpy-modern/scratch-bpx/` on b300 and are
listed in §7. Nothing below is an estimate; where a number is missing it says
**not measured**.

**Cross-spec decisions live in `docs/plans/ROADMAP.md`**, not here. Where this
spec and one of `bug-integrator.md`, `eigenvalues.md`, `riemannian-autodiff.md`
ask for the same object, the reconciled contract and its owner are recorded there
(§2), together with the dependency graph (§1), the milestone order (§4) and the
consolidated open questions (§6). **§4.2 of this spec won the preconditioner
disagreement** with `eigenvalues.md` §3.3 and `riemannian-autodiff.md` §6.4, and
both have been amended to point at the three-form contract in
`docs/plans/ROADMAP.md` §3, which also freezes what it forbids: no
`round(C @ A @ C)` handed to a solver (§1.6), no generic route to `T` (§2.4), no
global `prec=` on `amen_solve` (§4.1).

---

## 0. The short version

1. [BK20] is about **two** ill-conditionings, not one. The familiar one is
   `cond(A_L) = O(h^{-2}) = O(4^L)`. The new one is **representation**
   ill-conditioning: a TT decomposition can be so redundant that
   orthogonalizing it — the first step of *every* rounding — loses `4^L * eps`
   of accuracy. Reproduced here: at `L = 20`, orthogonalizing `A_L • v` for a
   smooth `v` changes the represented vector by a relative **2.47e-04**,
   against `4^20 * eps = 2.44e-04` (§1.3).
2. Preconditioning by multiplying the *matrix* fixes the first and **not** the
   second. Reproduced here: `round(C@A@C)` differs from the mathematically
   identical combined `B_L` by a relative **5.998e-04** at `L = 20` and
   **4.77e+14** at `L = 50` (§1.6).
3. The payoff is real and large. On `-u'' = 1, u(0)=0, u'(1)=0`, at `L = 30`
   (`2^30 ≈ 1.07e9` nodes), unpreconditioned `amen_solve` reaches a relative
   nodal error of **1.05e+00** after 40 sweeps / 25.1 s; the BPX-preconditioned
   solve reaches **1.23e-11** in **6 sweeps / 0.33 s** (§2.2).
4. **"Может можно лучше" is right in exactly one corner.** In `D = 1` with
   `A = M^T diag(a) M`, the exact inverse is `T diag(1/a) T^T` with
   `T = (I-S)^{-1}` of QTT rank **2**, so the whole solve is two matvecs: at
   `L = 40` it returns the nodal solution to a relative **5.55e-15** in **4 ms**
   at rank 3, which no preconditioned iteration can match (§2.4). Everywhere
   else the cheaper alternatives fail, and they fail for [BK20]'s *own* reason:
   using the same `T` as a *preconditioner* — assembling `T^T A T`, which is
   mathematically the identity, and iterating — gives a rank-13 representation
   of the identity and an answer **97 % wrong at `L = 30`, reported as
   `converged=True`** (§2.4). The exponential-sum preconditioner that works on
   Kronecker sums over physical modes has QTT ranks 8–21, not 1, so it does not
   transfer either (§2.3). **BPX's combined representation is the only
   construction measured here that survives an iteration past `L ≈ 15`.**
   §2.5 gives the build order.

---

## 1. What [BK20] actually says

### 1.1 The discretization, in our conventions

Model problem (BK (3)–(5)): on `Omega = (0,1)^D`, find `u` with `u = 0` on
`Gamma = {x : x_1 ... x_D = 0}` (Dirichlet on the faces through the origin),
`du/dn = 0` on the rest, such that

```
a(u,v) = int_Omega (grad v)^T A grad u + int_Omega c u v  =  f(v)     for all v.
```

The choice of `Gamma` is not cosmetic: with Dirichlet only at `x_k = 0`, the
level-`l` grid has exactly `2^l` degrees of freedom per dimension
(nodes `tau_{l,j} = 2^-l j`, `j = 1..2^l`; the node at `x_k = 1` survives
because it is a Neumann node). A power of two per level is what makes the
whole QTT-by-levels construction exact. **Pure Dirichlet gives `2^l - 1`
dofs and does not fold.**

Basis (BK (10)): the level-`l` hat functions are normalized in `L^2`, not by
their nodal value:

```
phi_{l,j}(tau_{l,j'}) = 2^{l/2} delta_{jj'},        || phi_{l,j} ||_{L2} ~ 1.
```

Consequence for us: with that normalization the `D = 1` stiffness matrix is

```
A_L = 2^{2L} * tridiag(-1, 2, -1)  with the last diagonal entry 1 (Neumann),
    = 2^{2L} * (I - S)^T (I - S),   S the sub-diagonal shift,
```

so `A_L` is `h^{-2}` times the familiar finite-difference matrix. **We call the
unscaled factor `A_DN` below**; `tt.qlaplace_dd([d])` is the Dirichlet–Dirichlet
sibling (last diagonal 2). Verified against dense: `lap_dn(L)` (prototype
`bpx.py`) equals the tridiagonal DN matrix exactly at `L = 3, 5`, QTT rank 4.

Index convention. [BK20] identifies `j - 1 = sum_l i_l prod_{k>l} n_k`, so
**BK level 1 is the coarsest scale = the most significant bit**. ttpy2's TT
flat index is `i_1 + n_1 i_2 + ...`, so **ttpy2 core 1 is the fastest = least
significant bit**. Measured (prototype `p0_conv.py`): cores `[J, I, I]` produce
a matrix with `j - i = 1`, cores `[I, I, J]` produce `j - i = 4`. Therefore a
BK core list maps to a ttpy2 core list by

```
ttpy2_cores = [ c.transpose(3, 1, 2, 0) for c in bk_cores[::-1] ]
```

(reverse the chain, transpose the two rank axes). Every construction below is
given in BK order; `bpx.to_ttpy` performs the conversion.

Structure of the operator (BK (26b)). The paper never writes `A_L` as one
object; it writes it as

```
A_L = sum_{(alpha,alpha') in D}  M_{L,alpha}^T  Lambda_{L,alpha,alpha'}  M_{L,alpha'},
```

where `alpha in {0,1}^D` is a derivative multi-index, `M_{L,alpha}` maps nodal
coefficients to the coefficients of `d^alpha phi` in the monomial basis on each
element (BK (12d), (20c)), and `Lambda_{L,alpha,alpha'}` is a **block-diagonal**
matrix carrying the coefficient function `c_{alpha alpha'}` and the element
integrals (BK (26a)). For the Laplacian, `Lambda` is diagonal with entries
`2^{-L}` and `2^{-L}/3` and QTT rank 1 (BK (27b)).

This factorization is the whole reason the paper works. `Lambda` is where the
*coefficient* lives (variable, oscillatory, high-contrast — all it needs is a
low QTT rank, BK (90a)); `M_{L,alpha}` is where the *differentiation* lives and
is coefficient-independent; and the preconditioner is fused into `M`, not into
`A`. In `D = 1` with `c(x)` piecewise constant on the fine mesh (BK Remark 1):

```
A_L = 2^{-L} M_{L,1}^T (diag c_L) M_{L,1},     M_{L,1} = 2^{3L/2} (I - S).
```

### 1.2 Why the QTT ranks of the solution stay bounded

Three separate statements, which the paper is careful to keep apart:

1. **Approximability of the exact solution** (BK Theorem 7, quoted from
   Kazeev–Schwab, Numer. Math. 138:133–190, 2018, Thm 5.16). For `D = 2`, under
   ellipticity and *analytic* data on `Omega`, for all `L, R` there is
   `u_{L,R} in V_L` with multilevel TT ranks `<= R` and
   ```
   || u - u_{L,R} ||_{H1}  <=  C e^{-b L}  +  C' e^{-b' sqrt(R)}.
   ```
   Note what the hypothesis is and is not: *analytic data on the closed
   polygon*, which **permits corner singularities of the solution** (the
   singularity comes from the geometry, not from the data). This is the exact
   sense in which "QTT handles singularities": `x^alpha` is self-similar under
   dyadic dilation, so the level-to-level transfer operator is asymptotically
   the same at every level, and the number of level-to-level directions — the
   TT rank — saturates. The rank cost of the singularity is `e^{-b' sqrt(R)}`,
   i.e. `R ~ |log eps|^2`, not `R ~ eps^{-1/alpha}`.
2. **Approximability of the Galerkin solution** (what an algorithm actually
   computes) is *not* implied by 1. BK are explicit (§6.2): "We are not aware
   of existing analysis that would allow to arrive at conclusions on Galerkin
   solution ranks". For `D = 1` Poisson the Galerkin solution *is* the nodal
   interpolant, so 1 applies verbatim; for `D > 1` it does not.
3. **Approximability of the preconditioned coefficient** `u_L = C_L^{-1} ubar_L`
   is a *third* statement (BK Assumption 1), and it is the one the complexity
   theorem needs. BK verify it only numerically (their Fig. 2, not rendered
   here): the action of `C_L^{-1}` preserves exponential singular-value decay
   "at a slightly modified rate". This is a real gap in the theory and it is
   worth knowing before building on it.

**Measured** (prototype `p7_sing.py`, b300, numpy, float64; nodal values on
`x_j = j 2^-L`, `j = 1..2^L`, normalized; rank = max TT rank of the TT-SVD at
relative Frobenius accuracy `eps`):

| function | `L=10` | `L=14` | `L=18` | `L=20` | `eps` |
|---|---|---|---|---|---|
| `x - x^2/2` (smooth) | 3 | 3 | 3 | 3 | 1e-6 |
| `x^{3/4}` (corner) | 5 | 5 | 5 | 5 | 1e-6 |
| `x^{1/2}` | 5 | 5 | 5 | 5 | 1e-6 |
| `x^{1/4}` | 5 | 5 | 5 | 5 | 1e-6 |
| `sin(2^10 pi x)` | 16 | 2 | 2 | 2 | 1e-6 |
| `exp(-x/1e-4)` (layer) | 1 | 1 | 1 | 1 | 1e-6 |
| `x - x^2/2` | 3 | 3 | 3 | 3 | 1e-10 |
| `x^{3/4}` | 7 | 7 | 7 | 7 | 1e-10 |
| `x^{1/2}` | 7 | 7 | 7 | 7 | 1e-10 |
| `x^{1/4}` | 7 | 7 | 7 | 7 | 1e-10 |
| `sin(2^10 pi x)` | 16 | 2 | 2 | 2 | 1e-10 |
| `exp(-x/1e-4)` | 1 | 1 | 1 | 1 | 1e-10 |

Read three things off it. (i) An algebraic vertex singularity costs **two extra
rank units** at `1e-6` and **four** at `1e-10`, *flat in `L`* — the singularity
is essentially free in QTT, and the exponent `alpha` does not matter. (ii) The
oscillatory function is rank 16 exactly at the level where the grid barely
resolves it (`L = 10`, `2^10` nodes against `2^10 pi` phase) and drops to rank
**2** once resolved — the QTT rank measures *unresolvedness*, not frequency.
(iii) The boundary layer is rank 1, because `exp(-x/delta)` is exactly separable
across dyadic scales. This is the entire empirical case for "textbook
discretization on a `2^50` grid, compressed" and it is very strong.

Counterweight: these are ranks of the *exact* nodal values, i.e. statement 1.
The ranks of what a solver produces are larger — measured 13–16 for the
BPX-preconditioned iterate against 3 for the exact solution on the same problem
(§2.2), and 162 for the unpreconditioned one, which is fitting round-off noise.

### 1.3 Two ill-conditionings

**Matrix.** `cond(A_L) = O(h^{-2}) = O(4^L)`. Closed form for the DN Laplacian:
`lambda_k = 4 sin^2((2k-1) pi / (2(2N+1)))`, `N = 2^L`, so
`kappa = sin^2((2N-1)pi/(2(2N+1))) / sin^2(pi/(2(2N+1)))`. **Validated against
`numpy.linalg.eigvalsh`** (prototype `p1_cond.py`, dense, float64): relative
error of `lambda_min` 1.5e-14 at `L=4`, 3.1e-9 at `L=12`; `kappa` agrees to all
printed digits.

| `L` | `N` | `kappa(A_DD)` (`tt.qlaplace_dd`) | `kappa(A_DN)` | `4^L * eps` |
|---|---|---|---|---|
| 6 | 64 | 1.712e+03 | 6.741e+03 | 9.09e-13 |
| 10 | 1024 | 4.258e+05 | 1.702e+06 | 2.33e-10 |
| 14 | 16384 | 1.088e+08 | 4.352e+08 | 5.96e-08 |
| 18 | 262144 | 2.785e+10 | 1.114e+11 | 1.53e-05 |
| 22 | 4194304 | 7.130e+12 | 2.852e+13 | 3.91e-03 |
| 24 | 16777216 | 1.141e+14 | 4.563e+14 | 6.25e-02 |

**Representation** (BK §4, the paper's own novelty). Define, for a tuple of
cores `X = (X_1..X_L)` with `tau(X)` the represented tensor (BK Definition 3):

```
ramp_l(X)  = lim_{e->0} (1/e) sup { ||tau(Xt) - tau(X)||_2 :
                                    ||Xt_l - X_l||_2 <= e ||X_l||_2, Xt_k = X_k otherwise }
rcond_l(X) = ramp_l(X) / ||tau(X)||_2
```

and, computably (BK Proposition 1),

```
ramp_l(X) = || tau^-_l(X) ||_{2->2} * || X_l ||_2 * || tau^+_l(X) ||_{2->2},
```

the product of the norms of the left partial contraction, the core itself, and
the right partial contraction. This is `O(1)` for an orthogonalized (TT-SVD)
representation and can be astronomically large for a redundant one. For
operators, `mramp_l(A) = sup_X ramp_l(A•X)/ramp_l(X)` (BK Definition 4), with the
computable upper bound `beta_l(A)` of BK Proposition 4 (65).

BK Proposition 5: for the standard rank-3 QTT representation `A` of the
Dirichlet Laplacian, `mramp_l(A) ~ 2^{2L}` and
`2^{(3L+l)/2} <~ mrcond_l(A) <~ 2^{2L}`. In words: **multiplying a
well-represented vector by the standard QTT Laplacian produces a representation
whose subsequent orthogonalization loses `4^L * eps`.** The cause is visible in
the cores (BK (54)): `A_1 .. A_{L-1}` have only non-negative entries, `A_L`
introduces the cancellation, and the cancellation is catastrophic exactly for
low-frequency grid functions.

**Measured** (prototype `p2_bpx.py`, §(c); b300, numpy, float64). Test:
`err = || full(M@V) - full(round(M@V, 1e-16)) ||_2 / || full(M@V) ||_2`, where
`M@V` is formed **without rounding** (ranks multiply) and the difference is
computed entrywise on the dense arrays — this is BK's "difference (a)" and it
avoids measuring the instability with an unstable instrument. `V` is `v1 = 1`,
`vmin = sin(pi x/2)`, `vmax = sin(pi (1+2^{L+1}) x / 2)`, all `l2`-normalized
and TT-SVD'd (so `rcond_l(V) <= sqrt(2)` by BK Proposition 2(iii)).

| `L` | `M` | rank `M` | `V=v1` | `V=vmin` | `V=vmax` |
|---|---|---|---|---|---|
| 10 | `B` combined | 17 | 2.07e-15 | 3.53e-15 | 1.01e-14 |
| 10 | `C.A.C` raw | 256 | 2.43e-10 | 2.27e-10 | 7.67e-13 |
| 10 | `A_DN` | 4 | 2.32e-14 | **2.50e-10** | 9.32e-16 |
| 14 | `B` combined | 17 | 2.73e-15 | 4.16e-15 | 6.23e-15 |
| 14 | `C.A.C` raw | 256 | 4.96e-08 | 1.11e-07 | 8.85e-12 |
| 14 | `A_DN` | 4 | 6.15e-14 | **6.24e-08** | 3.77e-15 |
| 18 | `B` combined | 17 | 1.69e-14 | 1.10e-14 | 1.92e-14 |
| 18 | `C.A.C` raw | 256 | 1.47e-05 | 1.02e-05 | 3.45e-10 |
| 18 | `A_DN` | 4 | 1.38e-13 | **2.15e-05** | 2.36e-14 |
| 20 | `B` combined | 17 | 1.40e-14 | 1.17e-14 | 1.85e-14 |
| 20 | `A_DN` | 4 | 5.81e-13 | **2.47e-04** | 2.34e-14 |

Compare the bold column with `4^L eps` from the previous table: 2.33e-10 vs
2.50e-10, 5.96e-08 vs 6.24e-08, 1.53e-05 vs 2.15e-05, 2.44e-04 vs 2.47e-04.
**BK Proposition 5 is reproduced to within a factor of 1.5.** Note also that
`A_DN • vmax` is *not* amplified (9e-16): the effect is specific to smooth
right factors, exactly as BK explain.

### 1.4 The BPX preconditioner: definition and why it is a sum over levels

BPX (Bramble–Pasciak–Xu 1990, BK ref. [10]) needs the nested spaces
`V_0 subset V_1 subset ... subset V_L` and the prolongations
`P_{l,L} : R^{J_l} -> R^{J_L}`, the matrix of the identity embedding
`span{phi_{l,j}} -> span{phi_{L,j'}}` in the bases of BK (10). In `D = 1`
(BK (14), (15)):

```
Phat_{l,L} = 2^{(l-L)/2} ( Ihat_l (x) etahat_{L-l}  +  Shat_l (x) (xihat_{L-l} - etahat_{L-l}) )
xihat_k    = (1,...,1)^T                     in R^{2^k}
etahat_k   = 2^{-k} (1, 2, ..., 2^k)^T       in R^{2^k}
```

(`Ihat_l` identity, `Shat_l` sub-diagonal shift, both of order `2^l`), and in
`D` dimensions `P_{l,L} = Phat_{l,L}^{(x)D}` (BK (22)). Verified against the
QTT form to 1e-12 for all `0 <= l <= L <= 7` (prototype `p0b_verify.py`).

The classical BPX preconditioner is

```
C_{2,L} = sum_{l=0}^{L} 2^{-2l} P_{l,L} P_{l,L}^T                       (BK (29))
```

and BK Theorem 1 (Dahmen–Kunoth, Oswald) states
`c <C_{2,L}^{-1} v, v> <= <A_L v, v> <= C <C_{2,L}^{-1} v, v>` with `c, C`
independent of `L`.

**Why a sum over levels.** The `H^1` energy of a function whose fine-grid
coefficients are `v` decomposes across dyadic scales; the level-`l` component
carries energy `~ 2^{2l}` times its `L^2` mass. `P_{l,L} P_{l,L}^T` is
(up to the Riesz/`L^2` normalization) the `L^2` projector onto `V_l` lifted to
level `L`, so `sum_l 2^{-2l} P_{l,L} P_{l,L}^T` inverts that scale-by-scale
weighting. It is a sum because a *norm equivalence over a multiresolution
decomposition* is a sum; there is nothing tensor-specific in it. The
tensor-specific part is §1.5.

**The paper's own new result** is that this is not the operator you want if you
want a *symmetric* preconditioner you can actually apply. `C_{2,L}^{1/2}` is not
implementable. BK instead take

```
C_L = sum_{l=0}^{L} 2^{-l} P_{l,L} P_{l,L}^T                            (BK (30))
```

— note `2^{-l}`, not `2^{-2l}` — and prove (**Theorem 2**, their new result,
proof in Appendix A) that

```
c ||v||_2^2  <=  <C_L A_L C_L v, v>  <=  C ||v||_2^2,      c, C independent of L.
```

So `C_L` is a *square root substitute*: `C_L A_L C_L` is well-conditioned, and
equivalently (BK Remark 2) `|| u ||_{H1} ~ || C_L^{-1} v ||_2`, i.e. the
functions `sum_i (C_L)_{ij} phi_{L,i}` form an `L`-uniform Riesz basis of
`V_L subset H^1`. That equivalence is what makes the *algebraic* residual
`||B_L u - g_L||_2` a meaningful `H^1` error estimator — which is the reason a
solver tolerance means something.

**Two different objects, two different customers.** `C_{2,L}` (weight
`2^{-2l}`) is the left preconditioner `B^{-1} ~ A^{-1}` that a preconditioned
gradient eigensolver wants; `C_L` (weight `2^{-l}`) is the two-sided change of
variables that a linear solver wants. They come from the *same* core template
with a different scalar per level. **Measured** (prototype `p8_c2.py`, dense
`numpy.linalg.eigvalsh` on the symmetrized similar matrix, `D = 1`, float64):

| `L` | rank(`Chat2`) | `kappa(A_DN)` | `kappa(Chat2 A)` | `kappa(Chat A Chat)` |
|---|---|---|---|---|
| 4 | 8 | 4.377e+02 | 5.1193 | 5.6674 |
| 6 | 8 | 6.741e+03 | 6.3648 | 7.9283 |
| 8 | 8 | 1.067e+05 | 7.2179 | 9.5217 |
| 10 | 8 | 1.702e+06 | 7.8222 | 10.6617 |
| 12 | 8 | 2.720e+07 | 8.2634 | 11.4910 |
| 13 | 8 | 1.088e+08 | 8.4401 | 11.8206 |

(`Chat2 = 2^{2L} C_{2,L}`, `Chat = 2^L C_L`; the `2^{wL}` factors only undo the
`2^{2L}` in `A_L` so that the unscaled `A_DN` can be used — see §1.5.) Both are
bounded; the left BPX `C_{2,L}` is slightly better conditioned, and has the same
QTT rank 8. **The eigensolver spec should ask for `C_{2,L}`, not `C_L`.**

### 1.5 Why BPX has a low-rank QTT representation, explicitly

This is the "хрен разберёшься" part. It rests on two three-line lemmas and one
observation about sums.

**Lemma A (BK Lemma 1).** With `I = [[1,0],[0,1]]`, `J = [[0,1],[0,0]]` and the
*strong Kronecker product* `⋊⋉` (BK Definition 1: block-matrix multiplication in
which scalar multiplication is replaced by the Kronecker product),

```
[ Ihat_l   Shat_l ]        [ I   J^T ] ⋊⋉ l
[   0      Jhat_l ]   =    [ 0    J  ]              ,   Jhat_l = J^{(x)l}.
```

Proof is one recursion: `Ihat_l = I (x) Ihat_{l-1}`,
`Shat_l = I (x) Shat_{l-1} + J^T (x) Jhat_{l-1}`, `Jhat_l = J (x) Jhat_{l-1}`.
So `Uhat := [[I, J^T],[0, J]]` is a **rank-2 QTT core carrying identity *and*
shift simultaneously**.

**Lemma B (BK Lemma 2).** With
`Xhat = (1/2) [[ (1,2)^T, (0,1)^T ], [ (1,0)^T, (2,1)^T ]]` (rank `2x2`, mode
size `2 x 1`) and `Phat = [1; 0]`,

```
[ etahat_l           ]
[ xihat_l - etahat_l ]  =  Xhat^{⋊⋉ l} ⋊⋉ Phat.
```

Again one recursion, on the pair `(eta, xi - eta)`. So the linear ramp `eta`
needs **rank 2**, which is the QTT statement that a linear function is rank 2.

**Lemma C (BK Lemma 4)** = A + B: with `Ahat = [1, 0]`,

```
Phat_{l,L} = 2^{-(L-l)/2} * Ahat ⋊⋉ Uhat^{⋊⋉ l} ⋊⋉ Xhat^{⋊⋉ (L-l)} ⋊⋉ Phat.
```

`L + 2` cores; the first `l` have mode size `2 x 2` (they carry the coarse index
bits, shared between rows and columns), the last `L - l` have mode size `2 x 1`
(they carry the fine bits, present only in the rows). **All ranks are 2.**
Verified exactly against BK (14) for all `0 <= l <= L <= 7`, max abs difference
`< 1e-12` (`p0b_verify.py`).

**One level of the preconditioner** (BK (80)). Using the core product `•`
(ranks Kronecker-multiply, modes contract),

```
Phat_{l,L} Phat_{l,L}^T = 2^{-(L-l)} * Ahat_b ⋊⋉ Uhat_b^{⋊⋉ l} ⋊⋉ Xhat_b^{⋊⋉ (L-l)} ⋊⋉ Phat_b
Ahat_b = Ahat • Ahat,  Uhat_b = Uhat • Uhat^T,  Xhat_b = Xhat • Xhat^T,  Phat_b = Phat • Phat
```

ranks `4 = 2^2` throughout. Explicitly `Ahat_b = [1 0 0 0]`,
`Phat_b = [1;0;0;0]`. In `D` dimensions each core is replaced by its `D`-fold
Kronecker power (BK (85)), so the ranks are `4^D`. Verified against dense for
all `0 <= l <= L <= 7`, `D = 1` (`p0b_verify.py`).

**The level sum (BK Theorem 3).** Now the point. Every level `l` uses *the same
two cores*, `Uhat_b` on the first `l` sites and `Xhat_b` on the last `L - l`,
differing only in where it switches and by the scalar `2^{-l}`. A sum of `L+1`
such chains is therefore a **two-state automaton**: state 1 = "still on
`Uhat_b`", state 2 = "already switched to `Xhat_b`", with the switch at level `l`
carrying weight `2^{-l}`. In core form:

```
Pi_L C_L Pi_L^T = [ A_b   A_b ] ⋊⋉ C_1 ⋊⋉ ... ⋊⋉ C_L ⋊⋉ [ 0 ; P_b ]

                  [ U_b     2^{-l} U_b ]
            C_l = [                    ]           l = 1..L
                  [  0      2^{-D} X_b ]
```

with `A_b = Ahat_b^{(x)D}`, `U_b = Uhat_b^{(x)D}`, `X_b = Xhat_b^{(x)D}`,
`P_b = Phat_b^{(x)D}`, and `Pi_L` the bit permutation from dimension-major to
level-major order (BK (49)). **Ranks `2 * 4^D = 2^{2D+1}`: 8, 32, 128 for
`D = 1, 2, 3`, independent of `L`.**

This is the answer to "why does the preconditioner have a low-rank QTT
representation at all": because in the *scale* separation (as opposed to the
spatial separation of BK ref. [5], where the preconditioner ranks grow with the
level count) the level index is the tensor index, the level sum is a sum of
`L+1` terms that differ in *one bond position*, and a sum of chains that differ
in one bond position is a bidiagonal transfer matrix of rank 2 per bond. It is
the same mechanism by which `sum_k S^k` (a triangular Toeplitz matrix) is rank 2
in QTT.

**Measured** (prototype `p6.py` §(i), b300):

| `D` | `L` | rank of the explicit Thm-3 cores | after `round(1e-14)` | build time |
|---|---|---|---|---|
| 1 | 20 / 30 / 50 | 8 / 8 / 8 | 8 / 8 / 8 | < 0.01 s |
| 2 | 20 / 30 / 50 | 32 / 32 / 32 | 32 / 32 / 32 | ≤ 0.01 s |
| 3 | 20 / 30 / 50 | 128 / 128 / 128 | 128 / 128 / 128 | 0.28 / 0.44 / 0.76 s |

and the explicit cores agree with an independent "assemble the `L+1` terms and
round" construction to a relative 1.2e-15 (`L=4`), 2.0e-15 (`L=6`), 7.1e-15
(`L=8`) in `D = 1`.

**Loud failure worth recording.** The naive assembly (build the `L+1` terms,
add, `round(1e-14)`) *works* in `D = 1` up to `L = 50` but **fails in `D = 2`
at `L = 30`**: it returns rank **102** where the theorem says 32
(`p5_alt2.py` §(b): `L=10` → 32, `L=20` → 32, `L=30` → 102, assembly 10.2 s).
The rounding of a rank-`32*31` sum with weights spanning `2^{-30}` is itself
representation-ill-conditioned. **Implement Theorem 3 directly; never assemble
the level sum numerically.** This is BK's own thesis applied to their own
construction.

### 1.6 The preconditioned operator, and why `C @ A @ C` is not it

Fusing the preconditioner into the *differentiation* factor is the second half
of the paper. BK Lemma 5 (proved in their Appendix B) gives the optimal-rank
form of the product `Mhat_{L,alpha} Phat_{l,L}`,

```
Mhat_{L,alpha} Phat_{l,L} = 2^{(alpha+1/2) l} * Ahat ⋊⋉ Uhat^{⋊⋉ l} ⋊⋉ That_alpha ⋊⋉ Yhat_alpha^{⋊⋉ (L-l)} ⋊⋉ Nhat_alpha
```

with (BK (78))

```
That_1 = [1; -1],   Yhat_1 = (1/2) [ (1,1)^T ],   Nhat_1 = [1]
That_0 = [[1,1],[1,-1]],  Yhat_0 = (1/2) [[ (2,2)^T, (-1,1)^T ], [ 0, (1,1)^T ]],
Nhat_0 = (1/2) [[ (1,0)^T ], [ (0,1)^T ]]
```

and hence (BK (82))

```
Mhat_{L,alpha} Phat_{l,L} Phat_{l,L}^T
   = 2^{(alpha+1/2)L - (L-l)} * Ahat_b ⋊⋉ Uhat_b^{⋊⋉ l} ⋊⋉ What_alpha ⋊⋉ Zhat_alpha^{⋊⋉ (L-l)} ⋊⋉ Khat_alpha
What_alpha = That_alpha • Ihat,  Zhat_alpha = Yhat_alpha • Xhat^T,  Khat_alpha = Nhat_alpha • Phat
```

of ranks `4, ..., 4, 4/2^{|alpha|}, ..., 4/2^{|alpha|}` — versus the ranks `8`
you would get by multiplying `Mhat_{L,alpha}` and `Phat Phat^T` as separate
QTT objects. **Verified against dense** for `alpha = 1`, all `0 <= l <= L <= 7`,
relative max error `< 1e-11` (`p0b_verify.py`).

Summing over `l` by the same two-state automaton (BK Theorem 4) gives
`Q_{L,alpha} = M_{L,alpha} C_L` of rank `4^D + 4^D/2^{|alpha|} <= 2^{2D+1}`, and
BK Theorem 5 then bounds the ranks of

```
B_L = C_L A_L C_L = sum_{(alpha,alpha')} Q_{L,alpha}^T Lambda_{L,alpha,alpha'} Q_{L,alpha'}
R_l = 2^{4D} sum_{(alpha,alpha')} (1 + 2^{-|alpha|})^2 r_{l,alpha alpha'} <= 12 D^2 2^{4D} r
```

(`r` = the QTT ranks of the coefficient). `D = 2` Laplacian: `R_l = 1152`. That
number is the honest cost of BPX in 2D and BK say so in their conclusion
("fairly large for `D > 1`"). The escape (their §5.5, (92), and their §7.2
practice) is to never form `B_L`: for the Laplacian

```
B_L = sum_{k=1}^{D} Theta_{L,k}^T Theta_{L,k},   Theta_{L,k} = Lambda^{1/2}_{L,k} • Q_{L,alpha}, alpha = e_k
```

and apply it as `Theta^T (round(Theta v))`, so the largest rank ever formed is
that of `Theta` — 6 in `D = 1`, 24 in `D = 2` (BK §7.4).

For `D = 1` all the scalars collapse: `Lambda = 2^{-L}`, `Mhat_{L,1} = 2^{3L/2}(I-S)`,
and with `Theta := 2^{-L/2} Q_{L,1}` the per-level factor
`2^{-l} * 2^{3L/2 - (L-l)} * 2^{-L/2} = 1` **exactly, for every `l`** — the
representation has no `L`-dependent scalars anywhere, which is precisely why it
is representation-stable. `Theta = (I - S) Chat` with `Chat = 2^L C_L`, and
`B = Theta^T Theta = Chat A_DN Chat`.

**Measured** (`p2_bpx.py` §(a), b300):

| `L` | rank `Chat` | `t_C` | rank `Theta` | `t_Theta` | rank `B = Theta^T Theta` | `t_B` | rank `round(C@A@C)` |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 0.00 s | 6 | 0.00 s | 17 | 0.00 s | 23 |
| 16 | 8 | 0.02 s | 6 | 0.01 s | 17 | 0.00 s | 79 |
| 20 | 8 | 0.04 s | 6 | 0.02 s | 17 | 0.00 s | 104 |
| 30 | 8 | 0.19 s | 6 | 0.09 s | 17 | 0.01 s | 156 |
| 40 | 8 | 0.37 s | 6 | 0.23 s | 17 | 0.01 s | 142 |
| 50 | 8 | 0.73 s | 6 | 0.43 s | 17 | 0.01 s | **38** |

`rank(Theta) = 6 = 4 + 4/2` exactly as Theorem 4 predicts; `rank(B) = 17`,
comfortably under BK's bound `2^4 * (1 + 1/2)^2 * 1 = 36`. The last column is
the interesting one: the rank of the *rounded* separate product grows, then
*collapses* to 38 at `L = 50`. That collapse is not compression, it is
destruction:

**The decisive measurement** (`p6.py` §(ii)): relative error of the action of
`round(C@A@C, 1e-14)` against the combined `B`, on a normalized smooth QTT
vector:

| `L` | rank of `round(C@A@C)` | relative action error |
|---|---|---|
| 10 | 36 | 1.315e-10 |
| 20 | 104 | 5.998e-04 |
| 30 | 156 | 2.614e+02 |
| 40 | 142 | 1.472e+08 |
| 50 | 38 | **4.770e+14** |

`4^L eps` is 2.3e-10 / 2.4e-4 / 2.6e+2 / 2.7e+8 / 2.8e+14. **The two agree to
within a factor of 1.1 at every `L`.** Preconditioning by multiplication is
worthless past `L ≈ 15`; the combined representation is exact to 1.4e-14 at
`L = 20` (§1.3 table). This is [BK20]'s central practical claim and it
reproduces on the nose.

### 1.7 The condition-number bound and what it depends on

BK Theorem 2: `c ||v||^2 <= <C_L A_L C_L v, v> <= C ||v||^2` with `c, C`
independent of `L`. What the constants *do* depend on:

* the ellipticity constants of the bilinear form, `Abar = ess inf inf_xi
  xi^T A xi / xi^T xi > 0` and `||A||_inf` — i.e. **the coefficient contrast
  enters multiplicatively**, and BPX does nothing about it;
* the spatial dimension `D` (through the norm equivalence constants and,
  computationally, through the `4^D` ranks);
* the choice of `Gamma`, through the Poincaré constant;
* **not** on `L`, and **not** on the presence of singularities in `u`.

**Measured** (`p2_bpx.py` §(b), dense `numpy.linalg.eigvalsh` on the
symmetrized `B`, float64, `D = 1`):

| `L` | `N` | `kappa(A_DN)` | `lambda_min(B)` | `lambda_max(B)` | `kappa(B)` | `max abs(B - C A C)` |
|---|---|---|---|---|---|---|
| 4 | 16 | 4.377e+02 | 2.0113 | 11.3987 | **5.6674** | 3.8e-14 |
| 6 | 64 | 6.741e+03 | 2.0008 | 15.8628 | **7.9283** | 3.2e-13 |
| 8 | 256 | 1.067e+05 | 2.0000 | 19.0439 | **9.5217** | 1.7e-12 |
| 10 | 1024 | 1.702e+06 | 2.0000 | 21.3235 | **10.6617** | 2.5e-12 |
| 12 | 4096 | 2.720e+07 | 2.0000 | 22.9819 | **11.4910** | 6.9e-12 |
| 13 | 8192 | 1.088e+08 | 2.0000 | 23.6411 | **11.8206** | 3.4e-11 |
| 14 | 16384 | 4.352e+08 | 2.000000 | 24.2116 | **12.1058** | (`p9_bigk.py`, 60 s) |
| 15 | 32768 | 1.741e+09 | 2.000000 | 24.7078 | **12.3539** | (`p9_bigk.py`, 395 s) |

`lambda_min(B) -> 2` exactly and `lambda_max(B)` increases with geometrically
shrinking increments (`+0.3296, +0.2852, +0.2481` at `L = 13, 14, 15`, ratios
`0.865, 0.870`), which extrapolates to `lambda_max -> ~26.4` and
`kappa(B) -> ~13.2..14.0`. **That extrapolation is not a measurement** (Q1). An
independent Lanczos check (`scipy.sparse.linalg.eigsh` on a matrix-free QTT
apply of `Theta^T Theta`, `tol=1e-8`) reproduces `kappa = 10.6617` at `L = 10`
exactly, but costs 20 s there and was abandoned as too slow to extend.
`D = 2` (`p5_alt2.py` §(b),
dense, level-major permutation applied):

| `L` | `N = 4^L` | rank `Chat` | `kappa(A)` | `lambda_min(B)` | `lambda_max(B)` | `kappa(B)` |
|---|---|---|---|---|---|---|
| 3 | 64 | 16 | 1.135e+02 | 2.0657 | 9.3218 | 4.5128 |
| 4 | 256 | 32 | 4.377e+02 | 1.9835 | 11.6247 | 5.8607 |
| 5 | 1024 | 32 | 1.709e+03 | 1.8425 | 13.8403 | 7.5118 |
| 6 | 4096 | 32 | 6.741e+03 | 1.7148 | 15.7910 | 9.2086 |

Same shape; `lambda_min` is now *decreasing* slowly, which the `D = 1` case did
not do. Not measured beyond `L = 6` in `D = 2`.

---

## 2. Is BPX the right answer here?

### 2.1 The baseline: unpreconditioned AMEn, measured

Problem: `A_DN v = h^2 b`, `h = 2^-L`, `b = (1,...,1,1/2)^T` (the load vector of
`f = 1` with the last node at the Neumann boundary), which is the discrete form
of `-u'' = 1, u(0) = 0, u'(1) = 0`. Oracle: for `P1` elements in 1D the Galerkin
solution **is** the nodal interpolant, so `v_j = x_j - x_j^2/2` **exactly**,
`x_j = j 2^-L`. Regime: b300, numpy, float64, `amen_solve(A, rhs, None, 1e-10,
kickrank=4, nswp=40, seed=0)`, single run.

| `L` | `kappa` | sweeps | wall | `true_res` | rank `v` | **rel. nodal error** | `4^L eps` |
|---|---|---|---|---|---|---|---|
| 6 | 6.74e+03 | 3 | 0.26 s | 1.92e-12 | 7 | 2.009e-12 | 9.09e-13 |
| 8 | 1.07e+05 | 3 | 0.01 s | 1.98e-11 | 7 | 4.252e-11 | 1.45e-11 |
| 10 | 1.70e+06 | **40** | 0.29 s | 1.22e-10 | 18 | 1.611e-10 | 2.33e-10 |
| 12 | 2.72e+07 | **40** | 0.42 s | 4.90e-09 | 20 | 2.171e-09 | 3.72e-09 |
| 14 | 4.35e+08 | **40** | 0.61 s | 1.08e-07 | 44 | 5.857e-08 | 5.96e-08 |
| 16 | 6.96e+09 | **40** | 1.09 s | 3.35e-06 | 42 | 1.377e-06 | 9.53e-07 |
| 18 | 1.11e+11 | **40** | 3.67 s | 1.39e-03 | 108 | 2.544e-05 | 1.53e-05 |
| 20 | 1.78e+12 | **40** | 7.20 s | 5.73e-02 | 132 | 3.949e-04 | 2.44e-04 |
| 22 | 2.85e+13 | **40** | 9.43 s | 3.87e-01 | 139 | 3.731e-03 | 3.91e-03 |
| 24 | 4.56e+14 | **40** | 11.90 s | 9.76e+00 | 159 | **1.029e-01** | 6.25e-02 |

Read this carefully, because it is *not* the story the eigensolver spec's §3.2
tells about gradient methods. **AMEn's sweep count does not grow like `kappa`** —
it is an ALS method, the local problems are solved directly, and the Galerkin
sweep is insensitive to the conditioning in that way. What grows is:

* the **accuracy floor**, which tracks `4^L eps` to within a factor 1.7 across
  eight orders of magnitude — this is representation ill-conditioning, measured
  end-to-end;
* the **ranks**, from 7 to 159, because AMEn's residual enrichment faithfully
  fits round-off noise once the true residual is below the floor;
* the **time**, from 0.01 s to 11.9 s, entirely because of the ranks.

And AMEn *tells the truth about it*: `converged=False` and a `true_res` of
9.76 at `L = 24`. ttpy2's `check_true_res=True` default is doing exactly its
job here. What it cannot do is fix the problem.

The number the parent question asked for, stated plainly: **unpreconditioned
AMEn on a QTT Laplacian is usable to about `L = 12` at `eps = 1e-10`, and to
about `L = 16` at `eps = 1e-6`. Past that the answer is wrong and the solver
says so.**

### 2.2 BPX, head to head

Same problem, same tolerance, interleaved runs (`p3_head.py`; b300, numpy,
float64, `eps = 1e-8`, `kickrank=4`, `nswp=40`, `seed=0`, `x0=None`, one run per
cell). Preconditioned system: `B u = Chat (h^2 b)` with `B = Theta^T Theta`
rounded to `1e-14`; the reported solution is `v = Chat u`.

| `L` | method | sweeps | converged | wall | `true_res` | rank | **rel. nodal error** | assembly |
|---|---|---|---|---|---|---|---|---|
| 10 | unprec AMEn | 3 | True | 0.27 s | 5.65e-10 | 7 | 1.801e-10 | — |
| 10 | **BPX AMEn** | 5 | True | 0.06 s | 1.87e-09 | 16 | **2.596e-11** | 0.01 s |
| 14 | unprec AMEn | 40 | False | 0.59 s | 1.36e-07 | 36 | 5.631e-08 | — |
| 14 | **BPX AMEn** | 6 | True | 0.09 s | 1.55e-09 | 13 | **5.146e-12** | 0.02 s |
| 18 | unprec AMEn | 40 | False | 4.04 s | 3.56e-04 | 68 | 3.292e-05 | — |
| 18 | **BPX AMEn** | 6 | True | 0.17 s | 1.86e-09 | 13 | **4.439e-12** | 0.06 s |
| 22 | unprec AMEn | 40 | False | 8.70 s | 4.48e-01 | 98 | 3.167e-03 | — |
| 22 | **BPX AMEn** | 6 | True | 0.36 s | 3.56e-09 | 14 | **3.299e-12** | 0.21 s |
| 26 | unprec AMEn | 40 | False | 19.16 s | 3.90e+02 | 162 | 4.846e-01 | — |
| 26 | **BPX AMEn** | 6 | True | 0.29 s | 2.80e-09 | 14 | **4.308e-12** | 0.17 s |
| 30 | unprec AMEn | 40 | False | 25.13 s | 7.70e+06 | 162 | 1.049e+00 | — |
| 30 | **BPX AMEn** | 6 | True | 0.33 s | 3.79e-09 | 14 | **1.226e-11** | 0.28 s |

Six sweeps, flat from `L = 14` to `L = 30`; wall time 0.09 → 0.33 s over a
`2^16`-fold increase in the number of nodes (`2^14` to `2^30 ≈ 1.07e9`); nodal
error flat at ~4e-12; iterate rank 13–14 against 3 for the exact solution. The
unpreconditioned column is a 76x slower and, at `L = 30`, **100 % wrong**.

Counterweights on the same table. (i) At `L = 10` the unpreconditioned run is
*more* accurate (1.8e-10 vs 2.6e-11 — no, the BPX run wins there too, but only
by 7x) and the assembly is not free; below `L ≈ 12` BPX buys nothing but the
extra code. (ii) The BPX iterate has rank 13–14 where the exact solution has
rank 3: the preconditioned unknown `u = Chat^{-1} v` is *less* compressible than
`v`, exactly the caveat BK flag as their Assumption 1 (§1.2 item 3). At `L = 30`
this is a factor ~20 in storage that buys 11 orders of magnitude of accuracy, so
it is a good trade; on a problem where the solution rank is already 100 it might
not be. (iii) The BPX error floor is ~4e-12, not `eps_machine`: the solver was
run at `eps = 1e-8` and `kappa(B) ≈ 12`, so this is the solver tolerance
showing through, not a stability limit.

### 2.3 Exponential sums / approximate inverses: do they transfer to QTT?

`docs/plans/riemannian-autodiff.md` §6.2–6.3 measured a sinc/exponential-sum
`B^{-1} = sum_q c_q exp(-t_q L) (x) ... (x) exp(-t_q L)` and got 10–12
iterations across `kappa = 4.4e2 .. 1.1e5`. That construction needs each summand
to be a **rank-1 TT-matrix**, which it is when the operator is a Kronecker sum
over *physical* modes. That spec already flagged that it should not transfer to
a 1D QTT operator. **Measured** (`p4_alt.py` §(ii)): QTT rank of the dense
`expm(-t A_DN)` obtained by TT-SVD,

| `L` | `t` | rank at `eps=1e-10` | rank at `eps=1e-6` |
|---|---|---|---|
| 6 | `1/lambda_max` = 2.50e-01 | 9 | 6 |
| 6 | 1.0 | 12 | 9 |
| 6 | `1/sqrt(l_min l_max)` = 2.05e+01 | 19 | 13 |
| 6 | `1/lambda_min` = 1.69e+03 | 8 | 7 |
| 8 | 2.50e-01 / 1.0 / 8.16e+01 / 2.67e+04 | 9 / 12 / 21 / 8 | 7 / 9 / 15 / 7 |
| 10 | 2.50e-01 / 1.0 / 3.26e+02 / 4.25e+05 | 9 / 12 / 21 / 8 | 7 / 8 / 15 / 8 |

The ranks are `L`-independent (good) but **not 1** (8–21), so the cheap
rank-1-per-term structure that makes eq. (24)–(25) of [RNO19] worth having is
gone: `rho_B ≈ 40..60` terms of rank ≈ 15 each would cost more than one AMEn
sweep on `B`. Confirmed: **the exponential-sum route does not transfer to the
QTT (by-scale) setting.** It remains the right answer for Kronecker sums over
physical modes, which is a different problem and is owned by
`docs/plans/riemannian-autodiff.md`.

### 2.4 What is actually cheaper in `D = 1`: the exact inverse, and integration

`A_DN = (I - S)^T (I - S)` and `(I - S)^{-1} = sum_{k>=0} S^k = T`, the
lower-triangular all-ones matrix, which is a **QTT rank-2** matrix (it is
`tt.Toeplitz(ones, kind='L')`). Hence

```
A_DN^{-1} = T T^T,        (A_DN^{-1})_{ij} = min(i, j),      QTT rank 4.
```

**Measured** (`p4_alt.py` §(i), (iii)):

| `L` | rank at `1e-14` | rank at `1e-10` | rank at `1e-6` | `\|\|A^{-1}\|\|_2` | rel. nodal error of `x = A^{-1}(h^2 b)` |
|---|---|---|---|---|---|
| 6 | 4 | 4 | 4 | 1.686e+03 | 1.976e-15 |
| 8 | 5 | 4 | 4 | 2.666e+04 | 1.717e-14 |
| 10 | 4 | 4 | 4 | 4.254e+05 | 7.298e-15 |
| 12 | 5 | 4 | 4 | 6.801e+06 | 3.123e-14 |

Machine precision, no iteration at all, at `4^L eps = 3.7e-09`. (Ranks were
obtained by TT-SVD of the dense `min(i,j)`, which is why `L <= 12`; the rank-2
`T` construction is exact at any `L` and is what an implementation would use.)

The generalization is better still. Symmetric preconditioning by `T` turns the
*variable*-coefficient operator into a diagonal one, **exactly**:

```
A_a = (I - S)^T diag(a) (I - S)   =>   T^T A_a T = diag(a),   since (I - S) T = I.
```

so `cond(T^T A_a T) = a_max / a_min` with **no `L` dependence and no constant**.
**Measured** (`p5_alt2.py` §(a), dense `eigvalsh`, float64):

| `L` | coefficient | contrast | `kappa(A_a)` | `kappa(BPX)` | `kappa(INT)` |
|---|---|---|---|---|---|
| 8 | `a = 1` | 1.0e+00 | 1.067e+05 | 9.5217 | **1.0000** |
| 8 | `1/(2+cos 64 pi x)` | 2.7e+00 | 1.661e+05 | 13.9384 | **2.7171** |
| 8 | 2 layers, ratio 1e2 | 1.0e+02 | 8.894e+06 | 888.9436 | **100.0000** |
| 8 | 2 layers, ratio 1e4 | 1.0e+04 | 8.878e+08 | 88892.4635 | **10000.0000** |
| 10 | `a = 1` | 1.0e+00 | 1.702e+06 | 10.6617 | **1.0000** |
| 10 | `1/(2+cos 64 pi x)` | 3.0e+00 | 3.181e+06 | 18.4175 | **2.9808** |
| 10 | 2 layers, ratio 1e2 | 1.0e+02 | 1.420e+08 | 1020.6932 | **100.0000** |
| 10 | 2 layers, ratio 1e4 | 1.0e+04 | 1.418e+10 | 102068.3819 | **10000.0000** |
| 12 | `a = 1` | 1.0e+00 | 2.720e+07 | 11.4910 | **1.0000** |
| 12 | `1/(2+cos 64 pi x)` | 3.0e+00 | 5.348e+07 | 24.9382 | **2.9988** |
| 12 | 2 layers, ratio 1e2 | 1.0e+02 | 2.271e+09 | 1115.8313 | **100.0000** |
| 12 | 2 layers, ratio 1e4 | 1.0e+04 | 2.267e+11 | 111582.5913 | **10000.0000** |

`kappa(INT)` equals the contrast to five digits in every row, at every `L`.
BPX is uniformly a factor 9–25 worse and its factor *grows* with `L` in the
oscillatory row (13.9 → 18.4 → 24.9), which the constant-coefficient row does
not do.

**The catch, measured** (`p10_int.py`; `T` built as an explicit rank-2 QTT,
verified to equal `numpy.tril(ones)` exactly at `L = 3, 5, 7`, and
`||T^T A_DN T - I||_max` = 3.9e-15 / 4.4e-14 / 8.7e-14 there). `T` may be used
in exactly one way. **As a direct solve** — `v = T diag(1/a) T^T (h^2 b)`, two
QTT matvecs and one Hadamard division, no iteration and no assembled operator —
it is unconditionally stable:

| `L` | rank `v` | rel. nodal error | `4^L eps` | wall |
|---|---|---|---|---|
| 10 | 3 | 1.604e-15 | 2.33e-10 | 0.002 s |
| 18 | 3 | 3.184e-15 | 1.53e-05 | 0.002 s |
| 26 | 3 | 2.222e-15 | 1.00e+00 | 0.003 s |
| 34 | 3 | 8.527e-15 | 6.55e+04 | 0.003 s |
| 40 | 3 | 5.552e-15 | **2.68e+08** | 0.004 s |

Machine precision at `L = 40` (`2^40 ≈ 1.1e12` nodes) in 4 ms, with the
iterate at the exact rank 3 of the solution. But **as a preconditioner in the
[BK20] sense — assemble `T^T A T` as a QTT matrix and iterate on it — it fails
exactly like everything else the paper warns about**:

| `L` | rank `round(T^T A T)` | sweeps | converged | `true_res` | rel. nodal error | wall |
|---|---|---|---|---|---|---|
| 14 | 13 | 2 | True | 3.85e-11 | 9.997e-08 | 0.40 s |
| 22 | 13 | 3 | True | 9.70e-10 | 4.790e-03 | 0.15 s |
| 30 | 13 | 9 | True | 2.64e-09 | **9.665e-01** | 1.10 s |
| 40 | 13 | 40 | **False** | 8.86e-01 | **1.000e+00** | 287.72 s |

`T^T A T` **is the identity** and `round(·, 1e-14)` returns it at rank 13.
That is [BK20]'s representation ill-conditioning in its purest possible form:
the exact answer has rank 1, the product representation cannot be reduced to it
numerically, and the resulting object is wrong by `4^L eps`. Note especially the
`L = 30` row: AMEn reports `converged=True` with `true_res = 2.6e-09` and the
answer is **97 % wrong**, because the residual it converged is the residual of
the corrupted operator. This is the one place in this document where a solver
returns a plausible wrong answer without complaining, and it is why V8 exists.

Counterweights, and they are decisive for `D >= 2`:

* **`T` is a 1D accident.** `(I - S) T = I` has no `D`-dimensional analogue:
  in 2D the operator is `A_1 (x) I + I (x) A_1` and there is no `T` with
  `T^T A T` diagonal. This is exactly why [COR16] needed a constrained
  minimization with Volterra operators for `D = 2`, and why [BK20] §1.4 reports
  that there "the matrix condition number still grows exponentially with respect
  to `L`". **Not measured here** — I did not read [COR16].
* **`T` is only safe as a direct solve**, per the two tables above. The moment
  the problem stops being exactly `M^T diag(a) M` — a reaction term, a
  first-order term, a non-uniform mesh, two dimensions — you have to iterate,
  and then you are back to needing [BK20]'s *combined* representation.
* The contrast enters `kappa(INT)` linearly, and it enters `kappa(BPX)`
  linearly too — **neither** fixes high contrast. That is a separate problem
  (coefficient-adapted / robust multilevel methods) and neither paper solves it.

### 2.5 Recommendation

The honest summary of §2.1–§2.4 in one sentence: **anything that assembles a
preconditioned QTT operator as a product of separately-represented QTT factors
and then iterates on it is dead past `L ≈ 15`, whatever the preconditioner is —
`C@A@C` (5.998e-04 at `L = 20`), `T^T A T` (4.790e-03 at `L = 22`), or nothing
at all (3.949e-04 at `L = 20`). [BK20]'s *combined* construction is the only
thing measured here that survives, and that is the paper's real contribution.**
Build order:

1. **`tt.qlaplace_dn`, `tt.qtri_ones`, `tt.qdiff`, `qtt_ell.solve_direct_1d`**
   (§3, items K1–K4). Half a day. This gives the `D = 1` **direct** solve
   `v = T diag(1/a) T^T f`, which is exact to machine precision at `L = 40` in
   4 ms with rank 3 (§2.4) — better than any iterative method can be, and
   currently impossible with ttpy2's public API. It must ship with a docstring
   and a check that refuse the case it does not cover: `qtt_ell.solve_direct_1d`
   is valid **only** for `D = 1` and `A = M^T diag(a) M`, and must not be
   reachable as a generic "preconditioner", because `T^T A T` inside an
   iteration is wrong by 97 % at `L = 30` while reporting `converged=True`
   (§2.4).
2. **`tt.bpx(d, D, weight=1|2)` from the explicit Theorem-3 cores** (§3, item
   K5). One day. Ranks exactly `2^{2D+1}` = 8/32/128, build < 1 s at
   `L = 50, D = 3`, measured. This is the object `eigenvalues.md` §3.3 and
   `riemannian-autodiff.md` §6.4 are waiting for; the hard part of [BK20] is not
   `C_L`, and giving them `C_L` (and `C_{2,L}`) unblocks both cheaply. It must
   ship with the warning that `bpx(d) @ A @ bpx(d)` is **not** a usable
   preconditioned operator (§1.6).
3. **`tt.bpx_theta(d, D, coeff)` = the fused `Theta_{L,k}` of BK Lemma 5 /
   Theorem 4** (§3, item K6). Two to three days, `D = 1` first. This is the only
   construction in this document that survives an iteration past `L = 15`, and
   the only route to `D = 2` at all. Measured: 6 AMEn sweeps flat from `L = 14`
   to `L = 30`, nodal error ~4e-12, 0.33 s at `L = 30` where the
   unpreconditioned solve is 100 % wrong after 25 s (§2.2).

**So: is BPX the right answer? For a general elliptic operator, yes, and it is
the only right answer measured here.** The user's "может можно лучше" is
correct in one specific and useful corner — `D = 1`, `A = M^T diag(a) M`, where
the exact inverse has QTT rank 4 and a direct solve beats every preconditioner
by ten orders of magnitude — and wrong everywhere else, because the cheaper
alternatives fail for [BK20]'s *own* reason, which is not conditioning of the
matrix but conditioning of the representation.

**Where this recommendation loses.** If the target is `D >= 2` from day one,
step 1 is a detour: skip it and start on step 3, which does not depend on it.
If the target is `L <= 12` (up to `4096` nodes per dimension), none of this is
needed — plain `amen_solve` on `tt.qlaplace_dd` converges in 3 sweeps to 5e-10
there (§2.1), and every line of code below is overhead.

---

## 3. The QTT construction kit

### 3.1 What `tt/core/tools.py` already covers

Audited from source (`tt/core/tools.py`, 830 lines):

* `qlaplace_dd(d)` — the `D`-dimensional QTT Laplacian, **Dirichlet–Dirichlet**,
  ranks 3 (`D=1`) / 4 (`D>1`). This is [KK12]'s construction. Verified here to
  equal `tridiag(-1,2,-1)` exactly at `d=3`.
* `qshift(d)` = `shift(d,-1)` — the sub-diagonal shift `S`, rank 2.
  **Verified**: `tt.qshift(4).full() == numpy.eye(16, k=-1)`. This is BK's
  `Shat_L`, and Lemma A is the reason it is rank 2.
* `IpaS(d, a)` — `I + a S`, rank 2. **Verified**: `tt.IpaS(4,-1).full() ==
  I - qshift`. So `IpaS(d,-1)` is BK's `2^{-3L/2} Mhat_{L,1}` and K2 is a
  three-line wrapper, not new code.
* `Toeplitz(x, d, D, kind)` with `kind='L'` — lower-triangular multilevel
  Toeplitz. **Verified**: `tt.Toeplitz(tt.ones(2,4), 4, kind='L').full() ==
  numpy.tril(numpy.ones((16,16)))`. This *is* BK's `T = (I-S)^{-1}`, so K3 is
  also a wrapper. (Note the argument: `x` must have exactly `d` cores for
  `kind='L'`; `d+1` raises `ValueError: dimension mismatch`, which is the
  `'F'` convention.)
* `xfun`, `linspace`, `stepfun`, `delta`, `unit`, `sin`, `cos` — the vector
  side; `xfun` gives `eta`-like ramps.
* `eye`, `kron`, `mkron`, `zkron`, `zkronv`, `zmeshgrid`, `zaffine`,
  `reshape`, `permute` — the plumbing, including the level-major /
  dimension-major re-ordering that BK's `Pi_L` performs (`permute`).
* Hadamard product `x * y` on `tt.vector`, so `diag(a)` from a QTT coefficient
  is `tt.diag(a)`.

That is a large fraction of the kit. **The QTT construction primitives are
essentially all present; what is missing is the elliptic layer on top of them.**

### 3.2 What is missing (exact signatures, one-line contracts)

All new code goes into a new module `tt/algs/qtt_ell.py` plus **at most** the
five additions to `tt/core/tools.py` marked (T). No existing file is modified
beyond adding new names to `__all__`.

```python
# --- K1 (T) boundary conditions -------------------------------------------
def qlaplace_dn(d, bc='DN'):
    """QTT Laplacian on 2^d nodes with mixed BCs; bc in {'DD','DN','ND','NN'},
    one letter per end per dimension.  DN = Dirichlet at x=0, Neumann at x=1,
    which is the only combination with exactly 2^d dofs per level and hence the
    one [BK20] uses.  Unscaled (no h^-2).  Ranks 3..4 (D=1), <=5 (D>1)."""

# --- K2 (T) first-order / differentiation factor ---------------------------
def qdiff(d, D=1, kind='backward'):
    """QTT of the difference operator I - S (backward) or S^T - I (forward),
    per dimension; the factor M in A = M^T diag(a) M.  Rank 2 per dimension."""

# --- K3 (T) triangular ones / discrete integration -------------------------
def qtri_ones(d, D=1, upper=False):
    """QTT of the lower- (or upper-) triangular all-ones matrix, = (I-S)^-1.
    Rank 2 per dimension.  Thin wrapper over Toeplitz(ones, kind='L')."""

# --- K4 variable-coefficient elliptic operator -----------------------------
def stiffness(coeff, d, D=1, bc='DN', reaction=None):
    """A = sum_k M_k^T diag(a_k) M_k  (+ mass * diag(c)) as a tt.matrix.
    `coeff`: a tt.vector of the coefficient nodal/midpoint values per dimension
    (a list of D of them for anisotropic A), or a scalar.  No rounding is done
    beyond 1e-14; the caller owns the accuracy of `coeff`."""

def load_vector(f, d, D=1, bc='DN', quad='midpoint'):
    """RHS f_L = (f(phi_{L,i}))_i for f given as a tt.vector of point values,
    with the BK (10) L^2 normalisation folded in.  For f = 1 and bc='DN' this
    is 2^{-L/2} (1,...,1,1/2)."""

# --- K5 the BPX preconditioner (BK Theorem 3) ------------------------------
def bpx(d, D=1, weight=1, scaled=True):
    """BPX preconditioner as a tt.matrix, built from the explicit cores of
    [BK20] Theorem 3.  weight=1 gives C_L = sum_l 2^-l P_l P_l^T (the symmetric
    two-sided preconditioner of BK Theorem 2, for linear solves); weight=2
    gives C_{2,L} = sum_l 2^-2l P_l P_l^T (the classical left preconditioner of
    BK Theorem 1, for eigensolvers).  `scaled=True` multiplies by 2^{weight*d}
    so that the result pairs with the *unscaled* operators of K1/K4.
    TT ranks exactly 2^{2D+1}, independent of d.  SPD."""

def prolongation(l, d, D=1):
    """P_{l,L} of [BK20] Lemma 4 as a rectangular tt.matrix (2^{Dd} x 2^{Dl}),
    ranks 2^D.  Provided for tests and for anyone building their own multilevel
    scheme; `bpx` does not call it."""

# --- K6 the combined preconditioned operator (BK Lemma 5, Theorems 4,5) ----
def bpx_theta(d, D=1, coeff=None, alpha=None):
    """The factors Theta_{L,k} of [BK20] (92): B_L = sum_k Theta_k^T Theta_k
    with Theta_k = Lambda_k^{1/2} M_{L,e_k} C_L, built from the fused cores of
    Lemma 5 so that no 4^d cancellation is ever represented.  Returns a list of
    D tt.matrices of TT rank <= 2^{2D} + 2^{2D-1}.  Apply as
    sum_k Theta_k^T round(Theta_k v) -- never form B_L for D > 1."""

def bpx_operator(d, D=1, coeff=None, eps=1e-14):
    """B_L = C_L A_L C_L assembled explicitly and rounded.  Rank 17 measured
    for D=1 Laplacian ([BK20] bound 36); rank <= 1152 for D=2.  Raises
    ValueError for D >= 2 unless the caller passes eps explicitly and
    acknowledges the rank -- forming this object in 2D is usually a mistake,
    use bpx_theta."""

# --- K7 the solver front ends ----------------------------------------------
def solve_direct_1d(coeff, f, d, bc='DN'):
    """Exact solve of (M^T diag(a) M) v = f in D=1: v = T diag(1/a) T^T f.
    Two matvecs, no iteration, no assembled preconditioned operator.
    Measured exact to 5.6e-15 at d=40 for a=1 (rank 3, 4 ms).
    Raises ValueError for D > 1 or for any operator not of that exact form --
    this routine has no approximate mode and must never become one."""

def solve(A, f, d, D=1, precond='bpx', eps=1e-10, **amen_kwargs):
    """Iterative solve, returning the *nodal* solution.
    precond in {'none','bpx'}.  There is deliberately no 'int' option: the
    integration operator T is correct only in the direct form above, and
    assembling T^T A T and iterating on it returns a 97 %-wrong answer with
    converged=True at d=30 (measured, see V12b).
    Returns (v, info); info carries the AMEn history, the preconditioner used,
    and the achieved residual in *both* the preconditioned and the nodal norm.
    With precond='none' and d > 14 it warns, quoting the measured 4^d*eps
    accuracy floor of Sect. 2.1."""
```

Not needed, and deliberately not listed: non-uniform meshes (BK use uniform
grids everywhere — the whole point is that the *representation*, not the mesh,
is adaptive; BK §8 mentions general domains via Kazeev–Schwab but that is a
different paper), and `alpha != e_k` cores (`That_0`, `Yhat_0`, `Nhat_0` of
BK (78)) which are only needed for first-order and reaction terms. They are
written down in §1.6 and can be added when a problem needs them.

### 3.3 Multi-dimensional Laplacians, mixed BCs, variable coefficients

* **Mixed BCs.** `bc='DN'` per end per dimension is the only combination that
  gives `2^l` dofs per level. `'DD'` gives `2^l - 1` and does *not* fold; the
  existing `qlaplace_dd` works only because it silently uses `2^d` nodes with
  Dirichlet conditions *outside* the grid. `qlaplace_dn` must document that and
  `bpx` must **refuse** (`ValueError`) to be paired with a `'DD'` operator —
  the prolongations `P_{l,L}` are wrong for it.
* **Variable coefficients** enter only through `Lambda_{L,alpha,alpha'}`
  (BK (90a)–(90d)): if the coefficient has QTT ranks `r_l`, the operator gains
  a factor `r_l` in rank and *nothing else changes*. This is the cleanest part
  of the paper and the reason it covers oscillatory coefficients with
  `K = 2^40` (BK §7.3) without any special treatment.
* **Anisotropy** is a diagonal `A` in BK (5) and is `D` separate `Lambda`s; a
  full (non-diagonal) `A` adds the `alpha != alpha'` cross terms, whose cores
  are `That_0/Yhat_0/Nhat_0`. Rank cost `12 D^2 2^{4D} r` (BK (91)).
* **Non-uniform meshes**: not in this paper, not in this spec.

---

## 4. Public API, and who owns what

### 4.1 What `amen_solve` accepts today

Read from `tt/algs/amen.py`. `amen_solve(A, f, x0, eps, kickrank=4, nswp=20,
local_prec='c', local_iters=2, local_restart=40, trunc_norm=1,
max_full_size=200, verb=1, *, rmax, seed, check_true_res, return_info)`.

**There is no global preconditioner argument.** `local_prec` is a *block-Jacobi
preconditioner for the local `r_k n_k r_{k+1}` GMRES solve* (`_jacobi`,
lines 318–416), which is a completely different object: it accelerates the inner
Krylov iteration on one core and has no effect on the outer sweep. `tt.GMRES`
(`tt/algs/solvers.py`) takes the operator as a closure `A(x, eps)` and therefore
*can* absorb a preconditioner, but only a left one, and it has no `prec=`
argument either.

So today the only way to use a preconditioner with `amen_solve` is to pass the
already-preconditioned matrix and right-hand side — which is exactly what §2.2
does, and for BPX it is also exactly what BK do (their §7.2 assembles `B_L` and
hands it to AMEn). **This is not a defect for the two-sided case.**

### 4.2 The one interface, and which spec owns it

`docs/plans/eigenvalues.md` §3.3 and `docs/plans/riemannian-autodiff.md` §6.4
both ask this spec for a preconditioner object, and both state four
requirements. Answering them from this side:

1. **"`B^{-1}` must be exposable as a list of rank-1 TT-matrices, not only as a
   black-box `apply`; a callable is the slow path."** — **This is the wrong
   shape for BPX, and the "slow path" framing is wrong.** BPX is naturally a
   *single* `tt.matrix` of TT rank `2^{2D+1}` (8 / 32 / 128, measured, §1.5).
   Multiplying a rank-`R` TT-matrix by it costs a rank-`8R` intermediate and one
   rounding — the same cost as one extra matvec by `A`, whose rank is 3–4. It is
   not a sum of rank-1 terms and cannot be made into one. The interface must
   therefore accept **three** forms and say so:
   * `list[tt.matrix]` of rank-1 factors — the exponential-sum form, for
     Kronecker sums over physical modes ([RNO19] eq. (24)); `riemannian.project`
     already sums a list internally, so this stays the cheap path *there*;
   * a single `tt.matrix` of small TT rank — **the BPX form**, and the *only*
     form that works in the by-scale (QTT) setting; not slow;
   * a callable `prec(z) -> tt.vector` — the escape hatch (multigrid V-cycles,
     anything iterative), genuinely the slow path.

   Both consumer specs should relax "list of rank-1" to "an object with an
   `apply` and a declared `tt_rank`", with the rank-1 list as a *special case*
   the projector can exploit.
2. **SPD.** `C_L = sum_l 2^{-l} P_{l,L} P_{l,L}^T` is a sum of PSD terms
   including `l = L`, where `P_{L,L} = I`, so `C_L >= 2^{-L} I > 0`. SPD holds
   by construction, no check needed. Measured: `lambda_min(Chat A Chat) = 2.0000`
   at every `L` from 4 to 13 (§1.7).
3. **Spectral equivalence with `L`-independent constants.** Measured
   `kappa = 5.67 -> 11.82` for `L = 4..13` in `D = 1` and `4.51 -> 9.21` for
   `L = 3..6` in `D = 2`. Bounded in the measured range; saturation **not
   measured** (§1.7). The practical proxy is stronger: 6 AMEn sweeps flat from
   `L = 14` to `L = 30` (§2.2).
4. **Which BPX.** The eigensolver wants `C_{2,L}` (weight `2^{-2l}`), the linear
   solver wants `C_L` (weight `2^{-l}`); `bpx(d, D, weight=1|2)` is one function
   with one scalar changed (§1.4). Do not build two.
5. **What BPX does not give**, restating the eigensolver spec's own point and
   adding one: no preconditioner for the indefinite `A - sigma I` of an interior
   shift-and-invert (BPX is built from a *norm equivalence*, which needs
   coercivity); and no help with **coefficient contrast** — measured
   `kappa(BPX) = 1.12e+05` at contrast `1e4` (§2.4), i.e. the contrast passes
   straight through.

**Ownership (SSOT).** This spec owns: the QTT elliptic operators (`qlaplace_dn`,
`qdiff`, `qtri_ones`, `stiffness`, `load_vector`), the preconditioner
constructors (`bpx`, `prolongation`, `bpx_theta`, `bpx_operator`), and the
`qtt_ell.solve` front end. It does **not** own: the `prec=` argument of any
eigensolver or optimizer (owned by `eigenvalues.md` and
`riemannian-autodiff.md` respectively), `riemannian.project`'s list handling
(owned by `riemannian-autodiff.md`), or `amen_solve`'s `local_prec` (owned by
`amen.py`). The contract between them is one Python object — a `tt.matrix` — and
the three-form rule of item 1.

---

## 5. Validation tests, with named oracles

Each test names its oracle and its tolerance. `eps` = `2.22e-16`.

**V1 — the prolongation cores are exact.** For all `0 <= l <= L <= 7`, `D = 1`:
`bpx.prolongation(l, L).full()` against the closed form BK (14) built from
`numpy.eye`, `numpy.eye(k=-1)` and `eta = 2^-k (1..2^k)`. Tolerance `1e-12`
absolute. **Measured: passes** (`p0b_verify.py`).

**V2 — the level term is exact.** Same range: `P_{l,L} P_{l,L}^T` from BK (80)
against the dense product of V1's matrices. Tolerance `1e-11`. **Measured:
passes.**

**V3 — the fused differentiation core is exact.** Same range, `alpha = 1`:
BK (82) against `dense(M_{L,1}) @ dense(P) @ dense(P).T` with
`M_{L,1} = 2^{3L/2}(I - S)`. Relative tolerance `1e-11`. **Measured: passes.**

**V4 — the ranks are the theoretical ones.** `bpx(d, D, weight=w).r` must equal
`2^{2D+1}` for every `d in {8, 20, 50}`, `D in {1,2,3}`, `w in {1,2}`, **exactly**
(not "at most"): the construction is explicit, so a different rank means a bug,
not compression. `bpx_theta(d, 1).r == 6`. **Measured: 8/32/128 and 6**
(`p6.py`, `p2_bpx.py`).

**V5 — SPD and the condition-number bound (the whole claim, tested directly).**
For `d = 4..12`, `D = 1`: form `B = bpx_theta(d)^T bpx_theta(d)`, dense
`numpy.linalg.eigvalsh`. Assert `lambda_min > 1.5`, `lambda_max < 30`,
`kappa < 15`. Oracle: `numpy.linalg.eigvalsh` on the dense matrix. Measured
values `2.0113..2.0000` / `11.40..22.98` / `5.67..11.49`. The assertion that
makes this a test of the *preconditioner* and not of arithmetic: also assert
`kappa(A_DN) > 1e7` at `d = 12` (measured 2.72e+07), so the test fails if the
operator being preconditioned is accidentally benign.

**V6 — manufactured solution, nodal exactness.** `-u'' = 1`, `u(0)=0`,
`u'(1)=0`, `d = 20`. Oracle: `u(x) = x - x^2/2`, and the theorem that `P1`
Galerkin is nodally exact in 1D. Assert
`||v - v_exact|| / ||v_exact|| < 1e-9` for the BPX solve at `eps = 1e-10`
(measured 4.4e-12 at `d = 18`, 1.2e-11 at `d = 30`, at `eps = 1e-8`), **and**
assert that the same call with `precond='none'` gives `> 1e-6` (measured
3.29e-05 at `d = 18`). Without the second assertion the test passes for the
wrong reason.

**V7 — a solution with a singularity.** `-u'' = f` on `(0,1)` with `f` chosen so
that `u(x) = x^{3/4}` (`f = (3/16) x^{-5/4}`, integrable against `H^1` test
functions; use the exact load vector
`f_i = int phi_i f`, computable in closed form). Oracle: the nodal values of
`x^{3/4}`, whose QTT rank is **5 at `1e-6` and 7 at `1e-10`, flat in `L`**
(measured, §1.2). Assert (a) the solve at `d = 24` reaches relative nodal error
`< 1e-6`, (b) the iterate rank stays `< 40`. Note that the *Galerkin* solution
is not the interpolant here (`f` is not constant), so (a) must be stated against
`||u_h - u||_{L2} <= C h^{3/2}` rather than against the interpolant; the
oracle for the *discrete* problem is a dense solve at `d = 12`.
**Not measured** — this test is written, not run.

**V8 — the test that must fail loudly (representation).** `d = 40`, `D = 1`.
Build `B_comb = bpx_theta(40)^T bpx_theta(40)` and
`B_sep = (bpx(40) @ qlaplace_dn(40) @ bpx(40)).round(1e-14)`. These are the
*same matrix* in exact arithmetic. Apply both to a normalized rank-2 QTT
`sin(pi x / 2)` and assert
`||B_comb v - B_sep v|| / ||B_comb v|| > 1e+3`. **Measured: 1.472e+08.**
The test asserts that the naive route is *wrong*, so it fails loudly the day
someone "simplifies" `bpx_theta` into `C @ A @ C` — which is the single most
likely regression in this module, and the one that produces a plausible wrong
answer rather than an exception. Companion assertion at `d = 12`, where both
routes agree to `1e-10`, so the test cannot be satisfied by breaking `bpx`.

**V9 — the accuracy floor of the unpreconditioned solve is where theory says.**
`d in {14, 18, 22}`, `precond='none'`, `eps = 1e-10`, `nswp = 40`. Assert the
achieved relative nodal error is within a factor 10 of `4^d * eps` (measured
ratios 0.98, 1.66, 0.95). This pins the *baseline* so that a future improvement
to `amen_solve` cannot silently invalidate §2.1.

**V10 — the integration preconditioner is exact.** `A_a = qdiff(d)^T diag(a)
qdiff(d)`, `T = qtri_ones(d)`. Assert `||T^T A_a T - diag(a)||_F / ||diag(a)||_F
< 1e-12` for `d = 10` and `a` = ones, `a` = `2 + cos(64 pi x)`, `a` = a
two-layer field with contrast `1e4`. Oracle: the identity `(I - S) T = I`,
exact. **Measured**: `T.full()` equals `numpy.tril(ones)` at `d = 3, 5, 7` and
`||T^T A_DN T - I||_max` = 3.9e-15 / 4.4e-14 / 8.7e-14 there; and
`kappa(T^T A_a T)` equalled the contrast to five digits in all 12 cells of the
§2.4 table.

**V12 — the direct 1D solve is exact at `d = 40`, and the iterative form is
not (second loud-failure test).** `-u'' = 1`, `u(0)=0`, `u'(1)=0`, `d = 40`.
(a) Assert `qtt_ell.solve_direct_1d` gives relative nodal error `< 1e-13`
against `x - x^2/2` (measured 5.552e-15) **and** iterate rank `== 3` (measured
3). (b) Assert that `round(T^T A T, 1e-14)` has TT rank `> 3` even though the
exact product is the identity (measured 13), and that `amen_solve` on it at
`d = 30` returns `converged=True` with a nodal error `> 0.5` (measured
`converged=True`, `true_res=2.64e-09`, error `9.665e-01`). Part (b) is not a
test of correctness but a **frozen record of a trap**: it fails the day someone
exposes `T` through the generic `precond=` argument, which is the natural and
wrong way to package it.

**V11 — index convention.** `bpx(3, 1).full()` against the dense
`2^3 * sum_l 2^-l P_l P_l^T`, and `qlaplace_dn(3).full()` against the explicit
`8x8` tridiagonal. Tolerance `1e-14`. This is the regression test for the
BK-order → ttpy2-order reversal of §1.1, which is the one thing in this module
that is easy to get silently backwards (the DD Laplacian is *not* symmetric
under bit reversal, so the test bites).

---

## 6. Hard test problems

Marked (i) runnable today, (ii) after this lands, (iii) aspirational.

**H1 (i) — 1D Poisson, `L = 6..30`, DN.** `-u'' = 1`, oracle `x - x^2/2`,
exactly nodal. Reference numbers: §2.1 and §2.2 tables. This is the benchmark
that separates "works" from "returns noise" and it already runs
(`p1_cond.py`, `p3_head.py`).

**H2 (i) — algebraic corner singularity, `u = x^alpha`.** `alpha = 3/4, 1/2,
1/4`, `L = 10..20`. Reference: QTT rank 5 at `1e-6`, 7 at `1e-10`, flat in `L`
(measured, §1.2). Runnable today as an *approximation* benchmark; as a *solve*
benchmark it is (ii) because the load vector needs `qtt_ell.load_vector`.

**H3 (ii) — oscillatory diffusion, BK §7.3.** `-(a_K u')' = 1`, `u(0)=0`,
`u'(1)=0`, `a_K = (2 + cos(K pi x))^{-1}`, `K = 2^10, 2^20, 2^30, 2^40`,
`L = 50`. Exact solution known in closed form (BK (101)):
`u = x(2-x) + (K pi)^{-1}[(1-x) sin(K pi x) + (K pi)^{-1}(1 - cos(K pi x))]`,
with QTT ranks `<= 7` for `u` and `<= 6` for `u'` (BK's statement, **not
measured here**). BK's own reference numbers (their Table 5, AMEn, `K = 2^30`):
`H^1` error `3.21e-05` at `L = 30` / tol `1e-4`, `2.89e-07` at `L = 30` /
tol `1e-6`, `3.73e-08` at tol `1e-8`; and `3.65e-01` at `L = 10, 20` for every
tolerance (the grid does not resolve `K = 2^30` below `L ≈ 30`). That
"`3.65e-01` then a cliff" pattern is a *precise* acceptance criterion.
Note `a_K` itself has no exact low-rank form; BK obtain it by solving
`c(x_i) a_K(x_i) = 1` pointwise as an auxiliary elliptic problem.

**H4 (ii) — high-contrast two-layer and checkerboard coefficients.**
`a = 1` on `x < 1/2`, `a = rho` on `x > 1/2`, `rho = 1e2, 1e4, 1e6`.
Measured references (§2.4): `kappa(A) = 8.9e6 / 8.9e8` at `L = 8`,
`kappa(BPX) = 888.94 / 88892.46`, `kappa(INT) = 100.0000 / 10000.0000`. The
acceptance criterion is that `kappa(INT)` equals `rho` to 4 digits and that
`kappa(BPX)/rho` stays in `[8, 12]`. This is the benchmark that shows what
neither preconditioner fixes.

**H5 (ii) — 2D Poisson on `(0,1)^2` with mixed BCs, BK §7.4.** `-Lap u = 1`,
`u = 0` on `Gamma`, `du/dn = 0` elsewhere, `L = 50` (i.e. `2^100` unknowns).
BK's reference: converges with AMEn and with their STSolve; `Theta_{L,1}`,
`Theta_{L,2}` have max representation rank **24**, `B_L` has rank **1152**;
running times "several minutes" for AMEn, "several hours" for STSolve (their
§7.4, and their Fig. 6, **not rendered here**). Measured here: `rank(Chat) = 32`
at `L = 50`, build 0.01 s; `kappa(B) = 9.21` at `L = 6` (dense). This is the
problem that justifies BPX; nothing else in this document reaches it.

**H6 (iii) — 2D L-shaped domain / re-entrant corner.** The `r^{2/3}`
singularity is the canonical BPX/QTT showcase and the one Kazeev–Schwab's
Theorem 7 is stated for. Aspirational because it needs a non-product domain,
which [BK20] explicitly does not treat ("more general domains by techniques
developed in [28]") and which this spec does not either.

**H7 (iii) — `D = 3`.** `rank(bpx) = 128` measured at `L = 50` (build 0.76 s),
so the preconditioner itself is affordable; `B_L` would have rank
`12 * 9 * 2^12 ≈ 4.4e5` by BK (91) and is not. Only the `Theta` route can
work, and BK give no `D = 3` numbers. Aspirational and honestly uncertain.

**H8 (iii) — high-dimensional (`d >> 3`) elliptic problems.** Out of scope for
[BK20], which separates *scales*, not *dimensions*. A `d = 100` Laplacian is a
Kronecker sum and belongs to the exponential-sum preconditioner of §2.3, i.e. to
`docs/plans/riemannian-autodiff.md`. Listing it here only to say it is not ours.

---

## 7. Prototypes: what was run, and how to rerun it

All on b300 (`ssh b300`), `~/work/ttpy-modern/scratch-bpx/`, interpreter
`~/work/ttpy-modern/ttpy2/.venv/bin/python`, numpy backend, float64, CPU.

| file | what it does | wall |
|---|---|---|
| `bpx.py` | the construction library: BK cores, strong Kronecker product, core product, BK↔ttpy2 order conversion, `bpx_C`, `bpx_Theta`, `lap_dn` | — |
| `p0_conv.py` | index-convention check (§1.1) | < 1 s |
| `p0b_verify.py` | Lemmas 3/4/5 and (80)/(82) against dense, `L <= 7` (V1–V3) | 6 s |
| `p1_cond.py` | closed-form `kappa` validated against dense; unpreconditioned AMEn baseline `d = 6..24` (§1.3, §2.1) | 45 s |
| `p2_bpx.py` | assembly ranks/times `L = 8..50`; dense `kappa(B)` `L = 4..13`; BK Table 3 (§1.6, §1.7, §1.3) | 6 min |
| `p2b.py` / `p6.py` | BK Table 3 with the *unrounded* `C.A.C`; `round(C@A@C)` action error `L = 10..50`; explicit Thm-3 cores `D = 1,2,3` (§1.3, §1.5, §1.6) | 5 min + timeout |
| `p3_head.py` | interleaved head-to-head `L = 10..30` (§2.2) | 70 s |
| `p4_alt.py` | QTT ranks of `A^{-1}` and of `expm(-tA)`; explicit-inverse solve (§2.3, §2.4) | 40 s |
| `p5_alt2.py` | variable-coefficient `kappa` comparison; `D = 2` BPX (§2.4, §1.7) | 4 min |
| `p7_sing.py` | QTT ranks of singular / oscillatory / layer functions (§1.2) | 90 s |
| `p8_c2.py` | `C_{2,L}` vs `C_L` (§1.4) | 60 s |
| `p9_bigk.py` | dense `kappa(B)` at `L = 14, 15, 16` (§1.7, Q1) | 60 s / 395 s / ~50 min |
| `p10_int.py` | the `T` route: direct solve `L = 10..40` vs assembled-and-iterated `L = 14..40` (§2.4, Q2) | 5 min |

`p6.py` §(iii) (Lanczos `kappa(B)` beyond the dense range) was killed after
`L = 10`: 20 s per level and rising as `4^L`. It reproduced the dense
`kappa = 10.6617` exactly, which is the only thing it was needed for.
`p9_bigk.py`'s `L = 16` cell (34 GB dense `eigvalsh`) had not returned when
this document was written; `L = 14` and `L = 15` are in §1.7.

---

## 8. Open questions, each with the experiment that settles it

**Q1 — does `kappa(B)` actually saturate? Partly answered.** Extended to
`L = 14, 15` by dense `eigvalsh` on b300 (`p9_bigk.py`, 60 s and 395 s):
`kappa = 12.1058` and `12.3539`, `lambda_min = 2.000000` in both. The increments
are `+0.3296, +0.2852, +0.2481` at `L = 13, 14, 15`, ratios `0.865, 0.870` —
geometric to two digits, which extrapolates to `kappa -> 14.0`. **That is still
an extrapolation, from a geometric fit to three increments.** What would settle
it: `L = 16` dense (34 GB, ~50 min — was launched and is the natural next
point), or a TT-Lanczos at `L = 20, 24` with a rank cap and full
reorthogonalization of a 30-vector basis. If `kappa` were instead growing like
`c log L`, the increments would decay like `1/L` (ratios `0.93, 0.94` at these
`L`), which the data excludes at two digits — so the geometric reading is the
better-supported one, but not proved.

**Q2 — is the `T` (integration) preconditioner representation-stable at large
`L`? ANSWERED, and the answer is "only as a direct solve".** Measured
(`p10_int.py`, §2.4): the direct form `v = T diag(1/a) T^T f` is exact to
5.55e-15 at `L = 40`; the iterative form (assemble `T^T A T`, run AMEn) is wrong
by 4.79e-03 at `L = 22`, 9.67e-01 at `L = 30`, and fails outright at `L = 40`.
The remaining part of the question: does the *variable-coefficient* direct form
stay exact? `diag(1/a)` needs `a` inverted elementwise in QTT, which is a
nonlinear operation (`multifuncrs` or a cross), and its rank and accuracy were
**not measured**. Experiment: `a_K = (2 + cos(K pi x))^{-1}` at `K = 2^10`,
`L = 30`; compare `T diag(1/a) T^T f` against a dense solve at `L = 12` and
against BK (101) at `L = 30`. Two hours.

**Q3 — does the fused `Theta` construction survive a variable coefficient?**
Everything measured here is `a = 1`. BK (90a)–(90d) say the coefficient enters
only through `Lambda`, but `Lambda^{1/2}` is taken core-wise (BK §5.5), which is
exact only because `Lambda` is diagonal *and* rank 1. For a rank-`r`
coefficient, `Lambda^{1/2}` is not obtainable core-wise. Experiment: build
`Theta` for `a_K = (2 + cos(K pi x))^{-1}` at `K = 2^10`, `L = 30`, compare
`Theta^T Theta` against `Chat A_a Chat` at `L = 12` (dense), and check whether
the representation conditioning of the `L = 30` object is still `1e-14`. If
`Lambda^{1/2}` has to be computed by a nonlinear TT algorithm (`multifuncrs`),
the cost model of §2.5 step 3 changes.

**Q4 — what is the right stopping criterion in the preconditioned variable?**
BK Remark 2 says `||v||_{H1} ~ ||C_L^{-1} v||_2`, so the algebraic residual of
`B u = g` *is* an `H^1` error estimator with `L`-independent constants — that is
the whole reason to prefer `C_L` over `C_{2,L}` for solves. But `amen_solve`'s
`true_res` is `||Bu - g||/||g||`, a *relative* quantity, and the relation to the
absolute `H^1` error carries the factor `||g||`. Experiment: on H1, tabulate
`true_res`, `||v - v_exact||_{l2}`, and the discrete `H^1` error
`||Theta u - Theta u_exact||` for `L = 10..30` and fit the constants. Until this
is done `qtt_ell.solve` must return **both** residuals and let the caller
choose, which is what the §3.2 signature does.

**Q5 — does the `2^{-l}` weight need tuning?** BK take `2^{-l}` and `2^{-2l}`
because the theory dictates them, but the measured `lambda_max(B)` growth is
entirely driven by the coarsest levels. Experiment: replace `2^{-l}` by
`omega^l` in `bpx(weight=...)` and scan `omega in [0.35, 0.7]` measuring
`kappa(B)` at `L = 12` densely. A one-line change to `C_l` in Theorem 3. If a
tuned `omega` buys a factor 2 in `kappa`, it buys ~1 AMEn sweep and is probably
not worth losing the theorem for; if it buys 5x in `D = 2` it is.

**Q6 — does BPX survive contact with a Riemannian iterate?**
`docs/plans/riemannian-autodiff.md` §13 Q6 asks this and it is still open. What
this document adds: the *matrix* is fine (§1.7) and the *combined
representation* is fine (§1.3, 1e-14 at `L = 20`), so the failure mode that
spec worries about would have to come from the iterate, not from the
preconditioner. Experiment, refined from theirs: run truncated Rayleigh-quotient
descent on `B = Theta^T Theta` for `L = 10..20` at manifold rank 4, tracking
(a) the eigenvalue error against `lambda_min(A_DN) = 4 sin^2(pi/(2(2N+1)))`
mapped through the change of variables, and (b) the smallest singular value of
the unfoldings of the iterate. Note the change of variables is the subtlety:
minimizing the Rayleigh quotient of `B` gives the eigenvector of `B`, **not** of
`A`; the eigenproblem needs the *left* preconditioner `C_{2,L}` (§1.4), not the
two-sided `C_L`. That correction is this spec's answer to their §6.4, and it
should be measured before either spec builds on it.

**Q7 — is `bpx_operator` ever the right call in `D = 2`?** BK assemble `B_L`
(rank 1152) for AMEn because "in the available version of AMEn, the
decomposition of `B_L` needs to be used directly" (their §7.4) and report that
it works, with memory as the limit. ttpy2's `amen_solve` accepts a *list of
matrices meaning their sum* (its docstring, `_matrix_cores`), so we may be able
to hand it `[Theta_1^T Theta_1, Theta_2^T Theta_2]` and never form the rank-1152
object. Experiment: read `tt/algs/amen_mv.py::_matrix_cores`, then run H5 at
`L = 12` both ways and compare peak memory and sweep count. If the list form
works this is the difference between `D = 2` being feasible and not.

---

## 9. What I did not verify

* **The proofs.** BK Theorem 2 (Appendix A, 12 pages) and Lemma 5 (Appendix B)
  were read for their statements and their core definitions, not verified line
  by line. What I did instead is check the *conclusions* numerically: Lemma 4,
  (80) and (82) reproduce dense products to 1e-11 for all `0 <= l <= L <= 7`,
  and Theorem 3's rank is exactly `2^{2D+1}` for `D = 1,2,3` up to `L = 50`.
  Theorem 2 itself (the spectral equivalence) is verified only as
  `kappa(B) <= 11.83` for `L <= 13`, `D = 1` and `<= 9.21` for `L <= 6`,
  `D = 2` — the *uniformity* is an extrapolation (Q1).
* **The figures.** [BK20]'s Figures 1–6 were not rendered. Everything this
  document says about `beta_l(A) ~ 2^{2L}` growth (their Fig. 1b), about the
  singular-value decay of preconditioned solutions (their Fig. 2), and about the
  `L = 50` convergence histories (their Figs. 3–6) is from their **text**, which
  is unambiguous, not from the plots.
* **`alpha = 0` cores.** `That_0`, `Yhat_0`, `Nhat_0` (BK (78)) are transcribed
  in §1.6 but **never verified numerically** — only `alpha = 1` was. Anything
  with a reaction term, a first-order term, or a non-diagonal diffusion matrix
  rests on untested transcription.
* **`D >= 2` beyond ranks and `L <= 6`.** The `D = 2` `kappa` table stops at
  `L = 6` (`N = 4096`) and there is no `D = 2` *solve* anywhere in this
  document. BK's own `D = 2` results at `L = 50` are quoted, not reproduced.
  `D = 3` is ranks only.
* **`bpx_theta` with a variable coefficient** (Q3) — every measurement of the
  fused construction in this document is for `a ≡ 1`. `Lambda^{1/2}` is taken
  core-wise, which is exact only for a rank-1 diagonal `Lambda`; for a rank-`r`
  coefficient it is not, and I do not know what replaces it.
* **The `T` route with a variable coefficient.** §2.4's `L = 40` exactness is
  for `a ≡ 1`, where `diag(1/a) = I`. With a genuine `a`, `1/a` has to be formed
  in QTT by a nonlinear algorithm and its rank and accuracy were **not
  measured** (Q2, remaining half).
* **[COR16] and [KK12] were not read.** §2.4's claim that the `D = 1`
  integration trick has no `D = 2` analogue rests on [BK20]'s one-paragraph
  characterization of [COR16] plus the elementary observation that
  `(I-S)T = I` has no Kronecker-sum analogue — not on reading [COR16].
* **Nothing was run on a GPU**, and no timing here should be read as a
  performance claim about ttpy2's torch backend.
* **Single runs.** Every wall-clock number in §2.1 and §2.2 is one run on a
  shared 256-core machine with other jobs present (several GPUs were at 100 %
  utilization during the measurements, which does not affect CPU numpy work but
  the CPUs were not reserved either). The *iteration counts* and the *errors*
  are deterministic (`seed=0`); the *times* carry perhaps 30 % noise.
