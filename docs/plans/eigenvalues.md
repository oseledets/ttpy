# EIG: eigenvalue problems in the TT format, for `tt.algs.eig*`

Implementation spec. Sources actually read, and what each one is:

* **[RNO19]** M. Rakhuba, A. Novikov, I. Oseledets, *Low-rank Riemannian
  eigensolver for high-dimensional Hamiltonians*, J. Comput. Phys. **396** (2019)
  718–737 (received 19 Feb 2019, accepted 1 Jul 2019). This is the uploaded PDF,
  read in full (20 pp., text extracted with `pypdf`; the figures were not
  rendered, so every claim below that rests on a *figure* is marked as such).
  It **is** an eigenvalue paper, not a linear-solver paper. Contents: block
  computation of the `b` lowest eigenpairs of a TT-matrix `H` by LOBPCG in which
  every iterate, residual and search direction is projected onto **one** tangent
  space of the fixed-rank TT manifold (the "low-rank Riemannian alternating
  projection", LRRAP); the tangent-space schedule `t_k` (§3.2, eq. (17)); a
  nonstandard `3b+1` coefficient problem with a `diag(c)` constraint that does
  *not* reduce to a generalized eigenproblem, and an iterative solver for it
  (§5, Algorithms 1–2, `O(b^4)`); complexity table (Table 1); and numerical
  results on acetonitrile CH₃CN (`d=12`), ethylene oxide C₂H₄O (`d=15`) and
  Heisenberg spin chains (`d=40`), including a **direct comparison against our
  `eigb`** (Table 4) and a proof that the block-TT format used by `eigb` has
  ranks growing linearly in `b` (Appendix B, Table 5).
* **[DKOS14]** S. V. Dolgov, B. N. Khoromskij, I. V. Oseledets, D. V.
  Savostyanov, *Computation of extreme eigenvalues in higher dimensions using
  block tensor train format*, Comput. Phys. Commun. **185**(4):1207–1216, 2014,
  arXiv:1306.2269. **Not read** — it is not among the uploaded files. It is the
  reference of our own `tt/algs/eigb.py`, and everything this document says
  about the *block AMEn eigensolver* (§2.1) is reconstructed from
  `tt/algs/eigb.py` + `tt/algs/amen.py` and marked **[derived]**. Fidelity to
  the paper must be checked before the module claims to implement it (§11, Q1).
* **[AMEn]** S. V. Dolgov, D. V. Savostyanov, *Alternating minimal energy
  methods for linear systems in higher dimensions*, SIAM J. Sci. Comput.
  36(5):A2248–A2271, 2014 — read only through the docstring and code of
  `tt/algs/amen.py`, which is a faithful, tested implementation.

Everything marked **[derived]** is our reconstruction. Everything marked
**measured** was produced by a throwaway prototype run on this host before this
document was written: `.venv-tmp` (python 3.12.13, numpy 2.5.1, scipy 1.18.0),
numpy backend, float64, single machine with 2 cores and 4 GB of RAM, no GPU, no
`numba` (`tt.algs._fast.HAVE_NUMBA` is False here, so every AMEn local solve
took the interpreted path). Prototypes are in
`/home/ivan/.claude/jobs/b5968f66/tmp/`. Nothing below is an estimate; where a
number is missing it says "not measured".

**Cross-spec decisions live in `docs/plans/ROADMAP.md`**, not here. Where this
spec and one of `bug-integrator.md`, `riemannian-autodiff.md`,
`qtt-elliptic-bpx.md` ask for the same function, the reconciled signature and its
owner are recorded there (§2), together with the dependency graph (§1), the
preconditioner contract (§3), the milestone order (§4) and the consolidated open
questions (§6). Two things below were amended by it: §4(b) (the `project_delta`
signature is owned by `riemannian-autodiff.md` §8(a)) and §3.3 item 1 (the
preconditioner is not always a sum of rank-1 terms). §1.2a's defect is
**already fixed**, commit `191bbc0`.

---

## 1. What `eigb` is, and where it breaks

### 1.1 What it is

`tt.algs.eigb.eigb(A, y0, eps, ...)` is one-site block ALS (DMRG-1) in the
**block TT format**: `B = y0.r[-1]` eigenvectors share one core list, and the
block index travels with the current site as a 4-index array
`blk : (r_k, n_k, r_{k+1}, B)`. At site `k` the projected local matrix

    B_k = (Y_{<k} (x) I_{n_k} (x) Y_{>k})^H A (Y_{<k} (x) I_{n_k} (x) Y_{>k})

of size `r_k n_k r_{k+1}` is formed densely (`bk.eigh`) below `max_full_size`
and handed to `scipy.sparse.linalg.lobpcg` above it; its `B` lowest eigenpairs
are the local update. The block is then split by an SVD, and — this is the only
mechanism by which the ranks can change — the block index is attached to the
*other* side of the split:

```
direction < 0 :  svd( blk : "a i b B -> (B a) (i b)" )   ->  r_k    <= min(B r_k,     n_k r_{k+1})
direction > 0 :  svd( blk : "a i b B -> (a i) (b B)" )   ->  r_{k+1} <= min(r_k n_k,   r_{k+1} B)
```

Sweeps run `d -> 1` and `1 -> d`; the stopping indicator `ermax` is the largest
relative drop of `sum(lambda)` over a full sweep. The returned block always
carries its measured eigenresidual (`history.res`, `history.res_rel`), and a
large one raises a `RuntimeWarning`. That residual check is the module's one
defence against a stalled sweep, and §1.2 shows it earning its keep.

### 1.2 Where it breaks: `B = 1` has no rank adaptation at all

Read the two bounds above with `B = 1`. Right-to-left, `r_k <= min(r_k, ...)`;
left-to-right, `r_{k+1} <= min(..., r_{k+1})`. **With `B = 1` the TT ranks of the
iterate are non-increasing along every sweep.** The block index *is* the
enrichment; remove it and `eigb` is plain one-site ALS, whose defining defect is
that it cannot grow a rank. So for a single eigenvector the accuracy of `eigb`
is decided entirely by the rank the *caller guessed* when building `y0`.

Measured. Problem: open Heisenberg chain of `d = 10` spin-1/2 sites,
`H = sum_i [ (S+_i S-_{i+1} + S-_i S+_{i+1})/2 + S^z_i S^z_{i+1} ]`, a TT-matrix
of rank 5 (prototype `ham.heisenberg(10)`, validated element-by-element against
an explicit Kronecker construction, max abs difference **0.0**). Oracle:
`numpy.linalg.eigvalsh` of the dense `1024 x 1024` matrix,
`E_0 = -4.258035207283`. Initial guess `tt.rand(2, 10, r=[1]+[r0]*9+[1])`,
`numpy.random.default_rng(0)`, `eps = 1e-8`, `nswp = 60`, defaults otherwise:

| `r0` | max rank of the returned vector | `lambda` | abs. error | `ermax` | `converged` | rel. eigenresidual | time |
|---|---|---|---|---|---|---|---|
| 4  | **4**  | -4.251935850857 | **6.10e-03** | 8.1e-09 | True | 2.84e-02 | 0.17 s |
| 8  | **8**  | -4.258020596623 | 1.46e-05 | 2.2e-09 | True | 1.63e-03 | 0.38 s |
| 16 | **16** | -4.258035204636 | 2.65e-09 | 1.6e-10 | True | 2.49e-05 | 3.49 s |
| 32 | **32** | -4.258035207283 | 3.55e-15 | 1.3e-15 | True | 2.66e-11 | 0.77 s |

The maximal rank of the answer equals the maximal rank of the guess in every
row (the full rank profile at `r0 = 8` is `[1,2,4,8,8,8,8,8,4,2,1]`, capped only
by the boundaries) — the method never added a single rank unit. At `r0 = 4` it returns `converged = True` with
`ermax = 8.1e-09 < eps = 1e-08` and an eigenvalue wrong in the **third**
significant digit. The residual check does fire (2.84e-02 relative, warning
raised), so ttpy2 does not lie about it — but the *iteration* has no way to fix
itself, and the user is told to "start from a random guess of larger rank" with
no indication of how large.

This is the single strongest argument for the block AMEn eigensolver: AMEn's
residual enrichment is exactly the missing mechanism, and `amen_solve` already
implements it for linear systems.

### 1.2a A defect in the current `eigb`: `res_warn` does not scale with `eps`

`eigb`'s residual check — the one thing standing between a stalled sweep and a
plausible wrong answer — fires at a **fixed** threshold, `res_warn=1e-2`, which
has nothing to do with the `eps` the caller asked for. Between the two there is
a six-decade window in which `eigb` returns a wrong eigenvalue, reports
`converged=True`, and says nothing at all.

Reproduced exactly (prototype file `ham.py` in the job directory; `heisenberg`
is the rank-5 MPO of §1.2):

```python
import warnings, numpy as np, tt, ham
from tt.algs.eigb import eigb
H  = ham.heisenberg(10)
y0 = tt.rand(2, 10, r=[1]+[8]*9+[1],
             samplefunc=np.random.default_rng(0).standard_normal)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    y, lam, h = eigb(H, y0, 1e-8, nswp=60, verb=0, return_history=True)
```

Output on this host:

```
lam       = -4.258020596623202          exact = -4.258035207282887
abs err   = 1.461e-05                   rel err = 3.431e-06     (eps requested: 1e-8)
converged = True   ermax = 2.240e-09    sweeps = 3
res       = [0.00692561]                res_rel = [0.00162648]
ranks     = [1, 2, 4, 8, 8, 8, 8, 8, 4, 2, 1]
WARNINGS  = NONE
```

`res_rel = 1.6e-03` is a real, correctly measured relative eigenresidual that
sits just under the hard-wired 1e-2, so no warning is raised, and the only
number the caller normally looks at (`converged`) says True. This is precisely
the "hidden unknown" the project forbids: the run *knows* its residual is
1.6e-03 against a requested 1e-08 and does not say so.

**Fixed, and it took two changes rather than one.** Recorded here because the
first one alone was wrong in an instructive way.

*The threshold.* `res_warn=None` now means `max(sqrt(eps), 8 * eps_machine)` —
the square root because for a symmetric problem the eigenvalue error is
`O(res^2)` while the eigenvector error is `O(res / gap)`, so `sqrt(eps)` is the
weakest threshold under which the *eigenvalue* can plausibly have the requested
accuracy. With `eps = 1e-8` that is `1e-4`, and the run above now warns. The
fixed 1e-2 stays reachable by passing it explicitly.

*The denominator.* Making only that change turned three previously silent tests
into warnings whose own accuracy assertions pass — that is, false alarms, which
would have trained the user to ignore the warning and undone the fix. Cause: the
threshold was being applied to `res_i / ||A y_i||`, which asks each eigenvalue to
be accurate *relatively*, and near the bottom of the spectrum that is unachievable
at any truncation accuracy. Measured on `qlaplace_dd([10])`, `B = 4`, numpy
float64, single run:

| quantity | value |
|---|---|
| `lam_1` | 9.394e-06 |
| absolute residual | 3.584e-09 |
| eigenvalues vs dense `eigh` | correct to 1e-9 absolute |
| `res / ‖A y‖` | **2.973e-04** — would warn at `eps=1e-8` |
| `res / ‖A‖_2` | 8.959e-10 — silent, correctly |

The warning now fires on the **backward error** `res_i / ||A||_2`: the returned
pair is exact for `A + E` with `||E|| = res`, so this is the perturbation of the
operator the answer corresponds to, and for symmetric `A` it bounds
`|lam - lam_exact|` directly. On the Heisenberg case above it is unchanged
(1.626e-03, because `|lam| ≈ ||A||` there), so the real defect still warns.

`||A||_F` was measured as a candidate denominator and rejected: 78.4 against
`||A||_2 = 4.0` for `qlaplace_dd([10])`, 41.6 against 4.26 for the 10-site
Heisenberg chain — a denominator 10–20× too large desensitizes the warning by
the same factor. The estimate now used is a power iteration in TT
(`tt.algs.eigb.spectral_norm_estimate`, 12 iterations, rounding 1e-3), which is
a *lower* bound and therefore can only make the warning more eager: measured at
0.94–0.97 of the dense `||A||_2` on both operators (both have a clustered top of
the spectrum, the slow case for power iteration), 25–40 ms, and identical to
1e-14 between the numpy and torch backends.

Regression tests: `test_eigb_warns_when_a_too_small_guess_rank_stalls_it`,
`test_eigb_does_not_cry_wolf_near_the_bottom_of_the_spectrum`,
`test_spectral_norm_estimate_is_a_close_lower_bound`, all in
`tests/test_eigb_ksl.py`. The Hamiltonians of §1.2 moved into
`tests/hamiltonians.py` as reusable fixtures, with the exact critical-TFIM
ground-state energy as an oracle that does not go through LAPACK.

### 1.3 Where it does **not** break

Honest counterweight, measured on the same host.

**Clustered and exactly degenerate spectra are fine.** The `d = 10` Heisenberg
spectrum has exact SU(2) degeneracies (dense gaps `3.3e-01, 3.6e-15, 1.3e-15,
4.0e-01, 1.2e-14, 4.4e-16, ...`). `eps = 1e-8`, `r0 = max(4, B+1)`, seed 0:

| `B` | max rank | sweeps | time | max abs. error vs `eigvalsh` | max rel. eigenresidual |
|---|---|---|---|---|---|
| 1 | 4  | 7 | 0.15 s | **6.10e-03** (see §1.2) | 2.84e-02 |
| 2 | 32 | 3 | 3.61 s | 1.55e-14 | 2.78e-11 |
| 4 | 64 | 2 | 7.67 s | 1.78e-14 | 4.42e-11 |
| 6 | 64 | 2 | 1.98 s | 1.20e-14 | 5.64e-11 |
| 8 | 64 | 2 | 5.26 s | 1.69e-14 | 5.43e-11 |

`B = 4` resolves the triply degenerate `-3.93067359` to 1.8e-14 — the local
dense `eigh` handles the degeneracy exactly, and the block-TT format has no
trouble representing a degenerate invariant subspace. Same story on the
transverse-field Ising chain in the ordered phase, where the two lowest states
are *nearly* degenerate: `d = 10`, `g = 0.5`, dense gap **1.465e-03**, `B = 2`
gives both to 2.7e-14 in 0.58 s.

**A small spectral gap does not hurt the ALS iteration.** At `g = 0.5` the gap
is 1.5e-3 and `eigb` still converges in 2 sweeps. What hurts is `B = 1` inside a
cluster, and the cause is again the rank, not the gap: `B = 1` on the same
problem returns `-9.7655030293` against `-9.7655039579` (9.3e-07), with the rank
pinned at `r0 = 4`.

**On smooth QTT problems `eigb` is excellent.** 1D QTT Laplacian
`tt.qlaplace_dd([8])` (`N = 256`, dense oracle `numpy.linalg.eigvalsh`),
`eps = 1e-8`, `r0 = max(4, B+1)`, seed 0:

| `B` | ranks of the answer | max rel. error | max abs. error | sweeps | time |
|---|---|---|---|---|---|
| 1  | `[1,2,2,2,2,2,2,2,1]`     | 3.31e-12 | 4.94e-16 | 4 | 0.04 s |
| 2  | `[1,2,4,4,4,4,4,4,2]`     | 1.96e-12 | 5.87e-16 | 3 | 0.20 s |
| 4  | `[1,2,4,5,6,7,8,8,4]`     | 3.75e-12 | 9.31e-16 | 2 | 0.82 s |
| 8  | `[1,2,4,6,7,9,11,14,8]`   | 5.86e-12 | 9.76e-16 | 2 | 0.29 s |
| 16 | `[1,2,4,7,9,11,15,20,16]` | 3.45e-12 | 1.65e-15 | 2 | 0.54 s |

Here `B = 1` works because the exact eigenvector `sin(k pi j/(N+1))` has QTT rank
2 and the guess already had rank 2. Note the max rank: `2, 4, 8, 14, 20` for
`B = 1, 2, 4, 8, 16` — **linear in `B`**, which is [RNO19] Appendix B measured on
our code (their Table 5 reports `max r = 5, 12, 25, 63, 120` for
`b = 1, 2, 4, 8, 16` on acetonitrile at `delta = 1e-2`, i.e. the same law with a
larger constant). The block-TT format pays for `B` eigenvectors with a rank
`~ B r`, hence a local dense eigensolve of size `(B r)^2 n^2` and cost
`(B r n)^3`. That is the scaling `eigb` cannot escape and the reason [RNO19]
exists.

### 1.4 Rank adaptation with `B >= 2`: real, but bounded by `B`

`eigb` *does* adapt ranks for `B >= 2` — that is what the `(B a)` and `(b B)`
groupings buy. The bound is `r_new <= B r_old` per site per half-sweep, so a
rank that must grow by more than a factor `B` needs more sweeps, and a rank that
must grow from a rank-1 guess with `B = 1` never grows at all.

---

## 2. The candidate methods

Conventions throughout: TT cores `G_k : (r_{k-1}, n_k, r_k)`; TT-matrix cores
`A_k : (R_{k-1}, n_k, m_k, R_k)` with the **row index first**; interfaces
`(bra, alpha, ket)` as built by `tt.algs._localops.phi_left` / `phi_right`;
merged vector mode `s = i + n*j`; mode 1 is the fastest index. Code is 0-based,
the mathematics below is 1-based.

### 2.1 Block AMEn eigensolver — **[derived]**, the recommended first build

The idea in one line: keep `eigb`'s sweep exactly as it is and add AMEn's
residual enrichment, so that the rank can grow even when `B = 1`.

State, on top of what `eigb` already keeps:

* `X` — the block core list, with the block index riding on the current site
  (`eigb` today);
* `phiax_l[k] : (r_{k-1}, R_{k-1}, r_{k-1})`, `phiax_r[k]` — the `X^H A X`
  interfaces (`eigb` today, built by `lo.phi_left` / `lo.phi_right`);
* `Z` — a **new** block TT of rank `kickrank` approximating the block residual
  `R = A X - X diag(lambda)`, plus the four mixed interfaces
  `phizax_l/r` (`Z^H A X`) and `phizx_l/r` (`Z^H X`), exactly the objects
  `amen.py` keeps under the names `phizax_*` and `phizf_*` with `f := X diag(lambda)`.

One left-to-right step at site `k`:

```
BLOCK_AMEN_STEP(k):
  # 1. local eigenproblem -- unchanged from eigb
  lam, v = local_eig(phiax_l[k], A_k, phiax_r[k+1], B)      # v : (r_{k-1}, n_k, r_k, B)

  # 2. truncate -- unchanged from eigb
  u, s, vh = svd( rearrange(v, "a i b B -> (a i) (b B)") )
  rnew     = chop(s, eps/sqrt(d) * ||s||)
  Gk       = u[:, :rnew].reshape(r_{k-1}, n_k, rnew)
  xold     = (u[:, :rnew] @ (s[:rnew] * vh[:rnew, :])).reshape(r_{k-1}, n_k, r_k, B)

  # 3. NEW: the local residual seen through (X frames on the left, Z frames on the right)
  enr[c,i,z,B] =  sum  phiax_l[k][c,b,p] A_k[p,i,j,q] phizax_r[k+1][z,f,q] xold[b,j,f,B]
                - lam[B] * sum phix_l[k][c,g] xold[g,i,h,B] phizx_r[k+1][z,h]

  # 4. NEW: compress the enrichment ACROSS the block index, else the added rank is rz*B
  ue, se, _ = svd( rearrange(enr, "c i z B -> (c i) (z B)") )
  enr_c     = ue[:, :kickrank].reshape(r_{k-1}, n_k, kickrank)

  # 5. NEW: enrich, pad, re-orthogonalize -- byte for byte amen.py's enrichment
  Gk        = concatenate((Gk, enr_c), axis=2)
  next_blk  = concatenate((next_blk, zeros), axis=0)
  Gk, R     = qr_left(Gk);  next_blk = R @ next_blk

  # 6. update interfaces, and Z's own core from the same residual seen through Z on both sides
  phiax_l[k+1]  = phi_left(phiax_l[k],  A_k, Gk, Gk)
  phizax_l[k+1] = phi_left_mixed(phizax_l[k], A_k, Gk, Zk)
  ...
```

Three things that are *not* transcriptions of `amen.py` and are where the risk
sits:

1. **The right-hand side moves.** For a linear system `f` is fixed; here the
   "right-hand side" is `X diag(lambda)` with `lambda` the current Ritz values,
   so `Z` must be rebuilt every sweep, and the enrichment must use the `lambda`
   of the local solve that just happened, not the previous sweep's.
2. **Step 4 is mandatory.** The residual of a `B`-column block is a `B`-column
   object; concatenating it uncompressed adds `kickrank * B` to the rank at
   every bond, which would make block AMEn *worse* than `eigb` for large `B` —
   the exact defect [RNO19] Appendix B accuses the block format of. Compressing
   across the block index first keeps the added rank at `kickrank`.
3. **`B = 1` is the cheap special case.** With `B = 1` the algorithm above is
   `amen_solve` with the right-hand side `lambda x` and one extra dense `eigh`
   per block — perhaps 60 lines on top of `tt/algs/amen.py`, reusing
   `_project`, `_apply`, `_phi_next`, `_phi_yy_next`, `_truncate` unchanged.
   It fixes precisely the failure of §1.2. **Build this first.**

Cost per sweep, relative to `eigb` with the same ranks (notation of §6): the
enrichment adds one `phi`-style contraction of the residual at rank
`kickrank`, one `(r n) x (kickrank B)` SVD and one `(r n) x (r + kickrank)` QR
per site, i.e. `O(d n r^2 (R kickrank + kickrank B))` against `eigb`'s
`O(d (n r)^3)` local eigensolve — a lower-order term as long as
`kickrank << r n r'`. Not measured; there is no prototype (§10).

### 2.2 LOBPCG in TT, and the LRRAP variant of [RNO19]

Plain TT-LOBPCG ("MP LOBPCG" in [RNO19], their [21]) is:

```
R_k = H X_k - X_k Lambda_k            # column b: rank r_A * r + r, then rounded
P_{k+1} = round( B^{-1} R_k C_2 + P_k C_3 )
X_{k+1} = round( X_k C_1 + P_{k+1} )
(C_1, C_2, C_3) = argmin Tr(X^T H X)  s.t.  X^T X = I     # 3b x 3b generalized eigenproblem
```

with every `round` a `_ops.round_cores` to the working tolerance or rank. Two
things break:

* **Rank growth.** `H X` has rank `R r`; a linear combination of `3b` TT vectors
  has rank `3b r` before rounding. Every rounding is a projection that the
  Rayleigh–Ritz step does not know about, so `X_{k+1}` is not the minimizer of
  the trace over the subspace the coefficients were computed for. [RNO19] §6.1
  states this as the reason MP LOBPCG is less accurate than LRRAP at equal rank.
* **The Gram matrices are formed from rounded vectors.** `V^T V` and `V^T H V`
  are perturbed by `O(delta)` with `delta` the rounding tolerance, so the pencil
  is inconsistent and the Ritz values inherit `O(delta ||H||)`. On top of that,
  classical LOBPCG's own instability (`P_k` becoming numerically parallel to
  `X_k`, [RNO19] footnote 4 citing Hetmaniuk–Lehoucq) is *accelerated* by
  rounding: the rounding of two nearly parallel vectors need not preserve the
  small angle between them.

[RNO19]'s answer (§3.1–3.2): project `X_k`, `B^{-1} R_k` and `P_k` onto **one**
tangent space `T_{x^{(t_k)}} M_r`. Then (i) every linear combination stays in
that linear space and is computed **exactly**, no rounding until the retraction;
(ii) rank of any combination is `<= 2r` regardless of `b`; (iii) inner products
of two tangent vectors are the sum of the Frobenius products of their gauge
cores (their eq. (22)), `O(d n r^2)` instead of `O(d n r^3)`. In our conventions
the tangent representation is exactly the one `tt/algs/riemannian.py` builds:
`delta G_k : (r_{k-1}, n_k, r_k)` with the gauge condition
`ML(delta G_k)^H ML(U_k) = 0` for `k < d`, and the rank-`2r` core stack

```
S_1(i) = [dG_1(i)  U_1(i)],   S_k(i) = [[V_k(i), 0], [dG_k(i), U_k(i)]],   S_d(i) = [V_d(i); dG_d(i)]
```

`riemannian.project(X, Z)` already returns that stack as a rank-`2r` `tt.vector`
— but it *discards* the `delta G_k`, which is exactly what makes the cheap Gram
matrix possible. See §4(b).

The price [RNO19] pays: the correction is sought in **one** tangent space, so
the coefficient problem acquires a `diag(c)` constraint (their (15)–(16)) that
"can not be reduced to a generalized eigenvalue problem"; §5 solves it by an
iteration over `b` small generalized eigenproblems with null-space projections
(Algorithm 2), `O(b^4)`, which they measure to become the dominant cost at
`b ~ 100` (their Fig. 4, not rendered here). And the method has **no rank
adaptation at all** — `r` is a user parameter, fixed for all `b` eigenvectors
and all iterations (their §2.2 and §8: "in its current version, the proposed
algorithm lacks the rank adaptivity").

### 2.3 Riemannian eigensolvers

Same manifold, but the plain single-vector version: minimize
`R(x) = <x, Hx>/<x, x>` over `M_r`, with

```
grad R(x)      = 2 (Hx - R(x) x) / <x, x>
Riemannian grad= P_x grad R(x)                       # riemannian.project, exists
x_{k+1}        = Retract( x_k - tau P_x B^{-1} grad R(x_k) )
```

`Retract` is either `round_cores(..., rmax=r)` (TT-SVD retraction, [RNO19] §2.2,
quasi-optimal and a valid retraction by Steinlechner) or
`riemannian.projector_splitting_add(Y, delta)` (exists, exact when `Y + delta` is
already of rank `r`). `tau` from the exact 1D minimization of the Rayleigh
quotient in `span{x, xi}` — a `2 x 2` generalized eigenproblem, no line search
needed. Adding the LOPCG momentum term `P_x p_k` (their (6)) turns it into
LOPCG; adding `b` columns turns it into §2.2.

This is the *cheapest* item on the list in code terms — `project`,
`projector_splitting_add`, `tt_qr` and `round_cores` all exist — and the most
expensive in iterations, because the convergence factor is the one of
preconditioned gradient descent and therefore proportional to the condition
number. Measured, §3.

### 2.4 Inverse iteration / shift-and-invert on top of `amen_solve`

```
x_{k+1} = amen_solve(A - sigma I, x_k, x_k, eps_solve) ; x_{k+1} /= ||x_{k+1}||
lambda_k = <x_k, A x_k> / <x_k, x_k>
```

Nothing new is needed: `tt.eye`, `matrix.__sub__`, `round`, `amen_solve`,
`tt.dot`, `tt.matvec` all exist. It was prototyped
(`p1b_invit.py`, `p1c.py`, `p1d_shift.py`, `p1e_rqi.py`) and it works — with
sharp limits. §7 has the numbers.

### 2.5 Recommendation

**Build the block AMEn eigensolver (§2.1) first, starting with `B = 1`.**

The measurement that decides it: on the Heisenberg chain `d = 10`, `eigb` with
`B = 1` and a rank-4 guess returns an eigenvalue wrong by **6.10e-03**, reports
`converged = True`, and *structurally cannot* improve — while `amen_solve`, the
solver that already has the missing enrichment, drives the same problem to
**5.97e-12** in 6 solves / 0.7 s when given a shift 1 % below `E_0` (§7.2).
The enrichment is the whole difference, and it is 60 lines away from code that
is already tested.

Second: **inverse iteration** (§2.4) as a thin, documented wrapper — it is a
day's work, it is the only thing here that gives *interior* eigenvalues, and its
failure modes are now measured (§7.2, §7.3). It should ship with the loud
refusals of §8, not as a silent loop.

Third: **Riemannian / LRRAP LOBPCG** (§2.2–2.3), and only with a preconditioner.
Unpreconditioned it is not competitive on QTT-discretized differential operators
— measured in §3.2 — and [RNO19]'s own Table 4 shows it losing to `eigb` at
small `b`: `d = 40` Heisenberg, `b = 5`, at *comparable* accuracy `eigb`
(`delta = 1e-3`, MAE 2.4e-06) takes **26 s** of CPU where LRRAP (`r = 45`, MAE
2.2e-06) takes **251 s** of CPU / 44 s of V100. It wins at `b = 35`, and by
about 1.5x: LRRAP `r = 45` reaches MAE 5.1e-06 in 21 min against `eigb`
`delta = 1e-3` at 3.0e-06 in 31 min (their own numbers, their own machine; the
column alignment of that table is a text-extraction reading, see §10). Its case
is `b >~ 30`, where the block-TT rank `~ B r` makes `eigb` cubically expensive
in the local solve.

---

## 3. Conditioning

### 3.1 How it enters

For a preconditioned gradient-type eigensolver (PINVIT, LOPCG, LOBPCG,
Riemannian gradient) with SPD preconditioner `B` the standard bound
(Knyazev–Neymeyr) is

```
(lambda' - lambda_1) / (lambda_2 - lambda')  <=  sigma^2 (lambda - lambda_1) / (lambda_2 - lambda),
sigma = 1 - (1 - gamma) (1 - lambda_1 / lambda_2),      gamma = ||I - B^{-1} A||_A .
```

Two independent quantities: the **preconditioner quality** `gamma` and the
**relative gap** `1 - lambda_1/lambda_2`. With `B = I` on an elliptic operator,
`gamma = (kappa - 1)/(kappa + 1)`, so `1 - sigma ~ (1 - lambda_1/lambda_2)/kappa`
and the iteration count grows like `kappa`. For the 1D QTT Laplacian on
`N = 2^d` points, `kappa = lambda_max/lambda_min = 4 / (4 sin^2(pi/(2(N+1)))) ~
(2(N+1)/pi)^2 = O(4^d)`.

For the ALS family (`eigb`, block AMEn, DMRG) `kappa` does **not** enter the
sweep count in the same way: the local problems are solved directly, and the
sweep is a Galerkin method whose convergence is governed by how well the frames
approximate the invariant subspace. This is the main reason ALS-type solvers are
the right default in QTT.

### 3.2 Measured: the unpreconditioned rate

Prototype `p3_grad.py`: truncated steepest descent on the Rayleigh quotient,
`x_{k+1} = round(c_1 x_k + c_2 (-(H x_k - lambda_k x_k)), rmax = 4)` with
`(c_1, c_2)` from the exact `2 x 2` Rayleigh–Ritz in `span{x, r}`. Problem:
`tt.qlaplace_dd([d])`, oracle `lambda_1 = 4 sin^2(pi/(2(N+1)))`, seed 0,
iteration cap 4000.

| `d` | `N` | `kappa` | iterations to rel. 1e-3 | `lambda` after the cap | rel. error | time |
|---|---|---|---|---|---|---|
| 6  | 64   | 1.71e+03 | **1413** | 2.33787316e-03 | 9.96e-04 | 12.6 s |
| 8  | 256  | 2.68e+04 | **never** (4000) | 1.72420551e-04 | 1.54e-01 | 52.3 s |
| 10 | 1024 | 4.26e+05 | never (4000) | 3.66674612e-04 | **3.80e+01** | 67.6 s |
| 12 | 4096 | 6.80e+06 | never (4000) | 2.41476003e-04 | **4.10e+02** | 82.5 s |

Compare with `eigb` on the *same operator* at `d = 8`, `B = 1`: relative error
3.31e-12 in **4 sweeps and 0.04 s** (§1.3). The gap between the two families on
a QTT-discretized elliptic operator is not a constant factor; it is the
condition number. Note also `d = 10, 12`: at 4000 iterations the Rayleigh
quotient is not merely inaccurate, it is 39x and 411x too large. It *has*
descended a long way from the initial random vector (whose Rayleigh quotient is
`O(||A||) = O(1)`) down to `~2e-04`, and then stopped making relative progress —
which is the signature of a rate `1 - c/kappa` with `kappa = 4e5..7e6`, not of a
broken iteration. An unpreconditioned Riemannian or LOBPCG eigensolver
shipped for QTT problems would fail exactly like this, and it would fail
*silently* unless it measures the eigenresidual (which is `O(1)` here and would
say so).

### 3.3 What the eigensolver needs from a BPX-type preconditioner

BPX now has its own spec, `docs/plans/qtt-elliptic-bpx.md`; this section states
the interface so the two meet, and **item 1 below was answered from that side
and is superseded** — see the note after item 4 and `docs/plans/ROADMAP.md` §3.
The eigensolver needs `B^{-1}` to be:

1. **Applicable to a TT vector with bounded rank growth.** Either a TT-matrix of
   small TT rank, or (the form [RNO19] §4.5 relies on, their eq. (24), citing
   Khoromskij's exponential sums, Constr. Approx. 30:599–620, 2009) a sum of
   `rho_B` **rank-1** TT-matrices, `B^{-1} = B_1 + ... + B_{rho_B}`. Rank 1
   matters: multiplying a TT-matrix of rank `R` by a rank-1 TT-matrix leaves the
   rank at `R`, so `P_x B^{-1} H x` can be assembled term by term without ever
   forming a rank-`rho_B R r` intermediate. Any BPX we build should expose this
   decomposition, not only a black-box `apply`.
2. **Symmetric positive definite.** Rayleigh–Ritz on `span{X, B^{-1}R, P}` is a
   symmetric pencil only if `B` is SPD; a nonsymmetric preconditioner destroys
   the variational characterization and with it the "Ritz values decrease
   monotonically" invariant that every test in §8 relies on.
3. **Spectrally equivalent with mesh- and dimension-independent constants**,
   `c_1 (x, Ax) <= (x, Bx) <= c_2 (x, Ax)`. Only `c_2/c_1 = O(1)` makes `gamma`
   bounded away from 1 and the iteration count independent of `d`.
4. **Cheap in the ALS setting too, but differently.** Block AMEn does not need a
   global `B^{-1}`: what its GMRES local solves need is a preconditioner for
   the *local* operator, and `amen.py` already has three (`local_prec='c'/'l'/'r'`,
   block Jacobi). A BPX would help the *outer* iteration of §2.2–2.3 and the
   *linear solves* of §2.4, not the local solves of §2.1.

**Item 1, corrected by measurement.** `docs/plans/qtt-elliptic-bpx.md` §4.2
answers this section from the BPX side and item 1 is the wrong shape for it: BPX
is a **single `tt.matrix`** of TT rank `2^{2D+1}` — measured exactly 8 / 32 / 128
for `D = 1, 2, 3`, independent of `L` up to `L = 50` (their §1.5) — and it
cannot be written as a sum of rank-1 terms. Nor is a small-rank matrix a "slow"
form: one matvec plus one rounding costs the same as one extra matvec by `A`,
whose rank is 3–4. The rank-1-sum form remains right for Kronecker sums over
*physical* modes, and was measured **not** to transfer to the by-scale (QTT)
setting: the QTT ranks of `expm(-t A_DN)` are 8–21, not 1 (their §2.3). The
reconciled contract — three accepted forms, each declaring its side and its
rank — is `docs/plans/ROADMAP.md` §3. One further correction from there: an
eigensolver wants the **left** BPX `C_{2,L}` (weight `2^{-2l}`), not the
two-sided `C_L` (weight `2^{-l}`) that a linear solver wants; both come from one
constructor, `tt.bpx(d, D, weight=1|2)`.

What BPX does **not** give: a preconditioner for the indefinite `A - sigma I` of
shift-and-invert with an interior shift. §7.3 measures what happens without one.
Nor any help with coefficient contrast, which passes straight through
(`kappa(BPX) = 1.12e+05` at contrast `1e4`, qtt-elliptic-bpx.md §2.4).

### 3.4 The accuracy floor, measured

Conditioning also sets a floor on the *achievable* relative accuracy of a small
eigenvalue, independent of the method: a backward-stable computation returns the
exact eigenvalue of `A + E` with `||E|| ~ eps_machine ||A||`, so
`|delta lambda| / lambda <~ eps_machine ||A|| / lambda = eps_machine * kappa`.

Measured on the 1D QTT Laplacian with the inverse iteration of §7.1
(`eps_machine = 2.22e-16`, float64, `||A||_2 = 4` up to `O(1/N^2)`):

| `d` | `lambda_1` | `kappa = 4/lambda_1` | predicted floor `eps_machine * kappa` | **best relative error observed** |
|---|---|---|---|---|
| 8  | 1.494267e-04 | 2.677e+04 | 5.9e-12 | **1.77e-11** |
| 10 | 9.394024e-06 | 4.258e+05 | 9.4e-11 | **3.10e-11** |
| 14 | 3.676265e-08 | 1.088e+08 | 2.4e-08 | **4.85e-09** |
| 20 | 8.976336e-12 | 4.456e+11 | 9.9e-05 | **1.50e-05** |
| 24 | 3.506387e-14 | 1.141e+14 | 2.5e-02 | **2.47e-04** |
| 30 | 8.560517e-18 | 4.673e+17 | 1.0e+02 | **4.36e-01** |

Up to `d = 20` the prediction holds within a factor of 4 across seven decades.
At `d = 24, 30` the observed error is 10–200x *better* than the bound, which is
what one should expect: `eps_machine * kappa` is a worst case over perturbation
directions, and the Rayleigh quotient of an eigenvector that is accurate *in
shape* is second-order accurate in the eigenvector error. The bound is therefore
the right thing to warn on and the wrong thing to assert as an equality.

The consequence for the API: a solver asked for `eps = 1e-10` on a `d = 20` QTT
Laplacian is being asked for something float64 cannot deliver, and it should say
so from the input alone (`||A|| / |lambda|` is estimable once one Ritz value is
known) rather than iterate to `nswp` and report a plausible number. See §8
test 5.

---

## 4. What the core is missing

Everything the ALS sweep needs exists and is reusable: `_localops.operator_cores`,
`ones_interface`, `phi_left`, `phi_right`, `local_matvec`, `local_matmat`
(already takes a *stack* of blocks `(r, m, r, k)` — exactly a block local
matvec), `local_matrix`, `left_orthogonalize`, `right_orthogonalize`;
`_ops.chop`, `round_cores`, `add`, `sub`, `dot` (already returns the **block
Gram** `(ra0, rb0, rad, rbd)` when the boundary ranks exceed 1), `norm`,
`matvec_cores`, `randomized_round`; `amen._truncate`, `amen._qr_left`,
`amen._push_left`, `amen._jacobi`, `amen._gmres`; `amen_mv._project`, `_apply`,
`_phi_next`, `_phi_yy_next`; `eigb.block_residuals`, `eigb._local_eig_dense`,
`eigb._local_eig_lobpcg`; `riemannian.project`, `projector_splitting_add`,
`tt_qr`. Genuinely missing:

**(a) A block-of-tensors container — needed for speed, not for correctness.**
`eigb`'s block TT (`y.r[-1] == B`) already represents `B` eigenvectors and needs
nothing new. `eig_lobpcg` and the Riemannian methods want `B` *separate* TT
vectors, and for those a plain `list[tt.vector]` is sufficient and is what §5
proposes. A batched container (`cores[k]` of shape `(B, r, n, r)`, the T3F layout
of [RNO19] Appendix A) buys one thing: on a GPU backend it turns `B` small
`einsum` calls into one batched call, which is the entire reason their `b = 84`
runs are fast. Building it before there is a torch benchmark would be
speculative; the API of §5 is written so that it can be introduced later without
changing signatures (a list and a batched container both satisfy "a sequence of
`tt.vector`"). Recorded here as a deliberate deferral, not an oversight.

**(b) The tangent representation, not just the tangent vector.**

**Owned by `docs/plans/riemannian-autodiff.md` §8(a)–(c); do not restate the
signature here.** This section originally proposed
`project_delta(X, Z) -> list[array]`; that spec proposes
`project_delta(X, Z, *, weights=None, frames=None) -> (deltas, Frames)`, and
**its signature wins** — the caller almost always needs the frames immediately
afterwards (rebuilding them is the `O(d n r^3)` term), and the `weights`/list
form is what its §6.3 measured at `rho_B = 61` summands. The resolution is
recorded in `docs/plans/ROADMAP.md` §2.2, together with the same reconciliation
for `tangent_to_tt`, `tangent_inner` and `tangent_gram`.

What this spec needs from it, unchanged: the gauge cores `delta_G[k]` of shape
`(r_{k-1}, n_k, r_k)` satisfying `ML(delta_G[k])^H ML(U_k) = 0` for `k < d`, so
that `<xi, eta>` for two tangent vectors at the **same** `X` is
`sum_k <delta_G[k]^xi, delta_G[k]^eta>_F` ([RNO19] eq. (22)) and a Gram matrix
of `b` tangent vectors costs `O(b^2 d n r^2)` rather than `O(b^2 d n r^3)`,
without the cancellation of a TT contraction. That is what makes the LRRAP
eigensolver of §2.2 affordable.

Closest existing: `riemannian.project`, which computes these cores internally and
then assembles and discards them.

**(c) Rayleigh–Ritz on a set of TT vectors.**

```python
def rayleigh_ritz(A, basis, *, nev=None, gram_tol=1e-12):
    """Ritz pairs of ``A`` in the span of a list of TT vectors.

    Returns ``(theta, C)``: ``theta`` ascending, ``C[:, i]`` the coefficients of
    the i-th Ritz vector in ``basis``.  Solves the symmetric pencil
    ``(V^H A V) c = theta (V^H V) c``.
    Raises RuntimeError when ``V^H V`` has a numerical rank below ``len(basis)``
    (the basis collapsed -- the standard LOBPCG failure); it must NEVER
    silently drop directions, because the caller cannot tell the resulting
    plausible Ritz values from correct ones.
    """
```
Closest existing: nothing. `_ops.dot` gives one entry of the Gram matrix.

**(d) Block orthogonalization of a set of TT vectors.**

```python
def block_orthogonalize(vectors, *, eps=1e-14, method="chol"):
    """Orthonormalize a list of TT vectors: returns (vectors, R) with
    ``sum_i vectors_new[i] R[i, j] == vectors_old[j]`` and Gram = I to ``eps``.
    ``method='chol'`` is Cholesky-QR on the Gram matrix (one pass, then one
    reorthogonalization pass, which is what makes it stable enough here);
    it raises if the Gram matrix is not positive definite at ``eps``.
    """
```
Closest existing: `_ops.orthogonalize` (orthogonalizes the *cores* of one
tensor, a different thing entirely).

**(e) Block local eigensolver as a public, shared owner.** `eigb._local_eig_dense`
and `eigb._local_eig_lobpcg` are private to `eigb` and the block AMEn solver
needs the identical thing.

```python
def local_eig(left, acore, right, nblock, *, guess=None, tol=1e-8,
              max_full_size=1000, sym_tol=None, maxiter=200):
    """The ``nblock`` smallest eigenpairs of the projected local operator.

    Returns ``(lam, v, res_max)`` with ``v`` of shape ``(r, n, r', nblock)`` and
    ``res_max`` the largest MEASURED local residual (0.0 on the dense path).
    """
```
Move to `tt/algs/_localops.py`; `eigb` imports it from there. One owner.

**(f) Residual of a block, as cores.**

```python
def block_residual_cores(acores, xcores, lam):
    """Cores of ``A X - X diag(lam)`` for a block TT-vector (``r[-1] == B``)."""
```
Closest existing: the first half of `eigb.block_residuals`, which builds exactly
this and then immediately norms it. Split it in two; `eigb` keeps its behaviour,
the enrichment of §2.1 gets what it needs.

**(g) The enrichment step as a named owner.**

```python
def enrich(core, next_core, extra, *, direction="lr"):
    """Concatenate ``extra`` onto ``core`` along the moving bond, zero-pad
    ``next_core``, re-orthogonalize.  Returns ``(core, next_core)``.
    """
```
Closest existing: the seven inlined lines in `amen.amen_solve` (`bk.concatenate`
/ `bk.zeros` / `_qr_left` / `_push_left`). Block AMEn needs the same lines; two
copies of that truth is how they diverge.

**(h) `round_cores` with an absolute tolerance and a report of what it dropped.**
Identical to the request in `docs/plans/bug-integrator.md` §4(a) — the same
owner, extended once, for both specs:

```python
def round_cores(cores, eps=1e-14, rmax=None, abs_tol=None, return_discarded=False): ...
```
The eigensolvers need it because the natural truncation criterion for an
eigenvector is *absolute* (`||A x - lambda x||`-driven), not relative to `||x||`.

**(i) A history dataclass for the eigen-family.** `EigbHistory` exists and is
close; the new solver needs `kickrank`, per-sweep `enrichment_rank`, the Ritz
values of every sweep, the *converged* flag per eigenvalue (not per run), and
`res`/`res_rel` as today. Proposed in §5.

**(j) Nothing is missing for shift-and-invert.** `tt.eye`, `matrix.__sub__`,
`.round`, `amen_solve`, `tt.dot`, `tt.matvec` cover §2.4 completely; what it
needs is the *policy* of §8 (refuse loudly), not new algebra.

---

## 5. Proposed API

Module `tt/algs/eig.py`, exported as `tt.algs.eig`. `tt/algs/eigb.py` and the
legacy shim `tt/eigb/__init__.py` are **untouched**: `import tt.eigb;
tt.eigb.eigb(A, y0, eps)` and `tt.eigb_solve` keep working exactly as today
(`tests/test_eigb_ksl.py::test_legacy_imports` asserts identity of the
objects, and that assertion must keep passing). New names are added to
`tt.__init__._FUNCTIONS`, which is the existing lazy-resolution table:

```python
_FUNCTIONS = {..., "eig_amen": ("tt.algs.eig", "eig_amen"),
                   "eig_invit": ("tt.algs.eig", "eig_invit"),
                   "eig_lobpcg": ("tt.algs.eig", "eig_lobpcg")}
```

```python
def eig_amen(A, y0=None, eps=1e-8, *, nblock=1, kickrank=4, nswp=20, rmax=150,
             max_full_size=1000, local_prec='c', local_iters=2,
             local_restart=40, sym_tol=None, check_residual=True,
             res_warn=None, seed=None, verb=1, return_history=False):
    """The ``nblock`` smallest eigenpairs of a symmetric TT-matrix, block ALS
    with AMEn residual enrichment.

    Unlike :func:`tt.algs.eigb.eigb` the TT ranks are adapted for every
    ``nblock``, including ``nblock == 1``: the enrichment adds up to
    ``kickrank`` residual directions at every bond of every sweep.
    ``kickrank=0`` reduces the method to ``eigb`` exactly and is the intended
    A/B switch for the tests of §8.

    Args:
        y0: initial guess; ``None`` draws a random block of rank
            ``max(4, nblock+1)`` from ``seed``.  When ``y0`` is given,
            ``y0.r[-1]`` is the number of eigenvalues (the ``eigb`` contract);
            passing a ``nblock`` that disagrees with it raises, rather than
            silently picking one of two owners of the same number.
    Returns:
        ``(y, lam)`` or ``(y, lam, EigHistory)``.  ``lam`` ascending, ``y`` a
        block ``tt.vector`` with ``y.r[-1] == nblock`` and orthonormal columns.
    Raises:
        ValueError: non-square/non-Hermitian ``A``, mode or dimension mismatch,
            ``y0.r[0] != 1``, a local problem smaller than ``nblock``.
    Warns:
        RuntimeWarning: non-convergence (carries the achieved indicator), or a
            returned backward error ``||A y_i - lam_i y_i|| / ||A||_2``
            above ``res_warn`` (``None`` = ``sqrt(eps)``, as in ``eigb``
            after the fix of Sec. 1.2a -- same threshold, same denominator,
            one owner).
    """

def eig_invit(A, sigma=None, x0=None, *, eps=1e-8, solver_eps=None, maxit=50,
              rmax=None, refine=None, verb=1, return_history=False):
    """One eigenpair by (shifted) inverse iteration on top of ``amen_solve``.

    ``sigma=None`` means plain inverse iteration (smallest ``|lambda|``).
    ``refine='rayleigh'`` switches the shift to the current Rayleigh quotient
    once the residual drops below ``sqrt(eps)`` -- OFF by default, see the
    measurement in Sec. 7.3 of docs/plans/eigenvalues.md: RQI converged to a
    DIFFERENT eigenvalue than the one asked for on the d=10 Heisenberg chain.

    ``solver_eps`` defaults to ``eps/100``.  Every inner solve is checked:
    ``amen_solve``'s ``info.converged`` False on two consecutive iterations
    raises RuntimeError naming the shift, rather than iterating on a vector the
    linear solver could not produce.
    Returns ``(lam, x)`` or ``(lam, x, EigHistory)``.
    """

def eig_lobpcg(A, X0, *, nblock=None, prec=None, eps=1e-8, maxit=200, rank=None,
               schedule='argmax', warmup=20, verb=1, return_history=False):
    """The ``nblock`` smallest eigenpairs by LOBPCG on the fixed-rank TT manifold
    (LRRAP, [RNO19]).

    ``rank`` is a FIXED TT rank for every eigenvector -- the method has no rank
    adaptation ([RNO19] Sec. 8); the value is a required decision, not a
    tolerance.  ``prec`` is a callable ``tt.vector -> tt.vector`` or a list of
    rank-1 ``tt.matrix`` summing to ``B^{-1}`` ([RNO19] eq. (24)); the list form
    is what keeps ranks bounded and is preferred.
    ``schedule``: 'first' (all iterations in the tangent space of x^(1),
    [RNO19] (12)), 'argmax' ([RNO19] (17)), 'random'.
    """
```

```python
@dataclass
class EigHistory:
    eps: float
    lam: np.ndarray               # returned Ritz values, ascending
    res: np.ndarray               # ||A y_i - lam_i y_i||, MEASURED
    res_rel: np.ndarray           # res_i / ||A y_i||
    converged: bool               # the run
    converged_per_eig: np.ndarray # bool per eigenvalue: res_rel_i <= eps
    ranks: list
    ranks_per_sweep: list
    enrichment_rank: list         # actual added rank per bond of the last sweep
    lam_per_sweep: list           # every sweep's Ritz values -- the convergence plot
    ermax: float                  # movement of sum(lambda); NOT an error estimate
    max_local_res: float
    nswp_done: int
    n_local_solves: int
    n_linear_solves: int          # eig_invit only
    time: float
    message: str
```

`converged_per_eig` is the one field `EigbHistory` should have had: in a block
run the lowest eigenvalues converge many sweeps before the highest, and a single
`converged` flag forces a caller who wants three good eigenvalues to pay for
eight.

---

## 6. Cost

Notation: `d` modes, mode size `n`, TT rank of the iterate `r`, TT-matrix rank
`R`, block size `B`, enrichment rank `kickrank`, LOBPCG basis `3B`.

| step | cost | who pays |
|---|---|---|
| one `phi_left`/`phi_right` | `n r^2 R (r + n R)` | all ALS |
| one `local_matvec` | `n r^2 R (r + nR)` | all ALS |
| dense `local_matrix` + `eigh` | build `(n r^2)^2 R`, solve `(n r^2)^3` | `eigb`, block AMEn |
| block-TT rank | `r ~ B r_1` (measured §1.3: 2,4,8,14,20 for B=1,2,4,8,16) | `eigb`, block AMEn |
| enrichment (§2.1) | `n r^2 R kickrank + n r kickrank B min(nr, kB)` | block AMEn |
| `H x` for one column | `n^2 R^2 r^2` | LOBPCG, Riemannian, `eig_invit` |
| tangent projection `P_x z` | `d n r r_z^2` ([RNO19] §4.2) | LOBPCG, Riemannian |
| tangent Gram `V^T V` | `b^2 d n r^2` **if** the gauge cores are kept, `b^2 d n r^3` if not | LOBPCG |
| `find_coefficients` (Alg. 2) | `O(b^4)` | LRRAP LOBPCG |
| one `amen_solve` | `nswp * d * (local solve)` | `eig_invit` |

The one asymptotic statement worth acting on: because the block-TT rank grows
like `B r_1`, the dense local eigensolve of `eigb`/block-AMEn costs
`(n B^2 r_1^2)^3` — cubic in the local size, hence `B^6`. That is [RNO19]'s
entire motivation, and it is measured in §1.3 (rank `2 -> 20` as `B: 1 -> 16`)
and in their Table 5 (`5 -> 120` as `b: 1 -> 16`). LRRAP's tensor work is
*linear* in `b` (their Table 1) with an `O(b^4)` coefficient problem on top.
Honest caveat: `B^6` is the unconstrained worst case, not what was observed.
The measured growth on Heisenberg `d = 16` (§7.4) is about `B^2.2` over
`B = 4..16`, because the rank grew sublinearly (`112 -> 181 -> 200`) once
`rmax = 200` bound, and because part of each sweep runs in the matrix-free
LOBPCG path rather than the dense one.

Measured wall times on this host are in §1.3 and §7; they are single-run,
2 cores, no `numba`, and should not be compared with the published GPU numbers.

---

## 7. Measured: the prototypes

### 7.1 Prototype 1 — inverse iteration on `amen_solve`, QTT Laplacian

`A = tt.qlaplace_dd([d])` (tridiagonal `-1, 2, -1` on `N = 2^d` interior points;
verified against the dense matrix at `d = 3`, max eigenvalue difference
4.44e-16). Oracle: `lambda_k = 4 sin^2(k pi / (2(N+1)))`, exact, no dense
computation involved. `x_0 = tt.rand(2, d, r=2)` from
`numpy.random.default_rng(0)`, normalized; `eps_solve = 1e-10`;
`lambda = <x, Ax>/<x, x>` each step. "solve" = one `amen_solve` call.

| `d` | `N` | `lambda` computed | `lambda_1` analytic | rel. error | solves to 1e-6 | wall | solves to 1e-10 | wall | ranks | cap |
|---|---|---|---|---|---|---|---|---|---|---|
| 8  | 256      | 1.49426660537534e-04 | 1.49426660534890e-04 | **1.77e-11** | 8 | 0.50 s | 11 | 0.62 s | 4 | hit 1e-10 |
| 10 | 1024     | 9.39402419999208e-06 | 9.39402419970067e-06 | **3.10e-11** | 5 | 1.04 s | 8 | 1.49 s | 4 | hit 1e-10 |
| 14 | 16384    | 3.67626534684381e-08 | 3.67626536468265e-08 | 4.85e-09 | 7 | 7.93 s | never | 42.9 s (40 solves) | 2 | floor |
| 20 | 1048576  | 8.97647011297622e-12 | 8.97633579036887e-12 | 1.50e-05 | 4 | 19.8 s | never | 199 s (40 solves) | 2 | floor |

Answering the question asked directly: **six digits of the smallest eigenvalue
of a QTT Laplacian cost 5 AMEn solves and 1.04 s at `d = 10` (`N = 1024`), and 4
solves / 19.8 s at `d = 20` (`N = 10^6`).** The ranks stay at 2–4 throughout
(the eigenvector is a sine, QTT rank 2), so this is the *easy* regime; every
AMEn solve converged to its own `eps_solve = 1e-10` in 2–3 sweeps.

The relative gap is what makes it this cheap: `lambda_2/lambda_1 -> 4` for the
1D Laplacian, so plain inverse iteration contracts the eigenvector error by
`1/4` per solve and the Rayleigh quotient by `1/16`. On the shifted Heisenberg
problem of §7.2 at `sigma = -6`, where the corresponding ratio is
`(E_0-sigma)/(E_1-sigma) = 0.842`, the same code needs **41** solves for the
same six digits. The method has no defence against a
small relative gap other than a better shift.

Beyond `d = 20` the same code is measured in §3.4 (the accuracy floor) and §8
test 5 (`d = 24`: best relative error 2.47e-04 in 6 solves, 264 s; `d = 30`: the
Rayleigh quotient comes out **negative** for a positive definite operator,
372 s).

Interior eigenvalues by shift-and-invert on this operator (`sigma` halfway
between two neighbouring analytic eigenvalues, `d = 10`, targets `k = 4` and
`k = 11`) were attempted in `p1b_invit.py` and are **not reported here**: the
run did not finish inside the time budget of this investigation. Not measured.

### 7.2 Prototype 1b — shift-and-invert on the Heisenberg chain

`H = ham.heisenberg(10)` (§1.2), `B = (H - sigma I).round(1e-14)`,
`eps_solve = 1e-8`, oracle `E_0 = -4.258035207283` from
`numpy.linalg.eigvalsh`, cap 60 solves.

| `sigma` | `(E_0-sigma)/(E_1-sigma)` | solves to rel. 1e-6 | solves to rel. 1e-10 | final rel. error | time |
|---|---|---|---|---|---|
| -8.0 | 0.920 | **never** (60 solves) | never | 5.48e-05 | 4.6 s |
| -6.0 | 0.842 | 41 | never (60) | 1.36e-09 | 4.6 s |
| -5.0 | 0.694 | 20 | 32 | 8.94e-11 | 2.8 s |
| -4.5 | 0.425 | 9 | 14 | 5.07e-11 | 1.4 s |
| -4.3 | 0.114 | 4 | 6 | 5.97e-12 | 0.7 s |

The cost is a pure function of how well you already know the answer. With a
shift 1 % below `E_0` (sigma = -4.3) it beats `eigb` at `B = 2` (0.7 s against 3.6 s) and
matches its accuracy; with a shift 88 % below (sigma = -8) it never gets six digits at all.
**This is the honest counterweight to §2.5's second recommendation**: an inverse
iteration is only a solver if it comes with a way to choose `sigma`, and nothing
in ttpy2 provides one.

### 7.3 Prototype 1c — Rayleigh quotient iteration: converges to the wrong eigenvalue

Same problem, warm start of 5 solves at `sigma = -6`, then `sigma = lambda - 1e-8`:

```
it= 5 shift=-6.000000000 lam=-3.744944416882 relerr=1.20e-01   conv=True
it= 6 shift=-3.744944427 lam=-3.920252771369 relerr=7.93e-02   conv=True
it= 7 shift=-3.920252781 lam=-3.930675803586 relerr=7.69e-02   conv=True
it= 8 shift=-3.930675814 lam=-3.930673589502 relerr=7.69e-02   conv=True
it= 9 shift=-3.930673600 lam=-3.930673589502 relerr=7.69e-02   conv=False
     amen_solve did NOT reach eps=1.000E-08 in 20 sweeps: reached true residual 1.744
```

`-3.930673589502` is **exactly** `E_1` of the dense spectrum (12 digits) — the
first excited state, which is triply degenerate. RQI locked onto a perfectly
correct eigenpair that is not the one that was asked for, and stayed there for
the remaining 22 iterations while `amen_solve` correctly reported that it could
not solve the (now numerically singular) shifted system. Two lessons, both
encoded in §5 and §8:

* the inner solver's failure is **not** a sign that the outer iteration failed —
  here it is a sign that it *converged*;
* the outer iteration cannot tell "the smallest eigenvalue" from "an
  eigenvalue". `eig_invit` must therefore return the Ritz value with its
  residual and say explicitly which one it is not: without a lower bound or a
  deflation basis, it can only certify *an* eigenpair. `refine='rayleigh'` is
  off by default for this reason.

An earlier variant that switched to RQI immediately, from a cold start
(`p1d_shift.py`), was worse and *loud*: `amen_solve` refused at iteration 2 with
`true residual 5.502` because the shift `-0.481` landed in the middle of the
spectrum, where `H - sigma I` is indefinite and the local GMRES cannot solve it.
That is the correct behaviour and the reason §5 propagates it as a
`RuntimeError`.

### 7.4 Prototype 2 — `eigb` against `numpy.linalg.eigh`

Reported in §1.2 (Heisenberg `d = 10`, `B = 1`, the failure), §1.3 (Heisenberg
`B = 1..8`, QTT Laplacian `B = 1..16`, TFIM `g = 1` and `g = 0.5`) and §7.5.
The point at which `eigb` starts to lose accuracy is not a value of `B` — it is
`B = 1` together with a rank guess below what the eigenvector needs. For
`B >= 2` on every problem tried, `eigb` reached `1e-14`.

Two further measurements from the same prototype.

**Rank and cost against `B`, where the rank does not saturate.** Heisenberg
`d = 16` (dense oracle impossible: `65536^2`), `eps = 1e-6`, `nswp = 8`,
`rmax = 200`, `r0 = max(4, B+1)`, seed 0. Reference for the ground state: the
`B >= 2` runs, which agree on `-6.9117371456` to 1e-9 and carry a measured
relative eigenresidual below 5e-7.

| `B` | max rank | sweeps | `converged` | wall | `lambda_1` | max rel. eigenresidual | local solves |
|---|---|---|---|---|---|---|---|
| 1  | **4** (= `r0`) | 8 | False | 1.5 s | **-6.8934652792** | **2.76e-02** | 240 |
| 2  | 74  | 3 | True | 10.1 s | -6.9117371456 | 4.90e-07 | 90 |
| 4  | 112 | 2 | True | 6.6 s  | -6.9117371456 | 4.26e-07 | 60 |
| 8  | 181 | 2 | True | 31.5 s | -6.9117371456 | 5.17e-07 | 60 |
| 16 | **200** (= `rmax`, capped) | 2 | True | 110.5 s | -6.9117371454 | 6.68e-06 | 60 |

The `B = 1` row is the failure of §1.2 again, at a size where no dense oracle
exists to catch it: 1.8e-02 low, and only the eigenresidual says so. From
`B = 4` to `B = 16` the wall time grows 6.6 -> 31.5 -> 110.5 s, i.e. roughly
`B^2.2` here, with the rank growing `112 -> 181 -> 200` until `rmax` binds. This
is the scaling [RNO19] set out to fix.

**Degeneracy in a genuinely multidimensional QTT problem.**
`tt.qlaplace_dd([4,4,4])` (a `16^3` grid), `B = 8`, `eps = 1e-8`, `r0 = 10`.
Oracle: sums of analytic 1D eigenvalues, which contain two exactly threefold
degenerate groups inside the first eight. Result: max relative error
**4.64e-15**, max relative eigenresidual 2.32e-14, max rank 16, 2 sweeps,
0.35 s. `eigb` resolves symmetry-induced degeneracy without difficulty.

**Where `eigb` starts to lose accuracy**, stated precisely: not at any `B`, but
whenever the rank it is allowed to reach is below what the eigenvector needs —
which for `B = 1` is *always* the rank of the initial guess, and for `B >= 2` is
`rmax`. The mechanism is visible in the `rmax` table of §8 test 4.

---

## 8. Validation tests, with named oracles

Every test names its oracle. Nothing is compared against the module's own output
except where the test is explicitly about two code paths agreeing.

1. **QTT Laplacian 1D against the analytic spectrum.** Oracle:
   `lambda_k = 4 sin^2(k pi / (2(N+1)))` — no dense computation.
   `tt.qlaplace_dd([d])`, `d = 8, 12`, `B = 1, 4`. Assert
   `|lambda_k - exact_k| <= 1e-10 * exact_k + 1e-14`. Measured today with
   `eigb`: 3.3e-12 relative at `d = 8, B = 1` (§1.3), so the assertion has
   two orders of headroom. This is the test that catches an index-convention
   error (the `(B a)` versus `(a i)` groupings of §1.1) before anything else.

2. **QTT Laplacian 3D, degenerate by symmetry.** Oracle: sums of 1D analytic
   eigenvalues. `tt.qlaplace_dd([4,4,4])`, `B = 8`, which straddles the
   threefold degenerate `lambda_{112} = lambda_{121} = lambda_{211}`.
   `tests/test_eigb_ksl.py` already has the `d=5` version with `B=4`.

3. **THE RANK-ADAPTATION TEST — this is what the new solver is for.**
   Heisenberg `d = 10`, `B = 1`, initial rank 4, `eps = 1e-8`, seed 0.
   Oracle: `numpy.linalg.eigvalsh`, `E_0 = -4.258035207283`.
   * `eigb` (or `eig_amen(kickrank=0)`) is asserted to return
     `|lambda - E_0| > 1e-4` **and** `history.res_rel.max() > 1e-3` — i.e. the
     failure of §1.2 is pinned as a `pytest.mark.xfail`-style *expected* result,
     with the measured 6.10e-03 in the reason, so that a future change to `eigb`
     cannot silently alter it.
   * `eig_amen(kickrank=4)` on the identical input is asserted to reach
     `|lambda - E_0| <= 1e-9` and `res_rel <= 1e-6`, with a *reported* final
     rank (record it, do not assert a value until §11 Q2 is settled).

4. **THE FAIL-LOUD TEST: `rmax` below what the tolerance needs.** The method
   must not report success it did not achieve. Setup and the measured behaviour
   of today's `eigb`:

   Heisenberg `d = 10`, `B = 4`, `eps = 1e-8`, `nswp = 40`,
   `r0 = min(rmax, 8)`, seed 0; oracle `numpy.linalg.eigvalsh`
   (`-4.258035207, -3.93067359` threefold):

   | `rmax` | max rank | `converged` | `ermax` | max abs. error | max rel. eigenresidual | warned | time |
   |---|---|---|---|---|---|---|---|
   | 4   | 4  | **False** | 3.3e-02 | 1.51e-01 | 1.67e-01 | yes | 3.2 s |
   | 8   | 8  | **False** | 2.4e-03 | 1.92e-02 | 4.76e-02 | yes | 8.6 s |
   | 16  | 16 | **False** | 1.0e-05 | 7.53e-05 | 3.67e-03 | yes | 51.1 s |
   | 32  | 32 | True | 2.4e-09 | 1.33e-08 | 5.44e-05 | no | 7.4 s |
   | 64  | 64 | True | 2.2e-15 | 1.15e-14 | 5.19e-11 | no | 1.4 s |
   | 150 | 64 | True | 2.2e-15 | 1.15e-14 | 5.19e-11 | no | 1.0 s |

   Today's `eigb` **passes** this test: a binding `rmax` keeps the iterate
   moving, so `ermax` never falls below `eps`, `converged` stays False and the
   warning fires. Keep the test — it is the regression guard for the case the
   new solver must not break — and add the assertion that `rmax = 32` (which
   *does* report convergence) still leaves an absolute error of 1.33e-08 for a
   requested `eps = 1e-8`, i.e. the tolerance is met only barely and the
   `res_rel = 5.44e-05` should have been reported under the fix of §1.2a. The
   contrast with §1.2a is the point: `rmax` binding is loud, a *guess rank* that
   never grows is silent.

5. **THE SECOND FAIL-LOUD TEST: an eigenvalue below the floor of the
   arithmetic.** `tt.qlaplace_dd([30])`: `lambda_1 = 8.56e-18`,
   `kappa = 4.7e17 > 1/eps_machine`. No float64 method can return this to any
   relative accuracy; a solver that returns a plausible small number is lying.
   Assert that `eig_invit`/`eig_amen` **raises or warns** with a message naming
   `eps_machine * ||A|| / lambda`, and does not report `converged = True`.

   Measured today, inverse iteration on `amen_solve` (`eps_solve = 1e-10`,
   6 solves, `p1c.py`), against `lambda_1 = 8.560517e-18`:

   | iteration | `lambda` returned | rel. error | max rank | `amen_solve` true residual |
   |---|---|---|---|---|
   | 1 | **-4.699018890e-17** | 6.49e+00 | 78  | 3.9e+00 |
   | 2 | 1.639952151e-16 | 1.82e+01 | 147 | 1.2e+06 |
   | 3 | 2.994828557e-16 | 3.40e+01 | 202 | 4.3e+07 |
   | 4 | **-1.210725254e-16** | 1.51e+01 | 187 | 1.3e+04 |
   | 5 | 4.824179785e-18 | 4.36e-01 | 186 | 5.8e+03 |
   | 6 | 9.249816071e-17 | 9.81e+00 | 190 | 6.9e+03 |

   Two independent signals, both already available and both ignored by a naive
   loop: (a) the Rayleigh quotient of a **positive definite** operator comes out
   **negative** in iterations 1 and 4 — a mathematically impossible answer, and
   the cheapest possible assertion (`lam > 0` for an SPD `A`); (b) `amen_solve`
   reports a *relative* residual of `1.2e+06` and `4.3e+07`, i.e. the inner
   solves failed by seven orders of magnitude and said so, while the ranks ran
   away from 2 to 202 trying to represent a solution that does not exist in
   float64. The test asserts that the wrapper of §5 stops at the first of these,
   not at `maxit`.

   The same probe at `d = 24` (`kappa = 1.14e+14`) is the *borderline* case worth
   keeping as a second arm: the best relative error over 6 solves is 2.47e-04,
   the last is 4.80e-03, and `amen_solve`'s true residual reaches 9.4e+02 —
   wrong, loudly, but not absurdly. Total 264 s for `d = 24` and 372 s for
   `d = 30` on this host, so the test should run with `nswp` capped.

6. **Shift-and-invert returns *an* eigenpair, and says so.** Heisenberg `d=10`,
   `eig_invit(H, sigma=-3.9)`. Oracle: the dense spectrum. Assert the returned
   pair matches *some* dense eigenvalue to 1e-10 **and** that the history does
   not claim it is the smallest. The companion assertion is the RQI case of
   §7.3: `refine='rayleigh'` from a cold start must raise, not return.

7. **Transverse-field Ising chain at criticality — a published closed form.**
   `H = -sum_{i=1}^{L-1} sigma^z_i sigma^z_{i+1} - sum_{i=1}^{L} sigma^x_i`
   (open chain, `g = 1`). Exact ground-state energy (Pfeuty's free-fermion
   solution of the open chain):

   ```
   E_0(L) = 1 - 1 / sin( pi / (2(2L+1)) )
   ```

   **Verified numerically here** against `numpy.linalg.eigvalsh` of the dense
   matrix: relative differences 0.0 (L=4), 1.99e-15 (L=8), 4.88e-15 (L=10),
   3.57e-16 (L=12). Assert `|E_0^computed - E_0(L)| <= 1e-8 |E_0(L)|` at
   `L = 10` (runnable today) and `L = 64` (after the new solver: the formula
   holds for any `L`, and no dense oracle exists there).

8. **Monotonicity of the Ritz values.** An invariant, no oracle: in an ALS
   sweep with orthonormal frames every local eigensolve is a Galerkin
   restriction, so `sum(lambda)` cannot increase. Assert
   `sum(lam_{s+1}) <= sum(lam_s) + 8 eps_machine |sum(lam_s)|` over the sweeps
   of `history.lam_per_sweep`. This is the cheapest possible detector of a
   broken interface convention, and the enrichment of §2.1 must not break it.

9. **Orthonormality of the returned block.** `y.r[-1] == B` and the `B` dense
   columns are orthonormal to `1e-10`. Already asserted for `eigb`
   (`test_block_vector_columns_are_orthonormal_after_eigb`); the new solver must
   pass the same test unchanged.

10. **`kickrank = 0` reproduces `eigb` bit-for-bit** on the QTT Laplacian
    `d = 8`, `B = 4`, same seed. The one place where comparing against our own
    code is the right test: it pins that the enrichment is the *only* difference.

11. **Backend parity.** The same run on numpy and torch CPU agrees to 1e-12
    (mirrors `tests/test_algs_torch.py`).

---

## 9. Hard test problems

Each is marked (i) runnable today, (ii) runnable after the new eigensolver, or
(iii) aspirational. "Reference" says where the number comes from and whether it
was verified here.

### 9.1 (i) QTT Laplacian, 1D and `D`-dimensional

`A = tt.qlaplace_dd([d]*D)` — the Kronecker sum of `D` Dirichlet Laplacians on
`2^d` points each, TT-matrix ranks `<= 3` per 1D block.
Reference: `lambda_{k_1..k_D} = sum_i 4 sin^2(k_i pi / (2(2^d+1)))`, analytic,
**verified here** to 4.4e-16 against the dense matrix at `d = 3`.
Parameters worth running: `D = 1, d = 8..30`; `D = 3, d = 5, B = 4`
(already a test); `D = 3, d = 10` (10^9 grid points, `B = 4`) as the scaling
case. Difficulty: the *conditioning*, `kappa = O(4^d)`, and nothing else — the
eigenvectors are products of sines and have QTT rank 2. Measured behaviour of
inverse iteration in §7.1.

### 9.2 (i) Transverse-field Ising chain at criticality

`H = -sum_{i=1}^{L-1} sigma^z_i sigma^z_{i+1} - g sum_i sigma^x_i`, `g = 1`,
open boundaries. TT-matrix rank **3** (explicit MPO, row index first):

```
W[0,0] = I     W[1,0] = -sigma^z    W[2,0] = -g sigma^x
W[2,1] = sigma^z                    W[2,2] = I
first core = W[2, :],  last core = W[:, 0]
```
Reference: `E_0(L) = 1 - 1/sin(pi/(2(2L+1)))`, **verified here** to 5e-15 for
`L = 4, 8, 10, 12` (see test 7). At `L = 10`: `E_0 = -12.381489999654814`, gap
`0.2989`. At criticality the half-chain entanglement entropy grows like
`(c/6) log L` with central charge `c = 1/2` for the open chain, so
the TT rank needed for fixed accuracy grows with `L` — this is the benchmark
that makes a *rank-adaptive* method visibly better than a fixed-rank one, which
is exactly the §2.1-versus-§2.2 question. Run at `L = 10` (dense oracle
available), `L = 64`, `L = 256`.

### 9.3 (i/ii) Heisenberg spin chain

`H = sum_i [ (S+_i S-_{i+1} + S-_i S+_{i+1})/2 + S^z_i S^z_{i+1} ]`, open,
TT-matrix rank **5**, real (the `S^x S^x + S^y S^y` form is written with raising
and lowering operators to stay in float64). Explicit MPO, same layout as §9.2,
`S+ = [[0,1],[0,0]]`, `S- = [[0,0],[1,0]]`, `S^z = diag(1/2, -1/2)`:

```
W[0,0] = I         W[1,0] = S-/2      W[2,0] = S+/2      W[3,0] = S^z
W[4,1] = S+        W[4,2] = S-        W[4,3] = S^z       W[4,4] = I
first core = W[4, :],  last core = W[:, 0]
core layout (R, i, j, R): cores[k] = transpose(W, (0, 2, 3, 1))
```

Verified element-by-element against an explicit Kronecker construction for
`L = 2, 3, 4, 6, 8, 10` (max abs difference exactly **0.0**).
References:
* `L = 10`: dense `numpy.linalg.eigvalsh`, `E_0 = -4.258035207283`, with exact
  SU(2) degeneracies (gaps `3.3e-01, 3.6e-15, 1.3e-15, 4.0e-01, ...`) —
  measured here, runnable today.
* `L = 16`: no dense oracle; `eigb` at `B = 2, 4, 8` agrees on
  `E_0 = -6.9117371456` to 1e-9 with a measured relative eigenresidual below
  5e-7 (§7.4). Usable as a cross-method reference, not as ground truth.
* `L = 40`, `b = 5` and `b = 35`: [RNO19] Table 4 compares LRRAP LOBPCG
  (`r = 20/35/45`) against `eigb` (`delta = 1e-2..1e-5`) and ALPS, but their
  *reference* energies are themselves `eigb` at `delta = 1e-5`, so the table
  gives **relative** MAEs (1.0e-4 / 1.2e-5 / 2.2e-6 for LRRAP at `b=5`;
  2.2e-4 / 2.4e-6 for `eigb` at `delta = 1e-2/1e-3`), not absolute truth. Use
  it to reproduce a *comparison*, never as a ground-truth table.
* Asymptotic sanity only: `E_0/L -> 1/4 - ln 2 = -0.4431471805599...` (Bethe
  ansatz, thermodynamic limit). Not a finite-`L` oracle.
Difficulty: degeneracy, and for `b >~ 30` the block-TT rank blow-up of §1.3.

### 9.4 (ii) Molecular vibrational spectra — Hénon–Heiles

`H = -1/2 sum_i omega_i d^2/dq_i^2 + V(q)` with the standard scalable potential
`V = 1/2 sum q_i^2 + lambda sum_{i=1}^{d-1} (q_i^2 q_{i+1} - q_{i+1}^3/3)`,
`lambda = 0.111803`. Discretize with a DVR on a tensor product of Hermite meshes
([RNO19] §6.1 does exactly this, citing Baye–Heenen): every term `q_i^2`,
`q_i q_j q_k` is a rank-1 TT-matrix, and the sum is assembled with
`round(1e-12)`. `d = 6..20`, `n = 15..30`.
Reference: published MCTDH/DVR tables for the Hénon–Heiles levels.
**Not verified here** — I do not have the tables and did not build the operator.
Do not add this test until the reference is in hand (§10).

### 9.5 (iii) Acetonitrile CH₃CN (`d = 12`) and ethylene oxide C₂H₄O (`d = 15`)

The benchmark of [RNO19] §6.1. Hamiltonian TT ranks from their Table 2
(CH₃CN: `5, 9, 14, 21, 25, 26, 24, 18, 15, 8, 5`; C₂H₄O:
`5, 11, 17, 21, 23, 25, 27, 28, 25, 23, 21, 16, 11, 5` — read from a
text-extracted table, transcription must be re-checked against the PDF).
`b = 84` levels for CH₃CN, `b = 35` for C₂H₄O; MAE against the reference
energies of Rakhuba–Oseledets (J. Chem. Phys. 145:124101, 2016) and of
Thomas–Carrington (J. Chem. Phys. 146:204110, 2017). Reported MAE for LRRAP:
0.4 cm⁻¹ at `r = 15`, 0.05 cm⁻¹ at `r = 25` (CH₃CN); 1.5 / 0.3 cm⁻¹ at
`r = 25/35` (C₂H₄O). Aspirational because the potential energy surfaces were
"kindly provided by the group of Prof. Tucker Carrington" and are not public.

### 9.6 (iii) Anderson localization

`H = -Delta_h + diag(V)` on a 1D QTT grid with `V` i.i.d. uniform on
`[-W/2, W/2]`. The diagonal of an i.i.d. random vector has **full** QTT rank by
construction, so the operator itself is not compressible — this is the honest
"tensor methods do not apply unless you change the question" case, and it
belongs in the test suite as a *documented refusal*: assert that building the
operator hits `rmax` and that the solver says so.

---

## 10. What I did not verify

* **[DKOS14] was not read.** The block AMEn eigensolver of §2.1 is derived from
  our own `eigb.py` and `amen.py`. The published algorithm may differ in where
  the enrichment is inserted, in whether the residual is compressed across the
  block index (§2.1 step 4), and in the stopping rule.
* **No prototype of §2.1 exists.** Every claim about block AMEn is structural,
  not measured. In particular the claim "it fixes the `B = 1` failure of §1.2"
  is an inference from what AMEn does for linear systems (measured, §7.2),
  not a measurement of the eigensolver.
* **No prototype of §2.2 or §2.3 exists** beyond the unpreconditioned steepest
  descent of §3.2. The `O(b^4)` coefficient problem of [RNO19] §5 was read but
  not implemented, and I cannot confirm its cost claim.
* **[RNO19]'s figures were not rendered** (no poppler on this host); Figs. 1–5
  are cited from their captions and the surrounding text only.
* **Table 2 of [RNO19] (Hamiltonian TT ranks) came out of a text extractor** and
  the digit grouping for C₂H₄O is ambiguous in the extracted stream. Re-read the
  PDF before using those numbers.
* **The Pauli matrices printed in [RNO19] §6.2 are inconsistent** in the
  extracted text (`S_x` is given as `diag(1,-1)/2` and `S_z` as the
  off-diagonal), which is a transcription artefact of the extractor or of the
  paper; our Heisenberg MPO uses the standard convention and was validated
  independently against a Kronecker construction.
* **No torch/GPU measurement.** Everything here is numpy on 2 CPU cores.
* **No measurement above `d = 30` QTT or `L = 16` spins**, and none with `B` above
  16 — the 4 GB of RAM and 2 cores on this host are the limit, not the method.
* **Every wall-clock number is a single run**, not a median of repeats; the
  `eigb` timings in §1.3 are visibly non-monotone in `B` for that reason (7.67 s
  at `B = 4` against 1.98 s at `B = 6`, same problem), so treat them as orders of
  magnitude and not as a benchmark.

---

## 11. Open questions

**Q1 — Is §2.1 the published block AMEn eigensolver?** See §10. The specific
points to check in [DKOS14]: (a) is the enrichment the residual `A x - x lambda`
or the preconditioned residual; (b) is it compressed across the block index, and
if so to what rank; (c) does the block index stay on the moving site (as `eigb`
does) or is it kept at a fixed core with the enrichment moving instead.

**Q2 — What rank does the enrichment actually produce, and does it help `B >= 2`?**
For `B = 1` the case is clear (§1.2 measures the failure it removes). For
`B >= 2` `eigb` already adapts ranks up to a factor `B` per half-sweep and was
measured at 1e-14 on every problem tried, so the enrichment may buy nothing but
cost `kickrank` extra rank at every bond. The experiment that settles it: TFIM
at criticality, `L = 64`, `B = 4`, `kickrank in {0, 2, 4, 8}`, comparing the
error against `E_0(L) = 1 - 1/sin(pi/(2(2L+1)))` at equal wall time.

**Q3 — What is the right stopping rule?** `ermax` (movement of `sum lambda`) is
measured to be worthless as an error indicator: §1.2 has `ermax = 8.1e-09` at a
point with a relative eigenresidual of 2.8e-02. The residual `||A y_i - lam_i y_i||`
is the right quantity and `eigb` already measures it — but only *once, at the
end*, because it costs a matvec and a sweep. Should it be the per-sweep
criterion (paying ~15 % per sweep, not measured), or should the local residual
of the enrichment be used as a surrogate (free, but it is a projected quantity
and I do not know how tight it is)?

**Q4 — How is `lambda_1 < 0` handled by the block truncation?** `eigb` truncates
the block with a *relative* Frobenius criterion `eps/sqrt(d) * ||s||`, where the
singular values mix all `B` columns. When the eigenvalues have wildly different
magnitudes (a QTT Laplacian: `lambda_1/lambda_B = 1/100` at `B = 16`) the
truncation is dominated by the largest-norm column. §4(h)'s absolute tolerance
is the obvious fix but I have not measured that the relative one actually hurts.

**Q5 — Can `eig_invit` choose its own shift?** §7.2 shows the cost varying by a
factor of 10 with the shift, and §7.3 shows the naive adaptive rule (RQI)
converging to the wrong eigenvalue. Gershgorin bounds on a TT-matrix are cheap
(`||A||_1` per row is a TT contraction) and would give a valid starting shift
below the spectrum; whether that is enough to make the fixed-shift iteration
competitive is an experiment, not a derivation.

**Q6 — Does the LRRAP single-tangent-space trick survive without TensorFlow's
batching?** [RNO19]'s whole cost argument rests on `b` tensors being processed as
one batched operation (their Appendix A), which is what makes the `O(b^2 d n r^2)`
Gram matrices actually fast. In numpy, `b = 84` separate small `einsum` calls per
core per iteration is a dispatch-bound loop of the kind
`docs/plans/bug-integrator.md` §6 measures at ~7 µs per call. Whether the method
is worth building for a numpy backend at all is unresolved; on torch it would
batch naturally.

**Q7 — Deflation instead of blocking.** Every method here computes `B`
eigenvalues at once. The alternative — compute one, deflate, repeat — needs
`orthogonalize_against(x, [v_1..v_k])` in TT with rounding, and its convergence
degrades when the spectrum is clustered ([RNO19] §1 asserts this without
measuring it). With `eig_invit` already built, this is a cheap experiment on the
Heisenberg chain, whose first excited state is *exactly* triply degenerate.
