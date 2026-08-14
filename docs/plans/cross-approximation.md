# CROSS: cross approximation in the TT format — what to take from `ttcross` and from `teneva`

Owner of this document: the black-box side of ttpy2 — `tt/algs/cross.py`,
`tt/algs/maxvol.py`, and everything that goes through them
(`tt/algs/multifuncrs.py`, `tt/algs/qtt_ell.py::invert/sqrt`,
`tt/algs/completion.py`'s index bookkeeping). It owns no mathematics that
`docs/plans/ROADMAP.md` has already assigned elsewhere; where it touches the
rounding tolerance or the `history` contract it defers to `docs/REQUIREMENTS.md`
R4/R7.

**Sources actually read.**

| what | where | state |
|---|---|---|
| ttpy2 | `tt/algs/cross.py` (583 lines), `tt/algs/maxvol.py` (644), `tt/algs/_indexing.py` (220), `tt/algs/multifuncrs.py` (644) | read in full, run |
| ttcross | `github.com/savostyanov/ttcross` @ `9254e82` (2024-02-10): `dmrgg.f90` (936), `lr.f90`, `ttind.f90`, `rnd.f90`, `main.f90`, `README.md` | **read in full; built and run** on an M-series Mac (gfortran 15 + Accelerate + a single-process MPI stub; §2.1a) — b300 still has no toolchain. Measured numbers in §2.1a. |
| teneva | `github.com/AndreiChertkov/teneva` @ `5be12bc` (v0.14.11): `maxvol.py` (137), `cross.py` (287), `cross_act.py` (277), `utils.py`, `core.py` | read in full; **installed and run** on b300 (`uv pip install teneva`, 0.14.11, numpy 2.5.1) |
| legacy ttpy | `~/work/ttpy-modern/ttpy-src/tt/cross/rectcross/rect_cross.py` | read and run (micromamba `ttlegacy`, numpy 1.24.4) |

**Ground rule for numbers.** Every number below was produced by a script in
`b300:~/work/ttpy-modern/scratch-cross/` (`e1.py` … `e13.py`, §9) run with
`~/work/ttpy-modern/ttpy2/.venv/bin/python` (numpy 2.4.6, scipy 1.18.0) or, for
the teneva half, with `scratch-cross/tvenv/bin/python` (numpy 2.5.1, teneva
0.14.11, ttpy2 on `PYTHONPATH`). Nothing was run locally. Anything not measured
says "not measured".

**Baseline.** `pytest tests/ -q -n 8` on b300: **801 passed in 59.40 s**. That is
what every change below must leave green.

---

## 0. The short version

1. **Our `maxvol`/`rect_maxvol` are better than teneva's on every axis measured**
   — same pivots, same volume, same `K`, same row-norm bound, but **3.0×** faster
   for square maxvol and **18.0×** faster for rectangular maxvol at
   $N = 100000, r = 200$, and they raise where teneva silently returns a matrix
   full of `NaN`. Do not port teneva's maxvol. Port nothing from it. (§3.1, §4.1)
2. **Our cross throws away 60–76 % of its function evaluations by re-asking for
   entries it already has.** teneva's `cache` (`cross.py::_func_eval`) removes
   exactly that. Measured reduction in distinct evaluations: **2.5×** (smooth
   `d=5`), **3.1×** (QTT `d=20`), **4.2×** (kink `d=6`). It costs ~1.25 µs per
   probe in the prototype, so it is a *loss* for a black box cheaper than that
   and a **2.2× wall-clock win** for one at 50 µs/point. Must be opt-in. (§4.3, §5 P1)
3. **`tt.cross` returns a confidently wrong answer on a two-line function, with
   every indicator we own reading ~1e-15.** $f(i) = 1/(10^{-2} + |\sum_k i_k/9 - 5/2|)$,
   `d = 6`, `n = 10`, `eps = 1e-8`: true relative error **3.82e-04**,
   `converged=True`, `err_rel=2.9e-15`, `err_round=3.5e-15`,
   `err_check` on 4000 held-out points `3.7e-15`, **no warning**. It hits 5 of 6
   initial-guess seeds under numpy 2.4.6 and 4 of 6 under numpy 2.5.1 — *which*
   seed fails is roundoff. The legacy Fortran-era `tt.cross.rectcross` does the
   same (3.82e-04, 1.81e-06 on two of four seeds) and **so does teneva**
   (3.82e-04 on one of six seeds). This is a defect of the whole rect-maxvol
   cross family, not a ttpy2 regression. (§1.3)
4. **The cure is the one thing `multifuncrs.py`'s docstring currently says we
   reject**: a handful of *uniformly random* extra rows per micro-step
   (`kickrank2` in the legacy `multifuncrs2`, `dr2` in `teneva.cross_act`).
   `kickrank2 = 2` fixes **6 of 6** seeds on that problem (3.82e-04 → 2.84e-10)
   for 1.6–2.5× more distinct evaluations. The docstring's rejection is a
   projection of the concept that now lies and must be updated with the code.
   (§1.3, §5 P2)
5. **`rect_maxvol stopped at K = …` is a false alarm** and accounts for **80 of
   the 93** `RuntimeWarning`s the test suite emits. Measured cost of the
   early stop: **zero** — identical error, identical ranks, on three problems
   across `rf ∈ {0,2,5,10,30}`. It is `cross._select_rows` asking for a budgeted
   `K` and then being told off for respecting the budget. Silence it at that call
   site, exactly as `maxvol(warn_nonconvergence=False)` already does. (§1.2, §5 P3)
6. **From ttcross, port the error indicator, not the algorithm.** ttcross's
   greedy DMRG cross is a different animal (rank +1 per bond per sweep, MPI over
   bonds, rook pivoting on the *residual*), and porting it wholesale is a second
   engine we would have to keep correct. What is cheap and independently valuable
   is its stopping rule — `pivotmax <= accuracy * amax` sustained for **three
   consecutive sweeps** (`strike >= 3`, `dmrgg.f90:661-665`) — and its
   `dtt_accchk`, which reports the held-out error in **both** the ∞-norm and the
   Frobenius norm. (§3.2, §5 P4, §5 P5)
7. **Negative, and measured:** the post-hoc rook/alternating-maximisation
   residual probe does **not** catch the §1.3 failure (0/50 at 120 evaluations
   vs 2/50 for uniform Monte Carlo at the same budget), and the "one-sweep
   collapse of `err_rel`" signature is **not specific** (a tensor of exact TT rank
   3 collapses by 4.8e+14 in one sweep, correctly). Both were prototyped and both
   are rejected. (§5, "what not to take")

---

## 1. What we have, honestly

### 1.1 The algorithm in `tt/algs/cross.py`, stated precisely

One-site alternating cross with rectangular-maxvol row selection.

* `_init_right_indices` (`cross.py:259`) orthogonalises `x0` right-to-left and
  runs *square* maxvol on each unfolding to get the starting right sets `J_k`.
* `_sweep_lr` (`cross.py:282`) walks `k = 0 .. d-2`. It asks `fun` for the whole
  block $I_k \times n_k \times J_k$ ($r_1 \cdot n_k \cdot r_2$ values), takes an SVD basis of the
  $(r_1 n_k) \times r_2$ unfolding — `_left_basis`, **never truncated**, deliberately, see
  its docstring — and calls `_select_rows`, which runs
  `rect_maxvol(q, tau, maxK = min(npts, rho + kickrank + rf, max(rmax, rho)), min_add_K = kickrank)`.
  The chosen rows extend `I_{k+1}`. **The values themselves are discarded.**
* `_sweep_rl` (`cross.py:296`) walks back, asks for the same shaped blocks, and
  this time also builds the cores from `C = Q pinv(Q[ind])` (`_interp`).
* Stop: `||y − y_prev|| <= max(eps ||y||, eps_abs)`. Then `round(eps)`.

Cost per sweep: $2 \sum_k r_k n_k r_{k+1}$ function values, $O(d n r^3)$ flops.
Rank growth per bond per sweep: at most `kickrank + rf`, and capped from above by
the *other* side's set size, because `rho = min(q.shape[1], rmax) = min(r1 n_k, r2)`.

Public entry points: `rect_cross`, `cross`, `greedy_cross` (an alias for
`rect_cross`), `element`, `CrossHistory`.

### 1.2 The `rect_maxvol stopped at K = …` warning: measured, cosmetic, harmful anyway

It fires 80 times in the 801-test suite (86 % of all `RuntimeWarning`s), always
in the same shape:

```
tt/algs/cross.py:232: RuntimeWarning: rect_maxvol stopped at K = 29 (limit maxK = 29)
with a remaining candidate row norm 1.17167 > tol = 1.1.
```

**Where it comes from.** `_select_rows` computes an explicit budget
`max_k = min(npts, rho + kickrank + rf, max(rmax, rho))` and passes it as `maxK`.
`rect_maxvol` then stops because `K == maxK`, sets `stop_reason = "maxK"`,
`converged = False`, and warns (`maxvol.py:510-520`). The budget is the caller's
deliberate choice; the warning is `rect_maxvol` complaining about being obeyed.

**What the early stop costs, measured (E1a).** Re-running each captured
`rect_maxvol` call from `multifuncrs2([x, y], x/(1+y²))` with the budget removed
(`N = [10,12,14,9,11]`, `eps = 1e-8`, float64, b300):

| basis shape | `K` with budget | max‖C_i‖ | `K` free | max‖C_i‖ free |
|---|---|---|---|---|
| (378, 22) | 29 | 1.1240 | 30 | 1.0896 |
| (99, 29) | 36 | 1.2445 | 39 | 1.0764 |
| (504, 27) | 34 | 1.2088 | 37 | 1.0988 |

3 of 20 calls stop early; 1–3 extra rows would have satisfied `tau = 1.1`; the
interpolation constant is 1.24 instead of 1.10.

**What it costs end to end, measured (E1b).** Sweeping `rf` (the slack that
decides whether the warning fires at all), float64, b300, `kickrank=5`:

| problem | `rf` | warnings | error vs dense | ranks | `fun_eval` |
|---|---|---|---|---|---|
| `sin(s)/(1+s)`, `d=5`, `n=12`, `eps=1e-10` | 0 / 2 / 5 / 10 / 30 | 2 / 1 / 0 / 0 / 0 | 5.861e-11 (all five) | `[1,8,8,8,8,1]` (all five) | 54720 … 55764 |
| QTT `1/(1+t)`, `d=20`, `eps=1e-10` | 0 / 2 / 5 / 10 / 30 | 12 / 5 / 0 / 0 / 0 | `err_check` 3.471e-11 (all five) | max rank 5 (all five) | 18282 … 19262 |
| kink `1/(1e-2+|s−5/2|)`, `d=6`, `n=10`, `eps=1e-8` | 0 / 2 / 5 / 10 / 30 | 3 / 0 / 0 / 0 / 0 | 2.843e-10 (all five) | max rank 26 (all five) | 600820 … 610030 |

**Conclusion.** The accuracy cost is *zero to every digit printed*; the only cost
of silencing it by raising `rf` is 1.5–5.4 % more function evaluations. So the
warning is cosmetic — and that is precisely what makes it a defect: it is 86 % of
the warning traffic on a channel whose entire job is to be believed when
`tt cross:` or `maxvol did not converge` really does fire. Warning fatigue on the
loud-failure channel is a violation of the "unexpected states fail loud" contract
in the other direction. Fix in §5 P3.

### 1.3 A concrete input where our cross is confidently wrong

**Reproducer** (b300, `.venv/bin/python`, numpy 2.4.6, float64):

```python
import numpy as np, tt
from tt.algs.cross import cross
d, n = 6, 10
f = lambda I: 1.0 / (1e-2 + np.abs(I.sum(1) / 9.0 - 2.5))
y = cross(f, n, d, eps=1e-8, r=2, seed=0, kickrank=2, n_check=4000, check_seed=7)
# no warning is emitted
```

The dense oracle is `np.meshgrid`-built and independent of TT:
$1/(10^{-2} + |\sum_k i_k/9 - 5/2|)$ on the full `10**6` grid.

| what the run reports | value | the truth |
|---|---|---|
| `history.converged` | `True` | — |
| `history.err_rel` (change between the last two sweeps) | 2.9e-15 | — |
| `history.err_round` (exact, measured) | 3.5e-15 | — |
| `history.err_check` on 4000 held-out points | 3.7e-15 | — |
| warnings emitted | none | — |
| relative Frobenius error against the dense oracle | — | **3.82e-04** |
| relative ∞-norm error against the dense oracle | — | **9.24e-03** |

**Every number the API can produce is ~1e-15; the answer is 3.8e-4 wrong at a
requested `eps` of 1e-8.**

**How reproducible.** `kickrank=1`, `rf=2`, `eps=1e-8`, seeds 0–5 of the random
`x0`, numpy 2.4.6: seed 0 gives 2.84e-10 (correct); **seeds 1, 2, 3, 4, 5 all
give 3.82e-04**, all with `converged=True` and `err_rel` between 2.3e-15 and
3.1e-15. At `seed=0`, `kickrank ∈ {2, 3}` gives 3.82e-04 and
`kickrank ∈ {1, 5, 8}` gives 2.84e-10 — i.e. which side of the cliff you land on
is not a monotone function of the exploration budget.

**And it is not reproducible across numpy versions** (E14 — the same script under
both interpreters, `kickrank=1`, spelling of `n` and wrapping of `fun` held
invariant, which changes nothing):

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| numpy 2.4.6 (`.venv`) | 2.84e-10 | **3.82e-04** | **3.82e-04** | **3.82e-04** | **3.82e-04** | **3.82e-04** |
| `fun_eval` | 704650 | 394930 | 396910 | 230930 | 283610 | 270840 |
| numpy 2.5.1 (`tvenv`) | 2.84e-10 | 3.84e-10 | **3.82e-04** | **3.82e-04** | **3.82e-04** | **3.82e-04** |
| `fun_eval` | 572170 | 622400 | 316950 | 367180 | 518020 | 295170 |

Seed 1 lands on the other side of the cliff, and the evaluation counts differ by
up to 1.8× on runs that reach the same answer. Nothing in the code is
non-deterministic; the divergence is roundoff in the LAPACK the two numpy builds
call, amplified by a maxvol pivot choice that then changes every subsequent
index set. **A test written against a specific seed of this problem is a test of
a coin flip** — which is why T3 in §7 asserts on a *majority* of seeds and T4
asserts a disjunction rather than a value.

**Mechanism.** Per-sweep history at `seed=1`, `kickrank=1`:

```
swp  0 err_rel=3.84e+00 max_rank= 4      swp  7 err_rel=2.98e-01 max_rank=21
swp  1 err_rel=8.76e-01 max_rank= 6      swp  8 err_rel=1.73e-01 max_rank=26
swp  2 err_rel=7.71e-01 max_rank= 8      swp  9 err_rel=1.34e-01 max_rank=32
swp  3 err_rel=7.61e-01 max_rank=10      swp 10 err_rel=9.43e-02 max_rank=38
swp  4 err_rel=6.74e-01 max_rank=12      swp 11 err_rel=5.31e-02 max_rank=44
swp  5 err_rel=5.19e-01 max_rank=15      swp 12 err_rel=1.51e-02 max_rank=50
swp  6 err_rel=3.47e-01 max_rank=17      swp 13 err_rel=3.06e-15 max_rank=56
```

The index sets reach a **fixed point**: at sweep 13 the cross interpolant on the
current sets reproduces itself exactly, so `err_rel` collapses by 13 orders of
magnitude in one sweep. The interpolant is then rounded from rank 56 to rank 24 —
`round(eps)` correctly reports that rank 24 represents *the interpolant* to
1e-15. Nothing in that chain ever looks at the function outside the sampled
fibers, so nothing sees that the interpolant is 3.8e-4 from it.

**Where the error lives** (E3b, seed 1). 99 % of the squared error sits on **394
of the 1000000 entries** (0.039 %); 50 % on 145 entries. The worst entry is the
grid corner `(9,9,9,9,9,9)`, where `f = 0.2849` and the interpolant gives
`0.14396`. It is not a spike in the *function* — `f` is perfectly smooth there —
it is a region the index sets never reached, because the maxvol pivots all sit
near the peak at $s \approx 5/2$ where $|f|$ is 50× larger.

**Why `err_check` missed it.** With 394 bad entries out of `10**6`, a uniform
sample of $m$ points misses the region with probability $(1 - 3.94\times 10^{-4})^m$.
Measured detection rate over 200 independent samples, threshold `10*eps = 1e-7`
(E11):

| `n_check` | 200 | 500 | 1000 | 2000 | 4000 | 10000 |
|---|---|---|---|---|---|---|
| detects (Frobenius) | 7.5 % | 21.5 % | 36.5 % | 56.0 % | 77.5 % | 97.0 % |
| detects (∞-norm) | 7.5 % | 21.5 % | 36.5 % | 56.0 % | 77.5 % | 97.0 % |
| median reported (Frobenius) | 5.3e-15 | 5.4e-15 | 5.4e-15 | 3.6e-04 | 3.6e-04 | 3.8e-04 |
| median reported (∞-norm) | 6.0e-15 | 7.0e-15 | 7.8e-15 | 5.0e-03 | 6.5e-03 | 8.1e-03 |

Two honest readings of that table. (a) `n_check` is not a safety net at its
natural sizes: at 1000 points it is a coin-flip that lands on "everything is
fine" 63 % of the time. (b) **The ∞-norm buys no extra detection** — the
detection event is "did the sample hit the bad region at all", and once it does,
both norms are far above the threshold. What the ∞-norm buys is a 24× larger
*number* once detected, which is a better description of the damage, not a better
detector. Do not oversell it (§5 P5).

**This is inherited, not ours.** The legacy Fortran-era
`tt.cross.rectcross.cross` (numpy 1.24.4, `~/work/ttpy-modern/ttpy-src`), same
problem, `kickrank=1`, `eps=1e-8`:

| seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| relative error | **3.82e-04** | 8.19e-10 | 2.84e-10 | **1.81e-06** |
| max rank | 24 | 26 | 26 | 25 |

and **teneva** (v0.14.11, `dr_min=1, dr_max=2, e=1e-8`, then `truncate(1e-8)`)
gives **3.82e-04** on one of six seeds (§4.4). Same fixed point, same number.
It is a property of maxvol-driven rank-adaptive cross on this function.

**What fixes it.** Uniformly random extra rows per micro-step. Prototype (E5):
patch `_select_rows` to append `kickrank2` rows drawn without replacement from
the not-yet-selected rows. `kickrank=1`, `eps=1e-8`, seeds 0–5:

| seed | `k2 = 0` | `k2 = 2` | `k2 = 5` | `k2 = 10` |
|---|---|---|---|---|
| 0 | 2.84e-10 | 2.84e-10 | 2.84e-10 | 2.84e-10 |
| 1 | **3.82e-04** | 2.84e-10 | 2.84e-10 | 2.84e-10 |
| 2 | **3.82e-04** | 2.84e-10 | 2.84e-10 | 2.84e-10 |
| 3 | **3.82e-04** | 2.84e-10 | 2.84e-10 | 2.84e-10 |
| 4 | **3.82e-04** | 2.84e-10 | 2.84e-10 | 2.84e-10 |
| 5 | **3.82e-04** | 2.84e-10 | 2.84e-10 | 2.84e-10 |

6/6 correct at `k2 = 2`; the remaining 2.84e-10 is the accuracy of the rounded
interpolant, not of the sampling. Cost, at `k2 = 2` with the cache on (E10):
176832–229063 distinct evaluations against 88641–162106 without — **1.4–2.5×**.

`tt/algs/multifuncrs.py`'s module docstring currently states, as a design
decision, that "`kickrank2` (extra *random* rows) has no counterpart and is
rejected". That sentence is now contradicted by a measurement and is a projection
of the concept that lies; it must change in the same commit as the code (R7 /
IMMUNE-M).

### 1.4 What is good, and should not be touched

* `maxvol`/`rect_maxvol` themselves: fastest of the three implementations
  compared, with the strongest safeguards (§4.1).
* The `_left_basis` non-truncation decision and its docstring: the 12 %-error
  fixed point it describes is real and the reasoning is right.
* `_evaluate`'s refusal of non-vectorized `fun`, of a wrong-length return and of
  non-finite values; `element`'s refusal of negative indices; `_host_matrix`'s
  refusal of `NaN`. teneva has none of these and pays for it (§4.1 C).
* `CrossHistory` as a concept (R7). What is missing from it is listed in §6.
* The "no cross method can certify its own accuracy" note at `cross.py:426`.
  §1.3 is the *sharper* version of that note: the blind spot is not only spikes,
  it is any region the pivots avoid because $|f|$ is small there.

---

## 2. The two references

### 2.1 ttcross (Savostyanov) — read, not built

`dtt_dmrgg` (`dmrgg.f90:10-691`) is a **greedy DMRG cross**, i.e. the algorithm
of Savostyanov, *Quasioptimality of maximum-volume cross interpolation of
tensors*, LAA 2014, in the parallel form of Dolgov & Savostyanov,
arXiv:1903.11554. It is not our algorithm with different constants; it is a
different construction.

**State.** For each bond `p` it stores `vip(p)%p(4, r_p)` — for every rank index,
the quadruple `(i, j, k, q)` = (index into the left set at `p−1`, mode-`p` index,
mode-`p+1` index, index into the right set at `p+1`). A full multi-index is
reconstructed by walking that linked list (`dmrgg_fun`, `dmrgg.f90:694-707`).
The sets are therefore **nested by construction**, and the cross submatrix
`A(I_p, J_p)` is carried as a **growing LU factorisation** `inv(p)`, bordered by
one row and one column per added pivot (`dmrgg.f90:386-390`) and applied by
`d2_lual`/`d2_luar` (`lr.f90`). No inverse is ever recomputed.

**Rank adaptation.** Exactly **+1 per bond per sweep**. Iteration `it` is the
rank. There is no SVD anywhere in the sweep.

**Pivoting** — the pivot is the argmax of the *residual* `A − col·row` over a
candidate set, and there are three regimes selected by `pivpar`:

| `pivoting` | candidate set | evaluations per bond per sweep |
|---|---|---|
| `-1` (full) | the entire two-site superblock $r_{p-1} n_p n_{p+1} r_{p+1}$ | $r^2 n^2$ |
| `0` | a random "lottery" of $r_{p-1}+n_p+n_{p+1}+r_{p+1}$ entries, then one full column and one full row through the winner | $\approx 2 n r$ |
| $p \ge 1$ (rook) | as `0`, then alternate: maximise the residual along the column, then along the row, until a fixed point or $2p$ crossings | $\approx 2p \cdot 2 n r$ |

The lottery is a weighted sample (`lottery2`, `rnd.f90:105-126`) whose weights are
1 for every `(i,j)` not already a pivot and 0 for those that are — i.e. uniform
over the not-yet-selected entries. Rook pivoting is `dmrgg.f90:298-359`.

**Pivot rejection.** A pivot is only accepted if

```fortran
upd(p) = (abs(pivot) > small_element*amax) .and. (abs(pivot) > small_pivot*pivotmax_prev)
```
(`dmrgg.f90:369`), with `small_element = 10·eps` and `small_pivot = 1e-5` for
float64 (`dmrgg.f90:53`). `amax` is the largest $|A|$ seen anywhere so far,
`pivotmax_prev` the largest residual of the previous sweep. A tiny pivot is not
added — the rank does not grow rather than growing by a numerically meaningless
direction. **We have no analogue**: `_left_basis` takes the full numerical rank
unconditionally.

**Error indicator and stopping rule.** `pivotmax` = the largest `|residual|`
found this sweep, over all bonds. Stop when

```fortran
if (pivotmax <= accuracy*amax) strike = strike + 1 else strike = 0
ready = ready .or. (strike >= 3)
```
(`dmrgg.f90:661-665`). Three properties worth naming: it is a **max-norm residual
on freshly probed entries**, not a difference between iterates; it is normalised
by `amax`, the largest value ever seen, not by a norm of the iterate; and it must
hold for **three consecutive sweeps**, which is what makes a single lucky sweep
insufficient.

**Accuracy check.** `dtt_accchk` (`dmrgg.f90:708-752`) samples `nlot` uniform
random multi-indices and returns four numbers — `einf, efro, ainf, afro` — so the
caller reports **both** `einf/ainf` and `efro/afro`, and optionally the
multi-index of the worst entry (`pivot` argument). `main.f90` uses `nlot = 2**20`.

**Quantity of interest.** The optional `quad` argument is a TT of quadrature
weights; the sweep loop then also reports the running value of $\langle w, X\rangle$ and its
relative change `|1 − val/val_prev|` per sweep (`dmrgg.f90:634-657`). For the
Ising susceptibility integrals the code targets, the convergence of the
*integral* is the thing that matters, and it is monitored directly.

**Parallelism.** MPI over bonds (`own(0:nproc)`), OpenMP over the entries of a
fiber; ranks and pivots are broadcast by a "tape" of `(i,j,k,q)` quadruples after
every sweep. Requires `nproc < d`. Irrelevant to a pure-Python port and not
considered further.

**Precision.** The point of the repository is very high accuracy: it compiles at
`-fdefault-real-8/16` and ships an MPFUN/MPFR multiple-precision variant
(`dmrggmp.f90`, `mpblas.f90`). We are float64/float32 and out of scope here.

### 2.1a Built and measured (2026-08-09, Apple M-series)

The "not built" above stopped being true on a machine with a Fortran toolchain.
Build: gfortran 15 (brew) + Accelerate as LAPACK/BLAS + a ~100-line
single-process MPI stub (`mpif.h` with size-encoding datatype constants and a C
file implementing init/rank/size/barrier as no-ops, allreduce/reduce/bcast as
memcpy honouring `MPI_IN_PLACE`, send/recv as loud aborts — with one process the
neighbour exchanges are never reached, and the stub proves it by construction).
Three fixes to our stub header along the way (`MPI_2DOUBLE_PRECISION`,
`MPI_MAXLOC`, `MPI_PROD`); the ttcross sources themselves compiled untouched
with `-fallow-argument-mismatch`.

**Protocol.** `test_crs_ising.exe KIND m 65 RANK 1` against ttpy2 `rect_cross`
on the **identical discretized tensor**: $m-1$ variables, 65 Gauss–Legendre
nodes on $[0,1]$, weights halved to a measure, scaled by $n/2$ and baked into
the tensor entries exactly as the Fortran driver does; the integral is the
contraction with the rank-1 tensor of $(2/n)$'s, and the reference is Bailey's
constant embedded in `test_crs_ising.f90`. Both at `OMP_NUM_THREADS=4`, ttcross
single-process, ttpy2 at default `kickrank=1, rf=2`, numpy/Accelerate float64.

| problem | engine | knob | digits | evaluations | wall |
|---|---|---|---|---|---|
| C_6 | ttcross | rank 5 | 5.55 | 8 205 | 5 ms |
| C_6 | ttpy2 | eps 1e-5 | 7.13 | 71 110 | 40 ms |
| C_6 | ttcross | rank 10 | 7.54 | 26 315 | 8 ms |
| C_6 | ttpy2 | eps 1e-7 | 9.22 | 454 935 | 183 ms |
| C_6 | ttcross | rank 20 | 10.75 | 89 001 | 20 ms |
| C_6 | ttcross | rank 40 | 10.90 | 89 133 | 33 ms |
| C_6 | ttpy2 | eps 1e-11 | 13.63 | 1 092 000 | 415 ms |
| C_16 | ttcross | rank 20 | 11.19 | 88 671 | 25 ms |
| C_16 | ttpy2 | eps 1e-11 | 13.55 | 3 649 490 | 1.8 s |
| D_6 | ttcross | rank 25 | 10.39 | 88 275 | 20 ms |
| D_6 | ttpy2 | eps 1e-11 | 13.19 | 1 232 140 | 519 ms |
| E_6 | ttcross | rank 20 | 11.25 | 87 087 | 23 ms |
| E_6 | ttpy2 | eps 1e-11 | 12.62 | 1 111 565 | 400 ms |

**What the numbers say, in both directions.**

1. **At equal accuracy ttcross spends 5–10x fewer function evaluations** on
   these integrands, and its Fortran core is another order of magnitude faster
   per evaluation — 20 ms against 415 ms end-to-end at the ~11-digit level.
   This settles §4.2's "neither is uniformly cheaper in evaluations" for the
   Ising family: here the greedy +1-per-bond growth with rook pivoting on the
   residual **is** uniformly cheaper. The mechanism is visible in the counts:
   its evaluations saturate near 89k (the pivot-value stopping rule fires)
   while ours keep scaling with the requested `eps`.
2. **ttcross saturates at ~11 digits regardless of rank** (rank 40 buys 0.15
   digits over rank 20; its `accuracy = 500*eps_machine` and the three-strike
   pivot rule stop it there), while `rect_cross` pushed the same tensors to
   12.6–13.6 digits. The ceiling is the stopping rule's, not the method's —
   but as shipped, the last two digits belong to us.
3. ~~Dimension scaling favours the greedy growth~~ — **withdrawn, it was a
   lucky run.** The C_16-at-89k row below is real but atypical: the reference's
   lottery is unseeded and its accuracy rule is always armed, so occasionally
   the pivots freeze and it strikes out early (~88k evaluations, 11.19
   digits). Seven repeats measured later: **every one ran the full 19 sweeps
   at ~379.5k evaluations in 89–118 ms** — the same schedule as the port's.
   The greedy's per-sweep cost grows linearly in `d` for everyone.
4. The §5 verdict — port the indicator, not the engine — was overtaken by
   this table: the 5–40x evaluation gap on smooth integrands was judged worth
   a second engine, and the port was ordered and done the same day
   (`tt/algs/dmrg_cross.py`, 2026-08-09). The three-strike rule (P4), the
   two-norm held-out check (P5) and the pivot rejection (P6) ship inside it.

### 2.1b The port, measured against the table above

`tt.dmrg_cross` (`tt/algs/dmrg_cross.py`), same protocol, same machine, same
`OMP_NUM_THREADS=4`:

| problem | knob | Fortran ttcross | ttpy2 port |
|---|---|---|---|
| C_6 | rank 5 | 5.55 digits @ 8 205 | 4.94 @ **8 205** |
| C_6 | rank 10 | 7.54 @ 26 315 | 7.93 @ **26 315** |
| C_6 | rank 20 | 10.75 @ 89 001 | 11.17 @ 89 199 |
| C_6 | acc 1.1e-13 | 10.90 @ 89 133 (saturated) | **14.68** @ 288 631 |
| C_16 | rank 20 | 11.19 @ 88 671 | 14.05 @ 379 605 |
| D_6 | rank 25 | 10.39 @ 88 275 | 11.80 @ 130 079 |
| E_6 | rank 20 | 11.25 @ 87 087 | 12.02 @ 86 955 |

The evaluation counts at matched rank caps agree with the Fortran **to the
evaluation** on C_6 (8 205 and 26 315 exactly): the port reproduces the
original's evaluation schedule, not merely its idea. Two deliberate
differences show in the other rows. The ~11-digit ceiling of §2.1a turns out
to belong to the reference *build*, not the algorithm: with the same stopping
constant (their hard-coded `500*eps_machine`) the port reaches 14.7 digits,
because the cross matrix is factorised by LAPACK LU with partial pivoting
instead of the original's unpivoted bordered LU. And on C_16 the port spends
4.3x more evaluations for 2.9 more digits — a different point on the same
frontier, traceable to argmax tie-breaking in the rook search. Wall time is
37–350 ms against the Fortran's 20–30 ms: pure Python overhead, same ratio as
everywhere else in the package.

**Wall clock: the port is now faster than the Fortran per evaluation.**
Three rounds of measured work got it there (each profiled, best-of-5 timings,
`OMP_NUM_THREADS=4`, both binaries on the same cores):

1. vectorising the two Python-loop hotspots (lottery draw via cumsum +
   searchsorted — which is what `lottery2` does anyway — and the point-index
   assembly): 37 -> 26 ms on C_6;
2. raw LAPACK `getrf`/`getrs` instead of the scipy wrappers, whose per-call
   `check_finite` scans and Python overhead dwarfed the $r \times r$ solve;
3. the structural one: **never form the full $M^{-1} R$**. The port used to
   apply the interpolant by solving for the whole block once per bond visit,
   $O(r^2 n r_2)$ flops where the original's incremental factors pay
   $O(r n)$ per fiber; solving only for the one column, one row or one
   scattered batch actually needed removes the gap entirely. An explicit
   $M^{-1}$ cached per bond was tried first and **reverted**: multiplication
   by an inverse is not backward stable, the residual noise floor rises from
   $\varepsilon$ to $\mathrm{cond}(M)\,\varepsilon$, and the pivot-acceptance threshold (calibrated for
   a backward-stable residual) starts accepting duplicate pivots, which makes
   $M$ exactly singular — 6 tests caught it.

With the example's integrand numba-compiled (the Fortran driver's integrand
is compiled too — equal footing; the numpy fallback computes identical values
to 1 ulp), best of 5 against the Fortran's best of 3:

Medians over 7 runs (Fortran: 7 repeats, unseeded lottery) / 7 lottery seeds
(port), **identical stopping rules on both sides** (rank cap 20, accuracy
``500*eps_machine`` with three strikes -- the port run with exactly the
driver's configuration; every one of the port's 21 runs stopped at the rank
cap, as did 20 of the Fortran's 21):

| rank 20, medians [min..max] | evaluations | digits | wall |
|---|---|---|---|
| C_6 Fortran | 88 935 [88 869..89 067] | 11.14 [10.21..11.83] | 26.2 ms [22.7..31.3] |
| C_6 port, numpy | 89 001 [88 803..89 199] | 10.80 [10.29..11.17] | 22.6 ms [20.6..29.1] |
| C_6 port, compiled | same | same | **13.1 ms** [11.2..] |
| E_6 Fortran | 87 087 [86 955..87 285] | 11.86 [11.02..12.40] | 22.8 ms [21.9..26.2] |
| E_6 port, numpy | 87 219 [86 955..87 483] | 11.59 [10.94..12.43] | 24.2 ms [21.9..25.7] |
| E_6 port, compiled | same | same | **13.0 ms** [11.6..] |
| C_16 Fortran | 379 473 [..379 671] | 11.69 [11.43..12.12] | ≈94 ms [89..118] |
| C_16 port, numpy | 379 473 [379 209..379 605] | 11.68 [11.43..14.06] | 96.8 ms [87.1..123.1] |
| C_16 port, compiled | same | same | **59.2 ms** [53.6..] |

What the medians say: the evaluation schedules are **statistically
identical** (medians differ by under 0.15%, well inside each side's own
lottery spread), the delivered digits are statistically identical too, the
numpy engine runs at wall-clock parity with the compiled Fortran, and the
**compiled path** (`tt/algs/_dmrg_fast.py`, engages automatically when
``fun`` is a numba dispatcher — one jitted kernel per bond visit,
hand-written LAPACK-convention triangular solves, identical lottery draws to
the numpy path) is **1.6–2.0x faster than the Fortran**.

Three earlier single-shot claims died on the way to these medians and are
recorded as dead: the C_16 row was first published with a Fortran wall of
15.6 ms and a "4.3x smaller evaluation count" story — that was a rare
early-strike-out of the reference's always-armed accuracy rule under its
unseeded lottery (11.19 digits), not its typical cost; the port's "+2.9
digits on C_16" was one lucky seed of ours (the [11.43..14.06] spread above);
and "the numpy path is 8–14% faster per evaluation" from best-of timings is
inside session noise — parity is what the medians support.  The eps-driven
regime is a different story: at the driver's own ``500*eps_machine`` constant
*without* a rank cap the reference strikes out at ~89k evaluations / 10.9
digits even with rank headroom (rank cap 40), while the port continues to
rank 33+ and 14.7 digits — its residuals, computed through pivoted LAPACK
solves, stay meaningful below the reference build's ~1e-11 floor.  More
digits for more evaluations, priced explicitly.
One porting trap recorded for the next reader: scipy's raw ``getrf`` wrapper
already returns 0-based pivot indices (LAPACK's are 1-based); subtracting 1
again corrupts the swaps.

**The Fortran package's other two drivers** (`test_mc_ising`,
`test_qmc_ising`) also build and run on the same machine and reproduce the
paper's core claim on C_6: plain MC reaches 3.6 digits and lattice QMC 7.5
digits at 4.19M evaluations each, where the greedy cross has 7.9 digits at
26k — two orders of magnitude fewer calls at equal accuracy.
`examples/ising_integrals.py` is the ported driver; at `eps=1e-12`, n=65 it
delivers 13.7–14.7 digits on C_6/C_16/D_6/E_6 in 50–230 ms, and the
31-dimensional C_32 at 14.3 digits in 355 ms / 820k evaluations.

Acceptance tests live in `tests/test_dmrg_cross.py` (18, dense-truth and
closed-form oracles); `tt.greedy_cross` now resolves to this engine — the
alias stopped lying (P2).

### 2.1c The two in-package engines on the same integrals

The pre-port section 2.1a compared `rect_cross` with the Fortran; here are the
package's own two engines head to head, medians over 5 seeds, the same
numba-jitted integrand handed to both (the compiled bond kernel engages for
`dmrg_cross`; `rect_cross` calls the same dispatcher as a plain black box),
kernel compile off the clock:

| C_6 | digits | evaluations | wall |
|---|---|---|---|
| dmrg eps 1e-5 / rect eps 1e-5 | 8.16 / 7.00 | 31k / 72k | 11 / 48 ms |
| dmrg 1e-7 / rect 1e-7 | 10.30 / 9.23 | 67k / 373k | 14 / 162 ms |
| dmrg 1e-9 / rect 1e-9 | 11.95 / 11.47 | 114k / 686k | 23 / 323 ms |
| dmrg 1e-11 / rect 1e-11 | 14.21 / 13.60 | 192k / 1 111k | 42 / 523 ms |

| C_16 | digits | evaluations | wall |
|---|---|---|---|
| dmrg 1e-5 / rect 1e-5 | 8.00 / 6.86 | 86k / 307k | 33 / 223 ms |
| dmrg 1e-7 / rect 1e-7 | 9.34 / 8.72 | 169k / 1 137k | 60 / 601 ms |
| dmrg 1e-9 / rect 1e-9 | 11.27 / 11.40 | 345k / 2 177k | 112 / 985 ms |
| dmrg 1e-11 / rect 1e-11 | 13.81 / 13.49 | 623k / 3 717k | 168 / 1 354 ms |

At matched digits the greedy needs **3–7x fewer evaluations** (typically ~6x)
and **8–13x less wall time**, and at every eps it lands a fraction of a digit
to a digit *above* `rect_cross` for its budget.  Both engines top out near
14.5 digits on these tensors.  This corrects the "5–40x fewer evaluations"
claim first drawn from 2.1a: the 40x end of that range came from the same
lucky Fortran C_16 row that section 2.1b withdrew; the measured spread against
`rect_cross` at equal accuracy is 3–7x.  `rect_cross` keeps its own ground —
kickrank-sized rank jumps reach a target rank in far fewer sweeps, which is
what AMEn-style consumers want — but for quadrature-style smooth black boxes
the greedy is now the engine to reach for.

Artifacts: build recipe and stub in the session scratchpad
(`ttcross/mpif.h`, `ttcross/mpi_stub.c`); the ttpy2 side is
`cross_vs_ttcross.py` / `accept_dmrg.py` there, runnable against any
checkout.

### 2.2 teneva — read and run

`teneva.cross` (`cross.py:13-205`) is the same family as ours: one-site
alternating cross, `maxvol_rect` row selection.

Differences that matter:

* `_iter` (`cross.py:259-287`) takes a **QR** of the block, not an SVD, and
  returns three things: the new core `G = B` (the interpolation matrix), the
  interface `R = Q[ind] @ R` carried to the next core, and the extended index
  set. So the tensor is complete after **each half-sweep**, and the left-to-right
  half is not thrown away as it is in our `_sweep_lr`. This costs no extra
  function evaluations either way.
* Rank growth is **bounded per micro-step** by `dr_max` (default 1) with a floor
  of `dr_min`, so ranks climb slowly and the run takes many more sweeps.
  `dr_max = 0` degrades to fixed-rank square maxvol.
* A **cache** (`_func_eval`, `cross.py:229-256`): a `dict` keyed by the tuple of
  the multi-index. Only the misses go to `f`. It also yields a *convergence*
  criterion — `info['m_cache'] > m_cache_scale * info['m']` means the sweeps are
  re-asking for what they already know, i.e. the index sets have stopped moving
  (`cross.py:196`).
* A hard **evaluation budget** `m`: if the next batch would exceed it, the run
  stops with `info['stop'] = 'm'` and returns what it has.
* `f` may return `None` to abort cleanly (`info['stop'] = 'func'`).
* A **validation set** `(I_vld, y_vld)` with threshold `e_vld`, evaluated after
  every sweep and usable as the stopping criterion — our `n_check` is the same
  measurement but only once, at the end, and never as a criterion.
* A `cb(Y, info, opts)` callback with the index sets and the cache in `opts`.
* `info['stop']` is an explicit enum: `func | m | e | nswp | conv | e_vld | cb`.
* The docstring is explicit that the result has inflated ranks and the caller
  must `truncate` — we do that internally (`round_result=True`).

`teneva.maxvol` / `maxvol_rect` (`maxvol.py:15-137`) are transcriptions of the
same two papers we cite. `maxvol` uses `scipy.linalg.lu` + two
`solve_triangular`s; `maxvol_rect` maintains `B` by the same
Sherman–Woodbury–Morrison update we use, but **reallocates the whole `(n, K)`
matrix with `np.hstack` at every added row** (`maxvol.py:131`). Neither has a rank
test, an identity post-condition, a non-finite test, or complex support (`v =
B.dot(B[i])` and `F − l·v·v` are written without conjugation).

`teneva.cross_act` (`cross_act.py`) is the analogue of our `multifuncrs`: an
AMEn-style cross for $f(X_1, \dots, X_D)$ with interface matrices `Rx` carried per
input — the fused sampling our `multifuncrs.py` docstring says it gave up. Its
own docstring says "This is a draft … There is a problem in the rank-1 case".
It carries `dr` (kickrank) **and `dr2`** — the random enrichment of §1.3
(`core_qr_rand`, `core.py:41-55`).

---

## 3. The comparison, axis by axis

### 3.1 maxvol

| axis | ttcross | teneva | ttpy2 | better, and why |
|---|---|---|---|---|
| square maxvol exists? | no separate routine — the cross submatrix is a bordered LU, never a "maxvol" call | `maxvol` (`maxvol.py:15`) | `maxvol` (`maxvol.py:185`) | n/a |
| start | — | `scipy.linalg.lu`, `P[:, :r].argmax(0)` | LAPACK `getrf` pivots, `_check_rank` on `diag(U)` | **ttpy2** — same start, plus it refuses a rank-deficient input instead of returning a zero-volume submatrix |
| swap update | — | `B -= np.outer(bj, bi/B[i,j])`, one `(n,r)` temporary per swap | in-place BLAS `ger`/`geru`, verified to have aliased (`_ger`, `maxvol.py:99`) | **ttpy2** — no temporary; 3.0× faster at `N=1e5, r=200` (§4.1) |
| complex | — | **no** (`B.dot(B[i])` unconjugated) | yes (`geru`, explicit `.conj()`) | **ttpy2** |
| singular / near-singular submatrix | pivot *rejected* if `|pivot| <= max(10·eps·amax, 1e-5·pivotmax_prev)` — the rank simply does not grow | returns; on an exactly rank-deficient input it returns a `B` that happens to reconstruct `A`, on a `1e-14`-deficient one `max|B[I] − I| = 9.7e-02` and no warning | raises `LinAlgError` with the measured `|U_ii|` ratio; independent post-condition `_check_identity` at `1e-3` (error) / `sqrt(eps)` (warning) | **ttcross** for a cross *sweep* (degrade gracefully), **ttpy2** for a library primitive (fail loud). We should have both: §5 P6 |
| `NaN` in the input | n/a | returns `piv` and an all-`NaN` `B`, silently | `ValueError` naming the first offending index | **ttpy2** |
| rectangular maxvol | — | `maxvol_rect` (`maxvol.py:68`) | `rect_maxvol` (`maxvol.py:331`) | — |
| rect update | — | SWM update, then `np.hstack` — $O(nK)$ allocation and copy per added row, $O(nK^2)$ traffic overall | growable F-ordered buffer with capacity doubling + in-place `ger` | **ttpy2** — 18.0× faster at `N=1e5, r=200` (§4.1) |
| rect stopping | — | `k >= r_min and F[i] <= e²` | `(row_norm_sqr[i] > tol² and K < maxK) or K < minK`, plus `top_k_index` | equivalent; ttpy2 additionally distinguishes *criterion met* from *post-condition met* and names the reason (`stop_reason`) |
| reporting | `neval`, per-sweep line | none | `info` dict: `K`, `max_row_norm`, `max_row_norm_bounded`, `converged`, `stop_reason`, nested square-maxvol `info` | **ttpy2** |
| wrappers | — | none | `maxvol_qr`, `rect_maxvol_qr`, `maxvol_svd`, `rect_maxvol_svd` | **ttpy2** (legacy API surface, COMPAT) |

**Verdict: port nothing from teneva's maxvol.** The one thing worth taking from
ttcross is the *pivot rejection idea*, and it belongs in `cross.py`, not in
`maxvol.py` (§5 P6).

### 3.2 cross

| axis | ttcross (`dmrgg.f90`) | teneva (`cross.py`) | ttpy2 (`cross.py`) | better, and why |
|---|---|---|---|---|
| block | two-site (DMRG) superblock $r n n r$ | one-site $r n r$ | one-site $r n r$ | **teneva/ttpy2** on cost per micro-step; **ttcross** sees $n^2$ pivot candidates at once, which is what makes rank +1 per sweep enough |
| rank adaptation | +1 per bond per sweep, pivot = argmax residual | `dr_min … dr_max` per micro-step (default 1..1), driven by `maxvol_rect` | `kickrank … kickrank + rf` per micro-step, driven by `rect_maxvol` | **ttcross** is the most parsimonious in rank; **ttpy2** reaches the target rank in the fewest sweeps; on the Ising family ttcross is uniformly cheaper in evaluations, 5–40x (measured, §2.1a) — §4.2's "neither is uniformly cheaper" survives only as a statement about worst cases |
| pivot choice | max of the **residual** `A − col·row`, by full / lottery / rook search | max 2-volume of an orthonormal basis (`maxvol_rect`) | same as teneva | **ttcross** — the residual is the quantity the error depends on; volume is a proxy for it. But rook pivoting needs `2p` extra fibers per bond per sweep and is only affordable because the rank grows by 1 |
| exploration beyond the pivots | the lottery is a uniform random sample of unselected entries — **random exploration is built into every pivot search** | none in `cross`; `dr2` exists in `cross_act` | none | **ttcross**, decisively. This is the same gap §1.3 measures |
| basis factorisation | none (bordered LU of the cross submatrix) | QR | SVD, untruncated | ttcross avoids it entirely; QR vs SVD is a wash in practice (§4.5) |
| stopping rule | `pivotmax <= accuracy·amax` for **3 consecutive** sweeps | relative change `e`, or validation error `e_vld`, or budget `m`, or cache saturation, or `nswp`, or `cb` | relative change `eps`/`eps_abs`, or `nswp`, or `stop_fun` | **ttcross** for the *indicator* (a residual, not a difference of iterates) and the 3-strike rule; **teneva** for the *variety* — `m` and `e_vld` are things we simply lack |
| what it can report | `neval`, `pivotmax`, running `<w,X>` from a quadrature TT, `dtt_accchk` in ∞ **and** Frobenius norm with the worst index | `info` enum `stop`, `m`, `m_cache`, `e`, `e_vld`, `nswp`, `r`; `cb` gets the index sets and the cache | `CrossHistory` — per-sweep `err_rel/err_abs/erank/max_rank/fun_eval/time`, `err_round` (exact), `err_check` (Frobenius only), `rmax_active`, `converged` | **ttpy2** for per-sweep detail and for `err_round` being a *measurement*; **ttcross** for the norm pair and the worst index; **teneva** for a named `stop` reason |
| repeated evaluations | the nested-index design re-asks constantly; no cache | `cache` dict, hit/miss counted, and used as a convergence signal | **none** | **teneva** — this is the single largest measured saving available (§4.3) |
| evaluation budget | `maxrank` only | `m`, hard | `nswp`, `rmax` only | **teneva** |
| black box may give up | no | `f` returns `None` → clean stop | no | **teneva** |
| non-finite value from `f` | not checked | not checked (`np.array(y, dtype=float)`) | `ValueError` naming the index | **ttpy2** |
| complex `f` | no (`double precision`) | no (`dtype=float` in `_func_eval`) | yes, with automatic promotion of a real `x0` | **ttpy2** |
| backends | Fortran + MPI + OpenMP | numpy only | numpy / torch through `tt.backend`; index bookkeeping stays numpy | **ttpy2** |
| result rank | inflated, no rounding | inflated, caller must `truncate` | rounded to `eps` internally, `err_round` measured | **ttpy2** |
| parallelism | MPI over bonds, OMP over fibers | none | none (batched numpy inside a fiber) | **ttcross**, out of scope |

---

## 4. Measured head to head

All on b300. `.venv` = numpy 2.4.6 / scipy 1.18.0; `tvenv` = numpy 2.5.1 /
teneva 0.14.11 with ttpy2 on `PYTHONPATH`. float64 everywhere. Times are single
runs on a shared 256-core machine (other jobs present) — read them as ±30 %; the
errors, ranks and evaluation counts are deterministic given the seed.

### 4.1 maxvol (E7)

`A = qr(randn(N, r)).Q`. Square maxvol, `tol = 1.05`, `max_iters = 100`, mean of
5 repeats:

| `N` | `r` | `log det A[I]` ours | teneva | ours (ms) | teneva (ms) | swaps |
|---|---|---|---|---|---|---|
| 200 | 20 | −23.9240 | −23.9240 | 0.37 | 0.54 | 7 |
| 2000 | 50 | −99.8141 | −99.8141 | 9.10 | 8.99 | 11 |
| 20000 | 100 | −286.7560 | −286.7560 | 218.77 | 612.35 | 29 |
| 100000 | 200 | −680.5949 | −680.5949 | 4952.89 | 15054.09 | 48 |

Identical volume to 4 decimals at every size; **3.04× faster** at the top size.

Rectangular maxvol, `tol = 1.1`, `K` free, mean of 3 repeats:

| `N` | `r` | `K` ours | `K` teneva | max‖C_i‖ ours | teneva | ours (ms) | teneva (ms) |
|---|---|---|---|---|---|---|---|
| 200 | 20 | 30 | 29 | 1.0853 | 1.0953 | 0.79 | 0.81 |
| 2000 | 50 | 85 | 85 | 1.0969 | 1.0969 | 16.97 | 22.00 |
| 20000 | 100 | 177 | 177 | 1.0984 | 1.0984 | 206.94 | 1434.87 |
| 100000 | 200 | 361 | 361 | 1.0991 | 1.0991 | 2606.50 | 46881.35 |

Same `K`, same bound, **18.0× faster** at the top size. The gap is entirely the
`np.hstack` rebuild per added row (`teneva/maxvol.py:131`).

Degenerate inputs, `N = 200`, `r = 10` (E7 C):

| input | ttpy2 | teneva |
|---|---|---|
| exact rank deficiency (col 9 = col 0) | `LinAlgError`, names `|U_ii|` ratio | returns; `max|B[I] − I| = 1.2e-16` |
| near deficiency `1e-14` | `LinAlgError` | returns; **`max|B[I] − I| = 9.7e-02`**, no warning |
| all zeros | `LinAlgError`, `U[0,0] = 0` | `LinAlgError` (from `solve_triangular`) |
| one `NaN` | `ValueError`, names the index | returns; **`B` is all `NaN`**, no warning |
| `cond ≈ 1e18`, full rank | `LinAlgError` | returns; `max|B[I] − I| = 2.2e-16`, `‖BA[I]−A‖/‖A‖ = 2.1e-16` |
| complex, well conditioned | `max|B[I] − I| = 3.2e-16` | `max|B[I] − I| = 4.5e-16` (accidentally correct: this matrix is Hermitian-free) |

Honest counterweight on the last-but-one row: **our rank test is stricter than
necessary and rejects a matrix teneva handles correctly.**
$A = Q\,\mathrm{diag}(1, 10^{-2}, \dots, 10^{-18})$ is numerically
full rank in the sense that matters here, and teneva
returns a `B` accurate to 2e-16. Our `_check_rank` threshold $r\,\varepsilon$ on
$\mathrm{diag}(U)$ refuses it. The escape hatches exist (`rcond=`, `maxvol_qr`) and the
cross path always passes an orthonormal `Q`, so this never bites inside ttpy2 —
but a user calling `maxvol` directly on a badly scaled matrix will hit it.
Documented, not changed (§10 Q4).

### 4.2 cross: distinct function evaluations at matched accuracy (E2)

`err_rand` = relative error on 5000 fixed uniform random multi-indices, computed
against `f` itself. `raw` = every value asked for; `uniq` = distinct
multi-indices — the number a cache would reduce the cost to and the fair
implementation-independent metric.

| problem | implementation | `err_rand` | raw | uniq | max rank | time (s) |
|---|---|---|---|---|---|---|
| `sin(s)/(1+s)`, `d=5`, `n=12`, `eps=1e-10` | ttpy2 | 5.57e-11 | 33048 | 14425 | 8 | 0.05 |
| | teneva | 5.57e-11 | 37056 | 14492 | 8 | 0.16 |
| | teneva + cache | 5.57e-11 | 14492 | 14492 | 8 | 0.20 |
| kink `1/(1e-2+|s−5/2|)`, `d=6`, `n=10`, `eps=1e-8` | ttpy2 (`kickrank=2`) | **5.67e-04** | 299200 | 97444 | 24 | 0.39 |
| | teneva | 3.09e-10 | 578360 | 125238 | 26 | 16.95 |
| | teneva + cache | 3.09e-10 | 125238 | 125238 | 26 | 17.49 |
| QTT `1/(1+t)`, `d=20`, `n=2`, `eps=1e-10` | ttpy2 | 3.51e-11 | 15386 | 5147 | 5 | 0.05 |
| | teneva | 4.61e-10 | 5906 | 1807 | 5 | 0.05 |
| | teneva + cache | 4.61e-10 | 1807 | 1807 | 5 | 0.06 |
| exact TT rank 3, `d=8`, `n=16`, `eps=1e-10` | ttpy2 | 1.81e-15 | 16064 | 11802 | 3 | 0.03 |
| | teneva | 1.24e-15 | 20592 | 12502 | 3 | 0.06 |
| | teneva + cache | 1.24e-15 | 12502 | 12502 | 3 | 0.10 |

Readings:

* **Distinct-evaluation counts are within ~10 % of each other on three of four
  problems.** Neither algorithm is fundamentally cheaper in samples. On the QTT
  problem teneva is 2.8× cheaper in distinct evaluations but 13× less accurate
  (4.6e-10 vs 3.5e-11) — it stopped earlier, not smarter.
* **`raw / uniq` is 2.3–3.1× for ttpy2 and 2.6–4.6× for teneva.** That ratio is
  pure waste, and it is what the cache recovers.
* ttpy2 is **3–43× faster in wall clock** at equal ranks. The kink row (0.39 s vs
  17 s) is dominated by teneva's `maxvol_rect` reallocation and its per-index
  Python `dict` loop.
* The kink row is the §1.3 defect showing up in a benchmark table.

### 4.3 The cache, measured on our own cross (E9)

Prototype: memoise `cross._evaluate` on the packed rows (`np.unique` over a
structured view, then one `dict` probe per **distinct** index, not per index).

| problem | mode | `fun_eval` | time (s) | error | cache hits |
|---|---|---|---|---|---|
| `sin(s)/(1+s)`, `d=5`, `n=12`, `eps=1e-10` | plain | 34932 | 0.02 | 5.86e-11 | — |
| | cache | **13884** (2.52×) | 0.06 | 5.86e-11 | 21048 |
| QTT `1/(1+t)`, `d=20`, `eps=1e-10` | plain | 5680 | 0.03 | — | — |
| | cache | **1852** (3.07×) | 0.04 | — | 3828 |
| kink, `d=6`, `n=10`, `eps=1e-8` | plain | 704650 | 0.43 | 2.84e-10 | — |
| | cache | **169421** (4.16×) | 1.31 | 2.84e-10 | 535229 |
| `sin(s)/(1+s)`, black box slowed to 50 µs/point | plain | 34932 | 1.85 | 5.86e-11 | — |
| | cache | **13884** | **0.84** (2.2× faster) | 5.86e-11 | 21048 |

Two facts worth stating separately.

* **Duplicates inside a single batch: zero, always.** Every saving is *across*
  micro-steps and sweeps — the left-to-right sweep asks for a block, the
  right-to-left sweep asks for an overlapping one, and the next sweep asks again.
* **Break-even.** On the kink the cache added 0.88 s over 704650 probes, i.e.
  **≈ 1.25 µs per probe** in this prototype. The cache is a net loss for a black
  box cheaper than that per point, and a win above it. That is why it must be
  opt-in and why the default must stay off (§5 P1). A numpy-side implementation
  (sorted key array + `searchsorted` instead of a Python loop) would lower the
  break-even; not measured.

### 4.4 The integrated prototype vs teneva (E10)

ttpy2 + `kickrank2 = 2` + cache, against ttpy2 as it stands and teneva+cache.
Kink, `d = 6`, `n = 10`, `eps = 1e-8`, six seeds of the initial guess. **This
table is the `tvenv` interpreter (numpy 2.5.1)** — it has to be, teneva lives
there — so the "ttpy2" column is the second row of the §1.3 numpy table, not the
first: seed 1 passes here and fails under numpy 2.4.6.

| seed | ttpy2 error | uniq | +k2+cache error | uniq | time (s) | teneva+cache error | uniq | time (s) |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.84e-10 | 139186 | 2.84e-10 | 176832 | 1.71 | 2.84e-10 | 125238 | 18.01 |
| 1 | 3.84e-10 | 162106 | 2.84e-10 | 229063 | 2.32 | **3.82e-04** | 98886 | 13.95 |
| 2 | **3.82e-04** | 88641 | 2.84e-10 | 219465 | 2.26 | 2.84e-10 | 91943 | 9.47 |
| 3 | **3.82e-04** | 111772 | 2.84e-10 | 225928 | 2.41 | 2.84e-10 | 113235 | 19.35 |
| 4 | **3.82e-04** | 131879 | 2.84e-10 | 212540 | 2.12 | 2.84e-10 | 112630 | 17.36 |
| 5 | **3.82e-04** | 102190 | 2.84e-10 | 226382 | 2.44 | 5.84e-10 | 97349 | 10.30 |

Smooth $\sin(s)/(1+s)$, `d = 5`, `n = 12`, `eps = 1e-10`, three seeds: all nine
runs give 5.86e-11; uniq 13724–13766 (ttpy2), 19606–21161 (+k2+cache),
10655–14733 (teneva+cache); time 0.08–0.09 s vs 0.16–0.30 s.

Readings: the enrichment is **6/6 correct** where the plain method is 2/6 (1/6
under numpy 2.4.6, §1.3) and teneva is 5/6; it costs **1.4–2.5×** distinct
evaluations on the hard problem and **1.4–1.5×** on the easy one, which is the
price of the insurance. The remaining 7–8× wall-clock gap to teneva is ours to
keep. Note that teneva fails on a *different* seed than we do — further evidence
that which seed fails is roundoff, and that the failure is a property of the
method rather than of either implementation.

### 4.5 SVD vs QR in `_left_basis` (E12)

teneva takes a QR where we take an SVD, and we never truncate, so the column
spaces are identical ($\max|U U^{T} - Q Q^{T}| \le 2.6\times 10^{-16}$ at every size tested).

| block `(m, k)` | `svd` (ms) | `qr` (ms) | speed-up |
|---|---|---|---|
| (240, 24) | 0.15 | 0.06 | 2.51× |
| (2400, 60) | 21.03 | 19.22 | 1.09× |
| (10000, 100) | 78.94 | 78.99 | 1.00× |
| (40000, 200) | 331.28 | 439.42 | **0.75×** |

End to end, `_left_basis` patched to QR: smooth 0.020 → 0.016 s, kink 0.411 →
0.365 s, **identical errors and ranks in all cases** (including the bad seed:
3.82e-04 either way). numpy's `gesdd` is not the bottleneck at these shapes and
`geqrf` is *slower* on the biggest one. **Marginal; not recommended as a
standalone change** (§5, "not recommended").

### 4.6 Scale, for the record (E13)

$f(t) = e^{-3t}\sin(12t) + 1/(1+t)$ on a binary QTT grid of `2**d` points,
`eps = 1e-10`, `kickrank = 1`, `n_check = 2000`:

| `d` | max rank | `err_check` | `fun_eval` | time (s) | `fun_eval / 2**d` |
|---|---|---|---|---|---|
| 10 | 6 | 1.10e-11 | 4066 | 0.02 | 3.97e+00 |
| 20 | 6 | 2.14e-11 | 12954 | 0.04 | 1.24e-02 |
| 40 | 6 | 1.53e-11 | 33318 | 0.08 | 3.03e-08 |
| 60 | 6 | 1.54e-11 | 54820 | 0.12 | 4.75e-14 |

No warnings; `fun_eval` grows linearly in $d$ as it should. Note `d = 10` asks
for **4× the whole tensor** — for a small `2**d` the method is a pessimisation,
which is worth saying in the docstring.

---

## 5. What to port, concretely, ranked

Each item: what it is, which function changes, the exact signature after the
change, what it buys (measured, or clearly marked as expected), what it costs.

---

**P1 — an evaluation cache. `tt/algs/cross.py`. Rank 1.**

*What.* teneva's `cross.py::_func_eval` idea: memoise `fun` on multi-indices;
only misses reach the black box.

*Signature.* `cross.rect_cross` gains one keyword; `cross.cross` forwards it:

```python
def rect_cross(fun, x0, eps=1e-6, nswp=20, kickrank=1, rf=2, verbose=False,
               eps_abs=0.0, rmax=None, tau=1.1, n_check=0, check_seed=0,
               stop_fun=None, round_result=True,
               cache=None):
    """...
    Args:
        cache: Memoise ``fun`` on multi-indices.  ``None`` (default) does not
            cache.  ``True`` uses a private dict discarded on return.  A dict is
            used and filled in place, so it can be reused across calls with the
            same ``fun``.  Cuts the number of black-box evaluations by 2.5-4.2x
            on the problems in docs/plans/cross-approximation.md §4.3 and costs
            about 1.3 us per lookup: it is a net loss for a black box cheaper
            than that per point.  ``history.fun_eval`` counts misses only;
            ``history.fun_eval_cached`` counts hits.
    """
```

`CrossHistory` gains `fun_eval_cached: int = 0`. The cache is keyed on the packed
`int64` row (`idx.view([("", np.int64)] * d)`), which is exact for any `d` and
any mode size and never collides.

*Buys.* Measured: 2.52× / 3.07× / 4.16× fewer black-box calls (§4.3); 2.2×
wall-clock on a 50 µs/point black box.

*Costs.* ≈1.25 µs per probe (prototype); a `dict` of $d \cdot 8 + 8$ bytes per distinct
index (169 421 entries ≈ 10 MB on the kink); a second owner of "what is the value
at this index" that must be invalidated if `fun` is not a function of the index —
which `rect_cross`'s contract already requires, and which the cache now *enforces*
by construction, turning a silent contract violation into a reproducible one.

---

**P2 — `kickrank2`: uniformly random extra rows per micro-step.
`tt/algs/cross.py::_select_rows`, `rect_cross`. Rank 2 — this is the correctness
item.**

*What.* The legacy `multifuncrs2`'s `kickrank2`, teneva's `cross_act` `dr2`
(`core.py::core_qr_rand`), and — in a different guise — ttcross's lottery, which
draws its pivot candidates uniformly from the *unselected* entries at every bond
of every sweep. All three inject exploration that the volume criterion cannot.

*Signature.*

```python
def rect_cross(fun, x0, ..., round_result=True, cache=None,
               kickrank2=0, kick_seed=0):
    """...
    Args:
        kickrank2: Extra rows per micro-step drawn uniformly at random from the
            rows rectangular maxvol did *not* select.  Volume-based selection is
            blind to a region where |f| is small relative to the pivots, and a
            cross that never samples such a region converges to a wrong fixed
            point while reporting a relative change of 1e-15 -- see
            docs/plans/cross-approximation.md §1.3 for the reproducer.  Two rows
            fixed 6 of 6 failing seeds there, for 1.4-2.5x more evaluations.
            Costs nothing when the sampling was adequate anyway (the extra rows
            are removed by the final rounding).
        kick_seed: Seed of that draw, so a run stays reproducible.
    """
```

`_select_rows(q, kickrank, rf, rmax, tau)` becomes
`_select_rows(q, kickrank, rf, rmax, tau, kickrank2=0, rng=None)`.

*Buys.* Measured: 3.82e-04 → 2.84e-10 on 5 failing seeds, 6/6 correct (§1.3, §4.4).

*Costs.* Measured: 1.4–2.5× distinct evaluations on the hard problem, 1.4–1.5× on
the easy one. Ranks before rounding grow by `kickrank2` per bond per sweep; the
final `round(eps)` removes them (measured: identical final ranks). A non-zero
default is a behaviour change for every existing caller — see §10 Q1 for the
experiment that decides the default. **Recommendation: default `0` in this
change, and a follow-up commit that flips it once Q1 is measured on the whole
suite.**

---

**P3 — stop `rect_maxvol` from crying wolf about a budget the caller set.
`tt/algs/maxvol.py::rect_maxvol`. Rank 3.**

*What.* Add the flag that `maxvol` already has (`warn_nonconvergence`), and use it
at the one call site that always passes an explicit `maxK`.

*Signature.*

```python
def rect_maxvol(a, tol=1.05, maxK=None, min_add_K=None, minK=None,
                start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1,
                rcond=None, info=None, warn_budget=True):
    """...
    Args:
        warn_budget: Emit the RuntimeWarning when growth stopped at ``maxK``
            or ``minK`` rather than at the tolerance.  Set to ``False`` when a
            budgeted ``K`` is the point (``tt.algs.cross._select_rows`` computes
            ``maxK = rho + kickrank + rf`` on purpose); ``info['converged']`` and
            ``info['stop_reason']`` still report the truth.  The ``tol < 1`` and
            ``all rows selected`` warnings are *not* covered by this flag --
            those are genuine contradictions in the request.
    """
```

`cross._select_rows` passes `warn_budget=False`.

*Buys.* 80 of 93 `RuntimeWarning`s disappear from the suite; the remaining 13 are
all real. Measured accuracy change: **none** (§1.2).

*Costs.* One more boolean in a signature. The risk that a genuine "the budget is
too tight" situation goes unmentioned is covered by `info['stop_reason']`, which
`cross` should start recording per micro-step (§6, `CrossHistory.maxvol_budget_hits`).

---

**P4 — ttcross's stopping rule: a residual indicator with a three-strike rule.
`tt/algs/cross.py::rect_cross`. Rank 4.**

*What.* `dmrgg.f90:661-665`. Two independent ideas, both cheap:

1. *The indicator is a residual on freshly probed entries*, `pivotmax / amax`,
   not a difference between iterates. Our one-site sweep already has the
   ingredients: at each micro-step, the block `A_k` is evaluated and the current
   interpolant's value on the same block is one contraction away, so
   `max |A_k − X_k| / amax` costs $O(d n r^2)$ flops and **zero extra function
   evaluations**.
2. *The rule requires three consecutive sweeps below the threshold.* Our §1.3
   failure passes a one-sweep test trivially (`err_rel = 3e-15` on the sweep it
   locks up). Whether a residual indicator plus three strikes would have caught it
   is **not measured** — see §10 Q2, which is the experiment that decides whether
   P4 is worth its complexity.

*Signature.*

```python
def rect_cross(fun, x0, ..., kickrank2=0, kick_seed=0,
               stop_strikes=1):
    """...
    Args:
        stop_strikes: How many consecutive sweeps must satisfy the stopping
            criterion before the run is declared converged (ttcross uses 3,
            dmrgg.f90:661).  ``1`` reproduces the current behaviour exactly.
    """
```

`CrossHistory` gains `err_resid: float` (the last `max|A_k − X_k| / amax` over
the sweep) and each sweep dict gains `err_resid`.

*Buys.* **Expected, not measured:** an indicator that is a residual rather than a
difference of iterates cannot be fooled by a fixed point of the index sets in the
way §1.3 is — the residual is evaluated on the *newly probed* rows, and with P2's
random rows those include unexplored regions. Cost of `stop_strikes=3` is two
extra sweeps per run, i.e. roughly $2 \cdot 2 \cdot \sum_k r n r$ evaluations.

*Costs.* Two more sweeps by default if the default changes (it should not, in the
first commit). The residual computation is $O(d n r^2)$ flops and $O(n r^2)$ memory
per micro-step, both already paid.

---

**P5 — report the held-out error in both norms and name the worst index.
`tt/algs/cross.py::_held_out_error`, `CrossHistory`. Rank 5.**

*What.* `dtt_accchk` (`dmrgg.f90:708-752`) returns `einf, efro, ainf, afro` and
optionally the multi-index of the worst entry.

*Signature.*

```python
def held_out_error(fun, x, n, n_check, rng=None, seed=0):
    """Relative error on random multi-indices the cross never asked for.

    Returns:
        dict with ``rel_fro``, ``rel_inf``, ``worst_index`` (list of d ints),
        ``worst_exact``, ``worst_approx``, ``n_check``.
    """
```

`CrossHistory.err_check` keeps its meaning (Frobenius, for compatibility) and
gains `err_check_inf: float | None` and `err_check_worst: list | None`. The
warning at `cross.py:540` reports both numbers and the worst index.

*Buys.* Measured: on the §1.3 failure the ∞-norm reads 9.24e-03 against the
Frobenius 3.82e-04 — a 24× more honest description of the damage — and the worst
index `(9,9,9,9,9,9)` points straight at the unexplored corner, which is the
single most useful thing a user can be handed when the check fires.

*Costs.* Nothing: the same samples, three more reductions. **Honest counterweight,
measured (§1.3, E11): the ∞-norm does not detect more often.** Detection is
governed by whether the sample hits the bad region at all — 36.5 % at
`n_check = 1000`, 97 % at 10000, identical for both norms. Do not let the extra
number be read as extra safety.

---

**P6 — refuse a numerically meaningless rank increment.
`tt/algs/cross.py::_left_basis`. Rank 6.**

*What.* ttcross's `upd(p)` test (`dmrgg.f90:369`): a pivot smaller than
`max(10·eps·amax, 1e-5·pivotmax_prev)` is not added, and the rank does not grow.
Our `_left_basis` takes `rho = min(q.shape[1], rmax)` unconditionally, i.e. it
happily builds index sets on singular directions of the sampled block.

*Signature.*

```python
def _left_basis(mat, rmax, rel_drop=0.0):
    """Orthonormal basis of the column space of ``mat``, capped at ``rmax``.

    ``rel_drop > 0`` drops the trailing directions whose singular value is below
    ``rel_drop * s[0]``.  This is *not* truncation at the target accuracy -- see
    the fixed point this docstring already describes, which is what that costs.
    It is the ttcross ``small_element`` guard (dmrgg.f90:369): a direction at the
    level of the roundoff of the block carries no information about the function
    and the row maxvol selects for it is arbitrary.
    """
```

exposed as `rect_cross(..., basis_drop=0.0)`.

*Buys.* **Not measured.** The expectation is a small reduction in wasted ranks on
blocks that are exactly rank deficient — which `_left_basis`'s own docstring says
is the *normal* case for `f(i_1 + … + i_d)`. §10 Q3 is the experiment.

*Costs.* This is the knob whose careless use produces the 12 %-error fixed point
`_left_basis` warns about. It must default to `0.0` and its docstring must keep
that warning. **Recommend implementing it and leaving it off**, so that the
experiment in Q3 can be run without another patch.

---

**P7 — a hard evaluation budget and a graceful abort. `tt/algs/cross.py`. Rank 7.**

*What.* teneva's `m` (`cross.py:231`) and its `f → None` convention.

*Signature.*

```python
def rect_cross(fun, x0, ..., max_eval=None):
    """...
    Args:
        max_eval: Hard cap on the number of black-box values.  When the next
            batch would exceed it the run stops *before* asking, returns the
            current interpolant, sets ``history.stop_reason = 'max_eval'`` and
            warns.  ``None`` (default) is no cap.
    """
```

`CrossHistory` gains `stop_reason: str` with values
`'eps' | 'nswp' | 'max_eval' | 'stop_fun' | 'rmax'`, replacing the current
"`converged` plus four ad-hoc `if`s at `cross.py:523-544`" with one named enum.
`converged` stays and means `stop_reason == 'eps'`.

*Buys.* Not measured (it is a control-flow feature, not a numerical one). It is
the only way to run a cross against a black box that costs seconds per point
without risking an unbounded run, and it is the natural way to write a
"same accuracy per evaluation budget" benchmark, which §8 needs.

*Costs.* One more early-return path. The returned tensor is *not* converged and
must warn loudly, which the existing note machinery already does.

---

### What **not** to take, and why

* **teneva's `maxvol` / `maxvol_rect`.** Same pivots, 3–18× slower, no rank test,
  no identity post-condition, no non-finite test, no complex support. Measured
  in §4.1. Nothing to gain.
* **teneva's per-index Python `dict` cache as written.** `I_new = np.array([i for
  i in I if tuple(i) not in cache])` plus `np.array([cache[tuple(i)] for i in I])`
  is two Python loops over *every* index in the batch. Our P1 prototype loops over
  the *distinct* indices only, after one `np.unique`, and is what should be
  implemented.
* **teneva's `m_cache_scale` "convergence" heuristic** (`cross.py:196`: stop when
  the cache is hit 5× more often than it misses). It is a proxy for "the index
  sets stopped moving", which is exactly the state §1.3 shows is compatible with a
  3.8e-4 error. Adopting it would add a *third* way to declare success while
  wrong. Reject.
* **Porting ttcross's DMRG sweep wholesale.** It is a second engine with its own
  index representation (nested `(i,j,k,q)` quadruples), its own bordered-LU
  arithmetic, and a two-site block that costs $n^2 r^2$ per micro-step against our
  $n r^2$. §4.2 shows no evaluation-count advantage for the maxvol family that
  would justify it, and `docs/REQUIREMENTS.md` R0 is explicit that ttpy2 exists to
  have *fewer* engines. Take the indicator (P4), the pivot rejection (P6) and the
  two-norm check (P5); leave the sweep.
* **ttcross's MPI/OpenMP layer.** Out of scope for a pure-Python package (R1).
* **ttcross's multiple-precision variant** (`dmrggmp.f90`, MPFUN/MPFR). Out of
  scope; R4 fixes float64 as the top of our ladder.
* **A post-hoc rook/alternating-maximisation residual probe.** Prototyped (E4) and
  **rejected on measurement**: on the §1.3 failure, at a budget of 120
  evaluations, alternating maximisation over fibers detected the error in **0 of
  50** random starts against **2 of 50** for plain uniform Monte Carlo at the same
  budget. The reason is visible once stated: post-hoc the residual field is a
  needle — it is ~1e-15 everywhere except on 394 entries, so the argmax along a
  fiber is roundoff noise and the search does not drift anywhere. Rook pivoting
  works *inside* ttcross's sweeps because there the residual is still large and
  smooth. On the spike blind spot (a single non-zero entry in `10**6`) it is
  blind too, 0/3 at 360 evaluations, as is everything else.
* **An "`err_rel` collapsed by more than `k` orders in one sweep" detector.**
  Prototyped (E6) and **rejected**: the signature is not specific. A tensor of
  exact TT rank 3 (`d=8`, `n=16`) drops from `err_rel = 1.00` to `1.7e-15` in a
  single sweep — correctly, it became exact — a 5.9e+14 collapse, larger than the
  4.9e+12 of the failing run. Any threshold that flags §1.3 flags the healthy
  exact case.
* **Replacing the SVD in `_left_basis` by a QR** as a performance change.
  Measured (§4.5): 2.5× on the smallest blocks, 1.00× at (10000, 100), **0.75×**
  at (40000, 200); 10–20 % end to end. Not worth a change on its own. (If
  `_left_basis` gains `rel_drop` under P6 it needs the singular values anyway.)
* **teneva's `dr_max`-bounded rank growth.** It makes the run take many more
  sweeps to reach the same rank; §4.2 shows no accuracy or evaluation benefit
  that our `kickrank`/`rf` pair does not already give.

---

## 6. Missing core functions, exact signatures and one-line contracts

`(T)` = needs a test before it is called done. "Closest" = the existing function
it is nearest to, i.e. the one whose contract and error style it must match.

```python
# --- C1 (T) the held-out check, promoted to a public, reusable function ------
def held_out_error(fun, x, n=None, n_check=1000, seed=0):
    """Relative error of a TT tensor against a black box on random unseen points.

    Returns dict: rel_fro, rel_inf, worst_index, worst_exact, worst_approx,
    n_check.  Closest: tt.algs.cross._held_out_error (private, Frobenius only).
    """

# --- C2 (T) the residual of a cross interpolant on a fiber block -------------
def block_residual(fun, x, iset, jset, k):
    """max |fun(I_k x n_k x J_k) - x(I_k x n_k x J_k)| and the argmax index.

    The ttcross ``pivotmax``.  Zero extra evaluations when called from inside a
    sweep, which is the only place it should be called from.
    Closest: tt.algs.cross._sweep_lr (which already forms the left-hand side).
    """

# --- C3 (T) an index-set cache with an explicit owner ------------------------
class EvalCache:
    """Memoised black box.  ``EvalCache(fun)`` is itself callable and is what
    rect_cross(cache=...) wraps ``fun`` in.

    Attributes: hits, misses, nbytes.  Methods: clear().
    Closest: tt.algs.cross._Counter (which counts but does not store).
    """
    def __init__(self, fun, store=None): ...
    def __call__(self, idx): ...

# --- C4 (T) explicit stop reasons --------------------------------------------
@dataclass
class CrossHistory:
    """... existing fields ...
    stop_reason: str          # 'eps' | 'nswp' | 'max_eval' | 'stop_fun' | 'rmax'
    err_resid: float          # ttcross pivotmax/amax, last sweep (nan if off)
    err_check_inf: float|None # held-out relative error in the infinity norm
    err_check_worst: list|None# multi-index of the worst held-out point
    fun_eval_cached: int      # cache hits (0 when cache is off)
    maxvol_budget_hits: int   # micro-steps where rect_maxvol stopped at maxK
    """

# --- C5 the black box for an elementwise function, factored out --------------
def funs_blackbox(X, funs, dtype=None):
    """``idx -> funs(stack of element(X_j, idx))``, the adapter multifuncrs builds.

    Exposing it lets a caller hand the same black box to rect_cross with a cache
    or a budget, which multifuncrs currently cannot pass through.
    Closest: tt.algs.multifuncrs._make_blackbox (private, multifuncrs.py:291).
    """

# --- C6 a fair benchmark harness (bench/, not tt/) ---------------------------
def cross_benchmark(problems, methods, budgets, seeds, out=None):
    """Accuracy-vs-evaluations table for a set of (fun, oracle) problems.

    The metric is DISTINCT multi-indices, not calls -- see §4.2 for why.
    Closest: bench/ (there is no cross benchmark today).
    """
```

---

## 7. Validation tests, with named oracles

Every test names its oracle, its expected number, where that number comes from,
and its tolerance. `tests/test_cross.py` and `tests/test_verify_cross.py` are the
files; the split follows the existing convention (behaviour vs. verification).

| # | test | problem | oracle | expected | tolerance |
|---|---|---|---|---|---|
| T1 | `test_cache_does_not_change_the_answer` | smooth `sin(s)/(1+s)`, `d=5`, `n=12`, `eps=1e-10`, seed 0 | the *same run without the cache* — bit-for-bit | identical cores | `== 0` exactly; the cache is a memo, not an approximation |
| T2 | `test_cache_cuts_evaluations` | as T1 | measured, §4.3 | `fun_eval` 34932 → 13884; `fun_eval + fun_eval_cached == 34932` | `fun_eval <= 0.45 * 34932`; the identity is exact |
| T3 | `test_kickrank2_fixes_the_documented_fixed_point` | §1.3 reproducer, seeds 0..5, `kickrank=1` | dense `numpy` grid | `k2=0` → 3.82e-04 on **at least 4 of 6** seeds (which seeds depends on the LAPACK build, §1.3); `k2=2` → ≤1e-8 on all 6 | error ≤ `10*eps` on the `k2=2` half; the `k2=0` half asserted as a *count* (`>= 4` seeds above 1e-6), never per seed, so it neither pins a coin flip nor silently stops testing anything if the method improves |
| T4 | **`test_a_confidently_wrong_answer_is_not_returned_quietly`** — *the fail-loud test* | §1.3 reproducer with `n_check=20000, kickrank2=0` | dense grid | the run must **either** be accurate to `10*eps` **or** emit a `RuntimeWarning` naming a measured error above `eps` | assert `err <= 1e-7 or warned`; at `n_check=20000` the detection rate is 97 % (§1.3 table), so pin `check_seed` and assert the warning is present |
| T5 | `test_rect_maxvol_budget_is_not_a_warning` | run the whole `tests/test_cross.py` + `test_multifuncrs.py` under `-W error::RuntimeWarning` | the suite itself | zero `rect_maxvol stopped at K` warnings; 80 today (§1.2) | exact count `== 0` |
| T6 | `test_rect_maxvol_still_warns_when_the_request_is_contradictory` | `rect_maxvol(a, tol=0.05, maxK=r+2, warn_budget=False)` | the post-condition | still warns (`tol < 1` is unreachable by construction) | `pytest.warns(RuntimeWarning)` |
| T7 | `test_max_eval_stops_before_asking` | QTT `1/(1+t)`, `d=20`, `max_eval=5000` | the counter | `fun_eval <= 5000`, `stop_reason == 'max_eval'`, warned | exact |
| T8 | `test_held_out_reports_both_norms_and_the_worst_index` | §1.3 reproducer, `n_check=20000`, `check_seed` pinned to a detecting seed | dense grid | `rel_fro ≈ 3.8e-04`, `rel_inf ≈ 9.2e-03`, `worst_index == [9,9,9,9,9,9]` | 20 % on the two errors (Monte Carlo); the index exactly |
| T9 | `test_maxvol_beats_the_reference_on_volume_and_never_loses` | `qr(randn(2000,50)).Q` | teneva's pivots, computed in the test from a 20-line transcription of `teneva/maxvol.py:15-65` (no dependency added) | `logdet(ours) >= logdet(theirs) - 1e-10` | as stated; §4.1 measured equality to 4 decimals |
| T10 | `test_cross_on_an_exact_tt_rank_tensor_is_exact` | random TT rank 3, `d=8`, `n=16` | the tensor itself | `‖y − x‖/‖x‖` | `<= 4 * eps(float64) * d * r` per R4; measured 1.81e-15 (§4.2) |
| T11 | `test_evaluation_count_is_linear_in_d` | QTT `exp(−3t)sin(12t)+1/(1+t)`, `d ∈ {10,20,40,60}` | §4.6 | `fun_eval` = 4066 / 12954 / 33318 / 54820, max rank 6 throughout | 25 % on the counts, exact on the ranks |
| T12 | `test_the_spike_blind_spot_is_still_documented_and_still_blind` | single non-zero entry in `10**6` | the definition | `‖y‖ == 0`, `converged=True`, and the docstring note is the only defence | exact; this test exists to make the blind spot a *decision* rather than a surprise |

T4 is the one that must **fail loudly rather than return a plausible wrong
answer**, and it is written as a disjunction on purpose: cross cannot promise
accuracy, so what is asserted is that the library never *simultaneously* claims
success and is wrong by four orders of magnitude. Note the honest weakness of T4,
which the test's own docstring must state: it is a Monte Carlo detector at 97 %
(§1.3, `n_check = 20000`), so it is pinned to a fixed `check_seed`. A test whose
pass depends on a random draw is not a test; a test whose *subject* is a random
draw must pin it and say so.

---

## 8. Hard test problems

**(i) Runnable today**

| # | problem | parameters | reference | source |
|---|---|---|---|---|
| H1 | `sin(s)/(1+s)`, `s = sum_k i_k/(n−1)` | `d=5`, `n=12`, `eps=1e-10`, `kickrank=1` | error 5.86e-11, max rank 8, `fun_eval` 34932 (14425 distinct) | measured, §4.2/§4.3 |
| H2 | QTT `1/(1+t)`, `t = i/2**d` | `d=20`, `eps=1e-10` | `err_check` 3.47e-11, max rank 5, `fun_eval` ≈ 1.9e4 | measured, §1.2 |
| H3 | kink `1/(1e-2+|s−5/2|)` | `d=6`, `n=10`, `eps=1e-8` | **3.82e-04** at `kickrank=1`, seeds 1–5; 2.84e-10 at `kickrank2=2` | measured, §1.3 |
| H4 | exact TT rank 3, random cores | `d=8`, `n=16`, `eps=1e-10` | 1.81e-15 | measured, §4.2 |
| H5 | QTT `exp(−3t)sin(12t)+1/(1+t)` | `d ∈ {10,20,40,60}`, `eps=1e-10` | table in §4.6 | measured |
| H6 | `x/(1+y²)` on two rank-2 sum-tensors | `n=[10,12,14,9,11]`, `eps=1e-8` | 3.01e-09, ranks `[1,7,8,8,7,1]`, `fun_eval` 54430 | measured, §1.2 (this is `tests/test_multifuncrs.py::test_two_arguments_matches_dense`) |
| H7 | single spike in `10**6` | `d=6`, `n=10` | `‖y‖ = 0`, `converged=True` — the documented blind spot | measured, E4 |

**(ii) After P1–P3 land**

| # | problem | why it needs the change | reference |
|---|---|---|---|
| H8 | H3 with `kickrank2=2` on 20 seeds | needs P2 | expect ≥19/20 at ≤1e-8; **not measured beyond 6 seeds** |
| H9 | H1/H2/H3 at a fixed `max_eval` budget, accuracy as the output | needs P7 | not measured — this is the table §4.2 should become |
| H10 | An expensive black box: `f` = a 1-D BVP solve per point, ~1 ms | needs P1 for the cache to pay | expect ≈2.5× wall-clock from §4.3's 50 µs/point row extrapolated; **extrapolation, not a measurement** |
| H11 | `qtt_ell.invert(a)` / `sqrt(a)` on the coefficients of `tests/test_qtt_ell.py`, with `n_check` on | needs P5 to report the ∞-norm; these are the calls that emit 2 of the 80 budget warnings today | not measured |
| H12 | float32 end to end | today `x0` sets the width but nothing checks that `tau` and the maxvol thresholds still make sense at `eps ≈ 1.2e-7` | not measured |

**(iii) Aspirational**

| # | problem | what stands in the way |
|---|---|---|
| H13 | Ising susceptibility integrals `C_n`, `D_n`, `E_n` (ttcross's own `test_crs_ising`) with a QTT quadrature | needs a quadrature-weighted running functional (`quad` in `dmrgg.f90`) and, at the accuracies ttcross targets (30+ digits), multiple precision, which we do not have. A float64 run against the published `C_6` is a legitimate, if much weaker, check |
| H14 | 100-dimensional integration against a Monte Carlo reference | needs H9's budget harness and a defensible reference value |
| H15 | torch/GPU cross | the sweep is already backend-generic; the maxvol pivots are numpy by design. Nothing measured; the bottleneck may well be the host round-trip per micro-step |
| H16 | Cross of a genuinely noisy black box (`f` + 1e-6 noise) | `tests/test_cross.py::test_noise_is_reported_as_failure` covers the current behaviour; what a *robust* cross would do (repeat, average, or refuse) is an open design question |

---

## 9. Prototypes: what was run, and how to rerun it

All scripts are on b300 in `~/work/ttpy-modern/scratch-cross/`. Two interpreters:
`A = ~/work/ttpy-modern/ttpy2/.venv/bin/python` (numpy 2.4.6),
`B = ~/work/ttpy-modern/scratch-cross/tvenv/bin/python` (numpy 2.5.1 + teneva
0.14.11). Both need `PYTHONPATH=$HOME/work/ttpy-modern/ttpy2`.

| script | interpreter | what it measures | §|
|---|---|---|---|
| `e1.py` | A | captures every `rect_maxvol` call of `multifuncrs2(x/(1+y²))` and re-runs the ones that stopped early with the budget removed | 1.2 |
| `e1b.py` | A | `rf ∈ {0,2,5,10,30}` on three problems: warnings, error, ranks, `fun_eval` | 1.2 |
| `e3.py` | A | the §1.3 failure: `kickrank` sweep, seed sweep, the plain-call transcript | 1.3 |
| `e3b.py` | A | where the error lives; held-out error vs sample size; a post-hoc `pivotmax/amax` | 1.3 |
| `e4.py` | A | rook vs uniform Monte Carlo detection at equal budget; the spike | 5 |
| `e5.py` | A | per-sweep history of the failure; `kickrank` sweep; the `kickrank2` prototype | 1.3 |
| `e6.py` | A | is the one-sweep `err_rel` collapse specific? (no) | 5 |
| `e7.py` | B | maxvol head to head: volume, `K`, row norms, time, degenerate inputs | 4.1 |
| `e8.py` | legacy `~/micromamba/envs/ttlegacy/bin/python`, `PYTHONPATH=~/work/ttpy-modern/ttpy-src` | the same problems through the Fortran-era `tt.cross.rectcross` | 1.3 |
| `e9.py` | A | the cache prototype: `fun_eval`, time, break-even | 4.3 |
| `e10.py` | B | the integrated prototype (`kickrank2` + cache) vs teneva, 6 seeds | 4.4 |
| `e11.py` | A | detection rate of the held-out check vs `n_check`, both norms, 200 samples | 1.3 |
| `e12.py` | A | SVD vs QR in `_left_basis`: micro and end to end | 4.5 |
| `e13.py` | A | scale, `d ∈ {10,20,40,60}` | 4.6 |
| `e14.py` | **A and B** | the same six seeds of the failure under both numpy versions, with the spelling of `n` and the wrapping of `fun` varied as controls | 1.3 |

`ttcross` and `teneva` are cloned at `~/work/ttpy-modern/scratch-cross/{ttcross,teneva}`
at the commits named at the top of this document.

---

## 10. Open questions, each with the experiment that settles it

**Q1 — what should `kickrank2` default to?** §1.3 shows `0` is a defect and `2`
is a fix on one problem; §4.4 shows `2` costs 1.4–2.5× evaluations. The default
is a trade the whole package pays. *Experiment:* run `tests/` with
`kickrank2 ∈ {0,1,2,3}` patched in as the `rect_cross` default and record, per
test, the wall time, the `fun_eval` totals and any accuracy assertion that moves.
Then rerun H1–H7 at each value. Decide on: the smallest value that fixes 6/6 of
H3's seeds and costs less than 1.5× on H1/H2/H6. Cheap — the whole suite is 59 s.

**Q2 — does a residual indicator with three strikes (P4) actually catch §1.3?**
This is the question that decides whether P4 is worth building. *Experiment:*
instrument `_sweep_lr`/`_sweep_rl` to compute `max|A_k − X_k| / amax` at each
micro-step with zero extra evaluations, run the six seeds of H3 with
`kickrank2=0`, and print the indicator per sweep next to the true error. If the
indicator is ~1e-15 on the sweep where the run locks up, P4 is worthless on its
own and only earns its place *combined* with P2's random rows — in which case
rerun with `kickrank2=2` and see whether the indicator now stays above
`accuracy*amax` for the three sweeps the rule demands. **My expectation, stated
so it can be wrong: the indicator alone will not catch it**, because the residual
is measured on the rows the sweep chose, and those are exactly the rows that are
interpolated exactly at the fixed point.

**Q3 — does `basis_drop` (P6) reduce ranks without reintroducing the 12 %
fixed point?** *Experiment:* H1, H3, H6 and `tests/test_multifuncrs.py`'s two
tensors with `basis_drop ∈ {0, 1e-14, 1e-12, 1e-10, 1e-8}`; record final ranks,
`fun_eval` and true error. The failure mode to watch for is the one
`_left_basis`'s docstring names — ranks locking at a fixed point far above `eps`
with `converged=True`. If any `basis_drop > 0` produces that on any problem, drop
P6 entirely and say so in the docstring.

**Q4 — is `maxvol`'s rank test too strict?** §4.1 measured one case
(`cond ≈ 1e18`, full rank) where teneva returns a correct factorisation and we
raise. *Experiment:* sweep $A = Q\,\mathrm{diag}(1, \sigma, \dots, \sigma^{r-1})$ for
$\sigma \in \{10^{-1} \dots 10^{-3}\}$ at `r = 10, 20, 50` and, for each, compare
$\max|B[I] - I|$ and $\|B A[I] - A\|/\|A\|$ from the two implementations against the
value of `rcond` that would have let ours through. If the post-condition
`_check_identity` holds at `1e-3` everywhere teneva succeeds, then `_check_rank`
is redundant with it and the default `rcond` should be loosened to the point
where `_check_identity` becomes the binding test — one owner for "is this input
usable", instead of two.

**Q5 — how much of the `raw/uniq` waste is structural?** §4.3 shows every cache
hit comes from *across* micro-steps, never within a batch. If the overlap is
systematic (the left-to-right sweep asks for a block that the right-to-left sweep
then asks for again with one index changed), it may be removable by bookkeeping
rather than by a hash table. *Experiment:* log every `(sweep, direction, k)` and
the index set it asked for, then compute, for each hit, which earlier micro-step
produced it. If >70 % of hits come from the immediately preceding half-sweep at
the same `k`, a two-block ring buffer replaces the cache at a fraction of the
memory.

**Q6 — does the cache change what the method converges to?** It must not: the
`rect_cross` contract already requires `fun` to be a genuine function of the
index. But a cached run *enforces* that contract while an uncached one does not,
so a user whose `fun` is secretly stochastic will see two different answers.
*Experiment:* run H1 and H6 with and without the cache and compare the cores
bit-for-bit (T1). If they differ, something upstream is non-deterministic and
that is a finding in its own right.

**Q7 — is our `element()` fast enough to be the bottleneck of `multifuncrs`?**
`multifuncrs` samples every input TT at every requested index; with a cache in
`rect_cross` (P1) the black box gets called less, but `element` is inside the
black box, so the saving is on `element`, not on the user's `funs`. *Experiment:*
profile H6 with and without the cache and report the split between `element`,
`funs` and the sweep. If `element` dominates, the fused interface-matrix sampling
that `multifuncrs.py`'s docstring says it gave up (and that `teneva.cross_act`
implements) becomes worth reconsidering — the docstring's claim that "the number
of *user function* evaluations is the same" is true and beside the point if the
adapter's own cost is the bottleneck.

---

## 11. What I did not verify

* **ttcross was never compiled or run.** b300 has no `gfortran`, no `mpif90`,
  no `mpirun`, and no system LAPACK/BLAS. Everything §2.1 and §3.2 say about
  ttcross comes from reading `dmrgg.f90`, `lr.f90`, `rnd.f90`, `ttind.f90` and
  `main.f90`, with line numbers given so the claims can be checked. **No
  timing, no accuracy, and no evaluation count for ttcross appears anywhere in
  this document**, and none should be inferred from the tables, which contain
  ttpy2, teneva and legacy ttpy only.
* **The papers behind ttcross were not read.** Savostyanov (LAA 2014) and
  Dolgov–Savostyanov (arXiv:1903.11554) are cited by the repository's README; I
  read the code, not the papers. In particular the *quasioptimality* claim that
  gives the method its name is quoted, not checked.
* **`teneva.cross_act` was read but not run.** Its own docstring calls it a draft
  with a known rank-1 bug. Everything §2.2 says about it is from the source.
* **P4, P6, P7 are unmeasured.** They are argued from the ttcross/teneva source
  and from the mechanism of §1.3. Q2 and Q3 are the experiments that decide P4
  and P6; P7 is control flow and has no numerical claim attached.
* **The cache break-even (≈1.25 µs/probe) is a property of my prototype**, a
  Python loop over the distinct keys of each batch. A numpy-side implementation
  would be faster and would move the break-even down. Not measured.
* **The numpy-version divergence of §1.3 was measured, not explained.** I
  established that the *same* script with the *same* seed gives 3.82e-04 under
  numpy 2.4.6 and 3.84e-10 under numpy 2.5.1 at seed 1, with the spelling of `n`
  and the wrapping of `fun` ruled out as causes (E14). I did not trace which
  LAPACK call diverges, nor whether the two builds link different BLAS. The
  conclusion drawn — "the outcome is roundoff-sensitive, so no test may pin a
  single seed" — is safe under either explanation; the explanation itself is not
  verified.
* **`kickrank2 = 2` was validated on one function and six seeds** (§1.3, §4.4),
  plus three seeds of a smooth problem where it changes nothing but the cost. It
  is not a general theorem, and there is no reason to believe there is no
  function on which it fails; Q1 is the experiment that broadens the evidence.
* **The failure of §1.3 is characterised, not explained in the sense of a
  theorem.** I measured that the error concentrates on 394 entries in a corner
  where $|f|$ is 50× smaller than at the pivots, and that the index sets reach a
  fixed point. I did not prove that "volume-maximal pivots avoid regions where
  $|f|$ is relatively small", and the literature reference for that (the
  quasioptimality bounds carry a $\sqrt{1 + \dots}$ factor in the max norm) was not
  read.
* **Nothing was run on a GPU.** No claim here applies to the torch backend.
* **`float32` was not exercised.** Every measurement is float64. The `tau`,
  `small_pivot` and `rcond` constants discussed in §3.1/§5 P6 all have
  precision-dependent values in ttcross (`dmrgg.f90:52-58`); ours do not, and
  whether they should is H12, unmeasured.
* **Single runs on a shared machine.** Every wall-clock number in §4 is one run
  on b300 with other jobs present. Errors, ranks and evaluation counts are
  deterministic given the seed; times carry perhaps 30 %. The 18× and 3× maxvol
  ratios are large enough to survive that; the 10–20 % QR-vs-SVD numbers in §4.5
  are not, which is another reason §5 does not recommend that change.
* **The legacy comparison is a correctness comparison, not a speed one.** §1.3
  reports that legacy `rect_cross` reproduces the same 3.82e-04 fixed point, on a
  different numpy (1.24.4). The times in that experiment (0.44–1.60 s) are *not*
  comparable to ours: the two runs did different amounts of work (different
  `fun_eval`, different final ranks), and per `docs/PERFORMANCE.md`'s rule a time
  comparison at unequal accuracy is not a comparison.
