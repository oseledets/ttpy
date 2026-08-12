# tAMEn: the time-dependent AMEn integrator (shipped -- see Status below)

Source: S. V. Dolgov, "A tensor decomposition algorithm for large ODEs with
conservation laws", CMAM 19(1):23-38, 2019 (arXiv:1403.8085), read in full;
reference implementation `github.com/dolgov/tamen` (MATLAB, cell-array TT).

## What it is

A solver for `dx/dt = A(t) x`, `x(0) = x0` that treats one *time interval* as
a single TT tensor with an extra time mode and solves the global space-time
system by an AMEn iteration:

    B x = f,    B = I_N (x) S  -  (I_N (x) P) A(t),    f = x0 (x) (S e),

where `S`/`P` are the stiffness/mass matrices of the time discretization on
`J` nodes.  The paper's default is **Chebyshev spectral differentiation**
(`S` = Lagrange derivative matrix on Chebyshev nodes, `P = I`): exponential
convergence in `J` for solutions analytic near `[0, T]`, so `J = 8..16`
replaces hundreds of steps of Crank-Nicolson.  The last TT block carries the
time mode; the rest is the spatial TT of all snapshots at once.

Two properties make it more than "amen_solve on a bigger system":

1. **Exact invariants.** The AMEn enrichment is doubled: besides the residual
   `zeta`, every core is enriched with the projected co-kernel vectors
   `C_k = (X^{<k})* C^{<k}` of the linear invariants `c_m* x(t) = const`
   (total probability of a CME, mass, charge).  At the last core the local
   problem *is* the Galerkin-projected ODE in a basis that contains `C`, so
   the invariants are conserved **up to machine precision, independently of
   the TT truncation accuracy** (paper Sec. 3.4).  For skew-symmetric `A` the
   2-norm is preserved by rescaling the projected initial state:
   `theta = sqrt((|x0|^2 - |C* x0|^2)) / |X* x0|` (paper eq. 5).
2. **Adaptive intervals with rejections.** Per-interval error estimate
   `E_{J,h}`; the next interval is `h (eps/E)^{1/q}` with `q = J` for the
   Chebyshev scheme; an interval that misses the target is shrunk and redone.

Complexity per interval: `O(d n (R r^3 + R^2 r^2))` -- AMEn's own.

## Why we want it

* It is the natural propagator for the **dissipative** problems where our KSL
  has a structural handicap: the S-steps integrate backward, and the
  stiffness guard (`docs/plans/bug-integrator.md`) refuses exactly the
  regimes tAMEn handles by construction.  The paper's own comparison
  (its Table 2, 2D periodic convection): tAMEn with Chebyshev `J = 8` beats
  KSL with 512 steps 10x in time at better accuracy, and KSL with 16 steps
  fails outright.  We can reproduce that table with our own `ksl` as the
  baseline -- an honest benchmark of one of our solvers against another.
* The CME family (`docs/plans/tamen.md` is the prerequisite of the SIR
  example, `examples/` #29 in the tracker): probability must sum to 1
  *exactly*, otherwise rare-event tails are noise.  That is the linear
  invariant `c = ones`, which tAMEn conserves and a step-and-round scheme
  does not.

## What exists in ttpy2 to build on

* `tt/algs/amen.py` -- the alternating solve with residual enrichment; tAMEn
  is this loop plus (a) the extra time block, (b) the co-kernel enrichment,
  (c) the Galerkin last-core solve with the theta correction.  The sweep
  machinery, local solvers and interfaces are reusable as is.
* `tt/algs/_localops.py` -- interfaces/projections (single owner).
* The Chebyshev differentiation matrix is 10 lines of numpy (nodes
  `T/2 (1 - cos(pi j/J))`, Lagrange derivative); no new dependency.
* Oracles already in the tree: `examples/fokker_planck_dumbbell.py` (its CN
  path becomes a cross-check), `tests/hamiltonians.py`, and the KSL suite.

## API sketch

```python
x_t, info = tamen(A, x0, T, eps,
                  J=8, scheme="chebyshev",     # or "cn", "euler"
                  invariants=None,              # list of tt.vector c_m
                  conserve_norm=False,          # skew-symmetric A only
                  rmax=..., nswp=...)
```

`x_t` is the space-time TT of the last interval (time mode last); `info`
carries the interval history (accepted/rejected `h`, `E_{J,h}`, ranks) and
the measured invariant drift -- which the tests pin at `< 10 eps_machine`.

## Acceptance (all against external truth)

1. **Convection** (paper Sec. 4.1): periodic 2D transport, exact solution
   repeats with period `T_p = 20`; error = distance to the initial state
   after a full period.  Reproduce the shape of the paper's Table 2 with our
   `ksl` as the second column.
2. **CME lambda-phage** (paper Sec. 4.2): 5-species stochastic kinetics via
   shift-matrix MPO; oracle = total-probability drift (must be machine-eps)
   plus the stationary state from `amen_solve` on `A^T pi = 0`.
3. **Invariant conservation vs truncation**: run at crude `eps = 1e-2` and
   verify the invariants still hold to 1e-14 -- the property that
   distinguishes tAMEn from "solve and round".
4. Chebyshev-in-time convergence: fixed `h`, error vs `J` exponential until
   the spatial error floor.

## Risks and open questions

* The global matrix `B` is nonsymmetric and its conditioning grows with the
  Chebyshev `S` (`docs/plans/qtt-elliptic-bpx.md` owns the conditioning
  lore); the paper solves local systems with GMRES -- our `amen_solve`
  local solver policy (`max_full_size`, matrix-free path) needs a
  nonsymmetric review before reuse.
* Complex `A` (Schroedinger): the paper stays real; our KSL complex path
  covers that side, so tAMEn can stay real-first.
* The restarted-interval bookkeeping (Alg. 1 of the paper) is where the
  MATLAB reference spends its subtlety; port it as data (a plain interval
  loop), not as cleverness.

## Status: shipped 2026-08-12 (`tt/algs/tamen.py`, `tests/test_tamen.py`)

Implemented as designed, with two deviations recorded here:

1. **Post-hoc enrichment.** The co-kernel vectors enter the basis once,
   before the final reduced solve, not inside every sweep (the conservation
   argument of the paper's Sec. 3.4 only needs the span at the last step).
2. **The Galerkin re-solve is residual-guarded.** The projection of a
   nonnormal operator is not stability-preserving: on the SIR master
   equation `X* A X` acquired right-half-plane eigenvalues and an unguarded
   re-solve compounded to `||p|| ~ 1e19` over 6 intervals while the embedded
   time-error estimate stayed quiet (both J and J/2 solves share the same
   bad projection).  The re-solved iterate is therefore accepted only when
   its residual in the *full* system is `<= 2 res(amen) + eps`; otherwise
   the endpoint comes from the amen iterate and the invariants are restored
   by an explicit Gram-solved shift along the `c_m` (O(eps) perturbation,
   machine-exact conservation either way).  On the CME the guard rejects
   the Galerkin path in every interval -- the conservation contract is
   carried entirely by the correction path there, and holds: drift 3e-14
   at eps = 1e-2 (`test_tamen_conserves_probability_at_crude_accuracy`).

Measured niche, part 1 -- dissipative dynamics (Mac, dense-`expm` oracle):

| problem | tamen | ksl |
|---|---|---|
| SIR CME, N=7 chain, T=30 | **0.8 s, err 9.2e-7, sum(p) drift 3.6e-14** | 1.8-19.5 s, err stuck at ~1e-3 (fixed-rank modelling error; more steps and rank 16 do not help), drift 2-5e-4 |

Measured niche, part 2 -- the paper's Table 2 regime, reproduced across
grids.  2D periodic convection on `[-10,10]^2`, T = 20, central differences
in QTT; the oracle is the *FFT-exact* solution of the discrete system (the
periodic difference operator is diagonal in Fourier), so the numbers are
pure time-integration error at any grid with no dense matrix anywhere.
ksl at fixed rank 30; n = 64 on a Mac, the rest on 8 cores of a loaded
h200:

| n per axis | tamen (eps=1e-6, J=12) | ksl, fixed tau |
|---|---|---|
| 64   | 4.3 s, err 1.9e-6 | 21.3 s, err 1.1e-6 (1600 steps) |
| 1024 | **14.6 s, err 2.4e-6** | 54 s, err 2.3e-5 (1600 steps); 100 steps -> err 0.30 |
| 4096 | **109 s, err 1.0e-5** | 282 s, err 7.7e-6 (6400 steps); **100 and 400 steps -> err 1.4 and 1.2, silently** |

This is the paper's Table 2 shape, reproduced with our own solvers: on
fine grids the splitting error forces KSL into thousands of steps (and a
fixed tau chosen too large returns garbage *with nothing to say so* --
which is why `ksl_adaptive` now exists: step-doubling control finds the
admissible tau and pays ~3x the matvec work for never being silently
wrong about the time error).  At n = 4096 tamen reaches the same accuracy
2.6x faster than the equal-error fixed-tau KSL run.  In *plain* TT at
n = 64 the balance flips (ksl 0.5 s vs tamen 8.5 s): amen sweeps pay for
the large mode there, KSL does not -- in QTT all modes are 2 and the
balance flips back.

The division of labour, then: tamen owns dissipative /
conservation-critical dynamics (master equations) and fine-grid stiff
transport; ksl owns smooth norm-preserving dynamics at moderate stiffness
(and everything complex/Schroedinger).  Identified while measuring, as
the next optimization target: the per-step KSL cost at QTT block sizes
(rank 30 -> local blocks of 1800) runs on the interpreted Krylov path --
the compiled kernels of `_ksl_fast.py` cover only the exact-expm regime
(blocks <= 40).  Compiling the Krylov path is the follow-up with the
largest measured payoff.

Remaining from the original plan: time-dependent `A(t)`, the
theta rescaling for 2-norm conservation, and the lambda-phage CME example.
