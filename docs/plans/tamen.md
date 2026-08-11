# tAMEn: the time-dependent AMEn integrator (planned, not shipped)

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
