# Contributions in the ttpy 1.x branches that ttpy 2 must not lose

The rewrite started from the master branch of
[oseledets/ttpy](https://github.com/oseledets/ttpy).  The other branches and
the unmerged parts of `develop` carry work by other people that has no
counterpart here yet.  This file is the ledger: what exists, whose it is,
and what happens to it.  Nothing on this list may be silently dropped.

## To port (algorithmic content)

1. **Interpolatory projector-splitting (DEIM-KSL)** — Alec Dektor,
   PR [#102](https://github.com/oseledets/ttpy/pull/102), merged into
   `develop` 2025-03-14; `tt/ksl_deim/` (314 lines + demo + test).
   A first-order dynamical TT integrator for **nonlinear** equations
   $dy/dt = A y + N f(y)$: oblique interpolatory projectors on nested index
   sets chosen by QDEIM pivots, so the nonlinearity is evaluated only at
   cross-like fibers.  References: A. Dektor, *Collocation methods for
   nonlinear differential equations on low-rank manifolds*, LAA 2025;
   Dektor & Einkemmer, arXiv:2411.15990.  **This fills a hole neither
   `ksl` (linear) nor `tamen` (linear) covers**.  **Ported**:
   `tt/algs/ksl_deim.py` with the author's name on the module and four
   oracles in `tests/test_ksl_deim.py` (dense expm, `solve_ivp` on the
   nonlinear case, the demo's eigenvector invariance, agreement with our
   `ksl`); observed order 1.0 on both linear and nonlinear cases.
2. **QTT-FFT options** — Dishi Liu, 2018 (`develop`): inverse-FFT mode and
   a bit-reversion switch in `qtt_fft1`, groundwork for multi-dimensional
   QTT-FFT.  ttpy 2 has no QTT-FFT at all yet (the 1.x one lives in
   `tt-fort`); **Ported**: `tt/algs/qtt_fft.py` (`tt.qtt_fft1`), unitary in both
   directions, Dishi Liu's `inverse` and `bitReverse` options verbatim,
   pinned against `numpy.fft` in `tests/test_qtt_fft.py`.
3. **Flexible GMRES** — Larisa Markeeva, 2018 (`develop`, `new_gmres`,
   commits `90adf42..600b891`): a rewritten flexible GMRES, part of her
   work on solving equations on complicated domains in QTT via z-order
   curves (the `zkron`/`zaffine`/`zmeshgrid` family in `tt/core/tools.py`
   is hers too, same line of work, now credited in the docstrings).
   The diff (done 2026-08-15): her rewrite added `maxit`, `callback`, the
   per-step relaxation printout and an incremental Givens QR; the relaxed
   `A(x, eps)` closure our GMRES has *is* her notion of "flexible", and
   our version already carries it with the legacy bugs fixed (loop instead
   of recursion, `u_0` not mutated, complex-safe small solve, measured
   residual).  What her code did **not** have is the Z-basis of Saad's
   FGMRES — the update was expanded in the orthonormal V-basis, so an
   actually iteration-varying preconditioner would have been silently
   wrong.  **Ported**: `prec=` keyword of `GMRES` in `tt/algs/solvers.py`
   stores `z_j = M_j^{-1} v_j` and expands the correction in them, with
   her credit in the docstring; tests in `tests/test_solvers.py` (variable
   preconditioner beats plain GMRES on the QTT Laplacian, fixed/identity
   preconditioner parity, externally recomputed residual).
4. **`tt_qr`** — Qbit-, 2018 (`develop`): explicit TT orthogonalization as
   a public function.  ttpy 2 exposes this via `tt.core._ops.orthogonalize`
   internally; a public wrapper with the 1.x name belongs in the compat
   surface.

## Superseded, but credited

* **Core refactoring branch** — Tigran Saluev (`corerefactoring`, 2 commits;
  also 40 commits in 1.x master).  The object-model cleanup it attempted is
  what ttpy 2's `tt/core` does from scratch; the branch itself is history.
* **Build and packaging work** — Daniel Bershatsky (29 commits: Python 3.10
  fixes, GCC 11 workarounds, docker isolation, numpy pins).  ttpy 2 is
  pure Python, so this layer dissolved — but it is exactly the pain that
  motivated the rewrite, and `docs/LEGACY_BUILD.md` documents it.
* **KSL rank-growth fix and its revert** — Daniel Bershatsky, 2022
  (`9e99211` reverting an earlier fix).  Our `ksl` keeps ranks fixed by
  construction and the rank-adaptive path is the planned BUG integrator
  (`docs/plans/bug-integrator.md`); whoever implements it should read that
  revert first to know what went wrong in 1.x.

## Contributors to ttpy 1.x (shortlog of master)

Ivan Oseledets, Tigran Saluev, Daniel Bershatsky, Alexander Novikov (the
`tt/riemannian` module that our Riemannian toolbox descends from), Pavel
Kharyuk, Qbit-, Alexey Boyko, Larisa Markeeva, Rafael Ballester-Ripoll,
Dishi Liu, Ivan Tsybulin, Maxim Rakhuba, Moritz August, Dima Pasechnik,
Matthias Koeppe, Alec Dektor.  The README credits section points here.
