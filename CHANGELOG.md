# Changelog

## 2.0.0rc1 — 2026-08-08

First release of the pure-Python rewrite. Relative to ttpy 1.x this is a new
implementation with the same public API; the full compatibility contract, every
deliberate behaviour change and every fixed legacy defect are in
[docs/COMPAT.md](docs/COMPAT.md). Highlights:

### Changed
- **Pure Python.** No Fortran, f2py, `numpy.distutils`, compilers or git
  submodules; the wheel is `py3-none-any`. Required dependencies: numpy, scipy,
  einops. Optional: `[torch]` (GPU/autograd backends), `[fast]` (numba kernels
  for the AMEn inner loops; without it the numpy path gives identical results).
- **Backends.** One code base runs on numpy and torch (CUDA, and MPS in
  float32); `tt.set_backend(...)`, per-array dispatch, mixing backends inside
  one tensor is a loud error.
- **Every iterative solver reports on itself**: per-sweep histories recorded
  even at `verb=0`, non-convergence warns with the value actually reached, and
  quantities that used to be assumed (eigenresiduals, tangent defects, held-out
  cross errors) are measured. A silent wrong answer is treated as the worst
  failure mode; see [docs/NUMERICS.md](docs/NUMERICS.md) for the measured
  limits behind the defaults.
- `amen_solve` default `max_full_size` is 1000 (was 50 in 1.x; the old value
  assumed a compiled local solver).
- `write()`/`read()` use `.npz`; the tt-fort binary format is gone.

### Fixed (present in ttpy 1.x, found against dense truth)
- `Toeplitz`, `IpaS`, `qshift` returned transposed (or, for `IpaS`, simply
  wrong) matrices.
- TT-matrix `reshape` mixed row bits with column bits.
- The real branch of the Fortran KSL applied K/S steps in the wrong order in
  the backward pass, breaking the Strang palindrome.
- EXPOKIT and PRIMME are replaced (Arnoldi with adaptive substepping;
  dense `eigh` + LOBPCG) and verified against `scipy.linalg.expm` and analytic
  spectra — see the measured comparison in docs/COMPAT.md.

### Testing
- 823 tests, all against dense ground truth, closed forms or mathematical
  invariants — never against the legacy implementation.
- Adversarial `test_verify_*` suites with oracles that share no code with the
  algorithms under test, including a double-double residual oracle
  (`tests/extended.py`) that is bit-identical across platforms.
- Benchmark problems with external references only
  (`bench/bench_showcase.py`): Pfeuty's closed form, Bethe ansatz,
  Jahnke–Huisinga, Genz integrals, Khoromskij's quantics ranks, and one
  documented refusal (Anderson localization).
