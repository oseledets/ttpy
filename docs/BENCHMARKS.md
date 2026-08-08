# Benchmarks: problems, not operations

`bench/bench_core.py` and `bench/bench_round.py` measure **operations** —
rounding, `dot`, `matvec` — and their only output is a time.
`bench/bench_showcase.py` measures **problems**, and a row gets in only if it
carries a reference we did not produce ourselves.

Three kinds of admissible reference:

1. **a closed form** — the critical Ising energy (Pfeuty), the spectrum of a
   quadratic Hamiltonian through its normal modes, `Im[((e^i−1)/i)^d]` for the
   sine integral, Genz's formula for the corner peak, the nodal solution
   `x − x²/2`;
2. **a published structural fact** — Khoromskij's quantics ranks: 1 for the
   exponential, 2 for the sine, `m+1` for a polynomial of degree `m`,
   independent of `d`;
3. **an independent dense or sparse computation** we can afford at a small size
   — `numpy.linalg.eigvalsh`, `scipy.sparse.linalg.eigsh`, `numpy.fft`.

Everything else — a time with nothing to compare it against — stays in
`bench_core.py`.

```bash
python bench/bench_showcase.py --out bench/results/showcase.json
python bench/bench_showcase.py --problems tfim integrals --scale large
```

`--scale small` is a smoke run (about 100 s end to end), `--scale large` is a
deliberate heavy one.

---

## Where each reference comes from

### Spin chains

**Transverse-field Ising at the critical point.** `H = −Σ σᶻᵢσᶻᵢ₊₁ − Σ σˣᵢ`,
open chain. The exact ground-state energy is

    E₀(L) = 1 − 1 / sin(π / (2(2L+1)))

— Pfeuty, *The one-dimensional Ising model with a transverse field*, Ann. Phys.
57:79 (1970), via the Jordan–Wigner transformation. Checked against a dense
`eigh` to 5e-15 at L = 4, 8, 10, 12.

**Heisenberg.** `H = Σᵢ Sᵢ·Sᵢ₊₁`, open chain. There is no closed form for a
finite chain — at L=12 the reference is a dense `eigh`. For long chains the
reference is **asymptotic**: the energy per site of the infinite chain is
`1/4 − ln 2` (Hulthén 1938, Bethe ansatz). That is not the energy of any finite
chain, so what is compared is the **difference** `(E(2L) − E(L))/L`, which
cancels the surface term of order `1/L`. The logarithmic correction remains, and
the measured 6.7e-04 is that correction, not solver error.

### Vibrational spectra

Units are dimensionless throughout: `ħ = 1`, mass-weighted normal coordinates, a
reference oscillator of unit frequency, so that `(1/2)(−d²/dq² + q²)` is exactly
`diag(k + 1/2)`. Any published number in cm⁻¹ or hartree has to be converted
before it can be compared.

**Coupled oscillators.** `H = Σᵢ (wᵢ/2)(−d²/dqᵢ² + qᵢ²) + α Σᵢ<ⱼ qᵢqⱼ`,
`wⱼ = sqrt(j/2)`, `α = 0.1` — section V.1 of Rakhuba & Oseledets, *Calculating
vibrational spectra of molecules using tensor train decomposition*, J. Chem.
Phys. 145:124101 (2016), arXiv:1605.08422. **All** pairs are coupled, not just
neighbours, and the MPO still has TT rank 3.

The reference is analytic and touches no tensor format: the Hamiltonian is
quadratic, `H = ½ pᵀA p + ½ qᵀB q`, and its spectrum is `Σₖ (mₖ + ½)Ωₖ` where
`Ωₖ²` are the eigenvalues of `AB`. So a dense `d × d` problem serves as the
oracle for a quantum problem of size `n^d`. The `α = 0` case is measured
separately: there the basis is exact and only solver error remains.

**Hénon–Heiles.** `H = Σ (1/2)(−d²/dqᵢ² + qᵢ²) + λ Σ (qᵢ²qᵢ₊₁ − qᵢ₊₁³/3)`,
`λ = 0.111803` — the value used in the MCTDH / DVR / TT literature. The
reference at d = 2, 3 is a dense `eigh`; at `λ = 0` it is the analytic `d/2`; at
large `d` no published number exists in our units, and the row is marked "no
reference".

An implementation detail that matters: `q²` and `q³` are taken as **Galerkin**
matrix elements `⟨i|q²|j⟩`, not as powers of the truncated `Q`. The exact
elements need intermediate states above the basis, so the products are built at
`n + 6` and truncated afterwards. Squaring the truncated `Q` loses
`⟨n−1|q²|n−1⟩` by `n/2` — a percent-level error on the top basis functions,
which silently changes the operator being diagonalized.

### High-dimensional integrals

**Sine integral.** `∫[0,1]^d sin(Σxᵢ) dx = Im[((e^i − 1)/i)^d]` — closed form,
good to `d = 50` and beyond.

**Genz corner peak.** `∫ (1 + Σ aᵢxᵢ)^{−(d+1)}` — one of the six Genz families,
*Testing multidimensional integration routines* (1984); the integral has a
closed form by inclusion–exclusion.

Both rows measure not only accuracy but the **number of function calls**: in a
cross approximation that is the cost, not the wall time.

### Quantics ranks

Khoromskij, *O(d log N)-quantics approximation of N-d tensors*, Constr. Approx.
34:257–280 (2011): on a uniform grid of `2^d` points the exponential has QTT
rank 1, the sine 2, a polynomial of degree `m` has `m+1`, independent of `d`.
Those are integers, i.e. a reference in the strict sense.

The check runs **through `tt.cross`**, which is given nothing but a black box
over bit patterns: if it comes back with rank 2 for the sine on a grid of `2^40`
points, the structure was found rather than assumed.

**Here the tolerance is part of the claim.** The measured singular values of the
middle unfolding at `d = 10`, relative to the first: for the degree-3 polynomial
`s₄ = 2.2e-10`, for degree 5 `s₅ = 4.3e-11` and `s₆ = 6.8e-15`. At `eps = 1e-10`
the degree-5 polynomial comes back at rank 5 — correct for that tolerance, and
useless as a check of the exact `m+1`. Hence `QTT_RANK_EPS = 1e-13`: it admits
both with room to spare and is still above the 1e-15 level where the cross would
start chasing rounding noise.

The Gaussian is in the table as the case where no exact rank has been published
— only a statement that it is bounded in `d`, which a sweep over `d` measures.

### QTT FFT and Poisson

**FFT**: the reference is `numpy.fft` on the same vector, plus the structural
fact about the rank. **Poisson**: `−u'' = 1` solved directly through the exact
inverse operator (`qtri_ones` inverts `qdiff` exactly), with the nodal solution
`x − x²/2` at `∫u = 1/2` as the reference. It runs to `2³⁰` unknowns.

### A failure recorded as a result

**Anderson localization.** `−Δ_h + diag(V)` with independent uniform `V`. The
reference here is a counting argument rather than a measurement: a typical
vector of length `2^d` has TT ranks exactly `min(2^k, 2^{d−k})`, i.e. the
largest possible, and the operator is incompressible. The row exists so that
"tensor methods do not apply here" is a measured fact of the suite rather than
folklore, and so that a future rank heuristic that silently truncates this
potential gets caught.

### Chemical master equation

The reaction chain `0 → S₁ → S₂ → … → S_d → 0`. Jahnke & Huisinga, *Solving the
chemical master equation for monomolecular reaction systems analytically*, J.
Math. Biol. 54:1–26 (2007): from the empty state such a system has the exact
solution `∏ᵢ Poisson(mᵢ(t))` with `m' = Am + b` — a closed form for a problem
with `n^d` states.

It is integrated by implicit Euler through `amen_solve`. **The reported number
contains two errors**: the `O(τ)` time-discretization error, which belongs to
the method and is bounded by halving `τ`, and the tensor solver's error. Both
are in the row; separating them is a separate piece of work, and until then this
row must not be read as solver accuracy.

---

## The rule

A number without its regime is useless, so the regime travels with the number:
backend, device, dtype, thread count, sizes, repetitions. A time is the median
of `repeats` runs after a warm-up when `repeats > 1`, and a single run otherwise
(recorded in the `warmup` field). The JSON carries the machine.

And separately: **a rank claim only means something together with the tolerance
that admits it.** The same polynomial has rank 5 at 1e-10 and 6 at 1e-13, and
both answers are correct for their tolerance.
