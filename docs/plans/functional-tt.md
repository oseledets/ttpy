# FUNCTIONAL TT: the continuous tensor train, and the Marzouk sampling line

This document owns two questions `docs/plans/full-scale-examples.md` opened and
did not answer: **what a tensor train of *functions* would be in ttpy2**, and
**who "Marchuk" was** (its Q20). It owns no algorithm that another spec owns.
`docs/plans/cross-approximation.md` has already decided what to take from
ttcross, teneva and xfac, and nothing here reopens that: the question here is
not how to pick better indices, it is what happens when there are no indices.
`docs/plans/ROADMAP.md` stays the single owner of the milestone order; every
example below names the milestone it waits on.

**Sources actually read, and how.**

| what | where | state |
|---|---|---|
| Gorodetsky, Karaman, Marzouk — the FT paper | `arXiv:1510.09088v3` PDF, downloaded and text-extracted | **read in full** (31 pages): the definition, Algorithms 1–3, Theorem 1, all five experiment sections |
| C3 (`Compressed-Continuous-Computation`) | `github.com/goroda/Compressed-Continuous-Computation` @ `3096f3b`, cloned to `b300:~/work/ttpy-modern/scratch-ftt/c3` | **read** (the ~30 headers and sources named with file:line in §1.2); **built on b300** including the `c3py` bindings, and **one smoke example run** (F6) |
| c3sc | `github.com/goroda/c3sc` @ `0236698`, cloned to the same directory | cloned and sized; **not built** — it needs `goroda/cdyn`, which was not fetched (§1.3) |
| `lanl/pyftt` | `github.com/lanl/pyftt` via the GitHub API | exists, created 2026-07-29, **empty** (§1.5) |
| Bigoni, Engsig-Karup, Marzouk — spectral TT | `arXiv:1405.5713` abstract + CrossRef + the ar5iv rendering | **abstract and metadata verified; body read only through ar5iv**, which this survey caught being wrong about a *negative* claim on the companion paper (§8) |
| `TensorToolbox` | PyPI JSON API and the 1.0.22 sdist's `setup.py`; **install attempted on b300** under numpy 2.4.6 | **does not import** (F7, §1.6) |
| Martinelli, Manzini — the Rust FT library | `doi:10.1007/978-3-031-56208-2_22`, Springer landing page | **abstract read verbatim; body paywalled; no public repository found** (§1.5) |
| Dolgov, Anaya-Izquierdo, Fox, Scheichl — TT-IRT | `arXiv:1810.01212` abstract + CrossRef; `github.com/dolgov/TT-IRT` via the GitHub API | abstract and metadata verified; **code not cloned, not run** |
| Cui, Dolgov — squared IRT; Cui, Dolgov, Zahm — conditional DIRT | `arXiv:2007.06968`, `arXiv:2106.04170` abstracts + CrossRef; `github.com/DeepTransport/deep-tensor` README | abstracts verified; **MATLAB, not run** |
| ttpy2 itself | `tt/algs/cross.py`, `tt/algs/iga.py`, `tt/algs/multifuncrs.py`, `tt/core/vector.py`, `examples/iga_ring.py` | read; run on b300 for F1–F5 |
| the sibling specs | `full-scale-examples.md`, `cross-approximation.md`, `ROADMAP.md`, `REQUIREMENTS.md` | read before writing |

**Ground rule for numbers.** Every number is either cited from a named source
with its section, or measured on b300 by a script in
`b300:~/work/ttpy-modern/scratch-ftt/` — `grid.py` (F1), `pw.py` (F2), `irt.py`
(F3), `iga.py` (F4), `round.py` (F5) — and marked **[measured]** with its
regime. Nothing was run locally. §8 lists what was not verified.

---

## 0. The short version

### 0.1 The Marchuk/Marzouk question: settled, and the answer is that the pairing never existed

**"Marzouk" is indeed the name behind "Marchuk" — and it does not rescue the
reference.** Youssef Marzouk (MIT) is a real and central author in this
literature, and he *is* the missing name in the sense that he is the person a
half-remembered "Dolgov, March…" most plausibly points at. But:

1. **There is no Dolgov–Marzouk paper at all.** The arXiv API query
   `au:Marzouk_Y AND au:Dolgov` returns `totalResults = 0`. Marzouk's arXiv
   author listing (`arxiv.org/a/marzouk_y_1`, 100 entries, 2011–2026) contains
   no Dolgov and no Cui co-authored entry.
2. **Marzouk has no paper on vibrational or molecular spectra.** None of the 100
   titles concerns vibrational spectra, molecular spectra, or Hamiltonian
   eigenvalue computation. An arXiv full-text query `all:vibrational AND
   au:Marzouk` returns exactly one hit and it is a different person (Osama A.
   Marzouk, on cylinder-wake drag). His Google Scholar profile (`user=TwVbNZ4AAAAJ`,
   fetched, not blocked) lists "computational mathematics, uncertainty
   quantification, inverse problems, data assimilation, Bayesian statistics".
3. **Marzouk's entire tensor-train output is two papers**, both function
   approximation: spectral TT (§1.6) and the continuous/functional TT (§1.1).

So the original note **conflated two disjoint literatures**:

| the line | who | what | where it already lives |
|---|---|---|---|
| vibrational spectra in TT | Dolgov, Khoromskij, Oseledets, Savostyanov (CPC 2014); Rakhuba, Oseledets (J. Chem. Phys. 2016) | block-TT extreme eigenvalues; 84 states of CH₃CN | `full-scale-examples.md` §1.3, §3.3 — already correct |
| TT sampling and inverse problems | Dolgov with Cui, Zahm, Scheichl, Fox, Anaya-Izquierdo | inverse Rosenblatt transports, squared TT densities | §1.7–§1.9 below — new |
| functional / continuous TT | Gorodetsky, Karaman, **Marzouk**; Bigoni, Engsig-Karup, **Marzouk** | a TT of functions, not of arrays | §1.1–§1.6 below — new |

Marzouk is adjacent to the sampling line only through Zahm, Cui, Law, Spantini,
Marzouk, *Certified dimension reduction in nonlinear Bayesian inverse problems*,
Math. Comp. **91** (2022) 1789–1835, doi:10.1090/mcom/3737 — a transport-map
paper whose abstract contains no tensor train, and which is the *ancestor* of
the dimension-reduction step in Cui–Dolgov–Zahm (§1.9). That is how the name
gets next to the TT sampling line without ever being on it.

**`full-scale-examples.md` Q20 is answered and should be closed** (§7). The
correct action there is not to add a citation but to delete a phantom: there is
no Dolgov–Marchuk and no Dolgov–Marzouk vibrational-spectra paper, and the two
references that document actually needs are already in its §1.3.

### 0.2 What functional TT would cost us, in one paragraph

Less than it looks, for the useful half, and much more than it looks for the
rest. **F5 [measured]** shows that ttpy2's existing `tt.vector.round` *is* the
continuous rank truncation, provided the cores are first transformed by the
Cholesky factor of the univariate mass matrix: the Frobenius norm of the
transformed coefficient tensor equals `‖f‖_{L²}` to **1.42e-16**, and truncating
in those variables is **1.17–1.40×** more accurate in `L²` than truncating the
raw coefficients at the same rank. **F1/F2 [measured]** show that `tt.cross`
already works unchanged on any node family — Chebyshev–Lobatto, Gauss, graded
piecewise-Chebyshev — because it only ever indexes a list. So a *nodal*
functional TT with a fixed basis is **a ~300-line leaf module that reuses the
whole TT algebra**, and it buys, on a peaked function in 4D at a comparable
evaluation count, **7.26e-07** off-node `L²` error against **1.47e-03** for the
best uniform grid — four orders of magnitude at constant TT rank. What it does
*not* buy is the thing that makes C3 **55 912 lines of C** (F6): continuous
pivots and online per-fiber basis adaptation, where the cross grows the
univariate approximation while it runs. That is a different algorithm and a
different data structure, and it is a separate package, not a module.

### 0.3 The measurements made for this document

Seven, all on b300. F1–F5 with `~/work/ttpy-modern/ttpy2/.venv/bin/python`
(numpy 2.4.6, scipy 1.18.0), numpy backend, float64, `OMP_NUM_THREADS=1`, single
run each. F6–F7 in throwaway environments.

| # | what | headline result |
|---|---|---|
| F1 | `tt.cross` on a peaked `f = 1/(δ²+Σ(x_k−c_k)²)`, `d=4`, on uniform vs Chebyshev nodes, evaluated **off** the nodes | error **at** the nodes 1e-11 in every row; error **between** them 1e-3…1e-1. Uniform nodes + global polynomial reconstruction diverges to **1.7e+43** (Runge). Chebyshev + barycentric beats uniform + linear by **11×** at `n=65`, `δ=0.05`, and **loses** to it by 13× at `δ=0.01` |
| F2 | the same cross on a **graded piecewise-Chebyshev** node family | `δ=0.01`, 90 nodes/mode: off-node `L²` **7.26e-07** against **1.47e-03** (uniform, 65 nodes) and **9.49e-02** (Chebyshev, 65 nodes). TT rank moves from 24 to 26 — **the entire gain is univariate** |
| F3 | squared-TT inverse Rosenblatt sampling built from today's ttpy2 (~60 lines), `d=5`, `n=33`, 200 000 samples | works: ESS/N = **1.0000** at cross `eps ≤ 1e-2` on both densities, **0.4687** for Rosenbrock at `eps=1e-1`. Trap: rounding the Hadamard square at `eps²=1e-16` takes rank 16 → **256** and the sampler from 2.9 s to **49.6 s** |
| F4 | how continuous is `tt/algs/iga.py`? `R₀₀` off the quadrature nodes | on the **ring** the field is affine and piecewise-linear reconstruction is exact (3.1e-15); on a **twisted** duct it is **2.4e-10 at the nodes and 2.4e-04 between them**, converging as `O(h²)`. Quadrature order at a fixed mesh changes the ring's solution error by **nothing** (1.624e-07 for `per_element ∈ {3,4,6,10}`) |
| F5 | is `tt.round` already the continuous truncation? | **yes, in the right variables**: `‖L^T c‖_F = ‖f‖_{L²}` to **1.42e-16**, and rounding in those variables is 1.17–1.40× better in `L²` at ranks 2…12 |
| F6 | does C3 build on b300, with no system BLAS/LAPACK/cmake/swig? | **yes**, after three fixes (a 5 136-symbol linker shim for `scipy-openblas32`'s prefixed exports, de-prefixed `cblas.h` through `CPATH`, and `-Dcomplex=_Complex` for SWIG 4.4). `libc3.so` + `c3py-0.0.7` wheel; the smoke example returns `‖x₁+x₂‖_{L²[−1,1]²} = 1.63299` at rank 2 — **exact** |
| F7 | does `TensorToolbox` install under numpy 2.4.6? | **no**. 1.0.22's `setup.py` does `import pip` under build isolation; the wheel-only path backtracks to 1.0.19, which then dies on **`np.float` at `tensor_wrapper.py:108`**, a default argument evaluated at import |

---

## 1. The sources, one entry each

### 1.1 The functional tensor train — Gorodetsky, Karaman, Marzouk

> A. Gorodetsky, S. Karaman, Y. Marzouk, *A continuous analogue of the
> tensor-train decomposition*, Comput. Methods Appl. Mech. Engrg. **347** (2019)
> 59–84, doi:10.1016/j.cma.2018.12.015; arXiv:1510.09088 (v1 2015-10-30,
> v3 2018-12-11). Read in full from the v3 PDF.

**The paper calls it FT, not FTT.** The decomposition is

    f(x₁,…,x_d) = Σ_α f₁^{α₀α₁}(x₁) f₂^{α₁α₂}(x₂) ⋯ f_d^{α_{d−1}α_d}(x_d)
                = 𝓕₁(x₁) 𝓕₂(x₂) ⋯ 𝓕_d(x_d),   𝓕_k : 𝒳_k → ℝ^{r_{k−1}×r_k}

so **the three-index core array `G_k[α, i, β]` becomes a matrix-valued function
of one variable**, and the discrete index `i` disappears. "Quasimatrix" is
reserved in the paper for the vector-valued case, defined in its footnote 5
verbatim: *"Called a quasimatrix because it corresponds to a matrix of infinite
rows and n columns."*

**What replaces the SVD.** Two continuous factorizations. A **continuous QR of
quasimatrices** — the paper is explicit that "the discrete and continuous QR
decompositions differ with respect to the inner product used to define
orthonormality… the continuous computation paradigm of Chebfun enables
maintaining a notion of orthonormality that is consistent with the original
function space" — and a **continuous pivoted LU** used inside pivot selection.
Rank reduction is called **rounding** (`ft-round`, their §4.3) and it is
*exactly* TT-rounding with the QR done in function space and the SVD done on the
resulting small matrices. **There is no continuous SVD.** This is the single
most important structural fact for us, and it is what F5 measures from the other
side: once the univariate inner product is folded into the coefficients, the
existing discrete rounding is the continuous one.

**Cross approximation, and how it selects points.** Their **Algorithm 1,
"Continuous cross approximation using fiber-adaptive approximations"** (§4.1). A
fiber is a univariate function obtained by fixing all but one coordinate; the
bivariate skeleton is a **continuous CUR**, `f(x,y) = C(x) F† R(y)` with `C`, `R`
quasimatrices. The paper lists four differences from discrete TT-cross: fibers
are functions, not vectors; each fiber is approximated *online* to a tolerance
`ε_approx`, which is **a new error source with no discrete analogue**; the maxvol
optimisation ranges over a continuous variable; and rounding uses the continuous
QR.

Point selection is their **Algorithm 2, `dominant`**, "approximate maxvol through
dominant submatrices" (§4.2). Their Definitions 1–2 define a submatrix of a
matrix-valued `A : 𝒳 → ℝ^{n×r}` by fixing `r` pairs `(i_k, x_k) ∈ {1,…,n} × 𝒳`,
and call it **dominant** when `B = A · Ā_mat⁻¹` satisfies `|B[i,k](x)| ≤ 1` for
all `(i, x, k)`. **The index of the discrete maxvol becomes an (index, point)
pair, and the pivot ranges over the continuum.** The algorithm is: continuous
pivoted LU for an initial independent set, then swap rows while
`max_{x,i,j} A[i,:](x) · Ā†_mat[:,j] > 1`. Their practical remark is the one
worth stealing: if the univariate functions are orthonormal-polynomial
expansions, **that maximisation reduces to an eigenvalue problem** — no local
minima — and for piecewise polynomials one solves it per piece.

**What is proved, and what is not.** **There is no convergence proof for the
cross algorithm.** The only theorem is **Theorem 1 (parameterisation error)**: if
`|f_k^{ij}| ≤ C` and every univariate function is approximated to `|f − f̂| ≤ εC`
with `εd < 1/e`, the FT error is bounded by their Eq. (10) — a bound whose
consequence is that `ε` must shrink **quadratically in `d`** to hold a fixed
relative `L∞` error, though the paper adds that "in practice we usually see good
approximation behavior… even with fixed thresholds". Quasi-optimality of true
maxvol is cited (Goreinov, Savostyanov), not proved here.

**Bases, and where the adaptivity lives.** Both linear parameterisations (basis
expansions) and nonlinear ones (**piecewise polynomials with adjustable knot
locations**). The number of basis functions `n_k^{α_{k−1}α_k}` **varies
independently for each univariate function** — per-fiber, not per-mode. In the
experiments: Legendre polynomials, pseudospectral projection with Gaussian
quadrature, and adaptive piecewise Legendre. Order adaptation is explicit
(their §5.1: degree-7 Legendre per fiber, split the domain into three regions if
the normalised squared highest-degree coefficient exceeds `ε_approx`; §5.3.1:
start at degree 2, step `k → k+7`, stop when the last two coefficients
contribute less than `ε_approx`). Rank adaptation is their **Algorithm 3,
`ft-rankadapt`**: cross → round → bump ranks by `kickrank` → repeat, which is
recognisably ttpy2's `rect_cross(kickrank=…)` outer loop.

**Reference values a reimplementation could be checked against** — this is the
part of the paper that is directly usable:

| their § | problem | published target |
|---|---|---|
| 5.1 | Gaussian bump `A exp(−Σ(x_i−c)²/2l²)` on `[0,1]³`, `c=0.2`, `l=0.05`, `A=1`; integrate | both FT and STT reach `O(1e-11)`; FT up to **3 orders of magnitude lower error at fixed evaluation count**. This is a rerun of the STT paper's own experiment, whose numbers came from `TensorToolbox` |
| 5.3.1 | `sin(x₁+…+x_d)` on `[0,1]^d` | exact integral **`Im[((e^i − 1)/i)^d]`**; they report well-behaved relative error for **`d > 600`** with evaluations growing linearly in `d` |
| 5.3.2 | discontinuous Genz: `0` if any `x_i > ½`, else `exp(Σ5x_i)` | exact integral **`((e^{5/2} − 1)/5)^d`**; stated **`I ≈ 3.131e3` at `d=10`**, **`I ≈ 9.05455e34` at `d=100`**; derivative target `D[f] = 5d·exp(Σ5x_i)` at `x_i = 0.2` |
| 5.4 | `1/√(Σx_i² + 1e-12)` on `[−1,1]^d` | RRMSE on 10 000 uniform samples; Table 2 gives per-iteration rank profiles at `ε_round = 1e-5` (e.g. `1 2 4 5 6 7 7 … 7 6 5 5 … 5 1`), piecewise polynomials of degree ≤ 3, four regions, initial rank 4, `kickrank=2`, ≤5 cross iterations |
| 5.5 | 24-dimensional elliptic PDE UQ, reparameterised to `[0,1]²⁴` by the normal CDF | three settings `(a, σ², l) = (0, 0.1, 0.125) / (0.5, 1, 0.045) / (0, 1, 0.045)`; `δ_cross = 1e-3`, ≤3 sweeps, `kickrank=5`, ≤4 rank adaptations |

**§5.3.2 is the one to build first** (§5 below): a closed-form integral in any
`d`, no data file, no geometry, and it is exactly the regime where a discrete
grid is hopeless and a piecewise basis with an adapted breakpoint is trivial.

**Honest counterweight.** The paper's headline comparison against STT (their
§5.1) is a comparison against **their own rerun** of someone else's method in
someone else's library, at settings they chose. And Theorem 1 is a
*parameterisation* bound: it says nothing about whether the cross found the right
fibers, which is precisely the failure `cross-approximation.md` §1.3 documents
in the discrete case and which the continuous version cannot be assumed to have
escaped.

### 1.2 C3 — `Compressed-Continuous-Computation`

> `github.com/goroda/Compressed-Continuous-Computation` @ **`3096f3b`**
> (2022-05-05, "resolved memory leak"). C. **`LICENSE` is a 22-line BSD-3-clause,
> © 2014–2016 MIT plus © 2016 Sandia Corporation** — the GitHub API reports the
> licence as `NOASSERTION` only because it does not recognise the dual header;
> the file itself is unambiguous, and it was read. Created 2015-09-08, last push
> 2023-12-15, 47.2 MB. Named by the FT paper's §5 verbatim: *"All of the
> experiments described below are performed with the Compressed Continuous
> Computation (C3) library available at
> http://github.com/goroda/Compressed-Continuous-Computation."*

**Size**, which is the argument of §2.5 in one line: **55 912 lines of `.c` and
6 546 of `.h`** in the library proper, plus 40 101 lines of tests and examples —
and, separately, **2 500 124 lines of machine-generated tabulated coefficients**
(`lib_funcs/legtens.h`, `hermtens.h`, `legtens_backup.h`). Per sub-library:
`lib_funcs` 21 750, `lib_clinalg` 13 687, `lib_superlearn` 5 208,
`lib_optimization` 3 850, `lib_probability` 2 607, `lib_quadrature` 2 418,
`lib_tensdecomp` 1 964, `lib_linalg` 1 869.

**[measured] F6 — C3 builds on b300, and the Python bindings work, after three
fixes.** All in a throwaway venv `~/work/ttpy-modern/scratch-ftt/buildvenv`; the
ttpy2 venv was untouched. The README declares BLAS, LAPACK, SWIG and CMake, and
b300 has none of them as system packages.

*The baseline failure, recorded because R0-style "does it build" evidence is the
point:*

```
CMake Error at .../Modules/FindPackageHandleStandardArgs.cmake:290 (message):
  Could NOT find BLAS (missing: BLAS_LIBRARIES)
Call Stack (most recent call first):
  .../FindBLAS.cmake:1509 (find_package_handle_standard_args)
  CMakeLists.txt:35 (find_package)
```

`pip install cmake scipy-openblas32 swig numpy setuptools` supplies **cmake
4.4.2, SWIG 4.4.1, scipy_openblas32 0.3.34** — so b300 does have SWIG, through
pip. Three obstacles remained, each real:

1. **`scipy-openblas32` exports prefixed symbols.** `nm -D` shows `scipy_dgemm_`,
   `scipy_cblas_dgemm`, not `dgemm_`/`cblas_dgemm`. C3 calls eleven LAPACK
   routines (`dgeev_ dgelsd_ dgeqrf_ dgerqf_ dgesdd_ dgesv_ dgetrf_ dgetri_
   dorgqr_ dpotrf_ dsyev_`) and twelve CBLAS ones by their unprefixed names.
   Fixed by generating an alias shim of 5 136 PLT tail-jumps
   (`.globl X / X: jmp scipy_X@PLT`) into `shim/libblasalias.so`.
2. **`cblas.h` missing**: `c3/lib_linalg/linalg.h:116: fatal error: cblas.h: No
   such file or directory`. Fixed by de-prefixing the wheel's headers and passing
   them through `CPATH` — **not** through `-DCMAKE_C_FLAGS`, because C3's
   `CMakeLists.txt` overwrites `CMAKE_C_FLAGS` wholesale.
3. **SWIG 4.4 cannot parse C3's `double complex * ccoeff;`** at
   `c3/lib_funcs/polynomials.h:211`: `Error: Syntax error - possibly a missing
   semicolon (';')`. Fixed by a `swig` wrapper injecting `-Dcomplex=_Complex`.
   (`setup.py` also passes the removed `-py3` flag, which SWIG 4.4 warns about
   and ignores.)

Result: `[100%] Built target c3` → `libc3.so`; then
`c3py-0.0.7-cp312-cp312-linux_x86_64.whl`, 6 805 560 bytes, installed. **Smoke
test passed**: `pyexamples/c3py_test_small.py`, exit 0 — adaptive cross of
`f(x) = x₁ + x₂` on `[−1,1]²`, Legendre, 7 parameters per dimension:

```
...... Error L/R Sweep = 1.136842E-08
...... Error R/L Sweep = 0.000000E+00,0.000000E+00
rounded ranks = 1 2 1
Final norm = 1.63299
```

`1.632993161855454 = ‖x₁+x₂‖_{L²([−1,1]²)}` and rank 2 is exact. **C3 is
reachable on b300**, which makes it usable as an oracle for anything §2.4 builds
— unlike `ttpy` (`full-scale-examples.md` §1.4) and `qtt-laplace` (its §1.6).

**The API, read from source — and it does not match the names one would guess.**
Reported here because `docs/plans` rule 2 forbids reconstructing an API from
memory, and because these are the names a `tt.functional` design should be
checked against:

| concept | C3 | file:line |
|---|---|---|
| a univariate function | `struct GenericFunction { size_t dim; enum function_class fc; void * f; void * fargs; }` | `lib_funcs/functions.h:85` |
| the function classes | `enum function_class {PIECEWISE, POLYNOMIAL, CONSTELM, LINELM, KERNEL}` — **five, and no `RATIONAL`** | `functions.h:63` |
| polynomial types | `enum poly_type {LEGENDRE, CHEBYSHEV, HERMITE, STANDARD, FOURIER}` | `polynomials.h:70` |
| a piecewise polynomial | `struct PiecewisePoly` — a **tree**: `int leaf; size_t nbranches; struct PiecewisePoly ** branches; struct OrthPolyExpansion * ope;` | `piecewisepoly.h:87` |
| a vector-valued function | `struct Quasimatrix { size_t n; struct GenericFunction ** funcs; }` | `quasimatrix.c:66` |
| a **matrix**-valued function (the core) | `struct Qmarray { size_t nrows, ncols; struct GenericFunction ** funcs; /* fortran order */ }` | `qmarray.h:61` |
| the train | `struct FunctionTrain { size_t dim; size_t * ranks; struct Qmarray ** cores; … }` | `ft.h:63` |
| cross | `ftapprox_cross(...)`, `ftapprox_cross_rankadapt(...)`, wrapper `c3approx_do_cross(struct C3Approx *, struct Fwrap *, int adapt)` | `ft.c:3361`, `ft.c:3769`, `approximate.c:432` |
| its options | `struct FtCrossArgs { … double epsilon; size_t maxiter; int adapt; double epsround; size_t kickrank; size_t * maxranks; … }`, defaults `epsilon=1e-10, maxiter=5, epsround=1e-10, kickrank=5, maxranks[i]=ranks[i+1]+4·kickrank` | `ft.c:3055`, `ft.c:3075` |
| the pivots | `qmarray_lu1d(A, L, u, size_t * piv, double * px, …)` and `qmarray_maxvol1d(A, Asinv, size_t * pivi, double * pivx, …)` — **`piv` is a row index and `px` a continuous coordinate**, exactly §1.1's `(index, point)` pair | `qmarray.c:1928`, `qmarray.c:2262` |
| rounding | `function_train_round(ain, epsilon, aopts)`: `delta = ‖a‖₂ ε / √(d−1)`, then `function_train_orthor` (an `LQ` sweep) and a sweep of `qmarray_truncated_svd` | `ft.c:2579`, threshold at `ft.c:2594`, sweep at `ft.c:2620/2638` |
| the "continuous SVD" | `qmarray_truncated_svd` = **`qmarray_householder_simple("QR", …)` in function space, then a dense LAPACK SVD of the `R` factor**, with `U = qmam(Q, u, ·)` | `qmarray.c:3118`; `quasimatrix_householder` doxygen cites "(Trefethan 2010)", `quasimatrix.c:575` |
| the inner product | `generic_function_inner(a, b)` → `∫ a(x)b(x) dx`; `function_train_inner`, `function_train_integrate`, `function_train_norm2` | `funcs_mixed.c:62`; `ft.c:2356`, `ft.c:2138`, `ft.c:2505` |
| regression | `enum REGTYPE {ALS, AIO, REGNONE}`, `enum REGOBJ {FTLS, FTLS_SPARSEL2, REGOBJNONE}`; `ft_regress_run`, `ft_regress_run_rankadapt`; `struct FTparam` carries `params` and `nparams_per_uni` | `learning_options.h:51`; `regress.c:1249`, `regress.c:1306`; `parameterization.h:71` |

**The main loop, in one sentence, from `ft.c:3441/3459` and `ft.c:3621/3629`:
orthogonalise the fiber `Qmarray` by a function-space Householder QR (`"QR"`
left-to-right, `"LQ"` right-to-left), then run greedy `maxvol` on the orthogonal
factor to select `(row, x)` pivots.** That is structurally identical to
`tt/algs/cross.py::_left_basis` + `_select_rows`, with `x` added — which is the
best evidence available that §2.3 item 3 is a bounded change rather than a
rewrite, and simultaneously that it is not a small one.

**Basis adaptivity, per basis, from the source** — this is the table §2.3 item 4
is about:

| basis | adaptive? | evidence |
|---|---|---|
| Legendre, Chebyshev, Hermite | **yes, in order**: `orth_poly_expansion_approx_adapt` grows `N` (nested `N=2N−1` for Clenshaw–Curtis, else `N=N+7`) until the last `coeffs_check` coefficients fall below `tol`, absolutely and then relative to `Σc²` | `polynomials.c:3924`, loop at `:3960`, relative test at `:3987` |
| **Fourier** | **no** — the adaptation short-circuits and returns the starting order | `polynomials.c:3929-3937` |
| piecewise polynomial | **yes**, hierarchical splitting | `piecewisepoly.c:2522`, recursion at `:2505` |
| linear / constant elements | **yes**, knot insertion by bisection | `linelm.c:1915`, `constelm.c:1290` |
| Gaussian-kernel (RBF) | **no** `approx_adapt`; only a centre-adaptation flag used by the regression path | `kernels.c:548` |

**Two honest limits found in the source, not in the paper.**
(a) `generic_function_inner` **promotes both arguments to `PIECEWISE`** when
their classes differ, and asserts that neither is `LINELM`, `CONSTELM` or
`KERNEL` (`funcs_mixed.c:73-79`) — so the inner product is not defined across all
five function classes. (b) Rank adaptation grows the index set by **repeating the
last nodes** (`cross_index_copylast`, `ft.c:3841`), with the author's own comment
at `ft.c:3833`: *"simply repeat the last nodes / this is not efficient but I
don't have a better / idea. Could do it using random locations but / I don't want
to."* That is precisely the design point `cross-approximation.md` §5 P2 settled
in the *other* direction for ttpy2 — uniformly random extra rows (`kickrank2=2`)
fixed 6 of 6 seeds on its §1.3 counterexample. **The continuous library has the
weaker enrichment.**

**Not read:** `lib_probability`, `lib_quadrature`, `lib_tensdecomp`,
`lib_fft`, and all 40 101 lines of `c3_cex/`. **Not run** beyond the one smoke
example; **no C3 number appears anywhere else in this document.**

### 1.3 c3sc — stochastic optimal control on top of C3

> `github.com/goroda/c3sc`, cloned to `b300:~/work/ttpy-modern/scratch-ftt/c3sc`.
> The companion of A. Gorodetsky, S. Karaman, Y. Marzouk, *High-dimensional
> stochastic optimal control using continuous tensor decompositions*, Int. J.
> Robotics Research **37**(2–3) (2018) 340–377, doi:10.1177/0278364917753994,
> arXiv:1611.04706 (venue verified via CrossRef).

The idea, in ttpy2's conventions: the value function of a continuous-state
optimal control problem is represented as an FT and updated by value iteration,
so the dynamic-programming operator is applied *in the compressed continuous
format* and the state space is never gridded. This is the application that
motivates FT's continuous pivots — a control problem's value function has kinks
at switching surfaces whose location is not known in advance, which is the
`δ=0.01` column of F1/F2 in disguise.

**Cloned at `0236698` (2018-07-09), `LICENSE` a BSD-style MIT copyright, 6 632
lines of `.c`. Not built** — beyond C3 it needs `github.com/goroda/cdyn`, which
was not cloned, so the build was not attempted. **The paper was not read** — only
its abstract and CrossRef metadata.

### 1.4 FT regression — Gorodetsky, Jakeman

> A. A. Gorodetsky, J. D. Jakeman, *Gradient-based optimization for regression in
> the functional tensor-train format*, J. Comput. Phys. **374** (2018) 1219–1238,
> doi:10.1016/j.jcp.2018.08.010, arXiv:1801.00885. Verified via CrossRef.
> **Abstract only; full text not read.** Note the co-author is **Jakeman**, not
> Marzouk.

Learning the FT parameters from scattered samples by SGD / quasi-Newton on a
low-multilinear-rank parameterisation, rather than by cross. This is the
functional counterpart of the problem `tt/algs/completion.py` solves in the
discrete sparse-observation case and of `full-scale-examples.md` §1.7's
TT-regression-for-BSDEs question, and it is the same open question there: ttpy2
has no `tt.algs.regress`, and `ROADMAP.md` does not schedule one.

### 1.5 The Rust functional-TT library — Martinelli, Manzini

> M. Martinelli, G. Manzini, *A Functional Tensor Train Library in RUST for
> Numerical Integration and Resolution of Partial Differential Equations*, in
> I. Lirkov, S. Margenov (eds.), *Large-Scale Scientific Computations* (LSSC
> 2023, Sozopol, 5–9 June 2023, Revised Selected Papers), Lecture Notes in
> Computer Science **13952**, Springer, pp. 223–233, 2024,
> doi:10.1007/978-3-031-56208-2_22. Martinelli: IMATI "E. Magenes", CNR, Pavia;
> Manzini: T-5, Los Alamos.

Abstract, verbatim from the Springer landing page: *"…In recent years,
Function-Train decomposition, a continuous version of Tensor-Train
decomposition, was introduced. This decomposition permits the approximation of
high-dimensional functions without function sampling and provides an extensible
framework for function integration and differentiation. In this paper, we present
a new RUST-based library designed to provide functionality for Function-Train
decomposition. In addition, the library offers methods for continuous matrix
factorizations and continuous multilinear algebra operations, such as addition,
multiplication, integration, differentiation, etc."*

**Public repository: not found, but there is one live lead.** Neither the
abstract nor the landing page names a crate or a URL. GitHub repository searches
through the API returned `total_count = 0` for
`functional+tensor+train+language:Rust`, `"functional tensor train"+language:Rust`,
`martinelli+tensor+train`, `manzini+tensor+train` and `rust+quantics+tensor+train`;
`tensor+train+language:Rust` gives four unrelated repositories; crates.io's
`q=tensor train` gives nothing functional (closest: `tenrso-decomp`, a TT-SVD).

**The lead: `github.com/lanl/pyftt`**, described as *"Pytho wrapper for the
Functional Tensor Train (FTT) Library"*, created **2026-07-29**, no licence, no
language — and **empty**: `GET /repos/lanl/pyftt/contents/` returns
`{"message": "This repository is empty."}` and `raw.githubusercontent.com/.../
README.md` returns 404. Manzini is at LANL, so this is almost certainly the
intended wrapper, created and not yet populated. **Nothing was cloned because
there was nothing to clone.**

Caveat on the negative: GitHub's *code* search requires authentication
(`401 Requires authentication` unauthenticated), so only *repository* search was
usable and an oddly-named or private repository could have been missed. **"The
Rust FT library is publicly available" is unsubstantiated as of this search**, and
b300 has neither `cargo` nor `rustc` in any case.

The one thing worth taking from the abstract even without the code: it names the
same operation set this document proposes in §2.4 — *addition, multiplication,
integration, differentiation, and continuous matrix factorizations* — which is
independent evidence that that is the right minimal surface.

### 1.6 Spectral tensor train — Bigoni, Engsig-Karup, Marzouk

> D. Bigoni, A. P. Engsig-Karup, Y. M. Marzouk, *Spectral tensor-train
> decomposition*, SIAM J. Sci. Comput. **38**(4) (2016) A2405–A2439,
> doi:10.1137/15M1036919, arXiv:1405.5713. Authors, title, volume, issue, pages
> and year all verified against CrossRef. **Body read only through the ar5iv
> rendering** — see §8.

**What is "spectral" about it, and why it is *not* the same as §1.1.** Two
stages: sample `f` on a tensor grid of **Gauss quadrature points** and compute a
TT of that discrete tensor by **TT-DMRG-cross**; then approximate each resulting
**univariate core** in an orthogonal polynomial basis orthonormal with respect to
the measure `μ_k`. So STT **discretises first and projects second**, and the
"spectral" part is a polynomial fit *of the cores*. The convergence argument is a
truncation bound `‖R_TT‖²_{L²μ} ≤ Σ_i Σ_{α_i>r_i} λ_i(α_i)` from successive
Schmidt decompositions, plus the headline regularity result: the abstract states
that the regularity properties of `f` "are preserved" in the univariate
components, which is what licenses applying one-dimensional polynomial
approximation theory to them. Examples: modified Genz functions to `d = 100`, and
an elliptic PDE with random inputs, with "significant improvements… over an
anisotropic adaptive Smolyak approach".

**STT is exactly what ttpy2 can do today** — F1's Chebyshev column *is* a
poor-man's STT — and §1.1's whole criticism of it is that a fixed tensor-product
Gauss grid cannot adapt per fiber. **F1 and F2 measure that criticism directly
and confirm it** (§2.1): Chebyshev nodes beat uniform ones by 11× on a mildly
peaked function and **lose to them by 13×** when the peak is narrower than the
interior node spacing, while a graded piecewise basis at the same node count
beats both by four orders of magnitude.

**Code: `TensorToolbox`.** Named by the STT paper as
`http://pypi.python.org/pypi/TensorToolbox/` and independently cited by
Gorodetsky as the package that produced the STT numbers he compares against.
PyPI JSON metadata: **`TensorToolbox` 1.0.22**, author Daniele Bigoni,
"Tools for the decomposition of tensors", home page `www.limitcycle.it/dabi/`,
licence field `COPYING.LESSER` (LGPL-3 per the sdist's `setup.py` header, "DTU UQ
Library, Copyright (C) 2014-2016 The Technical University of Denmark"),
**latest release 2019-03-19**. The only URL in the metadata is
`http://www.limitcycle.it/dabi/`; **there is no GitHub repository** (a repository
search for `TensorToolbox bigoni` returns zero). Declared dependencies:
`scipy, SpectralToolbox, mpi4py, mpi-map, h5py, UQToolbox`, with `numpy` and
`Cython` in `setup_requires`, and **no `requires_python` constraint at all**.

**[measured] F7 — `TensorToolbox` resolves on PyPI and cannot be imported under
numpy 2.4.6.** Throwaway venv `~/work/ttpy-modern/scratch-ftt/ttbvenv`, created
from the ttpy2 Python 3.12, `numpy==2.4.6` pinned and verified unchanged
throughout.

1. `pip install TensorToolbox` — **fails**. pip prefers the 1.0.22 sdist, whose
   `setup.py` line 33 is literally `import pip`, a 2016-era hack that cannot run
   inside pip's isolated build environment:
   `× Getting requirements to build wheel did not run successfully. … File
   "<string>", line 33, in <module> / ModuleNotFoundError: No module named 'pip'`.
2. `pip install --only-binary=:all: TensorToolbox` — **succeeds, at 1.0.19**. pip
   rejects 1.0.22 (its `SpectralToolbox`, `mpi4py` and `UQToolbox` do not build),
   backtracks past 1.0.20, and installs 1.0.19, whose wheel declares no
   dependencies at all.
3. `import TensorToolbox` — fails three times, progressively:
   `ModuleNotFoundError: No module named 'scipy'` (`core/auxiliary.py:40`), then
   after installing scipy 1.18.0 `No module named 'h5py'` (`core/storage.py:35`),
   then after installing h5py 3.16.0 the hard wall:

```
  File ".../TensorToolbox/core/tensor_wrapper.py", line 108, in TensorWrapper
    data=None, dtype=np.float,
                     ^^^^^^^^
AttributeError: module 'numpy' has no attribute 'float'.
```

`np.float` was removed in numpy 1.24, and here it is a **default argument
evaluated at class-definition time**, so it fires on `import`, not on use.
TensorToolbox is Python-3-syntax-clean and frozen at the numpy ≤ 1.19 API; its
full feature set additionally wants `SpectralToolbox`, `mpi4py`, `mpi_map` and
`UQToolbox`, and the MPI dependencies are why the 1.0.22 resolution fails
outright.

**Consequence for this document.** The STT numbers that §1.1 §5.1 compares
against **cannot be reproduced today by installing the package that produced
them.** This is the third entry in the same pattern — `ttpy 1.2.0`, `qtt-laplace`,
`TensorToolbox` — and it is the same argument
`full-scale-examples.md` §1.4 makes: the reference implementations of this
literature are unreachable, and a pure-Python TT library that installs in seconds
is what unblocks them. C3 (§1.2) is the exception that proves it, and it took
three source-level fixes and a 5 136-symbol linker shim.

### 1.7 TT-IRT — Dolgov, Anaya-Izquierdo, Fox, Scheichl

> S. Dolgov, K. Anaya-Izquierdo, C. Fox, R. Scheichl, *Approximation and sampling
> of multivariate probability distributions in the tensor train decomposition*,
> Statistics and Computing **30**(3) (2020) 603–625,
> doi:10.1007/s11222-019-09910-z, arXiv:1810.01212. Abstract and metadata
> verified; **full text not read.**

**What it is, in ttpy2's conventions.** Build a TT of the target density by
**cross interpolation** — `tt.cross` on the density, nothing more — then sample
from it exactly by the **conditional-distribution construction**: walk the train
left to right, at mode `k` form the conditional probability vector
`p(i_k | i_{<k}) ∝ L_k · G_k[:, i_k, :] · R_{k+1}` where `R_{k+1}` is the
right-marginal recursion and `L_k` the accumulated left product, take its
cumulative sum, and invert it against a uniform. That map from the uniform cube
to the target is the **inverse Rosenblatt transport**. Because the TT is only an
approximation, the samples are debiased by one of three corrections:
Metropolis–Hastings accept/reject, control variates, or importance weighting with
a QMC lattice. Benchmarked against DRAM on failure-time modelling, an inverse
diffusion problem and Rosenbrock.

**F3 [measured] implements exactly this on top of today's ttpy2** and it is 60
lines (§4).

**Code: `github.com/dolgov/TT-IRT`**, BSD-2-Clause, primary language MATLAB,
created 2018-10-01, **last push 2025-03-17** (the liveliest repository in this
document), 368 KB. Its README states both a MATLAB and a Python part, with the
linear-spline IRT also in C (`matlab/utils/tt_irt1_int64.c`,
`python/tt_irt_py/tt_irt1_int32.c`) linkable as a MEX file or through `ctypes`,
and it depends on TT-Toolbox (MATLAB) or **`ttpy`** (Python). **Not cloned and
not run** — but note that its Python half depends on the `ttpy` that
`full-scale-examples.md` §1.4 measured to be unbuildable on any modern stack,
which makes it a second concrete consumer that ttpy2 would unblock.

### 1.8 Squared IRT and DIRT — Cui, Dolgov

> T. Cui, S. Dolgov, *Deep composition of tensor-trains using squared inverse
> Rosenblatt transports*, Found. Comput. Math. **22**(6) (2022) 1863–1922,
> doi:10.1007/s10208-021-09537-5, arXiv:2007.06968. Abstract and metadata
> verified; **full text not read.**

Two ideas, and both matter for §4. First, **squaring**: the transport is computed
from a **squared** TT decomposition, which makes the density non-negative by
construction and preserves monotonicity of the map — so one crosses `√π` and
squares, rather than crossing `π` and hoping. Second, **depth**: these
order-preserving transports are composed in a sequence ("deep"), each layer
bridging from the previous approximation to a sharper target, which is what makes
concentrated and strongly nonlinear targets tractable.

F3 implements the squaring and **does not** implement the depth, which is the
honest boundary in §4.3.

Related, in the same family and verified to exist but **not read**: T. Cui,
S. Dolgov, R. Scheichl, *Deep importance sampling using tensor trains with
application to a priori and a posteriori rare event estimation*, arXiv:2209.01941
(no journal reference on the abstract page); T. Cui, S. Dolgov, O. Zahm,
*Self-reinforced polynomial approximation methods for concentrated probability
densities*, arXiv:2303.02554 — the latter being the **sparse-polynomial** sibling
of the TT line and not itself a TT method.

### 1.9 Conditional DIRT — Cui, Dolgov, Zahm

> T. Cui, S. Dolgov, O. Zahm, *Scalable conditional deep inverse Rosenblatt
> transports using tensor trains and gradient-based dimension reduction*,
> J. Comput. Phys. **485** (2023) 112103, doi:10.1016/j.jcp.2023.112103,
> arXiv:2106.04170. Metadata verified; **full text not read.**

An **offline/online split**: the offline phase learns the *joint* law of
parameters and observables in TT; the online phase evaluates the order-preserving
*conditional* transport for newly observed data in real time — amortised
inference. Combined with gradient-based dimension reduction and heuristics for
reordering and reparameterising variables to lower the TT ranks. The reordering
heuristic is the piece with an immediate ttpy2 analogue: `tt.permute` exists
(`tt/core/tools.py:953`) and nothing in the repository ever chooses an ordering.

**Code: `github.com/DeepTransport/deep-tensor`**, MATLAB, licence reported by the
GitHub API as `NOASSERTION` ("Other"), created 2023-03-14, **last push
2023-10-26**, 370 KB. Its README states it implements all five papers of the
line (§1.7–§1.9 plus the two of §1.8) and builds on AMEn, functional TT and
ApproximationToolbox. **README read; code not cloned, not run.** There is no
repository called `TTDIRT`; that name appears to be a misremembering.

### 1.10 What is *not* in this line

For the record, so §0.1 is not re-litigated: **none of §1.1–§1.9 concerns
vibrational spectra, molecular Hamiltonians, or eigenvalue computation of any
kind.** The FT papers integrate and regress; the Dolgov sampling papers sample.
The eigenvalue line is `full-scale-examples.md` §1.3 and `eigenvalues.md`, and
they are already correct.

Two adjacent items checked and rejected as belonging here:
*Tensor train construction from tensor actions, with application to compression
of large high order derivative tensors* is **Alger, Chen, Ghattas** (SIAM J. Sci.
Comput. 42(5) 2020, arXiv:2002.06244), not Marzouk. Huan, Jagalur, Marzouk,
*Optimal experimental design: formulations and computations*, Acta Numerica **33**
(2024) 715–840, doi:10.1017/S0962492924000023, contains no tensor train in its
abstract.

---

## 2. What functional TT would mean for ttpy2, concretely

### 2.1 The measurement that motivates it

**[measured] F1 — a cross-approximated tensor is worth nothing between its
nodes, and which nodes you choose is the whole question.** b300, numpy 2.4.6,
float64, `OMP_NUM_THREADS=1`, single run.
`f(x) = 1/(δ² + Σ_{k=1}^{4}(x_k − c_k)²)` on `[0,1]⁴` with
`c = (0.3183, 0.4142, 0.5772, 0.6180)` — off every grid used —
`tt.cross(eps=1e-10, r=2, kickrank=2, rf=2, nswp=20, seed=0)`; "err on nodes" is
the relative Frobenius error against the **dense** `n⁴` reference; the off-grid
columns are max and `L²` relative error over **4000 uniform random points**,
reconstructed from the TT by per-mode weight matrices (piecewise-linear, or
global barycentric Lagrange on the node set).

`δ = 0.05` (the peak is wider than the uniform spacing at every `n`):

| nodes | n | rank | evals | s | err on nodes | off-grid, linear (max / L²) | off-grid, poly (max / L²) |
|---|---|---|---|---|---|---|---|
| uniform | 17 | 17 | 87 278 | 0.03 | 3.94e-11 | 1.23e-01 / 3.82e-02 | 4.24e+05 / 1.83e+05 |
| uniform | 33 | 19 | 381 942 | 0.07 | 5.73e-11 | 2.94e-02 / 8.89e-03 | 1.23e+17 / 3.87e+16 |
| uniform | 65 | 19 | 748 085 | 0.14 | 6.99e-11 | 3.19e-03 / 1.38e-03 | 5.85e+43 / 1.71e+43 |
| cheb | 17 | 16 | 87 771 | 0.02 | 2.61e-11 | 2.65e-01 / 8.17e-02 | 1.74e-01 / 1.26e-01 |
| cheb | 33 | 18 | 216 348 | 0.04 | 4.92e-11 | 1.94e-02 / 9.20e-03 | 1.23e-02 / 6.94e-03 |
| cheb | 65 | 19 | 714 805 | 0.13 | 6.57e-11 | 1.15e-02 / 4.09e-03 | **1.25e-03 / 3.71e-04** |

`δ = 0.01` (the peak is *narrower* than the interior Chebyshev spacing at `n=65`,
which is `π/128 ≈ 0.0245`):

| nodes | n | rank | evals | s | err on nodes | off-grid, linear (max / L²) | off-grid, poly (max / L²) |
|---|---|---|---|---|---|---|---|
| uniform | 17 | 18 | 94 979 | 0.02 | 3.28e-11 | 4.03e-02 / 1.92e-02 | 6.15e+06 / 3.20e+06 |
| uniform | 33 | 21 | 346 599 | 0.06 | 6.30e-11 | 1.54e-02 / 6.49e-03 | 1.65e+20 / 6.18e+19 |
| uniform | 65 | 24 | 1 100 970 | 0.21 | 5.87e-11 | 3.19e-03 / **1.47e-03** | 1.87e+48 / 6.76e+47 |
| cheb | 17 | 16 | 87 227 | 0.02 | 3.25e-11 | 1.92e-01 / 8.99e-02 | 5.79e-01 / 5.30e-01 |
| cheb | 33 | 19 | 322 641 | 0.06 | 6.95e-11 | 9.46e-02 / 3.48e-02 | 7.84e-02 / 4.53e-02 |
| cheb | 65 | 22 | 1 046 240 | 0.20 | 6.31e-11 | 7.32e-03 / 3.74e-03 | 2.63e-01 / **9.49e-02** |

Four things come out of that pair of tables.

* **The cross is not the problem.** `err on nodes` is 1e-11 in all twelve rows.
  ttpy2 does exactly what it promises: it reproduces the *array*. The array is
  simply not the function.
* **Reinterpreting a uniform-grid TT as a spectral one is a catastrophe, not an
  approximation.** `1.71e+43` is the Runge phenomenon at degree 64, and it is
  the reason "just evaluate the TT off the grid with a good interpolant" is not a
  design. The nodes and the reconstruction are one decision, not two.
* **Chebyshev nodes buy an order of magnitude when the feature is resolved**
  (3.71e-04 against 1.38e-03 at `n = 65`, `δ = 0.05`, at *fewer* evaluations)
  **and lose an order of magnitude when it is not** (9.49e-02 against 1.47e-03 at
  `δ = 0.01`). Chebyshev clustering is at the *boundary*; the peak is in the
  interior. **This is §1.1's criticism of §1.6, measured.**
* **A global spectral basis is therefore not the answer; an adaptive one is.**
  Which is F2.

**[measured] F2 — the same cross, the same `eps`, a piecewise basis graded to
the feature.** Identical regime; the only change is the node family: each mode
carries `n_el` elements with degree-`p` Chebyshev–Lobatto nodes inside, the
element breakpoints dyadically graded towards `c_k`, and the off-node
reconstruction is barycentric *inside the owning element*. This is C3's
`piecewise_poly` in 60 lines.

| δ | p | elements | n per mode | rank | evals | s | off-node max | off-node L² |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 4 | 2 | 10 | 16 | 33 780 | 0.02 | 2.43e-01 | 1.14e-01 |
| 0.05 | 4 | 6 | 30 | 15 | 111 000 | 0.02 | 6.14e-02 | 3.81e-02 |
| 0.05 | 4 | 10 | 50 | 19 | 511 150 | 0.10 | 2.20e-04 | 1.79e-04 |
| 0.05 | 6 | 10 | 70 | 19 | 825 300 | 0.16 | 6.50e-06 | 4.21e-06 |
| 0.05 | 8 | 10 | 90 | 19 | 1 154 070 | 0.25 | 1.25e-07 | **1.12e-07** |
| 0.01 | 4 | 2 | 10 | 15 | 24 120 | 0.01 | 4.76e+00 | 2.63e+00 |
| 0.01 | 4 | 6 | 30 | 14 | 116 610 | 0.02 | 5.97e-01 | 2.57e-01 |
| 0.01 | 4 | 10 | 50 | 21 | 467 700 | 0.08 | 4.06e-04 | 2.75e-04 |
| 0.01 | 6 | 10 | 70 | 24 | 1 135 400 | 0.22 | 3.66e-05 | 1.42e-05 |
| 0.01 | 8 | 10 | 90 | 26 | 1 361 520 | 0.26 | 1.97e-06 | **7.26e-07** |

At `δ = 0.01` and comparable evaluation counts (1.36e6 against 1.10e6), the
graded piecewise basis gives **7.26e-07** where the uniform grid gives
**1.47e-03** and the global Chebyshev grid gives **9.49e-02** — three to five
orders of magnitude, at **TT rank 26 against 24 and 22**. The TT structure is
doing the same work in every row. **The entire difference is univariate.**

**Honest counterweight, and it is a large one.** F2's grading was handed the
peak location `c_k`. A real adaptive method has to *find* it, and finding it is
the hard part of C3 (§1.1's `ε_approx` and the coefficient-decay refinement
criterion). F2 measures the size of the prize, not the cost of winning it. It
also does not contradict the trivial observation that a uniform grid with 10⁴
points per mode would do as well — it would need `10^{16}` evaluations in 4D and
the cross would need `10⁴`-long fibers, which is the point.

### 2.2 What survives from today's ttpy2

Almost all of it, on one condition: **the univariate basis must be nodal**, i.e.
the coefficients are values at points (Lagrange at the node set), not modal
expansion coefficients.

| ttpy2 piece | survives? | why |
|---|---|---|
| `tt.vector` core storage `(r,n,r)` | **yes, unchanged** | `n` is "how many univariate degrees of freedom", not "how many grid points". Nothing in `tt/core/vector.py` cares |
| `+`, scalar `*`, `kron`, `concatenate` | **yes, unchanged** | linear in the coefficients, and a nodal basis makes sums of coefficients sums of functions |
| Hadamard `x * y` | **only with a nodal basis** | values multiply pointwise; expansion coefficients do not. In a Legendre-coefficient basis `*` would need a triple-product tensor per mode and would be a different function |
| `round(eps)` | **yes, after a change of variables** — F5 | see below |
| `tt.cross` / `rect_cross` | **yes, unchanged** | it only ever indexes; F1 and F2 ran it on three different node families with no modification |
| `multifuncrs` | **yes, with the same nodal caveat as Hadamard** | it evaluates `funs` on sampled *values*, which is already the functional semantics |
| `tt.matrix`, `amen_solve`, `amen_mv` | **yes for a Galerkin discretisation** | a linear operator between two fixed bases *is* a `tt.matrix`; this is what `iga.py` already does (§3) |
| `tt.dot` | **no** — needs a mass matrix | `dot` is the Euclidean coefficient inner product; `∫fg` is `dot(x, M·y)`. One line, but a different line |
| QTT: `qtt_ell`, `zkron`, `zkronv`, `reshape`, `qlaplace_dd` | **no** | quantics is a statement about a *grid*, `n = 2^L`. There is no quantics of a Legendre basis |
| `riemannian.*`, `completion.*`, `ksl` | **yes, formally** | they manipulate cores; but their geometry is the Euclidean one, so their retractions and projections are `L²`-correct only in the transformed variables of F5 |

**[measured] F5 — `tt.round` is already the continuous truncation, in the right
variables.** b300, numpy 2.4.6, float64, `OMP=1`, single run. `d = 4`,
Chebyshev–Lobatto `n = 33` nodal Lagrange basis on `[0,1]`, mass matrix
`M[s,t] = ∫₀¹ φ_s φ_t dx` by 400-point Gauss–Legendre, `M = L Lᵀ` (Cholesky);
`f` as in F1 with `δ = 0.05`; `tt.cross(eps=1e-12)` → rank 21, 347 556
evaluations.

    ‖f‖_{L²}   (by contracting the TT against M in every mode)  = 6.264226022800e+00
    ‖Lᵀc‖_F    (Frobenius norm of the transformed coefficients) = 6.264226022800e+00
    relative difference                                          = 1.42e-16
    ‖c‖_F      (the raw coefficients)                            = 3.632666569798e+03

| target rank | `L²` error, rounding the RAW coefficients | `L²` error, rounding in the `L²` variables | ratio |
|---|---|---|---|
| 2 | 1.2080e-01 | 1.0350e-01 | 1.17 |
| 4 | 8.8330e-03 | 7.0111e-03 | 1.26 |
| 6 | 5.3603e-04 | 4.2384e-04 | 1.26 |
| 8 | 4.0899e-05 | 3.0151e-05 | 1.36 |
| 10 | 2.0595e-06 | 1.4845e-06 | 1.39 |
| 12 | 1.1932e-07 | 8.5117e-08 | 1.40 |

Two readings, and the first matters more than the second. **(a)** The identity
`‖Lᵀc‖_F = ‖f‖_{L²}` is exact, so "a continuous rank truncation" is not an
algorithm we lack — it is `tt.vector.round` composed with two `einsum`s. The
factor `L` is the only new object, it is `(n,n)` per mode, and for a Gauss or
Chebyshev *quadrature* basis it degenerates to `diag(√w)`. **(b)** The factor of
1.17–1.40 is the accuracy actually gained; it is real, consistent, and growing
with rank, but it is not the reason to do this. The reason is that without it
`round(eps)` controls a quantity — the Frobenius norm of a coefficient array
that here is 580× larger than the function's norm — that no user asked about.

### 2.3 What does not survive, and is genuinely new

Four things, in increasing order of difficulty.

1. **A univariate basis object.** Today `n` is an integer. A functional TT needs,
   per mode: the node set the black box is sampled at, an evaluation map to
   arbitrary points, a mass matrix (and its derivative-weighted variants), and a
   quadrature weight vector. `tt/algs/iga.py::gram_blocks` **already computes
   three of those four** (§3).
2. **An inner product that is not the Euclidean one.** `tt.dot(x, y)` must become
   `∫ f g`, i.e. `dot(x, apply_mass(y))`. Ten lines, one owner, and it must be
   the *only* owner — an `L²` `dot` and a coefficient `dot` living side by side
   with similar names is exactly the "two owners of one truth" that
   `docs/REQUIREMENTS.md` and `ROADMAP.md` §2 spend their length preventing.
3. **Adaptive point selection.** F2's grading was given. §1.1's Algorithm 2
   *computes* it: the pivot is an `(index, point)` pair and the maxvol
   maximisation runs over the continuum — reducible to an eigenvalue problem for
   orthonormal-polynomial fibers, and to a per-piece search for piecewise ones.
   This changes `cross._select_rows` from "pick rows of a matrix" to "pick rows
   of a matrix-valued function", and it changes the index bookkeeping in
   `cross.py` (which the module docstring is careful to note is plain integer
   combinatorics) into something that carries floats. **This is the boundary
   between a module and a package.**
4. **Online per-fiber basis adaptation.** §1.1's `ε_approx`: each fiber is
   refined *while the cross is running*, so `n_k` is not merely different per
   mode but different per `(α_{k−1}, α_k)` entry of the core. That breaks the
   `(r, n, r)` array outright — a core becomes a ragged collection of univariate
   approximations. **Nothing in `tt/core/` survives this**, and it is why C3 is a
   C library with its own object model rather than a layer on a TT library.

### 2.4 The smallest useful subset, with exact signatures

The proposal is the **nodal, fixed-basis** subset — items 1 and 2 of §2.3, not 3
and 4. It is additive, touches nothing under `tt/core/`, and is testable against
closed forms (§1.1's table).

```python
# tt/functional/basis.py -- the univariate layer, the only new concept

class Basis:
    """A univariate approximation space, sampled at `nodes`.

    Nodal by contract: a coefficient vector `c` of length `n` means the
    function `sum_j c_j phi_j`, with `phi_j(nodes[i]) = delta_ij`.  That is
    what makes tt.vector's Hadamard product and tt.multifuncrs mean what they
    already mean; a modal basis would need a different product and is out of
    scope (see functional-tt.md 2.2).
    """
    n: int                          # univariate degrees of freedom
    nodes: np.ndarray               # (n,) where the black box is evaluated
    domain: tuple[float, float]

    def eval(self, x: np.ndarray) -> np.ndarray:
        """(m,) points -> (m, n) reconstruction weights.  f(x) = eval(x) @ c."""

    def gram(self, a: int = 0, b: int = 0) -> np.ndarray:
        """(n, n) with G[s,t] = int D^a phi_s D^b phi_t.  gram(0,0) is the mass."""

    def quad(self) -> np.ndarray:
        """(n,) with w[j] = int phi_j, so `w @ c` is the integral of f."""

    def chol(self) -> np.ndarray:
        """(n, n) lower L with gram(0,0) = L L^T; cached.  See ftt.round."""


class Chebyshev(Basis):      # global, nodal at Chebyshev-Lobatto points
    def __init__(self, n: int, domain=(0.0, 1.0)): ...

class Legendre(Basis):       # global, nodal at Gauss-Legendre points; gram is diagonal
    def __init__(self, n: int, domain=(0.0, 1.0)): ...

class PiecewisePoly(Basis):  # the one F2 measures, and C3's workhorse
    def __init__(self, breaks: np.ndarray, p: int): ...
    def refine(self, elements: np.ndarray) -> "PiecewisePoly":
        """A new basis with the named elements bisected.  Not used by the
        fixed-basis path; the hook that item 3 of 2.3 would need."""

class Spline(Basis):         # exactly what tt/algs/iga.py already builds
    def __init__(self, p: int, n_el: int, domain=(0.0, 1.0)): ...
```

```python
# tt/functional/core.py -- the train

class ftt:
    """A tt.vector plus one Basis per mode.  The cores ARE the tt.vector's."""
    tt: tt.vector                   # cores (r_k, n_k, r_{k+1}) of COEFFICIENTS
    bases: list[Basis]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(m, d) points -> (m,) values.  One contraction per mode; the same
        loop F1 and F2 measure."""

    def round(self, eps: float = 1e-14, rmax: int | None = None) -> "ftt":
        """L2 rank truncation.  Transform every core by bases[k].chol().T, call
        tt.vector.round, transform back.  Measured exact (F5): the Frobenius
        norm of the transformed cores IS ||f||_{L2}, to 1.4e-16."""

    def integral(self) -> float:
        """int_Omega f.  Contract each core against bases[k].quad()."""

    def dot(self, other: "ftt") -> float:
        """int_Omega f g.  tt.dot(self.tt, mass_apply(other)).  THE inner
        product of this layer; tt.dot is not it."""

    def norm(self) -> float:        # sqrt(self.dot(self))
    def __add__, __mul__, __sub__   # delegate to tt.vector (nodal contract)


def cross(fun, bases: list[Basis], eps: float = 1e-8, **kw) -> ftt:
    """Cross approximation of a function of a CONTINUOUS argument.

    `fun` takes (m, d) float points and returns (m,) values -- note the
    difference from tt.cross, whose `fun` takes integer multi-indices.  The
    node lookup is this function's job, which is the whole of the wrapper.
    **kw goes to tt.algs.cross.rect_cross unchanged (F1, F2).
    """


def galerkin(coef: ftt, bases_test: list[Basis], deriv: list[tuple[int, int]]
             ) -> tt.matrix:
    """The operator whose (s,t) entry in mode k is
    sum_q coef_core[a,q,b] * bases_test[k].gram(deriv[k][0], deriv[k][1]).

    This is tt/algs/iga.py::_direction_core, generalised off B-splines and
    off three dimensions.  It is the bridge back to amen_solve.
    """
```

Estimated size: `basis.py` ~220 lines for the four bases, `core.py` ~120,
`galerkin` ~40, plus tests. **The tests are the cheap part and they are strong**:
partition of unity; `eval(nodes) == I`; `gram(0,0)` against dense quadrature;
`integral()` against §1.1 §5.3.1's `Im[((e^i−1)/i)^d]` and §5.3.2's
`((e^{5/2}−1)/5)^d`; and F5's identity `‖Lᵀc‖_F = ‖f‖_{L²}` as a regression test.

### 2.5 Module, or separate package? — the honest answer is "both, and the line is sharp"

**A module.** Items 1–2 of §2.3, i.e. everything in §2.4. It is a leaf: it
imports `tt.vector`, `tt.cross`, `tt.matrix` and nothing imports it. It adds no
dependency (`REQUIREMENTS.md` R1 is safe: pure numpy). It does not touch
`tt/core/`. It has one owner for the inner product and one for the truncation.
It is the natural home of `iga.py`'s spline layer (§3), so it *reduces*
duplication rather than adding it.

**A separate package.** Items 3–4. The moment the cross's pivots carry
floating-point coordinates and the cores become ragged, the `(r,n,r)` array is no
longer the representation, `tt.round` is no longer the truncation, and
`tt.vector` is no longer the object. That is not an extension of ttpy2; it is a
second library that would use ttpy2 for the small-matrix linear algebra and
nothing else. **C3 is the evidence, and F6 quantifies it**: 55 912 lines of C
built around its own `GenericFunction` / `Quasimatrix` / `Qmarray` /
`FunctionTrain` object model, of which `lib_funcs` — the univariate layer alone —
is 21 750. It did not grow inside a TT library, and it is not shaped like one:
`function_train_round` reimplements TT-rounding from scratch on top of
`qmarray_truncated_svd` because there is no `(r,n,r)` array for a discrete
rounding to act on.

**The counterweight against the module.** F2 measures a four-order-of-magnitude
prize and F2's grading was *given*. Everything the fixed-basis module buys, a
user could get today by choosing better `n` and doing the reconstruction by hand
— F1 and F2 are exactly that, in 120 lines of script. The module's real value is
that it makes the node family and the reconstruction **one object instead of two
independent choices**, which is what turns `1.71e+43` into a type error instead
of a plot. If that argument does not persuade, the module should not be built.

---

## 3. The relationship to `tt/algs/iga.py`

**`iga.py` is a functional TT with a fixed basis — on the test side only.** The
split is precise, and it is worth stating exactly because it is the shortest
description of what "functional" adds.

| the FT concept | `iga.py`'s realisation | complete? |
|---|---|---|
| a univariate basis | `open_knots` + `bspline_basis` (Cox–de Boor, degree `p`, open knot vector) | **yes** — it is `Basis` minus the interface |
| the basis's Gram matrices | `gram_blocks(p, n_el)` returns `B[(a,b)][q,s,t] = w_q D^a N_s(x_q) D^b N_t(x_q)` for `a,b ∈ {0,1}` | **yes** — this *is* `Basis.gram(a, b)`, un-contracted |
| the sample points | `quadrature(p, n_el)` — Gauss–Legendre, `p+1` per element, i.e. already a **piecewise-polynomial node family**, not a uniform grid | **yes**, and it is F2's node family by accident |
| the coefficient field as a function | `geometry_field` returns `R[(i,j)]` as a **`tt.vector` over the quadrature index** with no basis attached | **no.** This is the gap |
| the continuous truncation | `stiffness_tt(..., eps=)` and `cross(eps=)` round in the Euclidean coefficient norm | **no** — F5's change of variables is absent |
| the operator | `_direction_core` contracts the coefficient core against a Gram block and emits a `tt.matrix` core | **yes** — it is `galerkin()` of §2.4, specialised to `d=3` and B-splines |

So `iga.py` **is** an FT whose bases are B-splines and whose cross runs on a
Gauss node family, with one thing missing: the object `tt.cross` returns is a
tensor of *values at quadrature points*, and nothing in the module can evaluate
it anywhere else.

**[measured] F4 — what that gap is worth, on the module's own example.** b300,
numpy 2.4.6, float64, `OMP=1`, single run. `p = 2`; `geometry_field(..., jac=…,
eps=1e-10)`; "err AT nodes" is the max relative error of `element(R₀₀, idx)`
against the exact `R₀₀` on 4000 random quadrature multi-indices; "err OFF nodes"
is the same over 2000 uniform random points in the cube, reconstructed
piecewise-linearly from the same TT.

| geometry | n_el | quad pts/dir | rank(R₀₀) | err AT nodes | err OFF nodes |
|---|---|---|---|---|---|
| ring (`examples/iga_ring.py`) | 8 | 24 | 1 | 1.99e-15 | 2.12e-15 |
| ring | 16 | 48 | 1 | 1.13e-15 | 1.20e-15 |
| ring | 32 | 96 | 1 | 3.68e-15 | 3.11e-15 |
| twisted duct (M1's map) | 8 | 24 | 2 | 2.38e-10 | 2.39e-04 |
| twisted duct | 16 | 48 | 2 | 2.72e-10 | 5.59e-05 |
| twisted duct | 32 | 96 | 2 | 3.65e-10 | 1.32e-05 |

**The ring rows are a warning about the example, not a result about the method.**
For the analytic annulus `R₀₀ = θ_max H r / Δr` is *affine in `ξ₀`*, so
piecewise-linear reconstruction is exact and the question is invisible. On the
twisted duct — the map of `full-scale-examples.md` M1, where `R` is not a
polynomial in any direction — the same TT is right to **2.4e-10 where it looks**
and **2.4e-04 halfway between**, six orders of magnitude apart, and the gap
closes only as `O(h²)` (2.39e-04 → 5.59e-05 → 1.32e-05 for `n_el` doubling
twice: rates 2.10 and 2.08). `R` itself is *smooth*; the `h²` is purely the
reconstruction's.

**[measured] F4(b) — and the knob that looks like it should fix it does not.**
Same regime, ring, `p = 2`, `n_el = 16`, varying only `per_element` (the
quadrature order), solving the full Poisson problem to `amen_solve(eps=1e-10)`
and comparing against Eq. (46):

| per_element | quad pts/dir | rank(K) | solve s | max rel err vs Eq. (46) |
|---|---|---|---|---|
| 3 (the default, `p+1`) | 48 | 3 | 0.17 | 1.624e-07 |
| 4 | 64 | 3 | 0.01 | 1.624e-07 |
| 6 | 96 | 3 | 0.01 | 1.624e-07 |
| 10 | 160 | 3 | 0.01 | 1.624e-07 |

Identical to four digits. **The mesh, not the quadrature, is binding on this
problem** — which is the honest counterweight to everything above: making
`iga.py` continuous buys *nothing* for its own headline example. It buys
something when (i) `R` is not polynomial and (ii) somebody asks for a value at a
point that is not a quadrature point. Both conditions hold for post-processing
(visualisation, a functional at a physical location, coupling to another code)
and for any geometry that is not the analytic ring.

**The shortest path from `iga.py` to something genuinely continuous**, in the
order the work would be done:

1. **Give `gram_blocks` an interface instead of a return value.** `blocks[(a,b)]`
   becomes `Basis.gram(a, b)` and `quadrature()` becomes `Basis.nodes`; the
   B-spline code moves verbatim to `tt/functional/basis.py::Spline`. Zero
   behavioural change, and `iga.py` becomes the first consumer of the new layer.
   ~1 day, and the acceptance criterion is that `tests/test_iga.py` and
   `examples/iga_ring.py` produce the same numbers to the last digit.
2. **Make `geometry_field` return `ftt` instead of `tt.vector`.** The cross is
   already sampling at `Basis.nodes`; attaching the basis is one constructor call
   and it makes `R(ξ)` evaluable anywhere. This is the change F4 measures the
   value of. Note it needs a *second* basis: the quadrature node family
   (`(p+1)·n_el` nodes) is not the trial basis (`n_el+p` splines), and today
   `iga.py` keeps both only implicitly — `_direction_core` **is** the change of
   basis between them.
3. **Route `stiffness_tt`'s rounding through F5's change of variables.** Its
   `eps` currently controls the Frobenius norm of a coefficient array; after
   step 1 the mass matrix is available and it can control `‖K‖` in the energy
   sense. Ten lines, and it is the only step that changes a number.
4. **Stop there.** Adaptive knot insertion driven by the cross — refining the
   spline space where `R` is badly resolved — is item 3 of §2.3 and it is
   research (§6, Q34).

Steps 1–3 are a refactor plus twenty lines. They are **not** on the critical
path of anything in `ROADMAP.md` and should be done when `tt/functional/` is
built, not before.

---

## 4. Sampling and inverse problems — the Marzouk-adjacent line

### 4.1 What TT-based sampling buys

A TT of a density is a **sampler**, not just an approximation. The conditional
construction of §1.7 produces **independent, exact samples of the TT surrogate**
at `O(d n r²)` per sample, with no burn-in, no autocorrelation and no proposal
tuning — the three things that make MCMC expensive to use and hard to certify.
The approximation error does not go away; it moves into a **weight**, and the
honest metric becomes the effective sample size

    ESS/N = (Σ w_i)² / (N Σ w_i²),    w_i = π(x_i) / π_TT(x_i),

which is measurable without knowing either normalising constant. That is the
number below.

### 4.2 [measured] F3 — it is 60 lines on top of today's ttpy2, and it works

b300, numpy 2.4.6, float64, `OMP=1`, single run. `d = 5`, `n = 33` uniform nodes
per mode (39 135 393 dense entries, reference computed densely in ~5 s). Recipe,
following §1.8: `tt.cross` on **`√π`**, then `p = f * f` (Hadamard square — so
non-negativity is structural, not hoped for), rounded at `eps²`; then the
right-marginal recursion `R[k] = (Σ_i G_k[:,i,:]) @ R[k+1]`, a `cumsum` along
each mode's conditional, and a `searchsorted`. 200 000 samples. Two densities on
`[0,1]⁵`: a log-concave nearest-neighbour Gibbs density
`U = 10Σ(x_k−½)² + 8Σx_k x_{k+1}`, and a Rosenbrock ridge
`U = 4Σ[10(y_{k+1}−y_k²)² + (1−y_k)²]`, `y = 3x − 1.5`.

| density | cross eps | rank `√π` | rank `π` | evals | cross s | square s | 200k samples s | **ESS/N** | max mean err |
|---|---|---|---|---|---|---|---|---|---|
| Gibbs | 1e-1 | 2 | 3 | 17 028 | 0.01 | 0.00 | 1.53 | **0.9946** | 6.31e-04 |
| Gibbs | 1e-2 | 3 | 5 | 46 728 | 0.01 | 0.00 | 1.74 | **1.0000** | 4.48e-04 |
| Gibbs | 1e-4 | 5 | 8 | 46 728 | 0.01 | 0.00 | 2.10 | 1.0000 | 9.41e-04 |
| Gibbs | 1e-6 | 6 | 11 | 46 728 | 0.01 | 0.00 | 2.44 | 1.0000 | 7.70e-04 |
| Gibbs | 1e-8 | 8 | **64** | 104 511 | 0.03 | 0.02 | **8.77** | 1.0000 | 2.72e-04 |
| Rosenbrock | 1e-1 | 8 | 11 | 231 957 | 0.05 | 0.00 | 1.99 | **0.4687** | 7.09e-04 |
| Rosenbrock | 1e-2 | 11 | 13 | 231 957 | 0.05 | 0.01 | 2.32 | **0.9999** | 2.02e-04 |
| Rosenbrock | 1e-4 | 14 | 16 | 231 957 | 0.05 | 0.10 | 2.84 | 1.0000 | 2.09e-04 |
| Rosenbrock | 1e-6 | 15 | 16 | 231 957 | 0.05 | 0.20 | 2.88 | 1.0000 | 3.10e-04 |
| Rosenbrock | 1e-8 | 16 | **256** | 412 995 | 0.08 | 0.60 | **49.58** | 1.0000 | 1.90e-04 |

Four readings.

* **The primitive is not missing.** Everything above the sampler is `tt.cross`
  and `tt.vector.__mul__`. The sampler itself is the right-marginal recursion
  (4 lines), a `cumsum` (1) and a `searchsorted` (1). There is no third-party
  dependency and no new core representation.
* **The accuracy knob is `ESS`, and it moves where it should.** Rosenbrock at
  `eps = 1e-1` gives **0.4687** — half the samples wasted — and one step of
  `eps` recovers it. The Gibbs density never falls below 0.9946, which is what
  "log-concave, nearest-neighbour coupling" is supposed to mean.
* **`max mean err ≈ 2e-4…9e-4` in every row is Monte-Carlo noise, not TT
  error.** At `N = 200 000` and a coordinate standard deviation near 0.1 the
  sampling error of the mean is `0.1/√N ≈ 2.2e-4`. This column measures the
  measurement, and it is reported to say so rather than to be quoted.
* **The trap is the square's rounding tolerance, and it has no owner.** At
  `eps = 1e-8` the square is rounded at `eps² = 1e-16`, which is below float64
  resolution, so nothing compresses: rank 16 → **256**, and the sampler goes from
  2.9 s to **49.6 s** for no accuracy gain (ESS was already 1.0000 at `eps=1e-4`).
  `eps²` is the *mathematically* right tolerance for a squared quantity and the
  *numerically* wrong one. Any `tt.algs.irt` must own that choice explicitly
  rather than deriving it, and must warn when the squared rank exceeds the
  square of the root's rank.

### 4.3 What it needs from us, precisely

| ingredient | today | needed |
|---|---|---|
| a non-negative TT of the density | `tt.cross` on `√π`, then `x * x` | **have it.** The only addition is the tolerance decision above |
| conditional marginals | — | `R[k] = (Σ_i G_k[:,i,:]) @ R[k+1]`; 4 lines. Belongs in `tt/algs/irt.py`, not in `tt/core/` |
| a cumulative sum along a mode | `np.cumsum` on the `(m,n)` conditional | **discrete: have it.** *Continuous:* `Basis.antiderivative()` — `∫ᵃˣ φ_j` as an `(n,n)` map — is a genuinely new primitive of §2.4's `Basis`, and the only one F3 did not need |
| a root find | `np.searchsorted` on the discrete CDF | *continuous:* one monotone scalar solve per sample per mode. The CDF is monotone by construction, so bisection cannot fail; it is `O(d N log(1/tol))` and it is embarrassingly parallel |
| the debiasing | — | importance weights (F3 computes them), or MH accept/reject, or a QMC lattice (§1.7). All three are ~20 lines *given* the weights |
| the depth (DIRT) | — | a bridging-density schedule, a ratio function, and a composition of maps. **Not 20 lines** — §1.8 |

### 4.4 Verdict: is this ttpy2's kind of library?

**The discrete TT-IRT is**, and it should be `tt/algs/irt.py`: ~120 lines with
tests, no new representation, no new dependency, and it converts a TT the
repository already knows how to build into something a statistician can use.
F3 is the evidence that it works before it is written. It also has an
independently useful by-product: `ESS/N` is a **held-out accuracy indicator that
does not require a dense reference**, which is precisely what
`cross-approximation.md` §1.3's failure case lacked — the counterexample there
had every internal indicator reading 1e-15 at a true error of 3.82e-04.
Whether importance weights would have caught it is Q33 below.

**The deep / conditional version (DIRT) is not.** It owns a temperature ladder,
a bridging schedule, a composition of transports, and a debiasing strategy — four
decisions that are statistics, not tensor algebra, and that `deep-tensor`
(§1.9) implements in MATLAB in a package of its own. Building it inside ttpy2
would put a second owner of "what a good approximation is" next to
`REQUIREMENTS.md` R4. **It should be a package that depends on ttpy2**, and
`TT-IRT`'s Python half already wants exactly that, blocked today only by its
`ttpy` dependency (§1.7).

---

## 5. Full-scale examples these lines enable

Marked (i) buildable today, (ii) after a named milestone, (iii) needs research.
Each carries a reference value that does not come from ttpy2. Fitted into
`ROADMAP.md` §4's M-track and `full-scale-examples.md` §4's E-track as **stage
E9 onward** — none of them blocks or is blocked by anything already scheduled.

### 5.1 (i) The discontinuous Genz integral in `d = 10` and `d = 100`

`f(x) = 0` if any `x_i > ½`, else `exp(Σ 5x_i)`, on `[0,1]^d`. **Reference:
`I = ((e^{5/2} − 1)/5)^d` exactly; §1.1 §5.3.2 states `I ≈ 3.131e3` at `d = 10`
and `I ≈ 9.05455e34` at `d = 100`.** A closed form, no data file, no geometry,
any `d`.

This is the cleanest possible demonstration of the whole document, because the
discontinuity is at `x_i = ½` and **a basis with a breakpoint there is exact**
while a global polynomial basis cannot converge at all. Two arms: (a) today, with
`tt.cross` on a uniform grid plus trapezoid weights, which converges at `O(h)`
and is the honest baseline; (b) with `tt/functional/`'s `PiecewisePoly([0, ½, 1],
p)`, which should be exact to rounding. **~90 lines for both arms**, and the
comparison against §1.1's own `d = 100` value is a check on a *published number*,
not on ourselves. Arm (a) is (i); arm (b) is (ii) after the module of §2.4.

**Counterweight.** It is a quadrature demonstration, not an application: nobody
outside numerical analysis needs `((e^{5/2}−1)/5)^{100}`. It earns its place as
the acceptance test of §2.4, not as a full-scale example in
`full-scale-examples.md`'s strict sense.

**And it now has a running oracle.** F6 built `c3py` on b300, so this example can
be checked against **the authors' own implementation** at the same tolerances,
which is a stronger oracle than the paper's printed number and is available for
none of the other entries in either document.

### 5.2 (i) `sin(x₁+…+x_d)` at `d > 600`

**Reference: `Im[((e^i − 1)/i)^d]` exactly** (§1.1 §5.3.1), which the FT paper
reports holding for `d > 600` at evaluations linear in `d`. Rank 2 by
construction. This is a *scaling* example rather than an accuracy one, and it is
the cheapest possible check that a functional layer has not introduced a
`d`-dependent constant. **~40 lines, (i) today** on any node family — F1 shows
`tt.cross` does not care.

### 5.3 (ii) A Bayesian inverse problem sampled by TT-IRT

After `tt/algs/irt.py` (§4.4). The natural target is one of §1.7's three
benchmarks — failure-time modelling, an inverse diffusion problem, or Rosenbrock
— but **their published reference values were not read** (abstract only), so the
concrete oracle has to come from either the paper's text or from `TT-IRT`'s own
MATLAB test scripts, neither of which this survey obtained. **Until then the
honest oracle is a dense one**: F3's `d = 5`, `n = 33` reference is exactly that,
and it is what a first test should assert against.

The example that would make this *full-scale* rather than a unit test is the
inverse-diffusion one, because it is the same elliptic PDE that
`qtt-elliptic-bpx.md` and `iga.py` already solve — the forward map is a solve the
repository owns, and the posterior is a function of its output. That makes it the
only entry in this document that composes two existing capabilities instead of
adding one. **(ii)**, after `tt/algs/irt.py`; the forward map needs nothing new.

### 5.4 (ii)/(iii) FT regression from scattered data

§1.4. Fit an `ftt` to samples rather than crossing it. This is the same
unscheduled `tt.algs.regress` that `full-scale-examples.md` §1.7 identifies for
BSDEs and that `completion.py` half-implements for the discrete sparse case.
**(iii)** as stated; **(ii)** if it is restricted to the fixed-basis linear
least-squares case, where it is an ALS over cores and is genuinely close to
`completion.py`.

### 5.5 (iii) Stochastic optimal control in FT

§1.3. Value iteration with the value function in FT. It needs continuous pivots
(§2.3 item 3) because the value function's kinks move as the iteration proceeds
— which is the one problem where a *fixed* basis is provably the wrong tool.
**(iii)**, and it is the honest answer to "why would we ever build items 3–4 of
§2.3": this, and nothing smaller.

### 5.6 Where these sit in `ROADMAP.md`'s order

| stage | lands | needs | why here |
|---|---|---|---|
| **E9** — `tt/algs/irt.py` | the sampler of §4.3, the `ESS` diagnostic, `tests/test_irt.py` against F3's dense reference | **nothing** | measured working before being written (F3); 120 lines; independent of every M-milestone; and its `ESS` by-product is a free error indicator for `cross-approximation.md` §1.3's open problem |
| **E10** — `tt/functional/` | §2.4's `Basis`, `ftt`, `cross`, `galerkin`; §5.1 arm (b) and §5.2 as tests | **nothing** | a leaf module; but it should land **after E0** (`full-scale-examples.md`'s spline layer), because `iga.py`'s `gram_blocks` is its first and best-tested `Basis` and rewriting it twice is waste |
| **E11** — `iga.py` on `tt/functional/` | steps 1–3 of §3 | E10 | pure refactor plus F5's change of variables; the acceptance criterion is that `examples/iga_ring.py` prints the same digits |
| **E12** — the inverse-diffusion posterior | §5.3 | E9, and `qtt_ell.solve` (`ROADMAP.md` A10) | the only entry that composes two owned capabilities |
| **not scheduled** | continuous pivots, ragged cores, DIRT, FT regression | a spec of their own | §2.3 items 3–4, §4.4, §5.4–§5.5. Each is a package or a research question, and putting them on a milestone list would misrepresent them |

**The one ordering worth arguing about.** E9 before E10, i.e. the sampler before
the functional layer, inverts the intellectual order (the sampling papers of
§1.7–§1.9 use *functional* TT internally). The justification is measured: F3
shows the discrete sampler already works and needs nothing, while F1/F2 show the
functional layer's value is real but is a *refactor of a choice users can already
make by hand*. Cheap and done beats correct and pending.

---

## 6. Open questions

Numbered from Q31 to avoid colliding with `ROADMAP.md` §6 (Q1–Q15) and
`full-scale-examples.md` §6 (Q20–Q30).

**Q31 — Does continuous maxvol actually find pivots a discrete one misses, at
equal cost?** *Blocks:* the whole of §2.3 item 3, and therefore the
module-versus-package decision. §1.1's Algorithm 2 is stated but **not proved to
converge**, and its cost per pivot is an eigenvalue problem rather than a row
comparison. *Experiment:* on F2's `δ = 0.01` problem, compare (a) the graded
basis with the grading *given* — measured, 7.26e-07 — against (b) a uniform
initial basis refined by §1.1 §5.1's coefficient-decay criterion, at equal
evaluation budget. If (b) does not get within an order of magnitude of (a), the
adaptivity is not paying for its complexity on this class. One day, and it needs
only F2's script plus a refinement loop.

**Q32 — What is the right rounding tolerance for a squared TT density?**
*Blocks:* `tt/algs/irt.py` (E9). F3 measured `eps²` taking rank 16 → 256 and the
sampler 2.9 s → 49.6 s at **no** accuracy gain. *Experiment:* sweep the square's
tolerance independently of the cross's over `{eps, eps^{3/2}, eps², 1e-14}` on
both of F3's densities and plot `ESS/N` against rank. Two hours. The answer must
be a documented default with a warning, not a derived quantity — this is exactly
`REQUIREMENTS.md` R4's "never a plausible answer substituted quietly".

**Q33 — Would importance weights have caught `cross-approximation.md` §1.3's
silent failure?** *Blocks:* nothing, but it is the cheapest possible reuse of
E9. That counterexample —
`f(i) = 1/(1e-2 + |Σ_k i_k/9 − 5/2|)`, `d=6`, `n=10` — has true relative error
3.82e-04 with every internal indicator at 1e-15. If the tensor is non-negative
(it is), F3's sampler turns it into a proposal and `ESS/N` becomes an error
indicator that uses **the black box's own values at points the cross chose not to
look at**, which is structurally different from `n_check`'s uniform sample.
*Experiment:* sample 10⁴ points from the returned TT, weight by `f`, report
`ESS/N`, over the same six seeds. Half a day. If `ESS/N` collapses on the four
failing seeds and not on the two good ones, `cross-approximation.md` gains an
indicator it currently does not have.

**Q34 — Should `iga.py`'s knot vector be adapted by the cross?** *Blocks:*
nothing; it is step 4 of §3, deliberately left out. F4 shows the reconstruction
error on a twisted duct falls only as `O(h²)` under uniform refinement.
*Experiment:* refine only the elements where the cross's own fiber residual is
largest, and compare against uniform refinement at equal degrees of freedom on
the twisted map. It needs Q31's answer first.

**Q35 — Is a modal (coefficient) basis ever worth the loss of the Hadamard
product?** *Blocks:* the nodal contract in §2.4, which is stated as a design
decision with one line of justification. A Legendre-*coefficient* basis makes
`round` trivially `L²`-correct (the Gram is the identity) and truncation of the
*basis* meaningful, at the cost that `x * y` and `multifuncrs` need a
triple-product tensor per mode. *Settled by:* whether any consumer needs `*` on a
functional TT. `multifuncrs` does, `qtt_ell.invert/sqrt` do, and F3's squaring
does — so the answer is currently "no", but it is recorded because the decision
is invisible in the code and would be expensive to reverse.

**Q36 — Does the Rust FT library of §1.5 exist in public?** *Blocks:* nothing.
`github.com/lanl/pyftt` — "Pytho wrapper for the Functional Tensor Train (FTT)
Library", created 2026-07-29 — **exists and is empty** (§1.5). *Settled by:*
watching that repository, reading the paywalled chapter PDF, or one email to
Martinelli or Manzini. Recorded because "not found by unauthenticated GitHub
repository search" is weaker evidence than it looks.

**Q37 — Should ttpy2's cross adopt C3's rank enrichment, or C3 adopt ours?**
*Blocks:* nothing, but it is a free cross-check of a decision already taken.
`cross-approximation.md` §5 P2 settled that **uniformly random** extra rows
(`kickrank2=2`) fix 6 of 6 seeds on its §1.3 counterexample where the greedy
rectangular maxvol fails on 4. C3's `ftapprox_cross_rankadapt` enriches by
**repeating the last nodes** (`cross_index_copylast`, `ft.c:3841`), with the
author's comment saying explicitly that he considered random locations and
declined (§1.2). *Experiment:* now that `c3py` builds on b300 (F6), run C3's
adaptive cross on the discretised §1.3 counterexample and see whether the
continuous version fails the same way. One afternoon, and it is the only place in
this document where C3 could be used as an oracle rather than as a source.

---

## 7. Coherence: what this document requires of the others

Minimal, and listed so it can be checked. **Nothing under `tt/` was touched and
no sibling spec was edited by this pass except the two-line pointer named in
item 1.**

1. **`full-scale-examples.md` §6 Q20 is answered** and its §1.3 paragraph
   *""Dolgov and Marchuk": not found, and probably not a paper"* is confirmed
   correct in its conclusion and incomplete in its reasoning. A two-line pointer
   to this document has been added there. The `§9` bullet *""Marchuk" was not
   identified"* should be amended to *"identified as Marzouk, and the pairing
   itself is a conflation — see `functional-tt.md` §0.1"* when that document is
   next revised. **This document does not make that edit**; it names it.
2. **`full-scale-examples.md` §1.9** (the functional-TT *neural network*,
   arXiv:2510.13386) and its §7 dismissal of it — "it is a neural
   parametrisation, not a TT linear-algebra method" — stand, and §1.1 here is the
   non-neural ancestor it should have cited. Anyone comparing the two should note
   that Feng et al.'s cores are small neural networks where Gorodetsky's are
   adaptive piecewise polynomials, and that only the latter has a rounding
   operation.
3. **`cross-approximation.md`** gains a candidate error indicator it does not
   have (Q33) and loses nothing: nothing here proposes changing
   `cross.py`. The `kickrank2` and cache decisions of its §5 are orthogonal to
   the node family and apply unchanged to F1's and F2's runs.
4. **`ROADMAP.md` §7 "Explicitly out of scope"** should gain one row when this is
   acted on: *continuous pivots and ragged FT cores — a separate package; C3 is
   the evidence that it does not grow inside a TT library (`functional-tt.md`
   §2.3, §2.5)*. Recorded here so it is not re-litigated.
5. **`REQUIREMENTS.md` R1** is not threatened by anything proposed: §2.4 is pure
   numpy, and C3, `TensorToolbox`, `TT-IRT` and `deep-tensor` are all named as
   *sources*, never as dependencies.

---

## 8. What I did not verify

* **I read exactly one paper in full: arXiv:1510.09088v3.** Everything about
  §1.3 (IJRR), §1.4 (JCP), §1.6 (SISC), §1.7 (Stat. Comput.), §1.8 (FoCM) and
  §1.9 (JCP) comes from abstracts, CrossRef metadata, GitHub API responses and
  READMEs — **not from their text**. Every method paragraph in those entries is
  at that level of confidence, and the reference values in §5.3 in particular do
  **not** exist in this document.
* **The spectral-TT method description in §1.6 was read through `ar5iv`**, a
  third-party renderer, not the SISC PDF. This survey caught `ar5iv` making a
  false *negative* claim about the companion paper (it asserted that
  arXiv:1510.09088 "does not mention C3 library or other specific software
  implementations"; the extracted PDF says the opposite verbatim). §1.6's
  equation-level statements should be re-verified against the SISC PDF before
  being quoted anywhere else.
* **No MATLAB was run.** `TT-IRT` and `deep-tensor` are MATLAB, b300 has neither
  MATLAB nor Octave, and neither repository was cloned. Everything said about
  them is from the GitHub API and one README.
* **The Rust FT library was not found and its chapter body was not read.** §1.5
  rests on the Springer landing page's abstract and on GitHub *repository*
  searches; **GitHub code search requires authentication and was not available**,
  so the negative is weaker than it reads. `lanl/pyftt` exists and is empty.
  b300 has no `cargo`/`rustc`, so it could not have been built either.
* **C3 was built but barely exercised.** F6 ran exactly one example
  (`pyexamples/c3py_test_small.py`) and checked one number
  (`1.632993161855454`). `lib_probability`, `lib_quadrature`, `lib_tensdecomp`
  and `lib_fft` were not read at all, and none of C3's own 40 101 lines of tests
  were run. **No C3 output is quoted anywhere else in this document**, and in
  particular nothing here confirms or disputes §1.1's claim of three orders of
  magnitude over STT.
* **The C3 build is not clean.** It rests on a hand-generated 5 136-symbol
  alias shim, de-prefixed headers and a SWIG wrapper script (§1.2). It proves
  "reachable on b300 by a determined person", not "installable". `c3sc` was
  cloned and **not built** — its `cdyn` dependency was never fetched.
* **Gorodetsky's MIT thesis was not read.** `dspace.mit.edu/handle/1721.1/108918`
  returns **HTTP 405**; the bibliographic record comes from his CV PDF.
* **The Marzouk arXiv listing caps at 100 entries.** It spans 2011–2026 and
  appears complete, but a hard cap cannot be excluded, so §0.1's "zero papers on
  vibrational spectra" is a statement about those 100 titles plus one full-text
  query, not a proof. Google Scholar was reachable but **only the first
  pubdate-sorted page was read**.
* **§1.10's Alger–Chen–Ghattas page range and DOI are single-sourced** from
  search results, not from the SIAM page.
* **F1, F2 and F5's off-node reconstructions are mine, not a library's.** The
  barycentric weights, the piecewise grading and the `PWCheb` class are ~120
  lines written for this document and checked only against the exact function.
  They are not C3's algorithms and should not be quoted as measuring C3.
* **F2's grading was given the peak location.** It measures the size of the prize
  from adaptivity, not the cost of achieving it (Q31). Nothing here measures an
  *adaptive* method at all.
* **F3's sampler is discrete.** The samples are grid indices; a continuous
  sampler needs `Basis.antiderivative()` and a root find (§4.3), neither of which
  exists or was measured. The `ESS/N = 1.0000` rows therefore say "the TT is an
  excellent proposal for the tensor on that grid", not "for the density on the
  cube".
* **F3's `ESS` is a single run at one seed per row**, and its `max mean err`
  column is Monte-Carlo noise, stated as such in §4.2 and not to be quoted as an
  accuracy figure.
* **F5's factor of 1.17–1.40 is one function, one basis, one `d`.** The identity
  `‖Lᵀc‖_F = ‖f‖_{L²}` is exact and general; the *benefit* is not measured
  anywhere else.
* **F4's twisted-duct Jacobian is taken by central differences**, so its "err AT
  nodes" of 2.4e-10 is the differencing noise floor described in
  `iga.py::geometry_field`, not the cross's accuracy. The off-node column, six
  orders larger, is unaffected by this.
* **I did not run the test suite.** This document adds no code under `tt/`;
  `cross-approximation.md` records 801 tests in 59.40 s and nothing here can have
  changed that.
* **I re-measured nothing from any sibling spec.** M1's twisted map, M2's
  zero-component finding and `cross-approximation.md` §1.3's counterexample are
  cited with their sections and were not re-run.
* **No claim is made about C3's or `TensorToolbox`'s numerical results.** §1.2
  and §1.6 report whether they build and what their sources say; neither was used
  to produce a number that appears in this document.
