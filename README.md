# ttpy 2

The Tensor Train toolbox, rewritten in pure Python.

```bash
pip install ttpy          # or: uv pip install ttpy
```

No Fortran, no `f2py`, no `numpy.distutils`, no compiler, no git submodules.
The wheel is `py3-none-any`, 128 KB, and installs into a fresh environment in a
quarter of a second. For comparison, building ttpy 1.x on a current machine
needs six separate workarounds — they are written down in
[docs/LEGACY_BUILD.md](docs/LEGACY_BUILD.md).

```python
import tt

A = tt.qlaplace_dd([12])           # 3D Laplacian in the QTT format, 2^12 unknowns
b = tt.ones(2, 12)
x = tt.amen_solve(A, b, b, 1e-8)   # AMEn linear solver
print((tt.matvec(A, x) - b).norm() / b.norm())
```

## What it is

A tensor in the TT (tensor train) format is stored as `d` cores of shape
`(r, n, r)`, which turns `n**d` numbers into `d n r**2` and makes linear algebra
in dimension 100 possible. This package implements the format and the algorithms
around it: TT-SVD and rounding, cross approximation, elementwise functions,
AMEn linear solvers and matvecs, block eigensolvers, the KSL integrator,
Riemannian tools, and completion.

## Backends

The numerics run on numpy by default and on torch (CPU or GPU) on request:

```python
tt.set_backend("torch", device="cuda", dtype="float64")
```

Both produce the same numbers, including the same truncation ranks; that is a
test, not a hope (`tests/test_backend_torch.py`).

Which one is faster depends on the operation, and not in the direction you might
expect — see [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Short version: TT
rounding is a chain of many *small* sequential factorizations, so it is bound by
LAPACK call latency, not by FLOPs. A GPU does not help there, and neither do 64
BLAS threads (they hurt). Contractions such as `dot` are GEMM-bound and the GPU
wins by an order of magnitude. For large ranks use the randomized rounding, which
turns the SVD chain into matrix multiplications:

```python
y = x.round(rmax=100, method="randomized")
```

## Coming from ttpy 1.x

Your `import tt` scripts should keep working. Read
[docs/COMPAT.md](docs/COMPAT.md) for the exceptions — in particular three places
where the old package was quietly returning a transposed or wrong result, now
fixed and checked against dense ground truth.

## Development

```bash
uv venv && uv pip install -e ".[test]"
pytest -q                       # core + backend + algorithms
python bench/bench_core.py --backends numpy torch --out bench/results/core.json
```

Requirements live in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) and are the
source of truth: the code should be reproducible from them and the tests.

## Where it stands

521 tests, all against dense ground truth or a mathematical invariant — never
against the old implementation. Measured against the Fortran ttpy on one host
(details and raw data in [docs/PERFORMANCE.md](docs/PERFORMANCE.md)): faster on
rounding (1.2-1.5x), on `dot` (2x), on `tt_svd` (11.9x) and on `amen_solve`
(2.8x, and 60x more accurate on the same problem).

Testing against dense truth also turned up four defects in the old package
(transposed `Toeplitz` and `qshift`, a plainly wrong `IpaS`, a broken K/S order
in the real branch of the KSL integrator) and three in this one, all fixed and
documented in [docs/COMPAT.md](docs/COMPAT.md).

## References

* I. V. Oseledets, *Tensor-train decomposition*, SIAM J. Sci. Comput. 33(5), 2011.
* S. Dolgov, D. Savostyanov, *Alternating minimal energy methods for linear
  systems in higher dimensions*, arXiv:1301.6068, arXiv:1304.1222.
* A. Mikhalev, I. Oseledets, *Rectangular maximum-volume submatrices and their
  applications*, arXiv:1502.07838.
* H. Al Daas et al., *Randomized algorithms for rounding in the Tensor-Train
  format*, SIAM J. Sci. Comput. 45(1), 2023, arXiv:2110.04393.
* C. Lubich, I. Oseledets, *A projector-splitting integrator for dynamical
  low-rank approximation*, BIT 54, 2014.

## License

MIT, as before.
