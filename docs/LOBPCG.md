# Fixed-rank LOBPCG linear solver

`tt.lobpcg_solve` minimizes the SPD energy

\[
E(x)=\frac12\langle x,Ax\rangle-\langle f,x\rangle
\]

over the TT rank profile prescribed by `x0`.  Unlike `amen_solve`, it never
enriches or truncates a bond.  The method visits one core at a time and solves
the local correction equation by block-Jacobi PCG.  At the next visit of the
same core, the last search direction is transported from the old left/right
frames and used as an exact one-dimensional coarse correction.

At core `k` the local trial information is therefore

\[
[g_k,\ M_k^{-1}r_k,\ \widetilde p_k],\qquad
\widetilde p_k=(V_k^{\rm new})^*V_k^{\rm old}p_k.
\]

The remaining PCG iteration is performed in the local-operator-orthogonal
complement of `p_k`.  The full tensor `V_k p_k` and a dense tangent basis are
never formed: frame transport is one left and one right overlap contraction.

## Representable manufactured system

When the solution has the prescribed ranks, both stationarity and the ordinary
linear residual converge to zero:

```python
import numpy as np
import tt

bits = 6
depth = 2 * bits
profile = [min(8, 2 ** min(k, depth-k)) for k in range(depth+1)]

def random_tt(seed):
    rng = np.random.default_rng(seed)
    x = tt.vector.from_list([
        rng.standard_normal((profile[k], 2, profile[k+1]))
        for k in range(depth)
    ])
    return x / x.norm()

A = tt.qlaplace_dd([bits, bits])
exact = random_tt(1)
f = tt.matvec(A, exact)

x, info = tt.lobpcg_solve(
    A,
    f,
    random_tt(2),
    1e-8,
    local_steps=12,
    local_prec="c",
    check_true_res=True,
    return_info=True,
    verb=0,
)

print(info.projected_gradient)
print(info.true_res)
print((x-exact).norm() / exact.norm())
assert list(x.r) == profile
```

The complete runnable version is `examples/lobpcg_fixed_rank.py`.

## Constrained fixed-rank approximation

For a point-source Poisson problem the selected profile need not contain the
exact solution.  In that case the solver can reach a stationary fixed-rank
energy minimizer while the full residual remains nonzero:

```python
bits = 12
depth = 2 * bits
side = 2**bits
rank = 16
profile = [min(rank, 2 ** min(k, depth-k)) for k in range(depth+1)]

A = tt.qlaplace_dd([bits, bits])
f = tt.unit(2, depth, side//2 + side*(side//2))
x0 = tt.rand(2, depth, profile)

x, info = tt.lobpcg_solve(
    A, f, x0, 1e-6,
    nswp=100,
    local_steps=28,
    check_true_res=True,
    return_info=True,
    verb=0,
)

print(info.projected_gradient)  # fixed-rank stationarity
print(info.true_res)            # representation error remains
```

This distinction is intentional.  The stopping criterion is

\[
\frac{\|P_{T_x\mathcal M_r}(Ax-f)\|}{\|f\|},
\]

not `||Ax-f||/||f||`.  Requesting the full residual is optional because forming
`A*x` multiplies TT ranks and can be much more expensive than an entire sweep.

## Parameters that matter

- `local_steps`: total local trial budget.  Once memory is available, one slot
  is recycled and at most `local_steps-1` are fresh PCG directions.
- `local_prec="c"`: central block Jacobi.  In QTT each diagonal block is only
  `2 x 2`.  The alternatives are `"l"`, `"r"`, and `"n"` (identity).
- `forcing_gamma`, `forcing_min`, `forcing_max`: adaptive local accuracy.  The
  default is `clip(0.1 * projected_gradient, 1e-10, 1e-2)`.
- `check_true_res`: report the full linear residual in addition to fixed-rank
  stationarity.

The current implementation supports NumPy cores and an SPD/Hermitian operator.
It returns the best projected-gradient iterate seen and warns rather than
claiming convergence when `nswp` is exhausted.

## Measured package path

H200 host, one OpenBLAS thread, float64, rank cap 16, warmed numba cache:

| problem | depth | tolerance | local steps | sweeps | time | final projected gradient |
|---|---:|---:|---:|---:|---:|---:|
| manufactured `f=A*x` | 24 | `1e-8` | 4 | 8 | 0.13 s | `1.04e-9` |
| central point source | 24 | `1e-6` | 28 | 39 | 0.74 s | `8.34e-7` |
| central point source | 28 | `1e-6` | 36 | 54 | 1.31 s | `8.36e-7` |

The manufactured run ended with true residual `1.16e-9` and relative solution
error `1.69e-9`.  Timings include the structured projected-gradient checks;
the first process invocation additionally pays the optional numba compilation
cost.
