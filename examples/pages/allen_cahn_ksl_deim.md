# 3D Allen–Cahn on a fixed-rank manifold, by the interpolatory KSL

The showcase problem of Dektor's collocation paper (sec. 7.2) — phase separation under the Allen–Cahn equation on a $32^3$ periodic grid — integrated by `tt.ksl_deim` on the manifold of TT rank 16. The integrator's tangent-space projector is *interpolatory* rather than orthogonal, so the cubic nonlinearity $u-u^3$ is only ever evaluated entrywise on QDEIM-selected cross fibers — the integrator never forms the $n^3$ vector — and the discrete Ginzburg–Landau energy decreases monotonically along the whole run, a live correctness invariant printed at every checkpoint.

<img src="../../docs/media/allen_cahn_deim.gif" width="100%">

## The problem

On the torus $[0, 2\pi]^3$, the Allen–Cahn equation

$$\frac{\partial u}{\partial t} = \alpha \Delta u + u - u^3, \qquad \alpha = 0.1,$$

is the $L^2$ gradient flow of the Ginzburg–Landau free energy

$$E[u] = \int \left[ \frac{\alpha}{2} |\nabla u|^2 + \frac{(1-u^2)^2}{4} \right] dx,$$

so solutions separate into plateaus at the double-well minima $u = \pm 1$ while $E$ decreases monotonically.

Discretization, exactly as in the paper: Fourier pseudospectral collocation on the uniform grid $x_i = 2\pi i/n$. The one-axis second derivative is the real symmetric circulant $D_2 = F^{-1} \mathrm{diag}(-k^2) F$ with integer wavenumbers $k = -n/2, \dots, n/2 - 1$, and the operator is the Kronecker sum

$$A = \alpha (D_2 \otimes I \otimes I + I \otimes D_2 \otimes I + I \otimes I \otimes D_2)$$

— a TT-matrix of rank 2. The initial condition is the paper's eq. 7.2,

$$u_0 = g(x_1,x_2,x_3) - g(2x_1,x_2,x_3) + g(x_1,2x_2,x_3) - g(x_1,x_2,2x_3),$$

$$g = \frac{\big[e^{-\tan^2 x_1} + e^{-\tan^2 x_2} + e^{-\tan^2 x_3}\big] \sin(x_1+x_2+x_3)}{1 + e^{|\csc(-x_1/2)|} + e^{|\csc(-x_2/2)|} + e^{|\csc(-x_3/2)|}}.$$

Why the *interpolatory* KSL exists at all: the classical projector-splitting KSL of Lubich–Oseledets needs the full right-hand side $Au + u - u^3$ projected **orthogonally** onto the tangent space of the rank-$r$ manifold, and the orthogonal projector needs the right-hand side as a TT object — but $u - u^3$ has no cheap TT representation (an entrywise cube cubes the ranks). `tt.ksl_deim` replaces the orthogonal projector by an oblique, *interpolatory* one: at every substep the right-hand side is **evaluated entrywise on cross fibers selected by QDEIM**, which a pointwise nonlinearity supports at the cost of a numpy call on the sampled entries — here literally `lambda v: v - v ** 3`.

## The code, walked through

The 1D spectral derivative is materialized as a dense $n \times n$ matrix (`ksl_deim` samples $Ay$ on fibers, so $A$ must be a TT-matrix; a dense core is fine) and self-checks on the grid eigenfunction $\sin x$:

```python
def spectral_d2(n):
    k = np.fft.fftfreq(n) * n
    D2 = np.real(np.fft.ifft(np.fft.fft(np.eye(n), axis=0)
                             * (-(k ** 2))[:, None], axis=0))
    x, _ = grid(n)
    assert np.abs(D2 @ np.sin(x) + np.sin(x)).max() < 1e-11 * n, \
        "spectral D2 fails on sin(x)"
    return D2
```

The 3D operator is the rank-2 Kronecker sum, assembled from three one-core TT-matrices:

```python
def _mat3(ms):
    """Kronecker product of three 1D matrices as a TT-matrix (mode 1 = x1)."""
    return tt.matrix.from_list(
        [np.ascontiguousarray(m[None, :, :, None]) for m in ms])


def laplacian3d(n, alpha=ALPHA):
    """``A = alpha Laplace`` as a rank-2 TT-matrix from the Kronecker sum."""
    D2, I = spectral_d2(n), np.eye(n)
    A = (_mat3([alpha * D2, I, I]) + _mat3([I, alpha * D2, I])
         + _mat3([I, I, alpha * D2]))
    return A.round(1e-13)
```

The initial condition goes into TT by `multifuncrs` — TT-cross on pointwise calls — from the three rank-1 coordinate tensors. At the $\csc$ singularities the exponent is capped at 30, which leaves $|g| < 10^{-12}$ there instead of an inf/inf NaN:

```python
def u0_values(v):
    """Pointwise ``u0`` of eq. 7.2; ``v`` is ``(batch, 3)`` of coordinates."""
    x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
    return (_g(x1, x2, x3) - _g(2 * x1, x2, x3)
            + _g(x1, 2 * x2, x3) - _g(x1, x2, 2 * x3))


def initial_condition(n, eps=1e-8):
    """``u0`` in TT by cross approximation from rank-1 coordinate tensors."""
    x, _ = grid(n)
    c, o = x.reshape(1, n, 1), np.ones((1, n, 1))
    X = [tt.vector.from_list(cores) for cores in
         ([c, o, o], [o, c, o], [o, o, c])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return multifuncrs(X, u0_values, eps, verb=0)
```

The manifold of the method is fixed rank $r$: `ksl_deim` keeps the ranks of its input, so `u0` is padded with $10^{-8}$ random noise and cut to rank exactly $r$; the whole nonlinearity contract is one line:

```python
def pad_to_rank(u0, r, seed=0):
    rng = np.random.default_rng(seed)
    noise = tt.rand([int(m) for m in u0.n], r=r,
                    samplefunc=rng.standard_normal)
    noise = noise * (float(u0.norm()) * 1e-8 / float(noise.norm()))
    return (u0 + noise).round(0.0, rmax=r)


def nonlinearity(v):
    """``Nf(u) = u - u^3`` on sampled entries -- the ``ksl_deim`` contract."""
    return v - v ** 3
```

The time loop is one call per step, with a monotonicity tripwire on checkpoints. The diagnostic energy is written with the quadratic form of the *same* spectral Laplacian, so the semidiscrete ODE is exactly the gradient flow of this discrete $E$ and any increase is a bug, not a quadrature mismatch:

```python
    for k in range(nsteps):
        y = ksl_deim(A, nonlinearity, y, tau)
        if (k + 1) % every == 0 or k + 1 == nsteps:
            t = (k + 1) * tau
            U = np.asarray(y.full())
            E_new = energy(U, h)
            mono = "" if E_new <= E + 1e-10 else "   <-- E increased!"
            E = E_new
```

The steps inside `ksl_deim` are explicit-Euler substeps: first order in $\tau$, subject to the Euler stability bound $\tau < 2/(3\alpha (n/2)^2)$ (about 0.026 at $n=32$); the default $\tau = 0.005$ sits well inside it.

## What comes out

The default run (`n=32`, rank 16, $\tau=0.005$, $T=10$; timings from one development laptop, indicative only):

```
Allen-Cahn on [0,2pi]^3, alpha=0.1: n=32, TT rank 16 (u0 cross rank 31), 2000 steps of tau=0.005 (T=10)
       t            E     max|u|
   0.000    61.895030   0.479714
   1.000    61.006430   0.516858
   3.000    56.472611   0.794857
   5.000    44.471937   0.930541
   7.000    38.044290   0.978157
     ...          ...        ...
  10.000    36.172199   0.991950

12.3 s (6.1 ms/step), final TT ranks [1, 16, 16, 1] (fixed by the method)
```

The energy decreases monotonically through all 41 checkpoints (61.90 → 36.17, no `E increased!` flag ever fires), and $\max|u|$ climbs to the double-well plateau 1 — the phase separation visible in the animation. At $n \le 16$ the script also prints the dense oracle line:

```
$ python examples/allen_cahn_ksl_deim.py 16 16 0.005 2
...
vs dense solve_ivp oracle (same D2, same start): relative error 4.67e-03
```

## Checked against the dense method

The same run is cross-checked across grid sizes against `scipy.integrate.solve_ivp` (RK45, `rtol=1e-8`, `atol=1e-10`) on the full $n^3$ ODE built from the *same* spectral $D_2$ by `scipy.sparse` Kronecker sums, both started from the same rank-16 initial condition:

| $n$ | dense size $n^3$ | relative difference |
|---:|---:|---:|
| 8  | 512     | 4.2e-03 |
| 16 | 4 096   | 4.7e-03 |
| 32 | 32 768  | 5.4e-03 |
| 64 | 262 144 | 4.8e-03 |

The relative difference is TT-vs-dense at $T=2$ and sits at $\sim 5\cdot 10^{-3}$ throughout — the fixed-rank-manifold plus first-order-splitting error of `ksl_deim` against a tight reference — and it does *not* grow with $n$.

And this is still only 3D, where the dense state exists at all. In the six-dimensional setting this integrator was built for — the $3{+}3$D Boltzmann equation of Dektor–Einkemmer — a single dense state at $n=32$ is $32^6 \approx 10^9$ entries ($\sim 8.6$ GB), and RK45 needs several copies of it: the dense reference of this table simply does not exist there, while the TT solution keeps scaling as $n r^2$ per dimension.

## Why believe it

* `tests/test_examples.py::test_allen_cahn_ksl_deim_matches_dense_solve_ivp` runs $n=8$ on the *full* manifold: rank 8 is the TT-rank bound of an $8^3$ tensor, so the fixed-rank manifold is the whole space and the only gap to the dense RK45 solve (rtol $10^{-8}$) of the same pseudospectral ODE is the first-order time discretization of the interpolatory splitting. The same test asserts that the ranks stay exactly $[1, 8, 8, 1]$ and that the discrete Ginzburg–Landau energy never increases between checkpoints (with $10^{-10}$ headroom for noise).
* The energy monotonicity is not only a test but a live invariant of every run: $E$ is written with the quadratic form of the same spectral Laplacian, so the semidiscrete ODE is exactly its gradient flow, and the run flags any increase at any checkpoint.
* The 1D operator self-checks $D_2 \sin x = -\sin x$ on the grid at every assembly, and the table above is an independent dense cross-check at four grid sizes.

## Run it

```bash
python examples/allen_cahn_ksl_deim.py                 # n=32, rank 16, T=10, ~15 s
python examples/allen_cahn_ksl_deim.py 16 16 0.005 2   # with the dense solve_ivp oracle, ~5 s
python examples/allen_cahn_ksl_deim.py --gif docs/media/allen_cahn_deim.gif   # + animation
```

Times measured on one development laptop, imports included.

## References

* A. Dektor — Collocation methods for nonlinear differential equations on low-rank manifolds, *Linear Algebra Appl.* 705 (2025), 143–184, arXiv:2402.18721. Sec. 7.2 is this exact problem and the source of the initial condition.
* C. Lubich, I. Oseledets — A projector-splitting integrator for dynamical low-rank approximation, *BIT* 54 (2014) — the classical (orthogonal-projector) KSL this method replaces.
* Z. Drmač, S. Gugercin — QDEIM, *SIAM J. Sci. Comput.* 38(2), 2016 — the fiber-selection rule.
* A. Dektor, L. Einkemmer — the interpolatory low-rank machinery applied to the full $3{+}3$D Boltzmann–BGK equation: the six-dimensional setting where the dense column of the table above does not exist.
* oseledets/ttpy PR #102 (A. Dektor) — the original `tt.ksl_deim`.
