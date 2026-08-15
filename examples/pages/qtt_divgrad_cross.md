# Variable-coefficient diffusion, assembled and solved entirely in QTT

A conservative finite-difference discretization of $-\nabla\cdot(k\nabla u)=1$ on a $256^2$ grid is built from the package's public pieces — the coefficient by TT-cross, the differences by the QTT shift matrix, the boundary by rank-1 corner corrections — and solved by `amen_solve` with no preconditioner, without ever forming a dense matrix.

<img src="../../docs/media/divgrad_amen.gif" width="100%">

## The problem

On $[0,1]^2$, find $u$ with

$$-\nabla\cdot\big(k(x,y) \nabla u\big) = 1,$$

where the coefficient is a smooth checkerboard lens of contrast 100:

$$k(x,y) = 10^{ \sin 3\pi x \sin 3\pi y}, \qquad 10^{-1} \le k \le 10^{1}.$$

The scheme is the standard conservative one: with $n = 2^b$ interior points per axis and $h = 1/(n+1)$, the one-dimensional backward difference $D = (I - S)/h$ (where $S$ is the down-shift matrix) gives the operator as a sum of two Kronecker products,

$$A \quad =\quad  D_x^{\top} \mathrm{diag}(k_x) D_x \quad +\quad  D_y^{\top} \mathrm{diag}(k_y) D_y,$$

with $k_x$, $k_y$ sampled at the *flux faces* $(x_i - h/2,  y_j)$ and $(x_i,  y_j - h/2)$.

Boundary conditions, stated rather than hidden: with $D$ alone the flux through the right and top faces is dropped, which makes the problem $u = 0$ on the left/bottom sides and $k \partial u/\partial n = 0$ (natural) on the right/top. The `--dirichlet` variant restores the missing flux with the rank-1 corner corrections $(k_{\mathrm{face}}/h^2)  e_n e_n^{\top}$ per axis, clamping all four sides.

Everything lives in QTT: an index $0 \le i \lt 2^b$ is written in binary, so a 2-D grid function becomes a tensor with $d = 2b$ modes of size 2, and all the matrices above are QTT matrices with small ranks.

## The code, walked through

The coefficient is just a pointwise function of coordinates — no structure is declared up front:

```python
def coefficient(v):
    """Smooth checkerboard lens, contrast 100, low TT rank."""
    return 10.0 ** (np.sin(3 * np.pi * v[:, 0]) * np.sin(3 * np.pi * v[:, 1]))
```

Assembly starts from the QTT coordinate `x = tt.xfun(...)` (rank 2) and builds the two face-coefficient trains by `multifuncrs` — the TT-cross of Oseledets–Tyrtyshnikov applied to pointwise calls. Cross samples the function only on adaptively chosen fibers, so the rank of the result is what the function actually needs, not what the grid size suggests; the `tt.kron` arguments shift the sampling to the flux faces $x - h/2$:

```python
def assemble(bits, dirichlet=False, eps=1e-10):
    """The QTT operator; ``dirichlet=True`` adds the right/top face terms."""
    h = 1.0 / (2 ** bits + 1)
    one = tt.ones(2, bits)
    x = (tt.xfun(2, bits) + one) * h            # interior points, QTT rank 2
    I = tt.eye(2, bits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kx = multifuncrs([tt.kron(x - one * (h / 2), one), tt.kron(one, x)],
                         coefficient, eps, verb=0)
        ky = multifuncrs([tt.kron(x, one), tt.kron(one, x - one * (h / 2))],
                         coefficient, eps, verb=0)
```

The difference operator needs exactly one nontrivial ingredient: `tt.qshift`, the QTT representation of the shift matrix $S$ (Kazeev–Khoromskij). Then $D = (I-S)/h$ is the backward difference with $u(0)=0$ built in, and the operator is the conservative form $D^{\top}\mathrm{diag}(k)D$ summed over the two axes:

```python
    D = (I - tt.qshift(bits)) * (1.0 / h)       # d/dx with u(0) = 0
    Dx, Dy = tt.kron(D, I), tt.kron(I, D)
    A = Dx.T @ tt.diag(kx) @ Dx + Dy.T @ tt.diag(ky) @ Dy
```

The Dirichlet variant adds the flux through the far faces. The corner matrix $e_n e_n^{\top}$ is rank 1 in QTT — every core is the same $2\times 2$ block `[[0,0],[0,1]]`, picking out the all-ones bit pattern of the last index — and the face coefficient is cross-sampled at $x = 1 - h/2$:

```python
    if dirichlet:
        # the flux through the right/top faces: (k_face/h^2) e_n e_n^T
        corner = tt.matrix.from_list(
            [np.array([[0., 0.], [0., 1.]])[None, :, :, None]] * bits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kxb = multifuncrs([tt.kron(one * (1 - h / 2), one),
                               tt.kron(one, x)], coefficient, eps, verb=0)
            kyb = multifuncrs([tt.kron(x, one),
                               tt.kron(one, one * (1 - h / 2))],
                              coefficient, eps, verb=0)
        Cx, Cy = tt.kron(corner, I), tt.kron(I, corner)
        A = (A + (Cx @ tt.diag(kxb) @ Cx) * (1 / h ** 2)
             + (Cy @ tt.diag(kyb) @ Cy) * (1 / h ** 2))
    return A.round(1e-12), h
```

The oracle is the same scheme assembled a second time in `scipy.sparse`, with no tensor format anywhere — the only subtlety is that the Kronecker order must match `tt.full`'s Fortran ordering (mode 1 fastest):

```python
def dense_oracle(bits, dirichlet=False):
    """The same scheme in scipy.sparse -- no tensor format anywhere."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = 2 ** bits
    h = 1.0 / (n + 1)
    xg = (np.arange(n) + 1) * h
    kf = lambda X, Y: 10.0 ** (np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y))
    Dd = (sp.identity(n) - sp.diags(np.ones(n - 1), -1)) / h
    k2 = lambda a, b: sp.kron(b, a, "csr")      # mode 1 (x) fastest
```

The solve itself is one call — AMEn (Dolgov–Savostyanov), warm-started from the right-hand side, no preconditioner:

```python
    A, h = assemble(bits, dirichlet=True)
    f = tt.ones(2, 2 * bits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = amen_solve(A, f, f, 1e-8, verb=0)
    res = float((tt.matvec(A, u) - f).norm() / f.norm())
```

The README animation is nothing but the iterate after each sweep: `sweep_frames` calls `amen_solve(..., nswp=1)` in a loop, warm-starting each sweep from the previous iterate, and records the residual, the maximal TT rank and the cumulative wall clock. A throwaway small solve is run first so the first frame does not carry the one-time numba/import warm-up:

```python
    x, out, elapsed = f, [], 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for swp in range(1, nswp + 1):
            t0 = time.perf_counter()
            x = amen_solve(A, f, x, eps, nswp=1, verb=0)
            elapsed += time.perf_counter() - t0
            res = float((tt.matvec(A, x) - f).norm() / f.norm())
```

## What comes out

All numbers below are from the fully-clamped (`dirichlet=True`) operator; timings are from one development laptop and are indicative only.

The animation above (256² grid, `eps=1e-9`, one externally restarted `amen_solve` sweep at a time) ends at

> sweep 8: residual 2.4e-04, TT rank 33, t = 1.72 s

where the residual is $\Vert Ax - f\Vert/\Vert f\Vert$ and t is the cumulative solve time over the sweeps, warm-up excluded.

A single `amen_solve` call with its default sweep budget goes much further. The script's own output:

```
$ python examples/qtt_divgrad_cross.py 8
bits=8 (256^2 grid): residual 2.37e-09, solution TT ranks <= 56, operator ranks <= 50

$ python examples/qtt_divgrad_cross.py 6
bits=6 (64^2 grid): residual 5.49e-10, solution TT ranks <= 36, operator ranks <= 49
vs scipy.sparse oracle: 4.78e-11
```

The last line is the relative difference between the QTT solution and `spsolve` on the scipy.sparse rebuild of the same scheme — agreement to $\sim 5\cdot 10^{-11}$ at `bits=6`, i.e. down to the solver tolerances (the exact last digits of this run-dependent number vary between machines).

## Why believe it

* `tests/test_examples.py::test_divgrad_cross_assembly_matches_scipy_sparse` pins the assembly at `bits=5` for **both** boundary variants — the two-term operator (natural BC on the far faces) and the Dirichlet one with the rank-1 corner corrections — against the `scipy.sparse` oracle, requiring relative agreement below `1e-7` after an `amen_solve` at `1e-10`. Since the coefficient goes through `multifuncrs`, the same test also pins TT-cross on a smooth 2-D function against direct sampling.
* The solver itself is pinned separately in the same file: `test_amen_solve_laplacian_residual` and `test_amen_solve_matches_dense_for_small_d` (dense `np.linalg.solve` oracle).

## Run it

```bash
python examples/qtt_divgrad_cross.py            # bits=8 solve; dense oracle printed for bits<=6
python examples/qtt_divgrad_cross.py 6          # 64^2, with the scipy.sparse comparison, ~1.5 s
python examples/qtt_divgrad_cross.py 8          # 256^2, ~4 s
python examples/qtt_divgrad_cross.py 8 --gif docs/media/divgrad_amen.gif   # + animation, ~13 s
```

Times measured on one development laptop, imports included.

## References

* V. Kazeev, B. Khoromskij — the QTT shift/Laplacian representations, *SIAM J. Matrix Anal. Appl.* 33(3), 2012.
* I. Oseledets, E. Tyrtyshnikov — TT-cross approximation, *Linear Algebra Appl.* 432, 2010.
* S. Dolgov, D. Savostyanov — AMEn, *SIAM J. Sci. Comput.* 36(5), 2014.
