# Isogeometric Poisson on a curved 3D domain

Almost every tensor-train PDE example lives on a box or a spin chain, where the
grid is a Cartesian product and the operator is a sum of Kronecker terms. This
one does not: the domain is a **curved** annular duct, and the geometry enters
the discretization only through a handful of scalar metric fields that TT-cross
compresses. The stiffness matrix is then one contraction per core — there is no
element assembly loop anywhere in the example.

![the annular duct solution and its isogeometric convergence](../../docs/media/iga_ring.png)

## The problem

Steady conduction with no source in a quarter-annulus duct — inner radius
$r_{\mathrm{in}} = 0.5$, outer $r_{\mathrm{out}} = 1$, opening
$\theta_{\max} = \pi/2$, height $H = 1$:

$$-\Delta u = 0, \qquad r_{\mathrm{in}} \le r \le r_{\mathrm{out}}, \qquad 0 \le \theta \le \theta_{\max}, \qquad 0 \le z \le H,$$

with the temperature fixed on the two cylindrical faces and zero flux on the
other four:

$$u = u_{\mathrm{in}} \text{ on } r=r_{\mathrm{in}}, \qquad u = u_{\mathrm{out}} \text{ on } r=r_{\mathrm{out}}, \qquad \partial_n u = 0 \text{ on the } \theta \text{ and } z \text{ faces.}$$

Because nothing drives the flow along $\theta$ or $z$, the solution depends on
the radius alone, and it is known in closed form (Eq. (46) of Tran et al.) — it
satisfies the natural conditions identically, so the error below is the
method's and not a comparison against another code:

$$u(r) = \frac{u_{\mathrm{in}} \log(r_{\mathrm{out}}/r) + u_{\mathrm{out}} \log(r/r_{\mathrm{in}})}{\log(r_{\mathrm{out}}/r_{\mathrm{in}})}.$$

## The code, walked through

The unit cube is mapped bilinearly-in-radius to the sector, and the whole
influence of the curved geometry is the metric field
$R = J^{-1} J^{-\top} |J|$ built from the Jacobian $J$ of that map. It is handed
to `geometry_field`, which samples its six independent components by `tt.cross`:

```python
def ring_map(xi):
    r  = R_IN + (R_OUT - R_IN) * xi[:, 0]
    th = THETA_MAX * xi[:, 1]
    return np.stack([r * np.cos(th), r * np.sin(th), HEIGHT * xi[:, 2]], axis=1)

R, scale = geometry_field(ring_map, grids, jac=ring_jac, eps=eps_cross, r=2, ...)
K = stiffness_tt(R, blocks, eps=1e-12)
```

The Jacobian is passed analytically (`jac=ring_jac`) rather than differenced.
The map's columns are orthogonal — radial, tangential, axial — so three of the
six components of $R$ vanish *exactly*; central differences would instead return
them as $10^{-10}$ noise, and `tt.cross` would waste rank fitting that noise. With
the exact Jacobian the surviving fields are what they should be and the
stiffness train stays at TT rank 3. The Dirichlet data is lifted by a single
radial ramp, and one `amen_solve` on the interior block returns the correction.

## What comes out

Degree $p = 2$ B-splines, refined from 16 to 128 elements per direction:

| elements / direction | DOFs | relative error vs $u(r)$ | rank $K$ | order |
|---|---|---|---|---|
| 16  | 5,832     | 2.29e-6 | 3 | — |
| 32  | 39,304    | 2.79e-7 | 3 | 3.03 |
| 64  | 287,496   | 3.89e-8 | 3 | 2.84 |
| 128 | 2,197,000 | 4.26e-9 | 3 | 3.19 |

The error falls at the isogeometric rate $h^{p+1} = h^3$, straight down the
dashed reference slope in the right panel of the figure, reaching $4\times10^{-9}$
on a mesh of 2.2 million degrees of freedom — assembled, as promised, with no
element loop, the stiffness operator never leaving TT rank 3. The left panel is
the analytic field on the physical sector: $u$ rising from 1 on the inner
cylinder to 2 on the outer one, constant along each radius, the curved geometry
that the cross compresses drawn directly.

## Why believe it

The reference is the closed-form solution $u(r)$ above, exact at every point, so
the last column of the table is the discretization error with nothing else mixed
in. Its decay at exactly the expected order $p+1$ across four refinements — and
the operator holding rank 3 while the grid grows by three orders of magnitude —
is the whole claim: a curved-domain isogeometric solve carried entirely in the
tensor-train format.

## Run it

```bash
python examples/iga_ring.py                 # p=2, 32 elements per direction
python examples/iga_ring.py 2 64            # degree, elements per direction
python examples/iga_ring.py 2 16 32 64 128  # a convergence sweep
```

## References

* Q. T. Tran, D. P. Truong, K. Ø. Rasmussen, B. S. Alexandrov — *A tensor
  train-based isogeometric solver for large-scale 3D Poisson problems*,
  [Comput. Methods Appl. Mech. Engrg. 453:118802,
  2026](https://doi.org/10.1016/j.cma.2026.118802) — the annular-duct benchmark
  and its Eq. (46).
* I. Oseledets, E. Tyrtyshnikov — *TT-cross approximation for multidimensional
  arrays*, [Linear Algebra Appl. 432:70,
  2010](https://doi.org/10.1016/j.laa.2009.07.024) — the sampling that
  compresses the geometry fields.
