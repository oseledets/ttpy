# Turbulence in the tensor train: incompressible Navier–Stokes in QTT

The "quantum-inspired" turbulence solver of Gourianov et al. (Nature
Computational Science, 2022) encodes each velocity component of a flow as a
matrix product state whose indices are the *bits* of the grid coordinates. That
encoding is the quantized tensor train — the QTT format of Oseledets and
Khoromskij — so this example implements their solver with the toolbox's native
QTT machinery, and reclaims the name.

The point of the format is scale-locality. When the two grid-index bits at each
refinement level share one tensor mode (the **interleaved, z-order, quaternary**
layout), a bond of the train separates *length scales* rather than directions,
and the bond dimension $\chi$ is exactly the interscale correlation the paper
measures. Capping $\chi$ — keeping the flow on a bounded-rank manifold — is the
whole method.

![decaying 2D turbulence carried entirely in QTT at bounded bond dimension](../../docs/media/qi_cfd_turbulence.gif)

## The equations, and their encoding

The incompressible Navier–Stokes equations on the periodic torus:

$$\frac{\partial V}{\partial t} + (V\cdot\nabla)V = -\nabla p + \nu\nabla^2 V, \qquad \nabla\cdot V = 0.$$

Each component of $V$ on a $2^d\times2^d$ grid is a QTT vector of $d$ modes of
size 4: the mode at level $k$ carries the $k$-th bit of *both* coordinates
(`tt.zmeshgrid`). The discretization is the paper's DNS scheme, carried in that
representation: **8th-order central finite differences** in space (periodic
circulant stencils, assembled as `tt.zkron` of one-dimensional shift operators),
**second-order Runge–Kutta** in time, and incompressibility by the projection
method of Chorin — with the pressure Poisson equation solved by the toolbox's
own linear solvers.

## The method, walked through

The derivative operators are built once, in the z-order layout:

```python
d1 = _periodic_d1(d, h, order=8)          # 8th-order central, periodic
Dx = tt.zkron(d1, tt.eye(2, d))           # d/dx  (x = the low bits)
Dy = tt.zkron(tt.eye(2, d), d1)           # d/dy
Lap = tt.zkron(d2, eye) + tt.zkron(eye, d2)
```

One time step is a Heun predictor–corrector; each stage advances the velocity by
advection and viscosity, then projects onto the divergence-free manifold:

```python
def step(ops, u, v, dt, nu, project, rmax):
    du1, dv1 = rhs(ops, u, v, nu, rmax)                 # -(V.grad)V + nu Lap V
    u1, v1 = project((u + du1*dt).round(rmax), (v + dv1*dt).round(rmax))
    du2, dv2 = rhs(ops, u1, v1, nu, rmax)
    u2 = (u + (du1 + du2)*(0.5*dt)).round(rmax)
    v2 = (v + (dv1 + dv2)*(0.5*dt)).round(rmax)
    return project(u2, v2)
```

The projection solves one pressure Poisson equation and subtracts the pressure
gradient. Its operator is the composition of the same central first differences,
$-(D_x D_x + D_y D_y)$: being exactly $-\mathrm{div}(\nabla(\cdot))$ in this
discretization, a projected field is divergence-free to the accuracy of the
solve. It is stored in the SPD (positive) form — the central $D_x$ is
skew-symmetric, so $D_x D_x$ alone is negative semidefinite — which lets **both**
toolbox solvers drive it: the rank-adaptive `amen_solve` and the fixed-rank
`lobpcg_solve`, the latter staying on the same bounded-rank manifold as the flow.

```python
phi = amen_solve(ops.Lap_reg, (-div).round(rmax), eps, rmax=chi)   # or lobpcg_solve
u, v = (u - matvec(Dx, phi)).round(chi), (v - matvec(Dy, phi)).round(chi)
```

Every quantity — velocity, its derivatives, the pressure — is a tensor train of
bond dimension at most $\chi$, and nothing ever leaves the format.

## A cheaper route: the vorticity form

In two dimensions the pressure and the incompressibility constraint can be
removed entirely by going to the **vorticity–streamfunction** form
(`qtt_vorticity.py`): with $\omega = \partial_x v - \partial_y u$ and
$\nabla^2\psi = -\omega$, $V = (\partial_y\psi, -\partial_x\psi)$,

$$\frac{\partial\omega}{\partial t} = \nu\nabla^2\omega - (V\cdot\nabla)\omega,$$

a single scalar transport–diffusion equation whose velocity is a curl, so it is
divergence-free by construction. This is both cheaper and lower-rank: on the
same shear flow the scalar $\omega$ carries bond dimension $\sim 10$ where the
velocity pair needs $\sim 34$ — vorticity is the more compressible object. It is
also the natural shape for the projector-splitting integrator `tt.ksl`: the
stiff *linear* viscous flow $\partial_t\omega = \nu\nabla^2\omega$ is integrated
*exactly* on the fixed-rank manifold (on Taylor–Green the enstrophy decay comes
out to $10^{-13}$), with advection as an explicit substep. For the
advection-dominated flows here plain RK2 on the scalar is fastest; KSL pays off
when the diffusion is stiff. (The pointwise DEIM of `tt.ksl_deim` does *not*
apply — the advection $(V\cdot\nabla)\omega$ is non-local in $\omega$ through the
streamfunction solve.)

## What comes out

**The rigorous check is the Taylor–Green vortex**, the one nonlinear
incompressible flow with a closed-form solution:
$u = \cos x\sin y\ e^{-2\nu t}$, $v = -\sin x\cos y\ e^{-2\nu t}$, whose kinetic
energy decays as $E(t) = E_0 e^{-4\nu t}$.

![Taylor–Green: the energy tracks the analytic decay at bounded rank](../../docs/media/qi_cfd_taylor_green.png)

The QTT solver reproduces that decay to $1.7\times10^{-8}$, and matches an
*identical dense finite-difference scheme* to $2.3\times10^{-14}$ — the rank
truncation loses nothing on this low-rank flow. Both pressure solvers give the
same answer with the velocity divergence held at $\sim10^{-5}$ (the residual of
the Tikhonov-regularized projection, orders below the flow).

**The structure-resolving run** is the animation at the top: a field of random
vortices left to decay into two-dimensional turbulence (`run.py turbulence`).
The vortices advect, merge and draw out into filaments — and the whole time the
bond dimension stays bounded, in the range $\chi\sim 30$–$50$ on a $128^2$ grid.
On a $256^2$ grid the same boundedness holds with $\chi\sim 80$, a small
fraction of the $N=256$ a dense field would carry. That bounded interscale
correlation, growing only as the flow forms structure and never approaching the
grid size, is exactly what the paper measures and what makes the format cheap.

## In three dimensions

The same construction runs in 3D (`qtt_ns3d.py`): each velocity component is an
interleaved **octal** QTT — the three grid-index bits at each level share one
mode of size 8, built by a nested `tt.zkron` — so a bond still separates length
scales. Three dimensions have vortex stretching, so the vorticity is a vector
and the scalar shortcut is gone; the solver keeps the velocity–pressure form
with a 3D Chorin projection.

Its analytic oracle is the **ABC (Arnold–Beltrami–Childress) flow**, the 3D
analogue of Taylor–Green: it is a Beltrami field, $\nabla\times V = V$, so its
advection is a pure gradient the pressure absorbs and it decays as
$V(t) = V_0 e^{-\nu t}$. The QTT solver reproduces the resulting energy decay
$e^{-2\nu t}$ to $3.7\times10^{-7}$, with the divergence held at $10^{-13}$; the
octal operators match dense periodic differentiation to $10^{-15}$.

## Why believe it

* The Taylor–Green oracle is analytic and exact at any resolution, so the energy
  column is the solver's own error, and its match to $e^{-4\nu t}$ is the
  physical statement that advection, pressure and diffusion are all correct.
* The identical-dense-scheme comparison isolates the one thing the tensor train
  adds — rank truncation — and finds it contributes at the level of $10^{-14}$.
* `tests/test_quantum_inspired_cfd.py` pins the periodic z-order operators
  against dense differentiation, the projection's divergence-freeness for both
  solvers, the analytic decay, and the dense-scheme agreement. The scalar
  vorticity solver is pinned too: its KSL viscous decay is machine-exact, and
  it reproduces the velocity–pressure solver's vorticity field to $10^{-6}$.

## Run it

```bash
python examples/quantum_inspired_cfd/run.py tgv          # Taylor-Green, analytic oracle
python examples/quantum_inspired_cfd/run.py turbulence   # decaying 2D turbulence (vorticity)
python examples/quantum_inspired_cfd/run.py turbulence --gif docs/media/qi_cfd_turbulence.gif
```

## References

* N. Gourianov, M. Lubasch, S. Dolgov, Q. Y. van den Berg, H. Babaee, P. Givi,
  M. Kiffner, D. Jaksch — *A quantum-inspired approach to exploit turbulence
  structures*, [Nature Comput. Sci. 2:30,
  2022](https://doi.org/10.1038/s43588-021-00181-1) — the MPS turbulence solver
  reclaimed here.
* A. J. Chorin — *Numerical solution of the Navier–Stokes equations*,
  [Math. Comp. 22:745, 1968](https://doi.org/10.1090/S0025-5718-1968-0242392-2) —
  the projection method.
* I. Oseledets — *Tensor-train decomposition*, [SIAM J. Sci. Comput. 33:2295,
  2011](https://doi.org/10.1137/090752286); B. Khoromskij — *O(d log N)
  quantics approximation*, Constr. Approx. 34:257, 2011 — the QTT format the
  paper's encoding is.
