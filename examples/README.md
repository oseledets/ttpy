# Examples

Every example is self-contained, runnable from the repository root, and
checked against an oracle that is not this package: an analytic formula, a
dense or sparse rebuild of the same problem, or a published table with the
paper named in the docstring.  The acceptance versions live in
`tests/test_examples.py`.

## Gallery

<table>
<tr>
<td width="50%">
<img src="../docs/media/divgrad_amen.gif" width="100%"><br>
<b><a href="pages/qtt_divgrad_cross.md">qtt_divgrad_cross</a></b> (<a href="qtt_divgrad_cross.py">code</a>) — variable-coefficient
diffusion $-\nabla\cdot(k\nabla u)=1$ on a $256^2$ QTT grid: the contrast-100
coefficient sampled at flux faces by TT-cross, solved by <code>amen_solve</code>
with no preconditioner; the animation is the iterate after every sweep.
<i>Oracle: scipy.sparse assembly of the same scheme.</i>
</td>
<td width="50%">
<img src="../docs/media/lov15_run.gif" width="100%"><br>
<b><a href="pages/henon_heiles_ksl_paper.md">henon_heiles_ksl_paper</a></b> (<a href="henon_heiles_ksl_paper.py">code</a>) — Fig. 3 of
Lubich–Oseledets–Vandereycken (<a href="https://doi.org/10.1137/140976546">SINUM 2015</a>) end to end: the 10-D Hénon–Heiles
spectrum by KSL time integration, sine-DVR, complex absorbing potential, at
the paper's full scale; the frames are the run's real log.  The resulting
spectrum is <a href="../docs/media/lov15_fig3.png">here</a>.
<i>Oracle: the paper's own figure; f=2 pinned against dense expm.</i>
</td>
</tr>
<tr>
<td>
<img src="../docs/media/henon_heiles_spectrum.png" width="100%"><br>
<b><a href="pages/henon_heiles_spectrum.md">henon_heiles_spectrum</a></b> (<a href="henon_heiles_spectrum.py">code</a>) — the same
spectral method with the eigensolver as the cross-check, at $d=8$ ($10^8$
states, beyond any dense oracle): coherent packet, autocorrelation, windowed
FFT; the ground state agrees with <code>eigb</code> to 4.4e-6 and the
8-fold multiplet at 4.96–4.98 sits under one peak, matching to 5.8e-5.
100 s end to end, eigb's 12 levels in 29 s.
<i>Oracle: eigb on the same operator — two TT methods checking each other
where neither has a dense referee.</i>
</td>
<td>
<img src="../docs/media/fokker_planck_viscometric.png" width="100%"><br>
<b><a href="pages/fokker_planck_dumbbell.md">fokker_planck_dumbbell</a></b> (<a href="fokker_planck_dumbbell.py">code</a>) — the polymer
dumbbell in shear flow of Dolgov–Khoromskij–Oseledets (<a href="https://doi.org/10.1137/120864210">SISC 2012</a>):
Crank–Nicolson in TT, Kramers viscometric functions reaching the paper's
Table 3 ($\eta$ 1.03291 vs 1.03281).
<i>Oracles: analytic $\beta=0$ stationary state, sparse propagator, the
$\alpha=0$ Lyapunov solution.</i>
</td>
</tr>
<tr>
<td>
<img src="../docs/media/sir_rare_events.png" width="100%"><br>
<b><a href="pages/sir_network_cme.md">sir_network_cme</a></b> (<a href="sir_network_cme.py">code</a>) — the SIR master equation
on a network (Dolgov–Savostyanov, <a href="https://doi.org/10.1016/j.amc.2023.128290">AMC 2024</a>): the $3^N$-state distribution in
TT, rare-event tails as one dot product with an explicit indicator train —
down to $10^{-12}$, where SSA would need $\sim 5\cdot 10^{13}$ trajectories.
<i>Oracles: brute-force propagator at small N, Gillespie SSA.</i>
</td>
<td>
<img src="../docs/media/robust_completion.png" width="100%"><br>
<b><a href="pages/robust_completion.md">robust_completion</a></b> (<a href="robust_completion.py">code</a>) — completion with 2%
outliers at 100x the scale: the square loss (the only one ALS can have)
chases the corruption; Riemannian descent with a log-cosh loss
(<code>tt.rgd</code>, torch autodiff through the tangent parametrization)
recovers the tensor — six orders of magnitude apart on held-out entries.
<i>Oracle: dense ground truth on unobserved entries.</i>
</td>
</tr>
<tr>
<td width="50%">
<img src="../docs/media/allen_cahn_deim.gif" width="100%"><br>
<b><a href="pages/allen_cahn_ksl_deim.md">allen_cahn_ksl_deim</a></b> (<a href="allen_cahn_ksl_deim.py">code</a>) — Dektor's
interpolatory projector-splitting integrator (<code>tt.ksl_deim</code>) on the
3D Allen–Cahn equation of his paper
(<a href="https://doi.org/10.1016/j.laa.2024.11.001">LAA 2025</a>,
sec. 7.2): the cubic nonlinearity $u-u^3$ is evaluated only on QDEIM-selected
cross fibers — the case the orthogonal-projector KSL cannot afford — with the
paper's Fourier pseudospectral Laplacian as a rank-2 TT-matrix; the animation
is the central slice separating into the $\pm 1$ phases.
<i>Oracle: dense solve_ivp of the same ODE at small n, plus monotone decay of
the Ginzburg–Landau energy along the whole run.</i>
</td>
<td width="50%">
<img src="../docs/media/sample_dirt_banana.gif" width="100%"><br>
<b><a href="pages/sample_dirt_banana.md">sample_dirt_banana</a></b> (<a href="sample_dirt_banana.py">code</a>) — a sample-only
probit bridge from the uniform square to a thin banana: seven low-rank residual
TT densities fitted by orthogonal ALS, with exact TT square integrals and exact
inverse Rosenblatt conditionals.  The same reference cloud sharpens layer by
layer; physical sliced $W_2$ falls from 0.439 to 0.052 (sampling floor 0.025),
using 1,912 parameters in about 1.8 s.
<i>Oracle: the held-out analytic banana density for KL/TV, independent target
samples for sliced Wasserstein, and the exact forward/inverse round trip.</i>
</td>
</tr>
<tr>
<td width="50%">
<img src="../docs/media/qtt_fem_triangle.png" width="100%"><br>
<b><a href="pages/qtt_fem_triangle.md">qtt_fem_triangle</a></b> (<a href="qtt_fem_triangle.py">code</a>) — Poisson on a
triangle, a domain no single tensor-product grid fits: three glued QTT patches,
z-order finite-element assembly with TT-cross Jacobian fields, one
<code>amen_solve</code> on the coupled block train; the energies published with
<a href="https://github.com/RerRayne/qtt-laplace">qtt-laplace</a>
(<a href="https://doi.org/10.1016/j.jcp.2020.109835">JCP 2021</a>) reproduced to 1.5e-9.
<i>Oracle: the FEniCS energy curve shipped in that repository, approached from
above as a Galerkin energy must.</i>
</td>
<td width="50%">
<img src="../docs/media/smoluchowski_run.gif" width="100%"><br>
<b><a href="smoluchowski/README.md">smoluchowski</a></b> (<a href="smoluchowski/run.py">code</a>) — the multicomponent
Smoluchowski coagulation equation of Matveev–Zheltkov–Tyrtyshnikov–Smirnov
(<a href="https://doi.org/10.1016/j.jcp.2016.04.025">JCP 2016</a>): the gain term is a lower-triangular convolution, so it is an
FFT on every TT core and never leaves the format. The paper's $1000^2$
reference point — error 2.2e-3 at TT rank 13 — reproduced with the whole
tensor kept in TT; the frames are the run's real log. The paper's non-separable
ballistic kernel (its eq. 17) is built too — a TT-cross of the $2d$-dimensional
$K$, cut along the one bond that separates $\bar u$ from $\bar v$ — and lands
on its Table 5 to the last digit printed from $N = 400$ up.
<i>Oracle: the analytic solution of the paper's eq. (18) and the exact total
density $1/(1+t/2)$, every step; Table 5 of the paper for the ballistic
kernel.</i>
</td>
</tr>
<tr>
<td width="50%">
<img src="../docs/media/cross_run.gif" width="100%"><br>
<b><a href="pages/cross_approximation.md">cross_approximation</a></b> (<a href="ising_integrals.py">code</a>) — a
high-dimensional Ising susceptibility integral, no separable form and no
product quadrature that fits: <code>tt.dmrg_cross</code> samples the integrand
along adaptively chosen fibers and reconstructs it, touching $\sim 10^5$
nodes of a grid with $10^{27}$ of them at $d=16$, and lands on the analytic
constant to 14 digits.
<i>Oracle: the analytic Ising integrals of Bailey–Borwein–Crandall, known to
hundreds of digits.</i>
</td>
<td width="50%">
<img src="../docs/media/amen_laplace_wall.png" width="100%"><br>
<b><a href="pages/amen_laplace.md">amen_laplace</a></b> (<a href="amen_laplace.py">code</a>) — the QTT
Laplacian $-u''=1$ solved by <code>amen_solve</code>, and the $O(4^d)$
conditioning wall an unpreconditioned iteration runs into: the error climbs
past the requested tolerance near $d=12$, and the solver reports the failure
rather than returning a plausible wrong answer.
<i>Oracle: the analytic discrete solution $u_i=i(N+1-i)/2$, exact at any $d$.</i>
</td>
</tr>
<tr>
<td width="50%">
<img src="../docs/media/bpx_conditioning.png" width="100%"><br>
<b><a href="pages/bpx_elliptic.md">bpx_elliptic</a></b> (<a href="bpx_elliptic.py">code</a>) — BPX
multilevel preconditioning in QTT (Bachmayr–Kazeev,
<a href="https://doi.org/10.1007/s10208-020-09446-z">FoCM 2020</a>): the
$4^d$ conditioning tamed to a bounded $\kappa(BA)$, over a billion unknowns
solved to $2\cdot10^{-13}$ at flat TT rank 17 — and why the preconditioned
operator must never be assembled as $CAC$.
<i>Oracle: dense eigenvalues for $\kappa$, the analytic $u(x)=x-x^2/2$ for the solve.</i>
</td>
<td width="50%">
<img src="../docs/media/iga_ring.png" width="100%"><br>
<b><a href="pages/iga_ring.md">iga_ring</a></b> (<a href="iga_ring.py">code</a>) — isogeometric
Poisson on a curved 3D annular duct (Tran et al.,
<a href="https://doi.org/10.1016/j.cma.2026.118802">CMAME 2026</a>): the
geometry enters only through metric fields compressed by TT-cross, the
stiffness stays at rank 3, and order-$h^3$ convergence reaches $4\cdot10^{-9}$
on 2.2M dofs with no element loop anywhere.
<i>Oracle: the closed-form radial solution, Eq. (46).</i>
</td>
</tr>
<tr>
<td width="50%">
<img src="../docs/media/fixed_rank_solvers.png" width="100%"><br>
<b><a href="pages/fixed_rank_solvers.md">fixed_rank_solvers</a></b> (<a href="qtt_divgrad_solvers.py">code</a>) — the
same SPD div–grad system solved two ways: <code>amen_solve</code> rank-adaptively
(enriches and truncates bonds) and <code>lobpcg_solve</code> at fixed rank
(minimizes the energy over a prescribed profile, bonds never change); a
manufactured problem isolates when the fixed-rank stationary point is exactly
the solution.
<i>Oracle: a manufactured $f=Ax_{\star}$ of known rank, plus the true residual.</i>
</td>
<td width="50%">
<img src="../docs/media/qi_cfd_taylor_green.png" width="100%"><br>
<b><a href="quantum_inspired_cfd/README.md">quantum_inspired_cfd</a></b> (<a href="quantum_inspired_cfd/run.py">code</a>) — the
"quantum-inspired" turbulence solver of Gourianov et al.
(<a href="https://doi.org/10.1038/s43588-021-00181-1">Nature Comput. Sci. 2022</a>),
reclaimed as QTT: incompressible Navier–Stokes with each velocity component an
interleaved-bit (z-order) tensor train, 8th-order central differences, RK2, and
Chorin projection whose pressure Poisson is solved by <code>amen_solve</code> or
<code>lobpcg_solve</code>, all at bounded bond dimension. Validated on the
analytic Taylor–Green vortex.
<i>Oracle: the analytic Taylor–Green decay $e^{-4\nu t}$, and an identical dense
finite-difference scheme (matched to $2\cdot10^{-14}$).</i>
</td>
</tr>
</table>

## All examples

| example | what it shows | oracle |
|---|---|---|
| [qtt_divgrad_cross.py](pages/qtt_divgrad_cross.md) ([code](qtt_divgrad_cross.py)) | div–grad assembly by TT-cross, AMEn animation | scipy.sparse rebuild |
| [amen_laplace.py](pages/amen_laplace.md) ([code](amen_laplace.py)) | the QTT Laplacian and the $O(4^d)$ conditioning wall | analytic discrete solution |
| [bpx_elliptic.py](pages/bpx_elliptic.md) ([code](bpx_elliptic.py)) | BPX preconditioning: $4^d$ conditioning tamed, and why $CAC$ must never be assembled | dense eigenvalues + analytic solution |
| [qtt_fem_triangle.py](pages/qtt_fem_triangle.md) ([code](qtt_fem_triangle.py)) | Poisson on a triangle: three glued QTT patches ([qtt-laplace](https://github.com/RerRayne/qtt-laplace)) | the repository's FEniCS curve + its TT energies |
| [iga_ring.py](pages/iga_ring.md) ([code](iga_ring.py)) | isogeometric Poisson on a curved 3D annular domain in QTT | closed-form radial solution |
| [qtt_divgrad_solvers.py](pages/fixed_rank_solvers.md) ([code](qtt_divgrad_solvers.py)) | fixed-rank (`lobpcg_solve`) vs rank-adaptive (`amen_solve`) on one div–grad system | manufactured exact-rank solution + true residual |
| [cross_engines.py](cross_engines.py) | the two cross engines side by side on the same integrands | analytic Ising constants |
| [cross_engines.py](pages/cross_approximation.md) ([code](cross_engines.py)) | `dmrg_cross` vs `rect_cross` on identical integrands: 5–9x fewer evaluations at equal accuracy | analytic integral values |
| [robust_completion.py](pages/robust_completion.md) ([code](robust_completion.py)) | the loss ALS structurally cannot have | dense ground truth |
| [henon_heiles_spectrum.py](pages/henon_heiles_spectrum.md) ([code](henon_heiles_spectrum.py)) | spectra from one trajectory (autocorrelation method) | `eigb` on the same operator |
| [henon_heiles_ksl_paper.py](pages/henon_heiles_ksl_paper.md) ([code](henon_heiles_ksl_paper.py)) | [\[LOV15\]](https://doi.org/10.1137/140976546) Fig. 3, quantum dynamics at paper scale | the paper; dense expm at f=2 |
| [fokker_planck_dumbbell.py](pages/fokker_planck_dumbbell.md) ([code](fokker_planck_dumbbell.py)) | [\[DKO12\]](https://doi.org/10.1137/120864210) sec. 4.2, polymer rheology in TT | three independent oracles |
| [sir_network_cme.py](pages/sir_network_cme.md) ([code](sir_network_cme.py)) | [\[DS24\]](https://doi.org/10.1016/j.amc.2023.128290) epidemics on networks, rare events | brute force + SSA |
| [allen_cahn_ksl_deim.py](pages/allen_cahn_ksl_deim.md) ([code](allen_cahn_ksl_deim.py)) | Dektor's interpolatory KSL (`tt.ksl_deim`) on [his paper's](https://doi.org/10.1016/j.laa.2024.11.001) 3D Allen–Cahn | dense solve_ivp at small n + energy monotonicity |
| [sample_dirt_banana.py](pages/sample_dirt_banana.md) ([code](sample_dirt_banana.py)) | sample-only residual TT transports learned by orthogonal ALS | analytic banana density + independent samples + exact round trip |
| [smoluchowski/](smoluchowski/README.md) ([code](smoluchowski/run.py)) | [\[MZTS16\]](https://doi.org/10.1016/j.jcp.2016.04.025) multicomponent coagulation: the gain term as an FFT on every TT core; the ballistic kernel separated by a cross of the $2d$-dimensional $K$ | the paper's analytic eq. (18) + exact total density; Table 5 for the ballistic kernel |
| [quantum_inspired_cfd/](quantum_inspired_cfd/README.md) ([code](quantum_inspired_cfd/run.py)) | [\[Gourianov22\]](https://doi.org/10.1038/s43588-021-00181-1) incompressible Navier–Stokes entirely in QTT: z-order velocity, 8th-order central differences, RK2, Chorin projection via `amen_solve`/`lobpcg_solve` | analytic Taylor–Green decay + identical dense scheme |
