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
<b><a href="qtt_divgrad_cross.py">qtt_divgrad_cross.py</a></b> — variable-coefficient
diffusion $-\nabla\cdot(k\nabla u)=1$ on a $256^2$ QTT grid: the contrast-100
coefficient sampled at flux faces by TT-cross, solved by <code>amen_solve</code>
with no preconditioner; the animation is the iterate after every sweep.
<i>Oracle: scipy.sparse assembly of the same scheme.</i>
</td>
<td width="50%">
<img src="../docs/media/lov15_run.gif" width="100%"><br>
<b><a href="henon_heiles_ksl_paper.py">henon_heiles_ksl_paper.py</a></b> — Fig. 3 of
Lubich–Oseledets–Vandereycken (SINUM 2015) end to end: the 10-D Hénon–Heiles
spectrum by KSL time integration, sine-DVR, complex absorbing potential;
877 s where the paper reports 4425 s (and 54354 s for MCTDH); the frames
are the run's real log.  The resulting spectrum is
<a href="../docs/media/lov15_fig3.png">here</a>.
<i>Oracle: the paper's own figure; f=2 pinned against dense expm.</i>
</td>
</tr>
<tr>
<td>
<img src="../docs/media/henon_heiles_spectrum.png" width="100%"><br>
<b><a href="henon_heiles_spectrum.py">henon_heiles_spectrum.py</a></b> — the same
spectral method at teaching size: coherent packet, autocorrelation, windowed
FFT; every peak is cross-checked against <code>eigb</code> on the same
operator (ground state agrees to 1.2e-5).
<i>Oracle: eigb + the quasi-degenerate multiplet structure.</i>
</td>
<td>
<img src="../docs/media/fokker_planck_viscometric.png" width="100%"><br>
<b><a href="fokker_planck_dumbbell.py">fokker_planck_dumbbell.py</a></b> — the polymer
dumbbell in shear flow of Dolgov–Khoromskij–Oseledets (SISC 2012):
Crank–Nicolson in TT, Kramers viscometric functions reaching the paper's
Table 3 ($\eta$ 1.03291 vs 1.03281).
<i>Oracles: analytic $\beta=0$ stationary state, sparse propagator, the
$\alpha=0$ Lyapunov solution.</i>
</td>
</tr>
<tr>
<td>
<img src="../docs/media/sir_rare_events.png" width="100%"><br>
<b><a href="sir_network_cme.py">sir_network_cme.py</a></b> — the SIR master equation
on a network (Dolgov–Savostyanov, AMC 2024): the $3^N$-state distribution in
TT, rare-event tails as one dot product with an explicit indicator train —
down to $10^{-12}$, where SSA would need $\sim 5\cdot 10^{13}$ trajectories.
<i>Oracles: brute-force propagator at small N, Gillespie SSA.</i>
</td>
<td>
<img src="../docs/media/robust_completion.png" width="100%"><br>
<b><a href="robust_completion.py">robust_completion.py</a></b> — completion with 2%
outliers at 100x the scale: the square loss (the only one ALS can have)
chases the corruption; Riemannian descent with a log-cosh loss
(<code>tt.rgd</code>, torch autodiff through the tangent parametrization)
recovers the tensor — six orders of magnitude apart on held-out entries.
<i>Oracle: dense ground truth on unobserved entries.</i>
</td>
</tr>
</table>

## All examples

| example | what it shows | oracle |
|---|---|---|
| [qtt_divgrad_cross.py](qtt_divgrad_cross.py) | div–grad assembly by TT-cross, AMEn animation | scipy.sparse rebuild |
| [amen_laplace.py](amen_laplace.py) | the AMEn linear solver on the QTT Laplacian | residual + dense solve |
| [bpx_elliptic.py](bpx_elliptic.py) | BPX preconditioning: $4^d$ conditioning tamed, and why $CAC$ must never be assembled | analytic solution |
| [qtt_fem_triangle.py](qtt_fem_triangle.py) | QTT finite elements on a triangle | analytic solution |
| [iga_ring.py](iga_ring.py) | isogeometric ring domain in QTT | manufactured solution |
| [ising_integrals.py](ising_integrals.py) | Ising susceptibility integrals $C_m$ by greedy DMRG cross, racing the original Fortran `ttcross` | published values (Bailey–Borwein–Crandall) |
| [cross_engines.py](cross_engines.py) | `dmrg_cross` vs `rect_cross` on identical integrands, matched stopping | same integrals |
| [robust_completion.py](robust_completion.py) | the loss ALS structurally cannot have | dense ground truth |
| [henon_heiles_spectrum.py](henon_heiles_spectrum.py) | spectra from one trajectory (autocorrelation method) | `eigb` on the same operator |
| [henon_heiles_ksl_paper.py](henon_heiles_ksl_paper.py) | [LOV15] Fig. 3, quantum dynamics at paper scale | the paper; dense expm at f=2 |
| [fokker_planck_dumbbell.py](fokker_planck_dumbbell.py) | [DKO12] sec. 4.2, polymer rheology in TT | three independent oracles |
| [sir_network_cme.py](sir_network_cme.py) | [DS24] epidemics on networks, rare events | brute force + SSA |
| [sample_dirt_2d_gallery.py](sample_dirt_2d_gallery.py) | deep inverse Rosenblatt transport, 2-D gallery | known densities |
| [sample_dirt_correlated_gaussian.py](sample_dirt_correlated_gaussian.py) | DIRT on a correlated Gaussian | analytic density |
| [sample_dirt_predator_prey.py](sample_dirt_predator_prey.py) | DIRT posterior, predator–prey ODE | reference sampler |
| [sample_dirt_lorenz96.py](sample_dirt_lorenz96.py) | DIRT posterior, Lorenz-96 | reference sampler |
| [sample_dirt_corner_mixture.py](sample_dirt_corner_mixture.py) | DIRT on a hard corner mixture | known density |
| [sample_dirt_optimizer_benchmark.py](sample_dirt_optimizer_benchmark.py) | DIRT fitting variants, measured | held-out likelihood |
