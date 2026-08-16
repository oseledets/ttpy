# One trajectory, the whole low spectrum

[`examples/henon_heiles_spectrum.py`](../henon_heiles_spectrum.py) computes the vibrational spectrum of the Hénon–Heiles Hamiltonian at teaching size ($d=4$) by the autocorrelation method — evolve one wave packet with KSL, Fourier-transform one scalar signal — and cross-checks every peak against `tt.eigb` run on the *same* discretized operator.

![The autocorrelation spectrum with eigb levels as dashed lines, and the zoom on the 2.97-2.99 multiplet](../../docs/media/henon_heiles_spectrum.png)

## The problem

The spectral method of quantum molecular dynamics — the workhorse of MCTDH (Beck–Meyer). Instead of diagonalizing $H$, evolve a packet under the time-dependent Schrödinger equation

$$i \frac{\partial\psi}{\partial t} = H\psi, \qquad H = \sum_{i=1}^{d} \frac{1}{2}\left(-\frac{\partial^2}{\partial q_i^2} + q_i^2\right) + \lambda \sum_{i=1}^{d-1} \left(q_i^2  q_{i+1} - \frac{q_{i+1}^3}{3}\right), \qquad \lambda = 0.111803,$$

with $\psi(0)$ a displaced Gaussian (a product coherent state), and record only the **autocorrelation**

$$
a(t) = \langle \psi(0), \psi(t) \rangle .
$$

If $\psi(0) = \sum_l c_l \phi_l$ in the eigenbasis of $H$, then

$$
a(t) = \sum_l |c_l|^2  e^{-i\lambda_l t},
$$

so the windowed Fourier transform

$$
\sigma(\omega) = \mathrm{Re} \int_0^T a(t)  w(t)  e^{i\omega t}  dt
$$

peaks exactly at the eigenvalues $\lambda_l$ that the packet overlaps — the whole spectrum on an interval from **one** trajectory, where an eigensolver would need a block of that many states. The price is resolution: peaks are $\sim 2\pi/T$ wide, so close levels need long trajectories. That trade shows up as a *measured* event below: a quasi-degenerate multiplet whose spread is smaller than $2\pi/T$ lands under a single peak.

The discretization is an $n$-function harmonic-oscillator product basis per mode (Galerkin, not DVR — this example and `tests/hamiltonians.py` share the same single source of truth). The propagator is the projector-splitting KSL integrator of [LOV15] at fixed TT rank; the step $\tau = -ih$ promotes the whole sweep to complex128. On small local blocks that sweep runs the compiled complex kernels of `tt/algs/_ksl_fast.py` (`docs/PERFORMANCE.md` § 3b: the kernels are dtype-generic, and the compiled complex step is pinned to the interpreted path at $2.8\times 10^{-15}$ on a Hénon–Heiles step); at the block sizes of this example it takes the interpreted Krylov path.

## The code, walked through

The one-mode operators are formed at size $n + 6$ and *then* cut — squaring the already-truncated $q$ instead is a percent-level error on the top basis functions:

```python
def ho_operators(n, pad=6):
    """``(N, Q, Q2, Q3)`` in the HO basis; mirrors ``tests/hamiltonians.py``."""
    n, m = int(n), int(n) + int(pad)
    k = np.arange(m, dtype=float)
    q = np.diag(np.sqrt((k[:-1] + 1.0) / 2.0), 1)
    q = q + q.T
    q2, q3 = q @ q, q @ q @ q
    return (np.diag(k[:n] + 0.5), q[:n, :n], q2[:n, :n], q3[:n, :n])
```

The Hamiltonian is the standard rank-3 MPO for a nearest-neighbour chain — identity tracks above the diagonal, one-site terms flow into the corner, $\lambda Q^2$ hands off to $Q$ on the next site:

```python
    for k in range(d):
        W = np.zeros((3, 3, int(n), int(n)))
        W[2, 2] = I
        W[0, 0] = I
        W[2, 0] = N - (lam / 3.0) * Q3 if k > 0 else N
        W[2, 1] = lam * Q2
        W[1, 0] = Q
```

The initial packet is the coherent state $|\alpha\rangle$, $\alpha = q_0/\sqrt 2$ per mode — an exactly rank-1 TT. KSL keeps ranks fixed and cannot grow a rank-1 start, so the packet is padded to the manifold rank with $10^{-8}$ noise:

```python
    # pad the rank-1 packet to the KSL manifold rank
    rng = np.random.default_rng(seed)
    noise = tt.rand([n] * d, r=r, samplefunc=rng.standard_normal)
    noise = noise * (1e-8 / noise.norm())
    y = (psi0 + noise).round(0.0, rmax=r)
```

The propagation loop records one dot product per step; on the very first step it also asks the integrator whether the rank is adequate at all — the *tangent defect*, the share of $H\psi$ that the rank-$r$ manifold cannot follow:

```python
        for k in range(nsteps):
            hist = None
            if k == 0:                 # one defect check: is the rank enough?
                y, hist = ksl(A, y, -1j * h, verb=0, return_history=True)
            else:
                y = ksl(A, y, -1j * h, verb=0, check_rank=False)
            acorr[k + 1] = tt.dot(psi0, y)
```

The transform uses the MCTDH-standard $\cos^2$ (Hann) window; the zero-padding interpolates the frequency grid but cannot add resolution, and `ifft` carries the $e^{+i\omega t}$ sign that turns $e^{-i\lambda t}$ autocorrelations into peaks at $+\lambda$:

```python
def spectrum(acorr, h, pad=8):
    N = len(acorr)
    tgrid = np.arange(N) * h
    w = np.cos(np.pi * tgrid / (2.0 * tgrid[-1])) ** 2
    M = pad * N
    S = np.fft.ifft(acorr * w, M) * M * h / np.pi
    omega = 2.0 * np.pi * np.arange(M // 2) / (M * h)
    return omega, S.real[:M // 2]
```

Peaks are local maxima above a relative threshold, refined off the frequency grid by parabolic interpolation through the three points around each maximum (`find_peaks`).

## What comes out

The default run ($d=4$, $n=10$, rank 12, 2048 steps of $h=0.1$, so $T = 204.8$ and resolution $2\pi/T = 0.031$) prints, verbatim:

```
rank-adequacy check at step 1: defect_rel = 9.9e-08 (the share of H psi the rank cannot follow)
propagation: 34.0 s (16.6 ms/step), norm drift 4.4e-12 (the exact flow is unitary)

eigb oracle (10 lowest levels, 1.9 s): 1.9957, 2.9726, 2.9810, 2.9872, 2.9893, 3.9030, 3.9310, 3.9621, 3.9627, 3.9712

  peak omega     height   nearest eigb   difference
      1.9957     12.133         1.9957      1.2e-05
      2.9873     10.834         2.9872      1.1e-04
      3.9732      4.435         3.9712      1.9e-03
      4.9491      1.111         3.9712      9.8e-01   <-- off the eigb window
```

Reading it:

* The **ground state** comes out at **1.9957**, matching `eigb` to **1.2e-05** — three orders of magnitude below the $2\pi/T$ resolution, thanks to the parabolic refinement.
* The second peak sits on the **2.97–2.99 multiplet**: `eigb` resolves four levels there (2.9726, 2.9810, 2.9872, 2.9893), spread **0.017** — *below* the resolution $2\pi/T = 0.031$ — so the transform can only show one peak, and does (the zoom panel of the figure). That is the method's resolution limit demonstrated on a real quasi-degeneracy, not a failure of either solver.
* The peak at 4.9491 is real but lies beyond $\lambda_9 = 3.971$, the last level the 10-level `eigb` run computed — flagged as "off the eigb window" rather than silently matched.
* The norm drifts by $4\times 10^{-12}$ over 2048 steps: the exact flow is unitary (this $H$ is Hermitian — no CAP here), and the integrator respects that.

## Why believe it

* **`eigb` on the same operator is the oracle** — not just for the ground state but for everything the packet can see: the same MPO object `A` goes into both the propagation and the block eigensolver, so any assembly bug hits both sides identically and any *method* discrepancy shows in the table.
* **The transform + peak finder are tested against planted frequencies** — `tests/test_examples.py::test_spectrum_transform_recovers_planted_frequencies`: a synthetic $a(t) = \sum_l w_l  e^{-i\lambda_l t}$ with $\lambda = (1.0,\ 2.3,\ 4.7)$ must produce exactly three peaks, each within $10^{-3}$ of its planted frequency (far better than the grid spacing), and nothing else above threshold.
* **The rank is checked, not assumed**: `defect_rel = 9.9e-08` at step 1 says the rank-12 manifold can follow $H\psi$ to eight digits for this packet.
* The KSL machinery itself is pinned against a dense `expm` propagator in the companion test `test_ksl_paper_setup_matches_dense_propagation` (see the [paper-scale page](henon_heiles_ksl_paper.md)), and the compiled/interpreted complex-step parity is pinned in `tests/test_eigb_ksl.py`.

## Run it

```console
$ python examples/henon_heiles_spectrum.py              # d=4, n=10, rank 12
$ python examples/henon_heiles_spectrum.py 4 10 12      # d, basis, KSL rank
$ python examples/henon_heiles_spectrum.py 4 10 12 2048 0.1 0.7
                                           # ... nsteps, step h, displacement
```

The default run takes about 36 s in the repository venv on a Mac laptop (34 s propagation + 1.9 s `eigb`). The acceptance test is instant (pure numpy, no propagation):

```console
$ pytest tests/test_examples.py::test_spectrum_transform_recovers_planted_frequencies
```

## The same run at $d=8$: beyond any dense oracle

`python examples/henon_heiles_spectrum.py 8 10 16` runs the identical
pipeline at $10^8$ basis states, where no dense eigensolver can follow.
Measured on 8 cores of a shared h200 (100 s total, `eigb`'s 12 lowest
levels in 29 s at TT ranks the block sweep chooses itself):

| peak $\omega$ | height | nearest eigb | difference |
|---|---|---|---|
| 3.9900 | 4.57 | 3.9900 | 4.4e-06 |
| 4.9814 | 8.08 | 4.9814 | -5.8e-05 |
| 5.9701 | 7.10 | — | beyond the 12 computed levels |

The 4.98 peak covers the *eight* quasi-degenerate levels
4.9640–4.9836 (spread 0.020, below the $2\pi/T = 0.031$ resolution) —
the same multiplet effect as at $d=4$, now with a denser ladder.  At this
size the two TT methods are each other's only referee: the trajectory
transform and the block eigensolver agree to $10^{-5}$ from completely
different algorithms.

## References

* I. V. Oseledets, DSc dissertation, Sec. 3.8 — this exact scheme (evolution + autocorrelation + FFT) for the Hénon–Heiles potential, there with a Strang-split QTT propagator; the KSL integrator replaces the step-and-round splitting.
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor trains", SIAM J. Numer. Anal. 53(2):917–941, 2015, arXiv:1407.2042 **[LOV15]**.
* M. H. Beck, A. Jäckle, G. A. Worth, H.-D. Meyer, "The multiconfiguration time-dependent Hartree (MCTDH) method", Phys. Rep. 324:1–105, 2000 — the autocorrelation/spectrum methodology and the $\cos^2$ window.
* Hénon–Heiles as the standard TT/MCTDH benchmark: `docs/plans/eigenvalues.md` section 9.4.
