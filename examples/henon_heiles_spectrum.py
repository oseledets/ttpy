#!/usr/bin/env python
"""Vibrational spectrum of Henon-Heiles by KSL time evolution + autocorrelation.

    python examples/henon_heiles_spectrum.py              # d=4, n=10, rank 12
    python examples/henon_heiles_spectrum.py 4 10 12      # d, basis, KSL rank
    python examples/henon_heiles_spectrum.py 4 10 12 2048 0.1 0.7
                                             # ... nsteps, step h, displacement

The spectral method of quantum molecular dynamics (the workhorse of MCTDH; see
the references): instead of diagonalizing ``H``, evolve a wave packet,

    i dpsi/dt = H psi,   psi(0) = a displaced Gaussian,

record only the **autocorrelation** ``a(t) = <psi(0), psi(t)>``, and Fourier
transform it.  If ``psi(0) = sum_l c_l phi_l`` in the eigenbasis of ``H``,
then ``a(t) = sum_l |c_l|^2 exp(-i lambda_l t)``, so the windowed transform

    sigma(omega) = Re integral_0^T a(t) w(t) exp(i omega t) dt

peaks exactly at the eigenvalues ``lambda_l`` that the packet overlaps -- the
whole spectrum on an interval from ONE trajectory, where an eigensolver would
need a block of that many states.  The price is resolution: peaks are
``~2 pi / T`` wide, so close levels need long trajectories.

The propagator here is the projector-splitting KSL integrator of [LOV15] at
fixed TT rank -- ``tau = -1j h`` promotes the whole sweep to complex128 (on
small local blocks it runs the compiled complex kernels of
``tt/algs/_ksl_fast.py``; at the block sizes of this example, the interpreted
Krylov path).  The check that closes the loop: the peak positions are compared
against ``tt.eigb`` run on the *same* discretized operator, which is an oracle
for everything the packet can see -- not just the ground state.

Model and conventions (``tests/hamiltonians.py`` is the SSOT for both):

    H = sum_{i=1}^{d} (1/2)(-d^2/dq_i^2 + q_i^2)
        + lam sum_{i=1}^{d-1} (q_i^2 q_{i+1} - q_{i+1}^3 / 3),  lam = 0.111803,

in an ``n``-function harmonic-oscillator product basis with Galerkin ``q^2``,
``q^3`` matrix elements (formed at size ``n + 6`` and cut -- squaring the
truncated ``q`` instead is a percent-level error on the top basis functions).
The initial packet is the coherent state ``|alpha>``, ``alpha = q0 / sqrt(2)``
per mode: a *rank-1* TT, padded to the KSL rank with 1e-8 random noise (KSL
keeps ranks fixed; it cannot grow a rank-1 start).

References
----------
* I. V. Oseledets, DSc dissertation, Sec. 3.8: this exact scheme (evolution +
  autocorrelation + FFT) for the Henon-Heiles potential, there with a
  Strang-split QTT propagator; the KSL integrator replaces the
  step-and-round splitting.
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor
  trains", SIAM J. Numer. Anal. 53(2):917-941, 2015, arXiv:1407.2042 [LOV15].
* M. H. Beck, A. Jaeckle, G. A. Worth, H.-D. Meyer, "The multiconfiguration
  time-dependent Hartree (MCTDH) method", Phys. Rep. 324:1-105, 2000 -- the
  autocorrelation/spectrum methodology and the cos^2 window.
* Henon-Heiles as the standard TT/MCTDH benchmark: docs/plans/eigenvalues.md
  section 9.4.
"""

import sys
import time
import warnings
from math import factorial

import numpy as np

import tt
from tt.algs.eigb import eigb
from tt.algs.ksl import ksl

HENON_HEILES_LAMBDA = 0.111803


def ho_operators(n, pad=6):
    """``(N, Q, Q2, Q3)`` in the HO basis; mirrors ``tests/hamiltonians.py``."""
    n, m = int(n), int(n) + int(pad)
    k = np.arange(m, dtype=float)
    q = np.diag(np.sqrt((k[:-1] + 1.0) / 2.0), 1)
    q = q + q.T
    q2, q3 = q @ q, q @ q @ q
    return (np.diag(k[:n] + 0.5), q[:n, :n], q2[:n, :n], q3[:n, :n])


def henon_heiles(d, n, lam=HENON_HEILES_LAMBDA):
    """The Henon-Heiles MPO, TT rank 3; mirrors ``tests/hamiltonians.py``."""
    N, Q, Q2, Q3 = ho_operators(n)
    I = np.eye(int(n))
    cores = []
    for k in range(d):
        W = np.zeros((3, 3, int(n), int(n)))
        W[2, 2] = I
        W[0, 0] = I
        W[2, 0] = N - (lam / 3.0) * Q3 if k > 0 else N
        W[2, 1] = lam * Q2
        W[1, 0] = Q
        c = np.transpose(W, (0, 2, 3, 1))
        if k == 0:
            c = c[2:3]
        if k == d - 1:
            c = c[:, :, :, 0:1]
        cores.append(np.ascontiguousarray(c))
    return tt.matrix.from_list(cores)


def coherent_packet(d, n, q0):
    """Rank-1 TT of the product coherent state displaced by ``q0`` per mode."""
    alpha = q0 / np.sqrt(2.0)
    v = np.array([np.exp(-alpha ** 2 / 2.0) * alpha ** k / np.sqrt(factorial(k))
                  for k in range(n)])
    v = v / np.linalg.norm(v)          # renormalize the basis truncation away
    return tt.vector.from_list([v.reshape(1, n, 1)] * d)


def spectrum(acorr, h, pad=8):
    """Windowed transform: ``sigma(omega_m)`` on ``omega_m = 2 pi m / (M h)``.

    cos^2 (Hann) window, the MCTDH standard; zero-padding interpolates the
    grid, it cannot add resolution.  ``ifft`` carries the ``exp(+i omega t)``
    sign that turns ``exp(-i lambda t)`` autocorrelations into peaks at
    ``+lambda``.
    """
    N = len(acorr)
    tgrid = np.arange(N) * h
    w = np.cos(np.pi * tgrid / (2.0 * tgrid[-1])) ** 2
    M = pad * N
    S = np.fft.ifft(acorr * w, M) * M * h / np.pi
    omega = 2.0 * np.pi * np.arange(M // 2) / (M * h)
    return omega, S.real[:M // 2]


def find_peaks(omega, S, rel=0.02):
    """Local maxima above ``rel * max``, refined by parabolic interpolation."""
    peaks = []
    thr = rel * S.max()
    for m in range(1, len(S) - 1):
        if S[m] > thr and S[m - 1] < S[m] >= S[m + 1]:
            denom = S[m - 1] - 2.0 * S[m] + S[m + 1]
            shift = 0.5 * (S[m - 1] - S[m + 1]) / denom if denom != 0 else 0.0
            peaks.append((omega[m] + shift * (omega[1] - omega[0]), S[m]))
    return peaks


def run(d=4, n=10, r=12, nsteps=2048, h=0.1, q0=0.7, nlev=10, seed=0):
    A = henon_heiles(d, n)
    psi0 = coherent_packet(d, n, q0)

    # pad the rank-1 packet to the KSL manifold rank
    rng = np.random.default_rng(seed)
    noise = tt.rand([n] * d, r=r, samplefunc=rng.standard_normal)
    noise = noise * (1e-8 / noise.norm())
    y = (psi0 + noise).round(0.0, rmax=r)

    T = nsteps * h
    print(f"Henon-Heiles d={d}, n={n} basis, KSL rank {r}: "
          f"{nsteps} steps of h={h} (T={T:.0f}, resolution 2pi/T="
          f"{2 * np.pi / T:.3f})\n")

    acorr = np.empty(nsteps + 1, dtype=complex)
    acorr[0] = tt.dot(psi0, y)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(nsteps):
            hist = None
            if k == 0:                 # one defect check: is the rank enough?
                y, hist = ksl(A, y, -1j * h, verb=0, return_history=True)
            else:
                y = ksl(A, y, -1j * h, verb=0, check_rank=False)
            acorr[k + 1] = tt.dot(psi0, y)
            if hist is not None:
                print(f"rank-adequacy check at step 1: "
                      f"defect_rel = {hist.defect_rel:.1e} "
                      f"(the share of H psi the rank cannot follow)")
    t_prop = time.perf_counter() - t0
    drift = abs(float(y.norm()) - 1.0)
    print(f"propagation: {t_prop:.1f} s "
          f"({t_prop / nsteps * 1e3:.1f} ms/step), "
          f"norm drift {drift:.1e} (the exact flow is unitary)\n")

    omega, S = spectrum(acorr, h)
    peaks = find_peaks(omega, S)

    # the oracle: eigb on the same operator
    ranks = [1] + [max(r, 20)] * (d - 1) + [nlev]
    y0 = tt.rand(n, d, r=ranks, samplefunc=rng.standard_normal)
    t0 = time.perf_counter()
    _, lam = eigb(A, y0, 1e-8, nswp=30, rmax=80, verb=0)
    t_eig = time.perf_counter() - t0
    lam = np.sort(np.asarray(lam))
    print(f"eigb oracle ({nlev} lowest levels, {t_eig:.1f} s): "
          + ", ".join(f"{v:.4f}" for v in lam) + "\n")

    print(f"{'peak omega':>12} {'height':>10} {'nearest eigb':>14} "
          f"{'difference':>12}")
    for w, height in peaks:
        j = int(np.argmin(np.abs(lam - w)))
        diff = w - lam[j]
        mark = "" if abs(diff) < 2 * np.pi / T else "   <-- off the eigb window"
        print(f"{w:12.4f} {height:10.3f} {lam[j]:14.4f} {diff:12.1e}{mark}")
    print(f"\n(a peak matches its level when |difference| < 2pi/T = "
          f"{2 * np.pi / T:.3f}; peaks beyond lambda_{nlev - 1} = "
          f"{lam[-1]:.3f} have no oracle row)")
    return omega, S, peaks, lam


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    h = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    q0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7
    run(d, n, r, nsteps, h, q0)
