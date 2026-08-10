#!/usr/bin/env python
"""Reproduce Fig. 3 of [LOV15]: the 10-D Henon-Heiles spectrum by TT-KSL.

    python examples/henon_heiles_ksl_paper.py                # the paper setup
    python examples/henon_heiles_ksl_paper.py 10 32 18 60 0.01
                                             # f, DVR size, rank, T, step h
    python examples/henon_heiles_ksl_paper.py ... out.npz    # 7th arg: save

The experiment of section 6.2 of the paper, verbatim: the time-dependent
Schroedinger equation

    i dpsi/dt = H psi,
    H = -1/2 Delta + 1/2 sum_k q_k^2
        + lam sum_{k<f} (q_k^2 q_{k+1} - q_{k+1}^3 / 3),   lam = 0.111803,

with the initial packet a product of shifted Gaussians ``exp(-(q-2)^2/2)``,
discretized by a sine-DVR with 32 functions per mode, made non-Hermitian by a
cubic complex absorbing potential (CAP)

    W(q) = i eta sum_k ((q_k - 6)_+^3 + (q_k + 6)_-^3),   eta = -1,

and propagated to T = 60 by the second-order (Strang) projector-splitting
integrator at fixed TT rank 18 with time step h = 0.01.  The output is the
absolute value of the Fourier transform of the autocorrelation
``a(t) = <psi(t), psi(0)>`` -- the vibrational spectrum a la MCTDH, which is
what Fig. 3 of the paper shows next to the MCTDH curve (54354 s MCTDH against
4425 s TT-KSL on 2015 hardware).

Differences from the paper, recorded rather than hidden:

* the DVR interval is not stated there; ``[-9, 9]`` is used here (the CAP
  ramps on the outer 3 units on each side).  The spectrum is insensitive to
  the choice as long as the absorber has room to act;
* the local exponentials use this package's EXPOKIT-style Krylov substepping
  (``expmv_krylov``) at relative accuracy 1e-8 -- the same algorithm and
  tolerance the paper takes from the Expokit package itself.

The sine-DVR kinetic matrix is checked on the spot: the harmonic levels on
this grid must come out at ``k + 1/2`` before the run starts.

References
----------
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor
  trains", SIAM J. Numer. Anal. 53(2):917-941, 2015, arXiv:1407.2042 [LOV15],
  section 6.2 and Fig. 3.
* D. T. Colbert, W. H. Miller, J. Chem. Phys. 96:1982-1991, 1992 -- the
  sine-DVR kinetic energy matrix.
* M. Nest, H.-D. Meyer, J. Chem. Phys. 117:10499-10505, 2002 -- the MCTDH
  benchmark this model comes from, including the CAP.
"""

import sys
import time
import warnings

import numpy as np

import tt
from tt.algs.ksl import ksl

LAMBDA = 0.111803
CAP_ETA = 1.0          # |eta|; the sign convention is fixed in cap() below
CAP_EDGE = 6.0
Q_SHIFT = 2.0


def sine_dvr(n, a, b):
    """Sine-DVR on ``[a, b]``: grid and the exact kinetic matrix.

    Colbert-Miller: ``n`` interior points of a particle-in-a-box sine basis,
    ``T`` dense, every function of ``q`` diagonal on the grid.
    """
    n = int(n)
    L = float(b - a)
    j = np.arange(1, n + 1)
    x = a + j * L / (n + 1)
    T = np.empty((n, n))
    pref = np.pi ** 2 / (4.0 * L ** 2)
    for r_ in range(1, n + 1):
        for c in range(1, n + 1):
            if r_ == c:
                T[r_ - 1, c - 1] = pref * ((2.0 * (n + 1) ** 2 + 1) / 3.0
                                           - 1.0 / np.sin(np.pi * r_ / (n + 1)) ** 2)
            else:
                T[r_ - 1, c - 1] = pref * (-1.0) ** (r_ - c) * (
                    1.0 / np.sin(np.pi * (r_ - c) / (2.0 * (n + 1))) ** 2
                    - 1.0 / np.sin(np.pi * (r_ + c) / (2.0 * (n + 1))) ** 2)
    return x, T


def cap(x, eta=CAP_ETA, edge=CAP_EDGE):
    """``-i eta ((q-6)_+^3 + (-6-q)_+^3)``: absorbing under ``exp(-iHt)``.

    The paper writes ``W = i eta (...)`` with ``eta = -1``; this is the same
    operator with the sign folded in, so ``eta`` here is positive and
    absorption is what you get, not what you have to remember.
    """
    ramp = np.maximum(x - edge, 0.0) ** 3 + np.maximum(-edge - x, 0.0) ** 3
    return -1j * eta * ramp


def hamiltonian(f, n, a=-9.0, b=9.0, lam=LAMBDA):
    """The Henon-Heiles + CAP MPO in the sine-DVR basis, TT rank 3, complex."""
    x, T = sine_dvr(n, a, b)
    # the on-the-spot check: harmonic levels on this grid must be k + 1/2
    ho = np.linalg.eigvalsh(T + 0.5 * np.diag(x ** 2))
    worst = np.abs(ho[:10] - (np.arange(10) + 0.5)).max()
    # 32 points on 18 units of interval carry the first ten levels to ~4e-4 --
    # three orders below the 2 pi / T spectral resolution; a formula bug would
    # show as O(1) here, which is what the check is for
    if worst > 1e-3:
        raise RuntimeError(
            f"sine-DVR self-check failed: the first ten harmonic levels are "
            f"off by {worst:.2E} on [{a}, {b}] with n={n}; widen the interval "
            f"or add points")
    print(f"sine-DVR check: first ten harmonic levels within {worst:.1E} "
          f"of k + 1/2 (spectral resolution will be ~0.1)")
    h1 = T + 0.5 * np.diag(x ** 2) + np.diag(cap(x))
    Q1 = np.diag(x)
    Q2 = np.diag(x ** 2)
    Q3 = np.diag(x ** 3)
    I = np.eye(n)
    cores = []
    for k in range(f):
        W = np.zeros((3, 3, n, n), dtype=complex)
        W[2, 2] = I
        W[0, 0] = I
        W[2, 0] = h1 - (lam / 3.0) * Q3 if k > 0 else h1
        W[2, 1] = lam * Q2
        W[1, 0] = Q1
        c = np.transpose(W, (0, 2, 3, 1))
        if k == 0:
            c = c[2:3]
        if k == f - 1:
            c = c[:, :, :, 0:1]
        cores.append(np.ascontiguousarray(c))
    return tt.matrix.from_list(cores), x


def packet(f, x, q0=Q_SHIFT):
    """``prod_k exp(-(q_k - q0)^2 / 2)`` on the grid: a rank-1 complex TT."""
    v = np.exp(-((x - q0) ** 2) / 2.0).astype(complex)
    v = v / np.linalg.norm(v)
    return tt.vector.from_list([v.reshape(1, len(x), 1)] * f)


def run(f=10, n=32, r=18, T=60.0, h=0.01, seed=0, save=None):
    A, x = hamiltonian(f, n)
    psi0 = packet(f, x)

    rng = np.random.default_rng(seed)
    noise = tt.rand([n] * f, r=r, samplefunc=rng.standard_normal)
    noise = noise * (1e-8 / noise.norm())
    y = (psi0 + noise).round(0.0, rmax=r)

    nsteps = int(round(T / h))
    print(f"[LOV15] section 6.2: f={f}, sine-DVR n={n} on [-9, 9], "
          f"CAP at +-{CAP_EDGE}, rank {r}, {nsteps} steps of h={h} (T={T})\n")

    acorr = np.empty(nsteps + 1, dtype=complex)
    acorr[0] = tt.dot(psi0, y)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(nsteps):
            # use_normest=2: skip the power-iteration guess of ||B|| (4 extra
            # operator applications per local exponential).  It only seeds the
            # first Krylov substep, and at tau ||B|| << 1 the whole step is one
            # substep anyway -- measured 25% of the wall time on this problem.
            y = ksl(A, y, -1j * h, verb=0, check_rank=False, use_normest=2)
            acorr[k + 1] = tt.dot(psi0, y)
            if (k + 1) % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  step {k + 1}/{nsteps}: |a| = {abs(acorr[k + 1]):.4f}, "
                      f"||psi|| = {float(y.norm()):.4f}, "
                      f"{el / (k + 1) * 1e3:.0f} ms/step", flush=True)
    t_prop = time.perf_counter() - t0
    print(f"\npropagation: {t_prop:.0f} s total "
          f"({t_prop / nsteps * 1e3:.0f} ms/step; the paper's own run took "
          f"4425 s on 2015 hardware, MCTDH 54354 s)")

    # |a^(xi)| = |integral_0^T a(t) exp(i xi t) dt|, the transform of Fig. 3
    tgrid = np.arange(nsteps + 1) * h
    M = 8 * len(acorr)
    spec = np.abs(np.fft.ifft(acorr, M) * M * h)
    xi = 2.0 * np.pi * np.arange(M // 2) / (M * h)
    if save:
        np.savez(save, xi=xi[: M // 2], spec=spec[: M // 2], acorr=acorr,
                 tgrid=tgrid, f=f, n=n, r=r, T=T, h=h)
        print(f"saved: {save}")
    return xi[: M // 2], spec[: M // 2], acorr


if __name__ == "__main__":
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    h = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
    save = sys.argv[6] if len(sys.argv) > 6 else None
    run(f, n, r, T, h, save=save)
