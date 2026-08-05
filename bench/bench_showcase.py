"""Real TT/QTT problems from the literature, each against a reference we did
not compute ourselves.

The other benchmarks in this directory measure *operations* -- rounding, dot,
matvec -- and their only output is a time.  This one measures *problems*, and a
row earns its place only if it carries a reference value that does not come out
of ttpy2:

* a **closed form** (Pfeuty's critical Ising energy, the normal-mode spectrum of
  a quadratic Hamiltonian, ``Im[((e^i-1)/i)^d]`` for the sine integral, Genz's
  corner-peak formula, the nodal solution ``x - x^2/2``);
* a **published structural fact** (Khoromskij's quantics ranks: 1 for an
  exponential, 2 for a sine, ``m+1`` for a polynomial of degree ``m``);
* an **independent dense or sparse computation** we can afford at small size
  (``numpy.linalg.eigvalsh``, ``scipy.sparse.linalg.eigsh``, ``numpy.fft``).

Everything else -- a timing with nothing to compare it against -- is left to
``bench_core.py``.  Conventions follow it: the regime travels with the number
(backend, device, dtype, thread count, sizes, repeats), timings are the median
of ``repeats`` runs after a warm-up when ``repeats > 1`` and a single run
otherwise (recorded in the ``warmup`` field), and the JSON carries the machine.

    python bench/bench_showcase.py --out bench/results/showcase.json
    python bench/bench_showcase.py --problems tfim integrals --scale large

``--scale small`` shrinks every problem for a smoke run; ``--scale large`` is
the deliberate push (it is *not* bounded by the two-minute rule).

Sources of the reference values are in ``docs/BENCHMARKS.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from decimal import Decimal, getcontext

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import tt                                                   # noqa: E402
import hamiltonians as ham                                  # noqa: E402
from tt.algs.amen import amen_solve                         # noqa: E402
from tt.algs.cross import cross                             # noqa: E402
from tt.algs.eigb import eigb                               # noqa: E402
from tt.algs.qtt_ell import solve_direct_1d                 # noqa: E402

SEED = 0


# --- timing, in the shape bench_core.py established -------------------------

def run(fn, repeats):
    """``(value, median_s, best_s, warmup)`` for a callable returning a value.

    With ``repeats > 1`` the callable is run once to warm up and then
    ``repeats`` more times; with ``repeats == 1`` it is run exactly once and the
    time is a single sample, which the row records so nobody reads it as a
    median.
    """
    if repeats > 1:
        fn()
    ts, out = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts), min(ts), repeats > 1


def randstart(n, d, r, block=1, seed=SEED):
    """Block TT start for ``eigb``: the block index rides on the last rank."""
    ranks = [1] + [r] * (d - 1) + [block]
    rng = np.random.default_rng(seed)
    return tt.rand(n, d, r=ranks, samplefunc=rng.standard_normal)


def relerr(got, ref):
    ref = float(ref)
    return abs(float(got) - ref) / abs(ref) if ref != 0.0 else abs(float(got))


# --- 1. quantum spin chains -------------------------------------------------

def bench_tfim(scale, repeats):
    """Critical transverse-field Ising chain against Pfeuty's closed form.

    ``H = -sum sz_i sz_{i+1} - sum sx_i`` (Pauli operators, open chain, g = 1).
    The reference is ``E_0(L) = 1 - 1/sin(pi/(2(2L+1)))`` -- the free-fermion
    solution, exact for every ``L`` and with no LAPACK anywhere in it.  That is
    what makes this the one large eigenvalue problem in the suite whose oracle
    survives past the size where a dense matrix exists.
    """
    sizes = {"small": [16, 32], "default": [16, 32, 64],
             "large": [16, 32, 64, 128, 256]}[scale]
    rows = []
    for L in sizes:
        A = ham.tfim(L, g=1.0)
        ref = ham.tfim_critical_ground_energy(L)
        y0 = randstart(2, L, r=min(24, 2 ** (L // 2)), block=2)
        out, med, best, warm = run(
            lambda: eigb(A, y0, 1e-9, nswp=30, rmax=100, verb=0,
                         return_history=True), repeats)
        _, lam, h = out
        rows.append(dict(
            problem="tfim_critical", case=f"L={L}", solver="eigb(B=2)",
            reference="Pfeuty free-fermion closed form E0(L)=1-1/sin(pi/(2(2L+1)))",
            ref_value=float(ref), measured=float(lam[0]),
            rel_err=relerr(lam[0], ref), rank=int(max(h.ranks)),
            sweeps=int(h.nswp_done), converged=bool(h.converged),
            res_back=float(np.max(h.res_back)) if h.res_back is not None else None,
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(L=L, states=f"2^{L}")))
    return rows


def bench_heisenberg(scale, repeats):
    """Heisenberg chain: a dense oracle where one exists, Bethe where it does not.

    ``H = sum_i S_i . S_{i+1}`` on an open chain.  Two references, deliberately
    of different kinds:

    * ``L = 12``: ``numpy.linalg.eigvalsh`` on the 4096 x 4096 dense matrix.
    * ``L = 32, 64``: the Bethe-ansatz bulk energy ``1/4 - ln 2`` (Hulthen 1938).
      A finite open chain does not have that energy -- it has a surface term --
      so the comparison is made on the **difference** ``(E(L2)-E(L1))/(L2-L1)``,
      which cancels the surface term and leaves the ``1/L^2`` and logarithmic
      corrections.  The deviation reported is therefore physics, not solver
      error, and is labelled as such.
    """
    dense_L = 12
    rows = []
    A = ham.heisenberg(dense_L)
    ref = float(np.linalg.eigvalsh(ham.dense(A))[0])
    y0 = randstart(2, dense_L, r=24, block=2)
    out, med, best, warm = run(
        lambda: eigb(A, y0, 1e-10, nswp=30, verb=0, return_history=True), repeats)
    _, lam, h = out
    rows.append(dict(
        problem="heisenberg_dense", case=f"L={dense_L}", solver="eigb(B=2)",
        reference="numpy.linalg.eigvalsh on the 4096x4096 dense matrix",
        ref_value=ref, measured=float(lam[0]), rel_err=relerr(lam[0], ref),
        rank=int(max(h.ranks)), sweeps=int(h.nswp_done),
        converged=bool(h.converged), median_s=med, best_s=best,
        warmup=warm, repeats=repeats, extra=dict(L=dense_L)))

    pair = {"small": (16, 32), "default": (32, 64), "large": (64, 128)}[scale]
    energies = {}
    for L in pair:
        A = ham.heisenberg(L)
        y0 = randstart(2, L, r=40, block=2)
        out, med, best, warm = run(
            lambda: eigb(A, y0, 1e-9, nswp=30, rmax=120, verb=0,
                         return_history=True), repeats)
        _, lam, h = out
        energies[L] = float(lam[0])
        rows.append(dict(
            problem="heisenberg_chain", case=f"L={L}", solver="eigb(B=2)",
            reference="none for the finite open chain; feeds the bulk-energy row",
            ref_value=None, measured=float(lam[0]), rel_err=None,
            rank=int(max(h.ranks)), sweeps=int(h.nswp_done),
            converged=bool(h.converged),
            res_back=float(np.max(h.res_back)) if h.res_back is not None else None,
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(L=L)))
    l1, l2 = pair
    bulk = (energies[l2] - energies[l1]) / (l2 - l1)
    ref = ham.heisenberg_bulk_energy_per_site()
    rows.append(dict(
        problem="heisenberg_bulk", case=f"(E({l2})-E({l1}))/{l2 - l1}",
        solver="eigb(B=2), finite difference in L",
        reference="Bethe ansatz / Hulthen 1938: e_inf = 1/4 - ln 2 (L -> infinity)",
        ref_value=float(ref), measured=float(bulk), rel_err=relerr(bulk, ref),
        rank=None, sweeps=None, converged=None, median_s=None, best_s=None,
        warmup=False, repeats=repeats,
        extra=dict(note="the residual gap is the finite-size correction of the "
                        "open chain, not solver error")))
    return rows


# --- 2. molecular vibrations ------------------------------------------------

def bench_coupled_oscillator(scale, repeats):
    """Bilinearly coupled oscillators: a quantum problem with an analytic answer.

    The benchmark of Rakhuba & Oseledets (JCP 145:124101, 2016) section V.1:
    ``H = sum_i (w_i/2)(-d^2/dq_i^2 + q_i^2) + alpha sum_{i<j} q_i q_j``,
    ``w_j = sqrt(j/2)``, ``alpha = 0.1``, all pairs coupled.  Being quadratic it
    has an exact spectrum from a ``d x d`` dense eigenproblem, so a ``15^64``
    quantum problem is checked against 64 numbers that never saw a tensor.

    Two rows per size, because two different errors are in play:

    * ``alpha = 0`` -- the ground state is exactly a product of basis functions,
      so the finite basis is exact and the whole error is the eigensolver's;
    * ``alpha = 0.1`` -- the finite harmonic-oscillator basis now truncates, and
      the gap to the analytic value is basis error *plus* solver error.  It is
      reported as the total, with the basis size in the row.
    """
    sizes = {"small": [8, 16], "default": [8, 16, 32], "large": [8, 16, 32, 64]}[scale]
    nb = 15
    rows = []
    for d in sizes:
        for alpha in (0.0, 0.1):
            A = ham.coupled_oscillator(d, n=nb, alpha=alpha)
            _, ref = ham.coupled_oscillator_exact_levels(d, alpha=alpha)
            y0 = randstart(nb, d, r=6, block=2)
            out, med, best, warm = run(
                lambda: eigb(A, y0, 1e-10, nswp=25, rmax=60, verb=0,
                             return_history=True), repeats)
            _, lam, h = out
            rows.append(dict(
                problem="coupled_oscillator", case=f"d={d} n={nb} alpha={alpha}",
                solver="eigb(B=2)",
                reference=("normal-mode closed form E0 = (1/2) sum sqrt(eig(A B)), "
                           "exact for the continuous operator"),
                ref_value=float(ref), measured=float(lam[0]),
                rel_err=relerr(lam[0], ref), rank=int(max(h.ranks)),
                sweeps=int(h.nswp_done), converged=bool(h.converged),
                median_s=med, best_s=best, warmup=warm, repeats=repeats,
                extra=dict(d=d, n=nb, alpha=alpha, states=f"{nb}^{d}",
                           basis_exact=(alpha == 0.0))))
    return rows


def bench_henon_heiles(scale, repeats):
    """Henon-Heiles vibrational levels against an independent sparse eigensolver.

    ``H = sum_i (1/2)(-d^2/dq_i^2 + q_i^2)
          + lam sum_i (q_i^2 q_{i+1} - q_{i+1}^3/3)``, ``lam = 0.111803``,
    dimensionless oscillator units -- the classic TT eigenvalue benchmark
    (``docs/plans/eigenvalues.md`` section 9.4).

    **The published MCTDH/DVR level tables are still not in hand**, so the
    reference here is not a citation: it is ``scipy.sparse.linalg.eigsh`` on the
    same operator assembled by Kronecker products with no tensor format
    involved (:func:`hamiltonians.henon_heiles_sparse`).  That is an oracle for
    the discretized problem, affordable to ``d = 4`` at ``n = 15`` and no
    further, which is exactly the point: past it there is no oracle at all and
    the rows carry no reference.

    The harmonic limit ``lam = 0`` is added at every ``d`` because there the
    answer is ``d/2`` exactly, in any basis -- a reference that survives to the
    sizes where the sparse one has died.
    """
    import scipy.sparse.linalg as spla

    nb = 15
    oracle_d = {"small": [2, 3], "default": [2, 3, 4], "large": [2, 3, 4, 5]}[scale]
    blind_d = {"small": [6], "default": [6, 10], "large": [6, 10, 16, 20]}[scale]
    rows = []

    for d in oracle_d:
        A = ham.henon_heiles(d, n=nb)
        S = ham.henon_heiles_sparse(d, n=nb)
        t0 = time.perf_counter()
        ref = float(spla.eigsh(S, k=1, which="SA", tol=0, maxiter=20000)[0][0])
        t_ref = time.perf_counter() - t0
        y0 = randstart(nb, d, r=8, block=2)
        out, med, best, warm = run(
            lambda: eigb(A, y0, 1e-11, nswp=30, rmax=80, verb=0,
                         return_history=True), repeats)
        _, lam, h = out
        rows.append(dict(
            problem="henon_heiles", case=f"d={d} n={nb} lam=0.111803",
            solver="eigb(B=2)",
            reference="scipy.sparse.linalg.eigsh on the Kronecker-assembled matrix",
            ref_value=ref, measured=float(lam[0]), rel_err=relerr(lam[0], ref),
            rank=int(max(h.ranks)), sweeps=int(h.nswp_done),
            converged=bool(h.converged), median_s=med, best_s=best,
            warmup=warm, repeats=repeats,
            extra=dict(d=d, n=nb, states=nb ** d, oracle_s=t_ref)))

    for d in blind_d:
        A0 = ham.henon_heiles(d, n=nb, lam=0.0)
        y0 = randstart(nb, d, r=4, block=2)
        out, med, best, warm = run(
            lambda: eigb(A0, y0, 1e-11, nswp=25, rmax=40, verb=0,
                         return_history=True), repeats)
        _, lam, h = out
        rows.append(dict(
            problem="henon_heiles_harmonic", case=f"d={d} n={nb} lam=0",
            solver="eigb(B=2)",
            reference="analytic: the harmonic limit has E0 = d/2 in any basis",
            ref_value=0.5 * d, measured=float(lam[0]), rel_err=relerr(lam[0], 0.5 * d),
            rank=int(max(h.ranks)), sweeps=int(h.nswp_done),
            converged=bool(h.converged), median_s=med, best_s=best,
            warmup=warm, repeats=repeats, extra=dict(d=d, n=nb)))

        A = ham.henon_heiles(d, n=nb)
        y0 = randstart(nb, d, r=8, block=2)
        out, med, best, warm = run(
            lambda: eigb(A, y0, 1e-10, nswp=30, rmax=80, verb=0,
                         return_history=True), repeats)
        _, lam, h = out
        rows.append(dict(
            problem="henon_heiles_noref", case=f"d={d} n={nb} lam=0.111803",
            solver="eigb(B=2)",
            reference="NONE -- no oracle at this size and no published table in hand",
            ref_value=None, measured=float(lam[0]), rel_err=None,
            rank=int(max(h.ranks)), sweeps=int(h.nswp_done),
            converged=bool(h.converged),
            res_back=float(np.max(h.res_back)) if h.res_back is not None else None,
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, n=nb, states=f"{nb}^{d}")))
    return rows


# --- 3. high-dimensional integration ----------------------------------------

def _gauss_legendre_01(nq):
    x, w = np.polynomial.legendre.leggauss(nq)
    return 0.5 * (x + 1.0), 0.5 * w


def _quad_weight_tt(w, d):
    nq = w.size
    return tt.vector.from_list([w.reshape(1, nq, 1)] * d)


def bench_integral_sine(scale, repeats):
    """``int_{[0,1]^d} sin(x_1 + ... + x_d) dx`` -- the TT-cross integral.

    The example Oseledets & Tyrtyshnikov used to introduce TT-cross (LAA 432(1),
    2010) and Dolgov & Savostyanov reused for high-precision quadrature
    (arXiv:1903.11554).  Its value is closed form:
    ``Im[((e^i - 1)/i)^d]``, because the integrand is the imaginary part of a
    product.  Gauss-Legendre in each mode turns it into a tensor contraction; a
    cross approximation finds the tensor, and the metric that matters is the
    number of **function evaluations**, not the time: a 100-dimensional integral
    has 16^100 quadrature nodes and cross looks at a few tens of thousands.
    """
    sizes = {"small": [10, 50], "default": [10, 50, 100],
             "large": [10, 50, 100, 500]}[scale]
    nq = 16
    x, w = _gauss_legendre_01(nq)
    rows = []
    for d in sizes:
        z = (np.exp(1j) - 1.0) / 1j
        ref = float((z ** d).imag)

        def integrate():
            F = cross(lambda I: np.sin(x[I].sum(axis=1)), nq, d,
                      eps=1e-12, r=2, kickrank=2, nswp=12, seed=SEED)
            return F, tt.dot(F, _quad_weight_tt(w, d))

        (F, val), med, best, warm = run(integrate, repeats)
        rows.append(dict(
            problem="integral_sine", case=f"d={d} nq={nq}", solver="tt.cross",
            reference="closed form Im[((e^i-1)/i)^d]",
            ref_value=ref, measured=float(val), rel_err=relerr(val, ref),
            rank=int(max(F.r)), fun_eval=int(F.history.fun_eval),
            sweeps=len(F.history.sweeps), converged=bool(F.history.converged),
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, nq=nq, nodes=f"{nq}^{d}",
                       eval_fraction=F.history.fun_eval / nq ** min(d, 40))))
    return rows


def _cornerpeak_exact(d, digits=90):
    """Genz's corner-peak integral with ``a_i = 1/d``, in extended precision.

    ``int (1 + sum a_i x_i)^{-(d+1)} dx
       = (1/(d! prod a_i)) sum_{v in {0,1}^d} (-1)^{|v|+d} / (1 + a.v)``.

    The sum has ``2^d`` terms of size O(1) and a result of size ``e^{-d}``, so
    in float64 it loses every significant digit by ``d ~ 14``.  ``Decimal`` at
    90 digits makes the formula usable as a reference instead of a curiosity;
    this is the same integral, not a different one.
    """
    getcontext().prec = digits
    a = Decimal(1) / Decimal(d)
    fac = Decimal(1)
    for k in range(1, d + 1):
        fac *= Decimal(k)
    total = Decimal(0)
    for v in range(1 << d):
        bits = bin(v).count("1")
        total += Decimal(-1) ** (bits + d) / (Decimal(1) + a * bits)
    return float(total / (fac * a ** d))


def bench_integral_cornerpeak(scale, repeats):
    """Genz's corner peak, the non-separable member of the standard test family.

    ``f(x) = (1 + sum_i a_i x_i)^{-(d+1)}`` with ``a_i = 1/d`` (Genz, *Testing
    multidimensional integration routines*, 1984).  Unlike the sine integral
    this one is not a rank-2 tensor: the rank grows with the requested accuracy,
    so it measures what cross does when the tensor is only *approximately* low
    rank.  Its exact value is the alternating ``2^d``-term formula above,
    evaluated in 90-digit decimal because float64 cannot.
    """
    sizes = {"small": [6, 10], "default": [6, 10, 14], "large": [6, 10, 14, 18]}[scale]
    nq = 24
    x, w = _gauss_legendre_01(nq)
    rows = []
    for d in sizes:
        ref = _cornerpeak_exact(d)
        a = 1.0 / d

        def integrate():
            F = cross(lambda I: (1.0 + a * x[I].sum(axis=1)) ** (-(d + 1.0)),
                      nq, d, eps=1e-13, r=2, kickrank=2, nswp=15, seed=SEED)
            return F, tt.dot(F, _quad_weight_tt(w, d))

        (F, val), med, best, warm = run(integrate, repeats)
        rows.append(dict(
            problem="integral_cornerpeak", case=f"d={d} nq={nq}", solver="tt.cross",
            reference="Genz 1984 closed form, evaluated in 90-digit Decimal",
            ref_value=ref, measured=float(val), rel_err=relerr(val, ref),
            rank=int(max(F.r)), fun_eval=int(F.history.fun_eval),
            sweeps=len(F.history.sweeps), converged=bool(F.history.converged),
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, nq=nq, a=a)))
    return rows


# --- 4. quantics structure --------------------------------------------------

def _qtt_grid(d):
    """``x = (sum_k i_k 2^k)/2^d`` -- mode 1 is the least significant bit."""
    wt = 2.0 ** np.arange(d)
    return lambda I: (I @ wt) / 2.0 ** d


#: A rank claim is only meaningful together with the tolerance that resolves it.
#: Measured on the middle unfolding at ``d = 10``, singular values relative to
#: the first: the degree-3 polynomial has ``s4 = 2.2e-10`` and the degree-5 one
#: ``s5 = 4.3e-11``, ``s6 = 6.8e-15``.  At ``eps = 1e-10`` the degree-5
#: polynomial therefore comes back at rank 5 -- correctly for that tolerance,
#: and uselessly as a test of Khoromskij's exact-arithmetic ``m + 1``.  1e-13
#: resolves both with room to spare and still sits above the 1e-15 floor where
#: the cross would start chasing rounding noise.
QTT_RANK_EPS = 1e-13

_QTT_FUNCTIONS = [
    ("exp(3x)", lambda t: np.exp(3.0 * t), 1,
     "Khoromskij 2011: an exponential on a uniform grid has quantics rank 1"),
    ("sin(9x)", lambda t: np.sin(9.0 * t), 2,
     "Khoromskij 2011: a sine on a uniform grid has quantics rank 2"),
    ("poly deg 3", lambda t: 1.0 + 2.0 * t - 3.0 * t ** 2 + 0.5 * t ** 3, 4,
     "Khoromskij 2011: a polynomial of degree m has quantics rank m+1"),
    ("poly deg 5", lambda t: np.polyval([1.0, -2.0, 0.5, 3.0, -1.0, 2.0], t), 6,
     "Khoromskij 2011: a polynomial of degree m has quantics rank m+1"),
    ("gauss sigma=1e-1", lambda t: np.exp(-((t - 0.5) ** 2) / 2e-2), None,
     "no closed-form rank; the published claim is that it is bounded in d"),
    ("gauss sigma=1e-3", lambda t: np.exp(-((t - 0.5) ** 2) / 2e-6), None,
     "no closed-form rank; the published claim is that it is bounded in d"),
]


def bench_qtt_ranks(scale, repeats):
    """The published quantics ranks, rediscovered by a black-box cross.

    Khoromskij (*O(d log N)-quantics approximation of N-d tensors*, Constr.
    Approx. 34:257-280, 2011) proves that on a uniform grid of ``2^d`` points an
    exponential has QTT rank 1, a sine rank 2, and a polynomial of degree ``m``
    rank ``m+1``, **independently of d**.  Those are integers, so they are a
    reference in the strictest sense.

    The check is deliberately made through ``tt.cross``, which is given nothing
    but a black box over bit patterns: if it returns rank 2 for a sine on a
    ``2^40``-point grid, the structure was found, not assumed.  The Gaussian is
    in the table as the case where no exact rank is published -- only the claim
    that it stays bounded as ``d`` grows, which is what the ``d`` sweep measures.
    """
    sizes = {"small": [10, 20], "default": [10, 20, 30], "large": [10, 20, 30, 40]}[scale]
    rows = []
    for d in sizes:
        g = _qtt_grid(d)
        for name, fn, want, why in _QTT_FUNCTIONS:
            def build():
                return cross(lambda I: fn(g(I)), 2, d, eps=QTT_RANK_EPS, r=3,
                             kickrank=2, nswp=20, seed=SEED)

            F, med, best, warm = run(build, repeats)
            got = int(max(F.r))
            rows.append(dict(
                problem="qtt_rank", case=f"{name}, d={d}", solver="tt.cross",
                reference=why, ref_value=want, measured=got,
                rel_err=None if want is None else float(got - want),
                rank=got, fun_eval=int(F.history.fun_eval),
                sweeps=len(F.history.sweeps), converged=bool(F.history.converged),
                median_s=med, best_s=best, warmup=warm, repeats=repeats,
                extra=dict(d=d, points=f"2^{d}", function=name,
                           err_rel=float(F.history.err_rel))))
    return rows


def bench_qtt_fft(scale, repeats):
    """The superfast Fourier transform, against the DFT it is supposed to be.

    ``tt.vector.qtt_fft1`` implements Dolgov, Khoromskij & Savostyanov (*Superfast
    Fourier transform using QTT approximation*, J. Fourier Anal. Appl. 18(5),
    2012).  Two references, one small and one large:

    * ``d <= 14``: ``numpy.fft.fft`` on the fully expanded vector -- an external
      implementation of the same transform, which also pins the normalization
      and the index order;
    * any ``d``: the analytic DFT of ``sin(2 pi m j / N)``, which is two spikes
      at ``k = m`` and ``k = N - m`` of height ``sqrt(N)/2`` in the unitary
      scaling.  Built as ``tt.delta``, so the error is measured in the TT norm
      without ever expanding ``2^d`` numbers.
    """
    sizes = {"small": [10, 14], "default": [10, 14, 20, 26],
             "large": [10, 14, 20, 26, 32]}[scale]
    m = 5
    rows = []
    for d in sizes:
        N = 2 ** d
        x = tt.sin(d, alpha=2.0 * np.pi * m / N)
        y, med, best, warm = run(lambda: x.qtt_fft1(1e-12), repeats)
        # analytic: unitary DFT of sin(2 pi m j / N) = (sqrt(N)/(2i)) (d_m - d_{N-m})
        want = (tt.delta(2, d, center=m) - tt.delta(2, d, center=N - m)).round(1e-14)
        want = want * (np.sqrt(N) / 2.0)
        err_an = float((y.imag() + want).norm() / want.norm())
        err_re = float(y.real().norm() / want.norm())
        row = dict(
            problem="qtt_fft", case=f"d={d} (N=2^{d})", solver="qtt_fft1",
            reference="analytic DFT of a discrete sine: two spikes of height sqrt(N)/2",
            ref_value=float(np.sqrt(N) / 2.0), measured=None,
            rel_err=float(max(err_an, err_re)), rank=int(max(y.r)),
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, points=N, err_vs_analytic=err_an,
                       spurious_real_part=err_re))
        if d <= 14:
            xf = np.asarray(x.full()).reshape(-1, order="F")
            yf = np.asarray(y.full()).reshape(-1, order="F")
            ref = np.fft.fft(xf) / np.sqrt(N)
            row["extra"]["err_vs_numpy_fft"] = float(
                np.linalg.norm(yf - ref) / np.linalg.norm(ref))
        rows.append(row)
    return rows


def bench_qtt_poisson(scale, repeats):
    """A Poisson problem on ``2^d`` nodes, solved exactly, with a nodal oracle.

    ``-u'' = 1`` on ``(0,1)`` with ``u(0) = 0``, ``u'(1) = 0``: the finite
    difference solution is *nodally exact*, equal to ``u(x) = x - x^2/2`` at
    every grid point.  In QTT the inverse of the difference operator is the
    triangular all-ones matrix, so the solve is two matvecs -- no iteration, no
    preconditioner, no tolerance.  At ``d = 50`` that is a linear system with
    ``2^50`` unknowns whose answer is checked against a formula.
    """
    sizes = {"small": [20, 30], "default": [20, 30, 40, 50],
             "large": [20, 30, 40, 50, 60]}[scale]
    rows = []
    for d in sizes:
        n = 2 ** d
        h = 1.0 / n
        rhs = (tt.ones(2, d) - 0.5 * tt.unit(2, d, j=n - 1)) * (h * h)
        u, med, best, warm = run(lambda: solve_direct_1d(rhs, d), repeats)
        xg = (tt.xfun(2, d) + tt.ones(2, d)) * h
        exact = (xg - 0.5 * (xg * xg)).round(1e-14)
        err = float((u - exact).norm() / exact.norm())
        rows.append(dict(
            problem="qtt_poisson_direct", case=f"d={d} (N=2^{d})",
            solver="qtt_ell.solve_direct_1d",
            reference="analytic nodal solution u(x) = x - x^2/2 (nodally exact FD)",
            ref_value=0.5, measured=None, rel_err=err, rank=int(max(u.r)),
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, unknowns=f"2^{d}",
                       cond_estimate=f"~4^{d}")))
    return rows


def bench_anderson(scale, repeats):
    """The documented refusal: an i.i.d. potential has no low-rank structure.

    Anderson localization asks for ``-Delta_h + diag(V)`` with ``V`` i.i.d.
    uniform.  The reference is a counting argument, not a measurement: a generic
    vector of length ``2^d`` has TT ranks exactly ``min(2^k, 2^{d-k})``, the
    largest the format allows, so the operator is incompressible and every
    tensor method on it degenerates to dense.  The row exists so that the claim
    "tensor methods do not apply here" is a measured fact in the suite rather
    than folklore, and so that a future rank heuristic that quietly truncates
    this potential is caught.
    """
    sizes = {"small": [10, 14], "default": [10, 14, 18], "large": [10, 14, 18, 22]}[scale]
    rows = []
    for d in sizes:
        rng = np.random.default_rng(SEED)
        v = rng.uniform(-0.5, 0.5, 2 ** d)

        def build():
            return tt.vector(v.reshape([2] * d, order="F"), eps=1e-8)

        V, med, best, warm = run(build, repeats)
        maxpos = [min(2 ** k, 2 ** (d - k)) for k in range(d + 1)]
        got = [int(r) for r in V.r]
        rows.append(dict(
            problem="anderson_refusal", case=f"d={d} (N=2^{d})", solver="tt_svd",
            reference="counting argument: a generic vector has ranks min(2^k, 2^(d-k))",
            ref_value=int(max(maxpos)), measured=int(max(got)),
            rel_err=float(max(got) / max(maxpos) - 1.0), rank=int(max(got)),
            median_s=med, best_s=best, warmup=warm, repeats=repeats,
            extra=dict(d=d, ranks=got, ranks_max_possible=maxpos,
                       saturated=bool(got == maxpos))))
    return rows


# --- 5. chemical master equation --------------------------------------------

def _cme_chain_operator(d, n, lam, c):
    """Generator of the monomolecular chain, as a TT-matrix of rank 3.

    Reactions: ``0 -> S_1`` at rate ``lam``, ``S_i -> S_{i+1}`` at rate
    ``c x_i``, ``S_d -> 0`` at rate ``c x_d``.  Written with the shift matrices
    ``J`` (``J v`` = shift up) and ``J^T``, and ``D = diag(0..n-1)``:

    * birth of ``S_1``:      ``lam (J_1 - I_1)``
    * ``S_i -> S_{i+1}``:    ``c (J^T_i D_i) (x) J_{i+1} - c D_i (x) I``
    * ``S_d -> 0``:          ``c (J^T_d D_d - D_d)``
    """
    n = int(n)
    J = np.eye(n, k=-1)
    D = np.diag(np.arange(n, dtype=float))
    I = np.eye(n)
    Ws = []
    for k in range(d):
        W = np.zeros((3, 3, n, n))
        W[2, 2] = I
        W[0, 0] = I
        diag_loss = -c * D if k > 0 else -c * D + lam * (J - I)
        if k == d - 1:
            diag_loss = diag_loss + c * (J.T @ D)
        W[2, 0] = diag_loss
        if k < d - 1:
            W[2, 1] = c * (J.T @ D)     # gain: one molecule leaves mode k ...
            W[1, 0] = J                 # ... and arrives in mode k+1
        Ws.append(W)
    return ham._mpo_sites(Ws)


def bench_cme(scale, repeats):
    """Chemical master equation for a monomolecular chain, against Jahnke-Huisinga.

    Reaction network ``0 -> S_1 -> S_2 -> ... -> S_d -> 0`` with production rate
    ``lam`` and identical first-order rates ``c``.  Jahnke & Huisinga (*Solving
    the chemical master equation for monomolecular reaction systems
    analytically*, J. Math. Biol. 54:1-26, 2007) prove that such a system,
    started from the empty state, has the exact solution ``prod_i
    Poisson(m_i(t))`` with ``m'(t) = A m + b`` -- a closed form for a
    ``n^d``-state problem.

    The CME is integrated here by implicit Euler with ``amen_solve``, i.e.
    ``(I - tau A) p^{k+1} = p^k``.  Two errors are then in the number reported:
    the ``O(tau)`` time discretization, which is the method's and is bounded by
    halving ``tau``, and the tensor solver's.  Both are in the row.
    """
    from scipy.linalg import expm

    d = {"small": 6, "default": 10, "large": 16}[scale]
    n, lam, c, T, nsteps = 32, 3.0, 1.0, 2.0, 40
    A = _cme_chain_operator(d, n, lam, c)
    tau = T / nsteps

    # exact means: m' = M m + b with M the first-order rate matrix
    M = -c * np.eye(d) + c * np.eye(d, k=-1)
    b = np.zeros(d)
    b[0] = lam
    aug = np.zeros((d + 1, d + 1))
    aug[:d, :d], aug[:d, d] = M, b
    m = (expm(aug * T) @ np.concatenate([np.zeros(d), [1.0]]))[:d]

    def marginal_poisson(mu):
        k = np.arange(n)
        p = np.exp(-mu + k * np.log(np.maximum(mu, 1e-300))
                   - np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, n))])))
        return p

    ref = tt.vector.from_list([marginal_poisson(mi).reshape(1, n, 1) for mi in m])

    def integrate():
        p = tt.vector.from_list([np.eye(1, n).reshape(1, n, 1)] * d)
        B = (tt.eye(n, d) + (-tau) * A).round(1e-14)
        for _ in range(nsteps):
            p = amen_solve(B, p, p, 1e-10, nswp=20, verb=0)
        return p

    p, med, best, warm = run(integrate, repeats)
    err = float((p - ref).norm() / ref.norm())
    mass = float(tt.dot(p, tt.ones(n, d)))
    rows = [dict(
        problem="cme_chain", case=f"d={d} n={n} T={T} tau={tau}",
        solver=f"implicit Euler x{nsteps}, amen_solve",
        reference=("Jahnke & Huisinga 2007: the exact solution is a product of "
                   "Poissons with means from m' = A m + b"),
        ref_value=None, measured=None, rel_err=err, rank=int(max(p.r)),
        median_s=med, best_s=best, warmup=warm, repeats=repeats,
        extra=dict(d=d, n=n, states=f"{n}^{d}", total_mass=mass,
                   exact_means=[float(v) for v in m], steps=nsteps, tau=tau))]
    return rows


PROBLEMS = {
    "tfim": bench_tfim,
    "heisenberg": bench_heisenberg,
    "coupled_oscillator": bench_coupled_oscillator,
    "henon_heiles": bench_henon_heiles,
    "integral_sine": bench_integral_sine,
    "integral_cornerpeak": bench_integral_cornerpeak,
    "qtt_ranks": bench_qtt_ranks,
    "qtt_fft": bench_qtt_fft,
    "qtt_poisson": bench_qtt_poisson,
    "anderson": bench_anderson,
    "cme": bench_cme,
}


def fmt(row):
    def num(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "     -   "
    return (f"{row['problem']:22s} {row['case']:28s} "
            f"ref {num(row.get('ref_value'), '>16.9g')}  "
            f"got {num(row.get('measured'), '>16.9g')}  "
            f"err {num(row.get('rel_err'), '>9.2e')}  "
            f"r {num(row.get('rank'), '>4d')}  "
            f"f_ev {num(row.get('fun_eval'), '>8d')}  "
            f"{num(row.get('median_s'), '>8.2f')} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS),
                    choices=list(PROBLEMS))
    ap.add_argument("--scale", default="default",
                    choices=["small", "default", "large"])
    ap.add_argument("--repeats", type=int, default=1,
                    help="1 (default) means a single timed run, no warm-up; "
                         ">1 warms up once and reports the median")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet-warnings", action="store_true",
                    help="silence the solvers' non-convergence RuntimeWarnings; "
                         "the same facts stay in the JSON")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.simplefilter("ignore")

    env = {
        "host": platform.node(),
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ttpy": tt.__version__,
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cpu_count": os.cpu_count(),
        "backend": "numpy", "device": "cpu", "dtype": "float64",
        "scale": args.scale, "repeats": args.repeats,
    }

    rows, t_all = [], time.perf_counter()
    for name in args.problems:
        t0 = time.perf_counter()
        print(f"\n--- {name} " + "-" * (60 - len(name)), flush=True)
        try:
            got = PROBLEMS[name](args.scale, args.repeats)
        except Exception as exc:                       # a failing problem is data
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(problem=name, case="-", reference="-",
                             failed=f"{type(exc).__name__}: {exc}"))
            continue
        for r in got:
            r.setdefault("backend", "numpy")
            r.setdefault("device", "cpu")
            r.setdefault("dtype", "float64")
            print("  " + fmt(r), flush=True)
        rows.extend(got)
        print(f"  [{name}: {time.perf_counter() - t0:.1f} s]", flush=True)

    print(f"\ntotal {time.perf_counter() - t_all:.1f} s")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"env": env, "rows": rows}, fh, indent=1, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
