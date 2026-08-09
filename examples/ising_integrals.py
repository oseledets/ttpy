#!/usr/bin/env python
"""Ising susceptibility integrals by greedy cross -- ttcross's own benchmark.

    python examples/ising_integrals.py             # C_6, C_16, D_6, E_6
    python examples/ising_integrals.py c 32        # one integral: KIND INDEX
    python examples/ising_integrals.py e 6 129     # ... and quadrature size

The problem family of Bailey, Borwein & Crandall, *Integrals of the Ising
class*, J. Phys. A 39:12271 (2006): ``C_m``, ``D_m`` and ``E_m`` are
``(m-1)``-dimensional integrals from the magnetic susceptibility of the 2D
Ising model, with reference values known to hundreds of digits.  This is the
example the reference implementation of the greedy DMRG cross ships as its
test driver (``test_crs_ising.f90`` of ``github.com/savostyanov/ttcross``),
and the benchmark of Dolgov & Savostyanov, arXiv:1903.11554.

The discretization is theirs, exactly: ``n`` Gauss-Legendre nodes on
``[0, 1]`` per direction, the quadrature weights baked into the tensor entries
(scaled by ``n/2`` against underflow), the integral recovered by contracting
with the rank-1 tensor of ``(2/n)``'s.  What to watch is the **number of
function evaluations** against the digits obtained: the same integrals by
(quasi) Monte Carlo cost orders of magnitude more calls for fewer digits --
that trade, measured on this package against the Fortran original and against
its own MC/QMC drivers, is in ``docs/plans/cross-approximation.md`` 2.1a-b.
"""

import sys
import time

import numpy as np

import tt
from tt.algs.dmrg_cross import dmrg_cross

# Bailey's constants, quoted to the digits a float64 run can use
# (the full-precision values are embedded in ttcross's test driver).
TRUE = {
    ("c", 6): 0.648634209031007075263149843450351690889772509481627995615,
    ("c", 16): 0.630503946173237263505295657560687419484316217208103047750,
    ("c", 32): 0.630473504207339806379189843197962510193081141118420178126,
    ("d", 6): 0.000489141700188034775100662315350456033220552627530599883,
    ("e", 6): 0.000687832871826409437004784273690210703814803310322272717,
}


def integrand(kind, m, nodes, wscaled):
    """The discretized C/D/E integrand, weights included -- ttcross's tensor."""
    d = m - 1

    def fun(idx):
        idx = np.asarray(idx, dtype=np.int64)
        t = nodes[idx]                                        # (batch, d)
        if kind in ("c", "d"):
            v = 1.0 + np.cumprod(t[:, ::-1], axis=1).sum(axis=1)
            u = 1.0 + np.cumprod(t, axis=1).sum(axis=1)
            b = 1.0 / (v * u)
        if kind in ("d", "e"):
            a = np.ones(len(t))
            for i in range(0, d + 1):
                uij = np.ones(len(t))
                for j in range(i + 1, d + 1):
                    if j - 1 < d:
                        uij = uij * t[:, j - 1]
                    a = a * (((uij - 1.0) / (uij + 1.0)) ** 2)
        f = {"c": lambda: 2 * b, "d": lambda: 2 * a * b,
             "e": lambda: 2 * a}[kind]()
        return f * np.prod(wscaled[idx], axis=1)

    return fun


def run(kind, m, n=65, eps=1e-12):
    d = m - 1
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = (x + 1.0) / 2.0
    scale = float(n // 2)
    fun = integrand(kind, m, nodes, (w / 2.0) * scale)

    t0 = time.perf_counter()
    y = dmrg_cross(fun, [n] * d, eps=eps)
    dt = time.perf_counter() - t0
    val = float(tt.dot(y, tt.ones(n, d))) / scale ** d

    h = y.history
    line = (f"{kind.upper()}_{m:<3d} n={n:<4d} rank {max(h.ranks):3d}  "
            f"evals {h.fun_eval:9d}  {dt * 1e3:8.1f} ms")
    tru = TRUE.get((kind, m))
    if tru is not None:
        digits = -np.log10(abs(1.0 - val / tru))
        line += f"  digits {digits:6.2f}"
    line += f"  value {val:.15e}"
    print(line)
    return val


if __name__ == "__main__":
    if len(sys.argv) > 1:
        kind = sys.argv[1].lower()
        m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 65
        run(kind, m, n)
    else:
        for kind, m in [("c", 6), ("c", 16), ("d", 6), ("e", 6)]:
            run(kind, m)
