#!/usr/bin/env python
"""The package's two cross engines on one black box, side by side.

    python examples/cross_engines.py               # C_6, three accuracies
    python examples/cross_engines.py c 16          # a harder one: KIND INDEX
    python examples/cross_engines.py e 6 129 1e-9  # ... quadrature size, eps

ttpy2 ships two cross-approximation engines, and they are different
algorithms, not one algorithm with different constants:

* ``tt.dmrg_cross`` (alias ``tt.greedy_cross``) -- the greedy DMRG cross of
  Savostyanov: rank grows by at most one per bond per sweep, the pivot is the
  entry of largest *residual* found by a rook search over one column and one
  row of the two-site superblock.  Built for exactly this use case --
  expensive smooth black boxes, high-dimensional quadrature.
* ``tt.rect_cross`` -- alternating cross with rectangular-maxvol row
  selection: rank grows by ``kickrank`` per micro-step, pivots maximise the
  2-volume of an orthonormal basis.  Reaches a target rank in far fewer
  sweeps, which is what AMEn-style consumers want.

The black box here is the Ising susceptibility integrand (Bailey, Borwein &
Crandall, J. Phys. A 39:12271, 2006), discretized exactly as in
``examples/ising_integrals.py``, so every printed digit count is against a
constant known to hundreds of digits.  What to watch is **function
evaluations at equal digits**: on these tensors the greedy needs several
times fewer (the measured medians are in
``docs/plans/cross-approximation.md`` 2.1c).  Note that ``eps`` means
different things to the two engines -- a residual-pivot threshold for the
greedy, a sweep-change threshold for the rectangular one -- so the fair
comparison axis is digits versus evaluations, not eps versus eps.

References -- the original implementations and papers
-----------------------------------------------------
* ``github.com/savostyanov/ttcross`` (D. V. Savostyanov, GPL-2.0): the
  reference Fortran implementation of the greedy DMRG cross, whose test
  driver this problem family comes from.  ``tt.dmrg_cross`` is a from-scratch
  port of that algorithm (read, not copied); its measured parity with the
  Fortran binary is in ``docs/plans/cross-approximation.md`` 2.1b.
* D. V. Savostyanov, "Quasioptimality of maximum-volume cross interpolation
  of tensors", Linear Algebra Appl. 458:217-244, 2014, arXiv:1305.1818.
* S. Dolgov, D. V. Savostyanov, "Parallel cross interpolation for
  high-precision calculation of high-dimensional integrals", Comput. Phys.
  Commun. 246:106869, 2020, arXiv:1903.11554.
* I. V. Oseledets, E. E. Tyrtyshnikov, "TT-cross approximation for
  multidimensional arrays", Linear Algebra Appl. 432(1):70-88, 2010 -- the
  cross idea ``rect_cross`` descends from.
* A. Mikhalev, I. V. Oseledets, "Rectangular maximum-volume submatrices and
  their applications", Linear Algebra Appl. 538:187-211, 2018,
  arXiv:1502.07838 -- ``rect_cross``'s pivot rule.
"""

import sys
import time

import numpy as np

import tt
from tt.algs.cross import rect_cross
from tt.algs.dmrg_cross import dmrg_cross
from tt.core.vector import vector

from ising_integrals import TRUE, integrand


def seeded_start(n, d, seed):
    """A rank-2 random start for ``rect_cross``, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    cores = [rng.standard_normal((1 if k == 0 else 2, n,
                                  1 if k == d - 1 else 2)) for k in range(d)]
    return vector.from_list(cores)


def run(engine, fun, kind, m, n, eps, seed=0):
    d = m - 1
    t0 = time.perf_counter()
    if engine == "dmrg":
        y = dmrg_cross(fun, [n] * d, eps=eps, seed=seed)
    else:
        y = rect_cross(fun, seeded_start(n, d, seed), eps=eps)
    dt = (time.perf_counter() - t0) * 1e3
    val = float(tt.dot(y, tt.ones(n, d))) / float(n // 2) ** d
    h = y.history
    tru = TRUE.get((kind, m))
    digits = ("   n/a" if tru is None else
              f"{-np.log10(max(abs(1.0 - val / tru), 1e-17)):6.2f}")
    print(f"  {engine:5s} eps {eps:7.0e}: digits {digits}  "
          f"evals {h.fun_eval:9d}  rank {max(int(r) for r in y.r):3d}  "
          f"{dt:8.1f} ms")


if __name__ == "__main__":
    kind = sys.argv[1].lower() if len(sys.argv) > 1 else "c"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 65
    epss = ([float(sys.argv[4])] if len(sys.argv) > 4
            else [1e-5, 1e-8, 1e-11])

    d = m - 1
    x, w = np.polynomial.legendre.leggauss(n)
    fun = integrand(kind, m, (x + 1.0) / 2.0, (w / 2.0) * float(n // 2))
    # a throwaway run compiles the integrand and the greedy's bond kernel,
    # so the timings below are the algorithms and not the compiler
    dmrg_cross(fun, [2] * d, rmax=2, eps=None)

    print(f"{kind.upper()}_{m}, n={n} Gauss-Legendre nodes per direction, "
          f"{d} dimensions:")
    for eps in epss:
        run("dmrg", fun, kind, m, n, eps)
        run("rect", fun, kind, m, n, eps)
