#!/usr/bin/env python
"""AMEn on the QTT Laplacian, and the wall it runs into.

    python examples/amen_laplace.py            # d = 8, 12, 16, 20
    python examples/amen_laplace.py 8 14 20 26

Solves ``-u'' = 1`` with homogeneous Dirichlet data on ``2^d`` interior points,
discretized as ``tridiag(-1, 2, -1) u = h^2 f``.  The oracle is the analytic
solution of that *discrete* system, ``u_i = i (N + 1 - i) / 2``, so the number
in the last column is the error of the solver, not of the discretization.

What to watch: the sweep count and the error as ``d`` grows.  The condition
number of the operator is ``O(4^d)``, so an unpreconditioned iteration has to
work harder and harder for the same accuracy and eventually cannot reach it at
all in float64.  ``examples/bpx_elliptic.py`` is what fixes that.
"""

import sys
import time
import warnings

import numpy as np

import tt
from tt.algs.amen import amen_solve

EPS = 1e-10


def run(d):
    n = 2 ** d
    a = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d) * (1.0 / (n + 1) ** 2)

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x, info = amen_solve(a, rhs, None, EPS, verb=0, seed=0, return_info=True)
    dt = time.time() - t0

    i = np.arange(1, n + 1, dtype=float)
    analytic = i * (n + 1 - i) / 2.0 / (n + 1) ** 2
    got = np.asarray(x.full(asvector=True))
    err = np.linalg.norm(got - analytic) / np.linalg.norm(analytic)

    note = "" if info.converged else "  <- did not converge"
    if caught:
        note = "  <- warned: " + str(caught[0].message).split(".")[0][:60]
    print(f"{d:3d} | {n:11,d} | {info.nswp_done:6d} | {max(info.ranks):4d} | "
          f"{dt:7.2f}s | {info.max_res:9.2e} | {err:9.2e}{note}", flush=True)


def main(argv):
    ds = [int(v) for v in argv[1:]] or [8, 12, 16, 20]
    print(__doc__.split("What to watch")[0].strip())
    print()
    print(f"requested accuracy eps = {EPS:g}")
    print("  d |    unknowns | sweeps | rank |    time |  max_res |     error")
    print("----+-------------+--------+------+---------+----------+----------")
    for d in ds:
        run(d)
    print()
    print("The error grows with d and the last rows warn: that is correct, not a")
    print("bug.  kappa(A) is O(4^d), so a backward-stable solve leaves a relative")
    print("residual of order eps_machine * kappa, and eps = 1e-10 is already below")
    print("that floor at d = 12.  The solver reports the failure instead of")
    print("returning a plausible number.  examples/bpx_elliptic.py removes the")
    print("floor by preconditioning, and reaches 1e-13 at d = 30.")
    print()
    print("max_res is the local residual before each block solve -- the default")
    print("stopping criterion.  The exact global residual is available with")
    print("check_true_res=True; it is off by default because forming A x")
    print("multiplies the ranks (see the docstring of amen_solve).")


if __name__ == "__main__":
    main(sys.argv)
