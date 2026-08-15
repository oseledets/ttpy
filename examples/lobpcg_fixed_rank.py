#!/usr/bin/env python
"""Fixed-rank TT linear solve with transported local LOBPCG directions.

The right-hand side is manufactured as ``f = A x_exact`` with ``x_exact`` at
the same rank profile as the iterate.  Consequently both the fixed-rank
stationarity measure and the true linear residual can converge to zero.

Run from the repository root:

    python examples/lobpcg_fixed_rank.py
    python examples/lobpcg_fixed_rank.py 8 8 1e-10

The first two optional arguments are QTT bits per spatial coordinate and the
maximal TT rank.  The third is the requested projected-gradient tolerance.
"""

import sys

import numpy as np

import tt


def feasible_profile(depth, rank):
    """A constant interior cap, tapered where a binary TT cut is smaller."""
    return [min(rank, 2 ** min(k, depth - k)) for k in range(depth + 1)]


def random_tt(profile, seed):
    rng = np.random.default_rng(seed)
    depth = len(profile) - 1
    cores = [
        rng.standard_normal((profile[k], 2, profile[k + 1]))
        for k in range(depth)
    ]
    value = tt.vector.from_list(cores)
    return value / value.norm()


def main(argv):
    bits = int(argv[1]) if len(argv) > 1 else 6
    rank = int(argv[2]) if len(argv) > 2 else 8
    tol = float(argv[3]) if len(argv) > 3 else 1e-8
    depth = 2 * bits
    profile = feasible_profile(depth, rank)

    operator = tt.qlaplace_dd([bits, bits])
    exact = random_tt(profile, seed=1)
    rhs = tt.matvec(operator, exact)
    initial = random_tt(profile, seed=2)

    result, info = tt.lobpcg_solve(
        operator,
        rhs,
        initial,
        tol,
        nswp=100,
        local_steps=12,
        local_prec="c",
        verb=1,
        check_true_res=True,
        return_info=True,
    )

    relative_error = (result - exact).norm() / exact.norm()
    print()
    print(info)
    print(f"rank profile:       {list(result.r)}")
    print(f"projected gradient: {info.projected_gradient:.3e}")
    print(f"true residual:      {info.true_res:.3e}")
    print(f"solution error:     {relative_error:.3e}")


if __name__ == "__main__":
    main(sys.argv)
