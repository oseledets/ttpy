#!/usr/bin/env python
"""Variable-coefficient central differences in QTT, solved two ways.

The PDE is

    -div(k grad u) = 1  in (0, 1)^2,    u = 0 on the boundary,
    k = 1 + 0.5 sin(2 pi x) sin(2 pi y).

There are ``2**bits`` interior nodes per direction.  The matrix is assembled
from face samples by the ordinary conservative central stencil; neither the
Kazeev--Bachmayr operator nor a dense grid is used.

Run from the repository root, optionally choosing bits, fixed rank and tol:

    python examples/qtt_divgrad_solvers.py
    python examples/qtt_divgrad_solvers.py 10 64 1e-8
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tt


def coefficient(points):
    """Positive smooth diffusion coefficient, vectorized over rows."""
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * points[:, 0]) \
        * np.sin(2.0 * np.pi * points[:, 1])


def feasible_profile(depth, rank):
    return [min(rank, 2 ** min(k, depth - k)) for k in range(depth + 1)]


def initial_guess(profile, seed=2):
    rng = np.random.default_rng(seed)
    value = tt.vector.from_list([
        rng.standard_normal((profile[k], 2, profile[k + 1]))
        for k in range(len(profile) - 1)
    ])
    return 0.05 * value / value.norm()


def main(argv):
    bits = int(argv[1]) if len(argv) > 1 else 8
    rank = int(argv[2]) if len(argv) > 2 else 40
    tol = float(argv[3]) if len(argv) > 3 else 1e-8
    depth = 2 * bits

    started = time.perf_counter()
    operator, assembly = tt.qtt_divgrad(
        [bits, bits],
        coefficient,
        coefficient_eps=1e-12,
        round_eps=1e-13,
        n_check=256,
        return_info=True,
    )
    assembly_time = time.perf_counter() - started
    rhs = tt.ones(2, depth)

    started = time.perf_counter()
    amen_result, amen = tt.amen_solve(
        operator,
        rhs,
        None,
        tol,
        kickrank=8,
        rmax=max(64, rank),
        nswp=100,
        local_iters=4,
        verb=0,
        seed=0,
        check_true_res=True,
        return_info=True,
    )
    amen_time = time.perf_counter() - started

    profile = feasible_profile(depth, rank)
    started = time.perf_counter()
    fixed_result, fixed = tt.lobpcg_solve(
        operator,
        rhs,
        initial_guess(profile),
        tol,
        nswp=150,
        local_steps=24,
        local_prec="c",
        verb=0,
        check_true_res=True,
        return_info=True,
    )
    fixed_time = time.perf_counter() - started

    coefficient_evaluations = sum(
        history.fun_eval + history.fun_eval_check
        for history in assembly.cross_histories
    )
    print(f"grid:             {2**bits} x {2**bits} ({2**(2*bits):,} unknowns)")
    print(f"QTT depth:        {depth}")
    print(f"operator ranks:   {assembly.operator_ranks}")
    print(f"coefficient eval: {coefficient_evaluations:,}")
    print(f"assembly:         {assembly_time:.3f} s")
    print()
    print("solver              time   sweeps  max rank  projected grad   true residual")
    print("----------------  -------  ------  --------  --------------   -------------")
    print(
        f"AMEN adaptive      {amen_time:7.3f}  {amen.nswp_done:6d}  "
        f"{max(amen_result.r):8d}  {'n/a':>14}   {amen.true_res:13.3e}"
    )
    print(
        f"LOBPCG fixed       {fixed_time:7.3f}  {fixed.nswp_done:6d}  "
        f"{max(fixed_result.r):8d}  {fixed.projected_gradient:14.3e}   "
        f"{fixed.true_res:13.3e}"
    )
    print()
    print(f"AMEN converged:   {amen.converged}")
    print(f"LOBPCG converged: {fixed.converged}")
    print(
        "The fixed-rank stopping quantity is the projected gradient; if its "
        "true residual is larger, increase the prescribed rank."
    )


if __name__ == "__main__":
    main(sys.argv)
