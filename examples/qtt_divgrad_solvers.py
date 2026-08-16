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


def manufactured_run(bits=6, rank=8, tol=1e-8):
    """Fixed-rank LOBPCG on a system whose solution has the prescribed ranks.

    ``f = A x_exact`` with ``x_exact`` at the requested profile, so the
    projected gradient and the true residual both reach machine order.
    """
    depth = 2 * bits
    profile = feasible_profile(depth, rank)
    operator = tt.qlaplace_dd([bits, bits])

    def random_tt(seed):
        rng = np.random.default_rng(seed)
        cores = [rng.standard_normal((profile[k], 2, profile[k + 1]))
                 for k in range(depth)]
        value = tt.vector.from_list(cores)
        return value / value.norm()

    exact = random_tt(1)
    rhs = tt.matvec(operator, exact)
    result, info = tt.lobpcg_solve(
        operator, rhs, random_tt(2), tol,
        nswp=100, local_steps=12, local_prec="c", verb=0,
        check_true_res=True, return_info=True,
    )
    error = (result - exact).norm() / exact.norm()
    return info, error


def render_png(amen, fixed, path):
    """Two panels: fixed-vs-adaptive convergence, and the exact fixed-rank case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, green, red = "#1f4e79", "#2e7d32", "#b00020"
    tol = amen.tol

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)

    # -- left: the same div-grad system, adaptive vs fixed rank --------------
    a_swp = [e["sweep"] for e in amen.sweeps]
    a_res = [e["max_res"] for e in amen.sweeps]
    a_rnk = [e["max_rank"] for e in amen.sweeps]
    f_swp = [e["sweep"] for e in fixed.sweeps]
    f_pg = [e["projected_gradient"] for e in fixed.sweeps]
    f_rnk = [e["max_rank"] for e in fixed.sweeps]

    axl.semilogy(a_swp, a_res, "o-", color=blue, zorder=3,
                 label=r"AMEn adaptive: local residual")
    axl.semilogy(f_swp, f_pg, "s-", color=green, zorder=3,
                 label=r"LOBPCG fixed: projected gradient")
    axl.axhline(tol, ls="--", color=red, lw=1.2,
                label=r"requested tol $=10^{-8}$")
    axl.set_xlabel("sweep")
    axl.set_ylabel(r"stopping quantity  (relative)")
    axl.set_ylim(1e-10, 1e2)
    axl.set_title(r"$-\nabla\cdot(k\,\nabla u)=1$: adaptive vs fixed rank",
                  fontsize=10)
    axl.grid(True, which="both", ls=":", alpha=0.4)
    axl.legend(fontsize=8, loc="lower left")

    axrank = axl.twinx()
    axrank.plot(a_swp, a_rnk, ":", color=blue, lw=1.2, alpha=0.7,
                label="AMEn rank")
    axrank.plot(f_swp, f_rnk, ":", color=green, lw=1.2, alpha=0.7,
                label="LOBPCG rank")
    axrank.set_ylabel("max TT rank", fontsize=9)
    axrank.set_ylim(0, max(a_rnk) * 1.25)
    axrank.legend(fontsize=8, loc="upper right")

    # -- right: manufactured f = A x_exact at the prescribed rank ------------
    man, man_err = manufactured_run()
    m_swp = [e["sweep"] for e in man.sweeps]
    m_pg = [e["projected_gradient"] for e in man.sweeps]
    axr.semilogy(m_swp, m_pg, "s-", color=green, zorder=3,
                 label="projected gradient")
    axr.semilogy([m_swp[-1]], [man.true_res], "*", color=blue, ms=13,
                 zorder=4, label=r"final true residual $\|Ax-f\|/\|f\|$")
    axr.axhline(man.tol, ls="--", color=red, lw=1.2,
                label=r"requested tol $=10^{-8}$")
    axr.set_xlabel("sweep")
    axr.set_ylabel(r"relative measure")
    axr.set_ylim(1e-10, 1e1)
    axr.set_title("fixed rank is exact when the solution has that rank",
                  fontsize=10)
    axr.grid(True, which="both", ls=":", alpha=0.4)
    axr.legend(fontsize=8, loc="upper right")
    axr.text(0.04, 0.06,
             rf"true res $={man.true_res:.1e}$" "\n"
             rf"sol err $={man_err:.1e}$",
             transform=axr.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))

    fig.suptitle(r"Fixed-rank vs rank-adaptive TT linear solvers  "
                 r"($E(x)=\frac{1}{2}\langle x,Ax\rangle-\langle f,x\rangle$)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main(argv):
    png = None
    if "--png" in argv:
        index = argv.index("--png")
        png = argv[index + 1]
        argv = argv[:index] + argv[index + 2:]
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
    if png:
        render_png(amen, fixed, png)


if __name__ == "__main__":
    main(sys.argv)
