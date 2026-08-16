#!/usr/bin/env python
"""BPX multilevel preconditioning in QTT -- the most advanced example here.

    python examples/bpx_elliptic.py            # all three parts, ~4 min
    python examples/bpx_elliptic.py 1d         # ~40 s, up to 2^30 unknowns
    python examples/bpx_elliptic.py cond       # ~15 s, dense eigenvalues
    python examples/bpx_elliptic.py 2d 10      # ~2.5 min
    python examples/bpx_elliptic.py 2d 12      # ~5 min, 16.8M unknowns

[BK20] M. Bachmayr, V. Kazeev, *Stability of Low-Rank Tensor Representations and
Structured Multilevel Preconditioning for Elliptic PDEs*, Found. Comput. Math.
20 (2020) 1175-1236.

Three parts:

  1d     one dimension, ``-u'' = 1`` with ``u(0) = 0, u'(1) = 0``, solved with
         and without the preconditioner up to 2^30 unknowns;
  cond   the conditioning claim itself, in 1D and 2D, against dense eigenvalues;
  2d     a genuinely two-dimensional problem with three sharp Gaussian peaks,
         built by cross approximation, on a fine grid.

The 2D part is the slow one, and honestly so: ``amen_solve`` needs a matrix, so
``B`` has to be *assembled* at TT rank 161 instead of applying the rank-24
factors of ``bpx_theta`` one at a time.  The sweep algebra is linear in that
rank and it dominates -- the local solves themselves take 3.6 GMRES iterations
per block, exactly what ``kappa(B) ~ 7`` predicts.  Teaching the solver to take
a factored operator is the open item.

The one idea worth taking away is in part 1d: the preconditioned operator must
never be *assembled* as ``C A C``.  Its entries cancel over ``4^d``, so rounding
that triple product loses accuracy like ``4^d * eps`` and its rank grows with
``d``.  ``bpx_theta`` gives the fused factors of [BK20] Lemma 5 instead, with
``B = sum_k Theta_k^T Theta_k`` the same matrix at TT rank 17, flat in ``d``.
"""

import sys
import time
import warnings

import numpy as np

import tt
from tt.algs.amen import amen_solve
from tt.algs.multifuncrs import multifuncrs
from tt.algs.qtt_ell import bpx, bpx_theta, merge_levels

warnings.simplefilter("ignore")


# --- part 1: one dimension, with and without ---------------------------------

def part_1d(ds=(10, 14, 18, 22, 26, 30)):
    print("=== 1D:  -u'' = 1,  u(0) = 0,  u'(1) = 0,  AMEn at eps = 1e-10 ===")
    print("the exact solution is u(x) = x - x^2/2, so the error is measurable")
    print()
    print("  d |      unknowns | unpreconditioned          | with BPX")
    print("    |               | sweeps    time   rel.err  | sweeps    time   rel.err  rank(B)")
    print("----+---------------+---------------------------+--------------------------------")
    for d in ds:
        n = 2 ** d
        h = 1.0 / n
        a = tt.qlaplace_dn(d, "DN")
        rhs = (tt.ones(2, d) - 0.5 * tt.unit(2, d, j=n - 1)) * (h * h)
        x = (tt.xfun(2, d) + tt.ones(2, d)) * h
        exact = (x - 0.5 * (x * x)).round(1e-14)

        def err(u):
            return float((u - exact).norm() / exact.norm())

        t0 = time.time()
        plain, i1 = amen_solve(a, rhs, tt.ones(2, d), 1e-10, nswp=30, verb=0,
                               return_info=True)
        t1 = time.time() - t0

        t0 = time.time()
        th = bpx_theta(d, 1)[0]
        c = bpx(d, 1, weight=1, scaled=True)
        b = (th.T @ th).round(1e-14)
        w, i2 = amen_solve(b, tt.matvec(c, rhs).round(1e-12), tt.ones(2, d),
                           1e-10, nswp=30, verb=0, return_info=True)
        u = tt.matvec(c, w).round(1e-12)
        t2 = time.time() - t0

        print(f"{d:3d} | {n:13,d} | {i1.nswp_done:6d} {t1:7.2f}s {err(plain):9.2e} |"
              f" {i2.nswp_done:6d} {t2:7.2f}s {err(u):9.2e}  {max(b.r):4d}", flush=True)
    print()
    print("At d = 30 the unpreconditioned answer is not merely inaccurate, it is")
    print("wrong (relative error ~1) -- and amen_solve says so rather than hiding it.")


# --- part 2: the conditioning claim ------------------------------------------

def part_cond():
    print("\n=== conditioning: kappa(A) against kappa(C A C), dense eigenvalues ===")
    for ndim in (1, 2):
        print(f"\n  D = {ndim}:  preconditioner rank {2 * 4 ** ndim}, "
              f"independent of d")
        print("   d |   kappa(A) | lam_min(B) | lam_max(B) | kappa(B)")
        print("  ---+------------+------------+------------+---------")
        for d in ((4, 6, 8, 10) if ndim == 1 else (3, 4, 5, 6)):
            order = "level" if ndim > 1 else "dim"
            ad = np.asarray(tt.qlaplace_dn([d] * ndim, "DN", order=order).full())
            cd = np.asarray(bpx(d, ndim, weight=1, scaled=True).full())
            wa = np.linalg.eigvalsh(ad)
            wb = np.linalg.eigvalsh(cd @ ad @ cd)
            print(f"  {d:2d} | {wa[-1] / wa[0]:10.3e} | {wb[0]:10.4f} | "
                  f"{wb[-1]:10.4f} | {wb[-1] / wb[0]:8.4f}", flush=True)
    print("\n  kappa(A) grows like 4^d; kappa(B) creeps.  That is the whole claim.")


# --- part 3: a real 2D problem ------------------------------------------------

def part_2d(d=12):
    ndim = 2
    n, h = 2 ** d, 1.0 / 2 ** d
    print(f"\n=== 2D: three sharp Gaussian peaks on {n} x {n} = {n * n:,} unknowns ===")
    print("the solution is built by cross approximation, the right-hand side is")
    print("f = A u, so the error against u is exact\n")

    xg, yg = tt.zmeshgrid(d)          # coordinates in level-major (z) order
    peaks = [(0.5, 0.5, 2.0 ** -9, 1.0),
             (0.25, 0.7, 2.0 ** -6, 0.6),
             (0.8, 0.3, 2.0 ** -4, 0.3)]

    def fun(v):
        px, py = (v[:, 0] + 1) * h, (v[:, 1] + 1) * h
        out = np.zeros(len(v))
        for cx, cy, s, wgt in peaks:
            out += wgt * np.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * s * s))
        return out

    t0 = time.time()
    u_exact = multifuncrs([xg, yg], fun, eps=1e-11, nswp=20, verb=0)
    print(f"  u by cross:      rank {max(u_exact.r):4d}   {time.time() - t0:6.1f}s")

    t0 = time.time()
    order = tt.level_major_order([d] * ndim)
    a = merge_levels(tt.permute(tt.qlaplace_dn([d] * ndim, "DN", order="dim"),
                                order, 1e-14), ndim)
    f = tt.matvec(a, u_exact).round(1e-12)
    print(f"  operator, rhs:   rank {max(a.r):4d} / {max(f.r):4d}   "
          f"{time.time() - t0:6.1f}s")

    t0 = time.time()
    c = bpx(d, ndim, weight=1, scaled=True)
    b = None
    for th in bpx_theta(d, ndim):
        term = (th.T @ th).round(1e-14)
        b = term if b is None else (b + term).round(1e-14)
    print(f"  C / B:           rank {max(c.r):4d} / {max(b.r):4d}   "
          f"{time.time() - t0:6.1f}s")

    rhs = tt.matvec(c, f).round(1e-12)
    t0 = time.time()
    w, info = amen_solve(b, rhs, rhs, 1e-8, kickrank=24, nswp=25, verb=0,
                         return_info=True)
    u = tt.matvec(c, w).round(1e-12)
    dt = time.time() - t0
    err = float((u - u_exact).norm() / u_exact.norm())
    res = float((tt.matvec(a, u).round(1e-12) - f).norm() / f.norm())
    print(f"  solve:           {info.nswp_done} sweeps, {dt:.1f}s, rank(w)={max(w.r)}")
    print(f"  relative error   {err:.3e}   (nodal residual {res:.3e})")
    print()
    print("  Note the two norms: amen_solve stops on the *preconditioned* system,")
    print("  so its own residual is smaller than the nodal one printed here.")
    print("  Starting from the right-hand side and a larger kickrank matters a lot:")
    print("  from a rank-1 guess the enrichment adds 4 per sweep and never gets there.")


# --- figure ------------------------------------------------------------------

def make_figure(path, dfield=9):
    """Gallery figure: the conditioning claim, plus the 2D three-peak field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # left panel: kappa(A) vs kappa(B) as d grows, dense eigenvalues, D = 1
    ds = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    kA, kB = [], []
    for d in ds:
        ad = np.asarray(tt.qlaplace_dn([d], "DN", order="dim").full())
        cd = np.asarray(bpx(d, 1, weight=1, scaled=True).full())
        wa = np.linalg.eigvalsh(ad)
        wb = np.linalg.eigvalsh(cd @ ad @ cd)
        kA.append(wa[-1] / wa[0])
        kB.append(wb[-1] / wb[0])

    # right panel: the three sharp Gaussian peaks.  The 2D part builds this
    # field by cross approximation (multifuncrs) to TT rank ~78 at eps 1e-11;
    # since the field is exactly the sum of Gaussians below, we sample it on
    # the grid directly -- an identical picture at a fraction of the cost.
    n = 2 ** dfield
    gx = (np.arange(n) + 1) / n
    XX, YY = np.meshgrid(gx, gx, indexing="ij")
    peaks = [(0.5, 0.5, 2.0 ** -9, 1.0),
             (0.25, 0.7, 2.0 ** -6, 0.6),
             (0.8, 0.3, 2.0 ** -4, 0.3)]
    field = np.zeros((n, n))
    for cx, cy, s, wgt in peaks:
        field += wgt * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * s * s))

    fig, (axk, axf) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)

    axk.semilogy(ds, kA, "o-", color="#c1272d", lw=1.8, ms=5,
                 label=r"unpreconditioned  $\kappa(A)\sim 4^{d}$")
    axk.semilogy(ds, kB, "s-", color="#0b6e4f", lw=1.8, ms=5,
                 label=r"BPX  $\kappa(BA)$, bounded in $d$")
    guide = [kA[0] * 4.0 ** (d - ds[0]) for d in ds]
    axk.semilogy(ds, guide, "--", color="#999999", lw=1.0, zorder=0)
    axk.set_xlabel(r"levels $d$   (unknowns $2^{d}$)", fontsize=10)
    axk.set_ylabel(r"condition number", fontsize=10)
    axk.set_title(r"$-u''=1$ in QTT, $D=1$: conditioning vs depth",
                  fontsize=10)
    axk.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    axk.legend(fontsize=8.5, loc="center right")

    imf = axf.imshow(field.T, origin="lower", cmap="magma",
                     extent=[0, 1, 0, 1])
    axf.set_title(r"$2^{%d}\times 2^{%d}$ solution: three sharp peaks"
                  % (dfield, dfield), fontsize=10)
    axf.set_xlabel(r"$x$", fontsize=10)
    axf.set_ylabel(r"$y$", fontsize=10)
    fig.colorbar(imf, ax=axf, fraction=0.046)

    fig.suptitle(r"BPX multilevel preconditioning in QTT  [BK20]:  "
                 r"$B=\sum_k\Theta_k^{\top}\Theta_k$, never $CAC$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"saved {path}   kappa(A): {kA[0]:.1f}..{kA[-1]:.2e}, "
          f"kappa(BA): {kB[0]:.2f}..{kB[-1]:.2f}")


def main(argv):
    if "--png" in argv:
        i = argv.index("--png")
        make_figure(argv[i + 1])
        return
    what = argv[1] if len(argv) > 1 else "all"
    if what in ("all", "1d"):
        part_1d(tuple(int(v) for v in argv[2:]) or (10, 14, 18, 22, 26, 30))
    if what in ("all", "cond"):
        part_cond()
    if what in ("all", "2d"):
        part_2d(int(argv[2]) if len(argv) > 2 and what == "2d" else 10)


if __name__ == "__main__":
    main(sys.argv)
