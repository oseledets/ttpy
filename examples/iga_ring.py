#!/usr/bin/env python
"""Isogeometric Poisson on a curved 3D domain, assembled entirely by cross.

    python examples/iga_ring.py                 # p=2, 32 elements per direction
    python examples/iga_ring.py 2 64            # degree, elements per direction
    python examples/iga_ring.py 2 16 32 64 128  # a convergence sweep

Steady conduction in an annular duct: ``-Lap u = 0`` on the sector
``r_in <= r <= r_out``, ``0 <= theta <= theta_max``, ``0 <= z <= H``, with
``u = u_in`` on the inner cylinder and ``u = u_out`` on the outer one.  The
other four faces are natural (zero normal derivative), which the exact solution
satisfies identically because it depends on ``r`` alone:

    u(r) = [u_in log(r_out/r) + u_out log(r/r_in)] / log(r_out/r_in)

-- Eq. (46) of Tran, Truong, Rasmussen & Alexandrov, *A tensor train-based
isogeometric solver for large-scale 3D Poisson problems*, Comput. Methods Appl.
Mech. Engrg. 453 (2026) 118802.  A closed form, so the error below is the
method's and not a comparison against another code.

Why this example exists: every other problem in ``bench/`` and ``examples/``
lives on a box or a spin chain.  Here the domain is **curved**, and the geometry
enters only through the six scalar fields ``R = J^-1 J^-T |J|`` which
``tt.cross`` compresses.  The stiffness matrix is then one contraction per core
-- there is no element loop anywhere in this file.
"""

import sys
import time
import warnings

import numpy as np

import tt
from tt.algs.amen import amen_solve
from tt.algs.iga import (embed_vector, geometry_field, gram_blocks, restrict,
                         restrict_vector, stiffness_tt)

warnings.simplefilter("ignore")

R_IN, R_OUT, THETA_MAX, HEIGHT = 0.5, 1.0, 0.5 * np.pi, 1.0
U_IN, U_OUT = 1.0, 2.0


def ring_map(xi):
    """The unit cube to the annular sector; smooth, and rank-1 in each factor."""
    r = R_IN + (R_OUT - R_IN) * xi[:, 0]
    th = THETA_MAX * xi[:, 1]
    return np.stack([r * np.cos(th), r * np.sin(th), HEIGHT * xi[:, 2]], axis=1)


def ring_jac(xi):
    """The analytic Jacobian.  Given, not differenced -- see geometry_field.

    Its columns are orthogonal (radial, tangential, axial), so three of the six
    components of R vanish identically.  With central differences they come back
    at 1e-10 instead of at 0 and tt.cross fits the noise at TT rank 46.
    """
    dr, th = R_OUT - R_IN, THETA_MAX * xi[:, 1]
    r = R_IN + dr * xi[:, 0]
    c, s = np.cos(th), np.sin(th)
    J = np.zeros((len(xi), 3, 3))
    J[:, 0, 0], J[:, 1, 0] = dr * c, dr * s
    J[:, 0, 1], J[:, 1, 1] = -r * THETA_MAX * s, r * THETA_MAX * c
    J[:, 2, 2] = HEIGHT
    return J


def exact(r):
    return (U_IN * np.log(R_OUT / r) + U_OUT * np.log(r / R_IN)) / np.log(R_OUT / R_IN)


def solve(p, n_el, eps_cross=1e-9, eps_solve=1e-9, verbose=True):
    t0 = time.time()
    blocks, grids, nb = [], [], []
    for _ in range(3):
        b, x, _w, n = gram_blocks(p, n_el)
        blocks.append(b)
        grids.append(x)
        nb.append(n)
    t_basis = time.time() - t0

    t0 = time.time()
    R, scale = geometry_field(ring_map, grids, jac=ring_jac, eps=eps_cross,
                              r=2, kickrank=2, nswp=12, seed=0)
    t_geom = time.time() - t0
    live = {k: (max(v.r) if v is not None else 0) for k, v in sorted(R.items())}

    t0 = time.time()
    K = stiffness_tt(R, blocks, eps=1e-12)
    t_asm = time.time() - t0

    # Dirichlet in direction 0 only: with an open knot vector the first and last
    # basis functions are the only ones alive on those faces, and they are
    # interpolatory, so the lift is a statement about two coefficients.
    keep = [np.ones(n, bool) for n in nb]
    keep[0][0] = keep[0][-1] = False
    ramp = U_IN + (U_OUT - U_IN) * np.linspace(0.0, 1.0, nb[0])
    lift = tt.vector.from_list([ramp.reshape(1, -1, 1)]
                               + [np.ones((1, n, 1)) for n in nb[1:]])

    rhs = restrict_vector((-1.0) * tt.matvec(K, lift).round(1e-14), keep)
    Kin = restrict(K, keep)

    t0 = time.time()
    corr, info = amen_solve(Kin, rhs, rhs, eps_solve, nswp=30, verb=0,
                            kickrank=6, return_info=True)
    t_solve = time.time() - t0
    u = (embed_vector(corr, keep) + lift).round(1e-12)

    # sample the spline solution on a physical line and compare to the closed form
    from tt.algs.iga import bspline_basis, open_knots
    ns = 41
    s = np.linspace(0.02, 0.98, ns)
    Ns = [bspline_basis(p, open_knots(p, n_el), s, deriv=0)[0] for _ in range(3)]
    cores = [np.asarray(c) for c in u.cores]
    # u(s, 0.5, 0.5) for every s: contract each core with its basis row
    mid = bspline_basis(p, open_knots(p, n_el), np.array([0.5]), deriv=0)[0][0]
    left = np.einsum("asb,ms->mab", cores[0], Ns[0], optimize=True)
    m1 = np.einsum("asb,s->ab", cores[1], mid, optimize=True)
    m2 = np.einsum("asb,s->ab", cores[2], mid, optimize=True)
    vals = np.einsum("mab,bc,cd->mad", left, m1, m2, optimize=True).reshape(ns)

    r = R_IN + (R_OUT - R_IN) * s
    err = np.abs(vals - exact(r)).max() / np.abs(exact(r)).max()

    if verbose:
        dofs = int(np.prod(nb))
        print(f"  p={p} n_el={n_el:4d}  dofs {dofs:>10,d}  "
              f"basis {t_basis:5.2f}s  geom {t_geom:5.2f}s  asm {t_asm:5.2f}s  "
              f"solve {t_solve:6.2f}s ({info.nswp_done} sw)", flush=True)
        print(f"      R ranks {live}  rank(K)={max(K.r)}  rank(u)={max(u.r)}  "
              f"max rel err vs Eq.(46): {err:.3e}", flush=True)
    return err, max(K.r), int(np.prod(nb))


def make_png(p, sizes, errs, path):
    """House-style two-panel figure: the curved solution field, and the rate.

    Left: the analytic temperature ``u(r)`` of Eq.(46) painted on the physical
    annular sector, so the curved geometry that ``tt.cross`` sees is visible.
    Right: relative error against Eq.(46) versus elements per direction on a
    log-log axis, with the ``O(n^-(p+1))`` IGA reference slope.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axf, axc) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)

    nr, nth = 64, 96
    rr = np.linspace(R_IN, R_OUT, nr)
    th = np.linspace(0.0, THETA_MAX, nth)
    Rg, Tg = np.meshgrid(rr, th)
    X, Y, U = Rg * np.cos(Tg), Rg * np.sin(Tg), exact(Rg)
    im = axf.pcolormesh(X, Y, U, cmap="magma", shading="gouraud")
    axf.set_aspect("equal")
    axf.set_xlabel(r"$x$")
    axf.set_ylabel(r"$y$")
    axf.set_title(r"analytic $u(r)$ on the annular sector", fontsize=10)
    cb = fig.colorbar(im, ax=axf, fraction=0.046, pad=0.04)
    cb.set_label(r"$u$")

    sizes = np.asarray(sizes, float)
    errs = np.asarray(errs, float)
    axc.loglog(sizes, errs, "o-", color="C3", label=r"cross-assembled IGA")
    ref = errs[0] * (sizes / sizes[0]) ** (-(p + 1))
    axc.loglog(sizes, ref, "k--", lw=1.0,
               label=r"$\mathcal{O}(n^{-(p+1)})$")
    axc.set_xlabel(r"elements per direction $n$")
    axc.set_ylabel(r"max rel. error vs Eq.(46)")
    axc.set_title(r"convergence, degree $p=%d$" % p, fontsize=10)
    from matplotlib.ticker import NullFormatter, NullLocator
    axc.xaxis.set_minor_locator(NullLocator())
    axc.xaxis.set_minor_formatter(NullFormatter())
    axc.set_xticks(sizes)
    axc.set_xticklabels([r"$%d$" % int(v) for v in sizes])
    axc.grid(True, which="both", ls=":", alpha=0.4)
    axc.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def main(argv):
    png = None
    if "--png" in argv:
        i = argv.index("--png")
        png = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    p = int(argv[1]) if len(argv) > 1 else 2
    sizes = [int(v) for v in argv[2:]] or [32]
    print(__doc__.split("Why this example")[0].strip())
    print()
    print(f"=== annular sector, r in [{R_IN}, {R_OUT}], theta in [0, pi/2], "
          f"h = {HEIGHT}; u_in = {U_IN}, u_out = {U_OUT} ===")
    prev, errs = None, []
    for n_el in sizes:
        err, rk, dofs = solve(p, n_el)
        errs.append(err)
        if prev is not None:
            rate = np.log2(prev[0] / err) / np.log2(n_el / prev[1])
            print(f"      observed convergence order: {rate:.2f}", flush=True)
        prev = (err, n_el)
    if png is not None:
        make_png(p, sizes, errs, png)
    print()
    print("The geometry never enters the assembly as a mesh: only the six fields")
    print("R = J^-1 J^-T |J|, compressed by tt.cross, and one contraction per")
    print("core. Three of the six vanish identically on this map and are screened")
    print("out -- see SMALL_FIELD_RATIO for what happens when they are not.")


if __name__ == "__main__":
    main(sys.argv)
