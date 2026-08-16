#!/usr/bin/env python
"""Quantum-inspired (QTT) incompressible Navier-Stokes in 2D.

    python examples/quantum_inspired_cfd/run.py tgv        # Taylor-Green, analytic oracle
    python examples/quantum_inspired_cfd/run.py shear      # Kelvin-Helmholtz roll-up
    python examples/quantum_inspired_cfd/run.py shear --gif docs/media/qi_cfd_shear.gif

The solver is `qtt_ns.py`: each velocity component is an interleaved-bit
(z-order, quaternary) QTT vector on a 2^d x 2^d periodic grid, advanced by a
projection method whose pressure Poisson equation is solved by the toolbox's
own `amen_solve`.  The bond dimension chi (`--chi`) caps the interscale
correlation the method keeps -- that cap is the whole idea.

Reference: N. Gourianov et al., "A quantum-inspired approach to exploit
turbulence structures", Nature Comput. Sci. 2 (2022) 30-37.
"""

import sys
import time
import warnings

import numpy as np

import tt

try:
    from . import qtt_ns as q
except ImportError:
    import qtt_ns as q


# --- z-order <-> 2D grid, diagnostics ----------------------------------------

def _zorder_index(d):
    N = 2 ** d
    z = np.arange(N * N)
    ix = np.zeros(N * N, int)
    iy = np.zeros(N * N, int)
    for b in range(d):
        ix |= ((z >> (2 * b)) & 1) << b
        iy |= ((z >> (2 * b + 1)) & 1) << b
    return ix, iy


def to_grid(vec, d, idx=None):
    ix, iy = idx if idx is not None else _zorder_index(d)
    a = np.asarray(vec.full(asvector=True)).ravel()
    G = np.zeros((2 ** d, 2 ** d))
    G[ix, iy] = a
    return G


def energy(ops, u, v):
    return 0.5 * (tt.dot(u, u) + tt.dot(v, v)) * ops.h ** 2


def enstrophy(ops, u, v, eps=1e-9):
    w = (tt.matvec(ops.Dx, v) - tt.matvec(ops.Dy, u)).round(eps)
    return 0.5 * tt.dot(w, w) * ops.h ** 2


def vorticity(ops, u, v, eps=1e-9):
    return (tt.matvec(ops.Dx, v) - tt.matvec(ops.Dy, u)).round(eps)


# --- Taylor-Green vortex: the analytic oracle --------------------------------

def taylor_green(d=6, nu=0.05, T=0.5, chi=30, order=8, solver="amen",
                 cfl=0.1, dense_check=True):
    """Decaying 2D Taylor-Green vortex; energy must follow E0*exp(-4 nu t).

    u = cos x sin y * exp(-2 nu t),  v = -sin x cos y * exp(-2 nu t) is an
    exact solution of the incompressible Navier-Stokes equations on the torus.
    """
    ops = q.Operators(d, box=2 * np.pi, order=order)
    u = ops.field(lambda x, y: np.cos(x) * np.sin(y))
    v = ops.field(lambda x, y: -np.sin(x) * np.cos(y))
    project = q.make_projector(ops, solver=solver, eps=1e-10, rmax=chi, tol=1e-9)
    u, v = project(u, v)

    dt = cfl * ops.h
    nsteps = int(round(T / dt))
    E0 = energy(ops, u, v)
    idx = _zorder_index(d)

    # optional identical-scheme dense reference (small d only): isolates the
    # rank-truncation error from the discretization.
    dense = None
    if dense_check and d <= 6:
        Dx, Dy, Lap, Lr = (ops.Dx.full(), ops.Dy.full(),
                           ops.Lap.full(), ops.Lap_reg.full())
        U = np.asarray(u.full(asvector=True)).ravel()
        V = np.asarray(v.full(asvector=True)).ravel()

        def drhs(U, V):
            return (nu * (Lap @ U) - (U * (Dx @ U) + V * (Dy @ U)),
                    nu * (Lap @ V) - (U * (Dx @ V) + V * (Dy @ V)))

        def dproj(U, V):
            phi = np.linalg.solve(Lr, Dx @ U + Dy @ V)
            return U - Dx @ phi, V - Dy @ phi

    times, ener, ranks = [0.0], [float(E0)], [max(max(u.r), max(v.r))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(nsteps):
            u, v = q.step(ops, u, v, dt, nu, project, eps=1e-10, rmax=chi)
            if dense is not None or (dense_check and d <= 6):
                a, b = drhs(U, V)
                U1, V1 = dproj(U + dt * a, V + dt * b)
                c, e = drhs(U1, V1)
                U, V = dproj(U + 0.5 * dt * (a + c), V + 0.5 * dt * (b + e))
            times.append((s + 1) * dt)
            ener.append(float(energy(ops, u, v)))
            ranks.append(max(max(u.r), max(v.r)))

    t = np.array(times)
    out = dict(t=t, energy=np.array(ener) / float(E0),
               analytic=np.exp(-4 * nu * t), ranks=np.array(ranks),
               d=d, nu=nu, chi=chi)
    if dense_check and d <= 6:
        out["dense_err"] = (np.linalg.norm(
            np.asarray(u.full(asvector=True)).ravel() - U) /
            np.linalg.norm(U))
    return out


# --- Kelvin-Helmholtz shear layer: the structure-forming run -----------------

def shear_layer(d=7, nu=1e-4, T=1.2, chi=60, order=8, solver="amen",
                cfl=0.2, delta=1.0 / 15.0, pert=0.05, nframes=40):
    """A doubly-periodic shear layer rolling up into vortices.

    The same Kelvin-Helmholtz instability the paper's jet develops, in the
    periodic geometry the QTT torus solver wants.  Returns vorticity frames
    for a movie together with the bond-dimension, energy and enstrophy history.
    """
    ops = q.Operators(d, box=1.0, order=order)
    u = ops.field(lambda x, y: np.where(y <= 0.5, np.tanh((y - 0.25) / delta),
                                        np.tanh((0.75 - y) / delta)), eps=1e-8)
    v = ops.field(lambda x, y: pert * np.sin(2 * np.pi * x), eps=1e-8)
    project = q.make_projector(ops, solver=solver, eps=1e-9, rmax=chi, tol=1e-7)
    u, v = project(u, v)

    dt = cfl * ops.h
    nsteps = int(round(T / dt))
    idx = _zorder_index(d)
    every = max(1, nsteps // nframes)

    frames, times, ranks, ener, enst = [], [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(nsteps + 1):
            if s % every == 0:
                frames.append(to_grid(vorticity(ops, u, v), d, idx))
                times.append(s * dt)
                ranks.append(max(max(u.r), max(v.r)))
                ener.append(float(energy(ops, u, v)))
                enst.append(float(enstrophy(ops, u, v)))
            if s < nsteps:
                u, v = q.step(ops, u, v, dt, nu, project, eps=1e-9, rmax=chi)
    return dict(frames=frames, t=np.array(times), ranks=np.array(ranks),
                energy=np.array(ener), enstrophy=np.array(enst),
                d=d, nu=nu, chi=chi)


# --- figures -----------------------------------------------------------------

def render_tg_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axe, axr) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)
    axe.plot(res["t"], res["analytic"], "-", color="#b00020", lw=2,
             label=r"analytic $e^{-4\nu t}$")
    axe.plot(res["t"], res["energy"], "o", color="#1f4e79", ms=4,
             mfc="none", label="QTT solver")
    axe.set_xlabel(r"$t$")
    axe.set_ylabel(r"kinetic energy $E(t)/E_0$")
    axe.set_title(r"Taylor--Green decay ($\nu=%.3g$)" % res["nu"], fontsize=10)
    axe.legend(fontsize=9)
    axr.plot(res["t"], res["ranks"], "-", color="#1f7a1f", lw=1.6)
    axr.set_xlabel(r"$t$")
    axr.set_ylabel(r"bond dimension $\chi$")
    axr.set_title(r"the field stays low-rank", fontsize=10)
    axr.set_ylim(0, max(8, res["ranks"].max() + 2))
    fig.suptitle(r"Incompressible Navier--Stokes in QTT: $2^{%d}\times2^{%d}$ torus"
                 % (res["d"], res["d"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def render_shear_gif(res, path):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    frames, t, ranks = res["frames"], res["t"], res["ranks"]
    vmax = np.abs(frames[len(frames) // 2]).max()
    pngs = []
    for k, W in enumerate(frames):
        fig, (axw, axr) = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=100,
                                       gridspec_kw={"width_ratios": [1.15, 1]})
        axw.imshow(W.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0, 1, 0, 1])
        axw.set_title(r"vorticity  $\omega$,  $t=%.2f$" % t[k], fontsize=10)
        axw.set_xticks([]); axw.set_yticks([])
        axr.plot(t[:k + 1], ranks[:k + 1], "-o", color="#1f4e79", ms=3)
        axr.set_xlim(t[0], t[-1]); axr.set_ylim(0, ranks.max() + 4)
        axr.set_xlabel(r"$t$"); axr.set_ylabel(r"bond dimension $\chi$")
        axr.set_title(r"interscale correlation stays bounded", fontsize=10)
        fig.suptitle(r"Kelvin--Helmholtz roll-up on a $2^{%d}\times2^{%d}$ torus, "
                     r"entirely in QTT" % (res["d"], res["d"]), fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        buf.seek(0)
        pngs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    pngs[0].save(path, save_all=True, append_images=pngs[1:],
                 duration=[220] * (len(pngs) - 1) + [2500], loop=0)
    print(f"saved {path}")


# --- CLI ---------------------------------------------------------------------

def main(argv):
    kind = argv[1] if len(argv) > 1 else "tgv"
    gif = argv[argv.index("--gif") + 1] if "--gif" in argv else None
    png = argv[argv.index("--png") + 1] if "--png" in argv else None

    if kind == "tgv":
        t0 = time.time()
        res = taylor_green()
        dt = time.time() - t0
        print(f"Taylor-Green  d={res['d']}  chi<={res['chi']}  ({dt:.1f}s)")
        print(f"  final energy ratio {res['energy'][-1]:.6f}  "
              f"analytic {res['analytic'][-1]:.6f}  "
              f"|diff| {abs(res['energy'][-1] - res['analytic'][-1]):.1e}")
        if "dense_err" in res:
            print(f"  QTT vs identical dense scheme: {res['dense_err']:.2e}")
        print(f"  bond dimension stayed <= {res['ranks'].max()}")
        if png:
            render_tg_figure(res, png)

    elif kind == "shear":
        t0 = time.time()
        res = shear_layer()
        dt = time.time() - t0
        print(f"Shear layer  d={res['d']}  chi<={res['chi']}  ({dt:.1f}s)")
        print(f"  energy drift {abs(res['energy'][-1]/res['energy'][0]-1):.2e}, "
              f"enstrophy grew x{res['enstrophy'].max()/res['enstrophy'][0]:.1f}")
        print(f"  bond dimension peaked at {res['ranks'].max()} (cap {res['chi']})")
        if gif:
            render_shear_gif(res, gif)

    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
