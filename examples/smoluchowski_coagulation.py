#!/usr/bin/env python
"""Multicomponent Smoluchowski coagulation in TT, at the scale of the paper.

    python examples/smoluchowski_coagulation.py                      # Table 1, N = 1000
    python examples/smoluchowski_coagulation.py --N 2000 --tau 0.05  # Table 1, N = 2000
    python examples/smoluchowski_coagulation.py --kernel additive --T 0.2 --tau 0.005
    python examples/smoluchowski_coagulation.py --gif docs/media/smoluchowski_run.gif

The two-component coagulation equation

    dn(v)/dt = (1/2) int_0^{v_1} int_0^{v_2} K(v-u; u) n(v-u) n(u) du
               - n(v) int_0^inf int_0^inf K(u; v) n(u) du

on a uniform ``N x N`` grid over ``[0, V_max]^2``, integrated by the explicit
midpoint scheme of Matveev-Zheltkov-Tyrtyshnikov-Smirnov (JCP 316:164-179,
2016) with the gain term evaluated by their Algorithm 1: the low-rank
FFT convolution of :mod:`tt.algs.smoluchowski`.

Why the grid is not small
-------------------------
This is the whole point of the method, and it is a *timing* argument, so the
default run is the paper's own reference point rather than a toy.  Table 1 of
the paper, for exactly this problem (``K = 1``, ``V_max = 100``, ``T = 10``):

    N       tau     TT (paper)      direct (paper)      error (paper)
    500     0.1        --            12 225 s              --
    1000    0.1      1 024 s        215 580 s            2.2e-3
    2000    0.05     2 492 s           --                5.0e-4

215 580 s is two and a half days for the *same scheme* evaluated directly:
the double sum of the gain term costs ``O(N^4)`` per step, and ``N^4`` is
10^12 at ``N = 1000``.  The TT convolution costs ``O(R^4 N log N)``.  The
numbers this script prints at the end are the same run on the machine you
are sitting at; the reproduction table in the module page records what it
reached here.

The oracle is the analytic solution of the paper's eq. (18) for ``K = 1``
and ``n_0 = exp(-v_1 - v_2)``,

    n(v, t) = exp(-v_1 - v_2) / (1 + t/2)^2 * I_0(2 sqrt(v_1 v_2 t/(t+2))),

whose total density is exactly ``1/(1 + t/2)``; both are checked every step.

Kernels
-------
``--kernel constant``  ``K = 1``            -- rank 1, and the one with an
                                              analytic solution.
``--kernel additive``  ``K = sum u_i + sum v_i`` -- rank 2, no analytic
                                              solution here, but two exact
                                              identities: the mass
                                              ``int (v_1+v_2) n`` is conserved
                                              and ``N(t) = N_0 exp(-M_0 t)``.

The ballistic kernel of the paper's eq. (17),
``K = ((sum u_i)^{1/3} + (sum v_i)^{1/3})^2 sqrt(1/sum u_i + 1/sum v_i)``,
is **not implemented**: it is not separable, and using it here would require
first building a separable approximation of the ``2d``-dimensional ``K`` by
cross approximation and then splitting it into ``(kv, ku)`` pairs.  That
approximation, and the error it introduces, is a piece of work this example
does not do, so it is recorded as missing rather than faked.

The ``--gif`` output is the run itself: the left panel is the log lines below
as they are printed, the right panel is the mass concentration
``(v_1 + v_2) n(v_1, v_2, t)`` -- Fig. 1 of the paper -- at the same instant.
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import numpy as np

import tt
from tt.algs.smoluchowski import (additive_kernel, coagulation_rhs,
                                  component_sum, constant_kernel,
                                  trapezoidal_weights)
from tt.core.vector import vector

LOG_HEADER = (
    "  step        t   R_pred  R_corr    density     exact     "
    "err(N)    err(n)   ms/step",
    "  ---------------------------------------------------------"
    "---------------------------",
)


def exponential_ic(N, h):
    """``n_0 = exp(-v_1 - v_2)``: rank 1, and the initial datum of eq. (18)."""
    x = h * np.arange(N)
    e = np.exp(-x).reshape(1, N, 1)
    return vector.from_list([e, e.copy()]), x


class Analytic:
    """Eq. (18) of the paper, with the ``v_1 v_2`` product precomputed."""

    def __init__(self, x):
        from scipy.special import i0
        self._i0 = i0
        v1, v2 = np.meshgrid(x, x, indexing="ij")
        self.prod = v1 * v2
        self.decay = np.exp(-v1 - v2)

    def __call__(self, t):
        arg = 2.0 * np.sqrt(self.prod * t / (t + 2.0))
        return self.decay / (1.0 + t / 2.0) ** 2 * self._i0(arg)


def midpoint_step(n, kernel, tau, h, eps, rmax):
    """One step of eq. (5), with the intermediate stage kept visible.

    Identical to :func:`tt.algs.smoluchowski.predictor_corrector_step`; spelled
    out here only so the log can show what the ranks do *inside* the step --
    the predictor inflates them (every Hadamard product multiplies ranks) and
    the rounding of the corrector pulls them back.
    """
    half = (n + (tau / 2.0) * coagulation_rhs(n, kernel, h, eps, rmax)
            ).round(eps, rmax)
    new = (n + tau * coagulation_rhs(half, kernel, h, eps, rmax)
           ).round(eps, rmax)
    return new, int(max(half.r)), int(max(new.r))


def run(d=2, N=1000, vmax=100.0, T=10.0, tau=0.1, eps=1e-6, rmax=None,
        kernel_name="constant", gif=None, view=None, nframes=20):
    if d != 2:
        raise SystemExit("this example is written for d = 2 (the analytic "
                         "solution and the Fig. 1 panel are 2-D); the solver "
                         "in tt.algs.smoluchowski is dimension-agnostic")
    h = vmax / (N - 1)
    n0, x = exponential_ic(N, h)
    modes = [N] * d
    if kernel_name == "constant":
        kernel = constant_kernel(modes)
    elif kernel_name == "additive":
        kernel = additive_kernel(modes, h)
    else:
        raise SystemExit(f"unknown kernel {kernel_name!r}: "
                         "'constant' or 'additive' (see the module docstring "
                         "for why 'ballistic' is not offered)")

    w = trapezoidal_weights(modes, h)
    s = component_sum(modes, h)
    exact = Analytic(x) if kernel_name == "constant" else None
    dens0 = float(tt.dot(w, n0))
    mass0 = float(tt.dot(w, s * n0))

    nsteps = int(round(T / tau))
    if nsteps < 1:
        raise SystemExit(f"T = {T} is shorter than one step tau = {tau}")
    frame_every = max(1, nsteps // max(1, nframes))
    lines = []
    frames = []

    print(f"Smoluchowski, d={d}, kernel={kernel_name}, N={N}, "
          f"V_max={vmax:g}, tau={tau:g}, T={T:g}, eps={eps:g}"
          + (f", rmax={rmax}" if rmax else ""))
    print(f"grid step h = {h:.6g};  n_0 = exp(-v1-v2):  "
          f"N(0) = {dens0:.6f}, M(0) = {mass0:.6f}")
    for line in LOG_HEADER:
        print(line, flush=True)

    def snapshot(t, n):
        """Mass concentration (v1+v2) n, downsampled for the panel."""
        full = np.asarray((s * n).full())
        stop = full.shape[0] if view is None else \
            min(full.shape[0], int(np.ceil(view / h)) + 1)
        step = max(1, stop // 300)
        return t, full[:stop:step, :stop:step], x[:stop:step][-1]

    n = n0
    solver_seconds = 0.0
    if gif:
        frames.append(snapshot(0.0, n) + (list(lines),))
    for k in range(1, nsteps + 1):
        t0 = time.perf_counter()
        n, r_half, r_new = midpoint_step(n, kernel, tau, h, eps, rmax)
        dt = time.perf_counter() - t0
        solver_seconds += dt

        t = k * tau
        dens = float(tt.dot(w, n))
        if exact is not None:
            dens_ref = 1.0 / (1.0 + t / 2.0)
            ref = exact(t)
            err_n = float(np.linalg.norm(np.asarray(n.full()) - ref)
                          / np.linalg.norm(ref))
        else:
            dens_ref = dens0 * np.exp(-mass0 * t)
            err_n = float(abs(float(tt.dot(w, s * n)) - mass0) / mass0)
        err_dens = abs(dens - dens_ref) / dens_ref

        line = (f"{k:5d}/{nsteps:<5d} {t:6.3f}   {r_half:4d}    {r_new:4d}  "
                f"{dens:10.6f} {dens_ref:10.6f}  {err_dens:8.2e}  "
                f"{err_n:8.2e}  {dt * 1e3:8.1f}")
        lines.append(line)
        print(line, flush=True)

        if gif and (k % frame_every == 0 or k == nsteps):
            frames.append(snapshot(t, n) + (list(lines),))

    tail = ("err(n) is the relative Frobenius error against eq. (18)"
            if exact is not None else
            "err(n) is the drift of the conserved mass int (v1+v2) n")
    print(f"\n{tail}; ms/step excludes the diagnostics.")
    print(f"solver time {solver_seconds:.1f} s for {nsteps} steps "
          f"({solver_seconds / nsteps * 1e3:.1f} ms/step), "
          f"final TT rank {int(max(n.r))}")
    if kernel_name == "constant":
        print("paper Table 1 for this problem (V_max=100, T=10): "
              "N=1000, tau=0.1 -> TT 1024 s, direct 215580 s, error 2.2e-3; "
              "N=2000, tau=0.05 -> TT 2492 s, error 5.0e-4")

    if gif:
        render_gif(frames, gif, d, N, vmax, tau, eps, kernel_name)
    return n, {"error": err_n, "density_error": err_dens, "density": dens,
               "rank": int(max(n.r)), "seconds": solver_seconds,
               "steps": nsteps, "density_0": dens0, "mass_0": mass0}


# --- the animation -----------------------------------------------------------

def render_gif(frames, path, d, N, vmax, tau, eps, kernel_name, nlines=28):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    vmaxc = max(float(f[1].max()) for f in frames)
    pngs = []
    for t, conc, extent, log in frames:
        fig = plt.figure(figsize=(11.6, 4.9), dpi=92)
        fig.patch.set_facecolor("#f7f7f7")

        # left: the run's own log, as it was printed
        axl = fig.add_axes([0.006, 0.02, 0.545, 0.90])
        axl.set_facecolor("#101418")
        axl.set_xticks([])
        axl.set_yticks([])
        shown = list(LOG_HEADER) + log[-nlines:]
        axl.text(0.012, 0.985, "\n".join(shown), transform=axl.transAxes,
                 va="top", ha="left", family="monospace", fontsize=7.4,
                 color="#c8d4dc", linespacing=1.3)
        axl.set_title("examples/smoluchowski_coagulation.py", fontsize=8,
                      family="monospace", color="#333333")

        # right: Fig. 1 of the paper, the mass concentration
        axr = fig.add_axes([0.635, 0.115, 0.30, 0.775])
        # log scale, and the *same* scale in every frame: the peak drops by
        # orders of magnitude while the distribution spreads, and a linear
        # scale shared across frames would show the late ones as black
        im = axr.imshow(np.maximum(conc.T, vmaxc * 1e-6), origin="lower",
                        cmap="magma",
                        norm=matplotlib.colors.LogNorm(vmin=vmaxc * 1e-5,
                                                       vmax=vmaxc),
                        extent=[0, extent, 0, extent], aspect="equal")
        axr.set_xlabel("$v_1$", fontsize=9)
        axr.set_ylabel("$v_2$", fontsize=9)
        axr.set_title(f"$(v_1+v_2)\\,n(v_1,v_2,t)$   $t = {t:.2f}$",
                      fontsize=9)
        axr.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=axr, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("mass concentration (log)", fontsize=7)

        fig.suptitle(f"Smoluchowski coagulation in TT -- {kernel_name} kernel, "
                     f"$d = {d}$, ${N}^{{{d}}}$ grid, "
                     f"$V_{{max}} = {vmax:g}$, $\\tau = {tau:g}$, "
                     f"$\\varepsilon = {eps:g}$",
                     fontsize=10, y=0.985)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        pngs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE,
                                            colors=128))
    pngs[0].save(path, save_all=True, append_images=pngs[1:],
                 duration=[220] * (len(pngs) - 1) + [3000], loop=0,
                 optimize=True)
    print(f"saved {path} ({len(pngs)} frames)")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--d", type=int, default=2, help="number of components")
    p.add_argument("--N", type=int, default=1000, help="nodes per component")
    p.add_argument("--vmax", type=float, default=100.0, help="grid upper bound")
    p.add_argument("--T", type=float, default=10.0, help="final time")
    p.add_argument("--tau", type=float, default=0.1, help="time step")
    p.add_argument("--eps", type=float, default=1e-6, help="TT rounding accuracy")
    p.add_argument("--rmax", type=int, default=None, help="hard TT rank cap")
    p.add_argument("--kernel", default="constant",
                   choices=("constant", "additive"))
    p.add_argument("--gif", default=None, help="write the two-panel animation here")
    p.add_argument("--view", type=float, default=30.0,
                   help="upper bound of the animated window in v")
    p.add_argument("--frames", type=int, default=20, help="animation frames")
    a = p.parse_args(argv)
    run(a.d, a.N, a.vmax, a.T, a.tau, a.eps, a.rmax, a.kernel, a.gif,
        a.view, a.frames)


if __name__ == "__main__":
    sys.exit(main())
