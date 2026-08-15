#!/usr/bin/env python
"""3D Allen-Cahn by the interpolatory KSL (``tt.ksl_deim``), Dektor sec. 7.2.

    python examples/allen_cahn_ksl_deim.py                 # n=32, rank 16, T=10
    python examples/allen_cahn_ksl_deim.py 32 16 0.005 10  # n, rank, tau, T
    python examples/allen_cahn_ksl_deim.py --gif docs/media/allen_cahn_deim.gif

The showcase problem of Dektor's collocation paper (sec. 7.2): the Allen-Cahn
equation

    du/dt = alpha Laplace(u) + u - u^3,   alpha = 0.1,

on the torus ``[0, 2 pi]^3``, whose cubic nonlinearity is exactly what the
interpolatory projector-splitting integrator exists for.  The classical KSL
needs the full right-hand side ``A u + Nf(u)`` projected orthogonally onto the
tangent space -- but ``u - u^3`` has no cheap TT representation, so the
orthogonal projector is unaffordable.  ``tt.ksl_deim`` (ported from Alec
Dektor's ttpy PR #102) replaces it by an *interpolatory* (oblique) projector:
the right-hand side is only ever **evaluated entrywise on QDEIM-selected cross
fibers**, which a pointwise nonlinearity supports at the cost of a numpy call
on the sampled entries -- here literally ``lambda v: v - v ** 3``.

Discretization, exactly as in the paper: Fourier pseudospectral collocation on
the uniform periodic grid ``x_i = 2 pi i / n``.  The one-axis second-derivative
operator is the real symmetric circulant

    D2 = F^{-1} diag(-k^2) F,   k = -n/2, ..., n/2 - 1  (integer wavenumbers),

materialized as a dense ``n x n`` matrix (``ksl_deim`` samples ``A y`` on
fibers, so ``A`` must be a TT-matrix; a dense core is fine), and

    A = alpha (D2 x I x I  +  I x D2 x I  +  I x I x D2)

is the Kronecker sum -- a TT-matrix of rank 2.  The assembly self-checks that
``D2 sin(x) = -sin(x)`` on the grid to machine precision.  The paper uses
``n = 64``; the default here is ``n = 32`` so that the dense diagnostics stay
a 32768-vector (the integrator itself never forms it).

The initial condition (eq. 7.2 of the paper) is

    u0 = g(x1,x2,x3) - g(2 x1,x2,x3) + g(x1,2 x2,x3) - g(x1,x2,2 x3),

    g = [exp(-tan^2 x1) + exp(-tan^2 x2) + exp(-tan^2 x3)] sin(x1+x2+x3)
        / [1 + exp|csc(-x1/2)| + exp|csc(-x2/2)| + exp|csc(-x3/2)|],

sampled into TT by :func:`tt.multifuncrs` (TT-cross on pointwise calls) from
the three rank-1 coordinate tensors.  At grid points where ``csc`` blows up
(``x = 0``: ``csc(0) = inf``) the paper's denominator suppresses ``g`` to
zero; numerically the exponent is capped at 30 (``exp(min(|csc|, 30))``, about
1e13), which leaves ``|g| < 1e-12`` there -- indistinguishable from the exact
zero -- instead of an ``inf/inf`` NaN.

Integration: fixed TT rank ``r`` (the manifold of the method).  ``u0`` is
padded with 1e-8 random noise and cut to rank exactly ``r`` (``ksl_deim``
keeps ranks and does not support interior ranks of 1), then stepped by
``tt.ksl_deim(A, Nf, u, tau)``.  The scheme is built from explicit-Euler
substeps: first order in ``tau``, and subject to the Euler stability bound
``tau < 2 / (3 alpha (n/2)^2)`` (about 0.026 at ``n = 32``) -- the default
``tau = 0.005`` sits well inside it and was verified stable over the full
``T = 10`` run.

Diagnostics on checkpoints (dense, ``n^3`` entries -- honest and cheap at
this size): ``max|u|`` and the Ginzburg-Landau free energy

    E[u] = sum_i [ -alpha/2 u (Laplace u) + (1 - u^2)^2 / 4 ] h^3,

written with the quadratic form of the *same* spectral Laplacian (summation
by parts replaces ``|grad u|^2``), so the semidiscrete ODE is exactly the
gradient flow of this discrete ``E`` and **E must decrease monotonically** --
a live correctness invariant of everything at once (operator, nonlinearity,
integrator stability).  The run prints it at every checkpoint; the phase
separation is visible as ``max|u|`` climbing to the double-well plateau 1.

Oracle (printed when ``n <= 16``): ``scipy.integrate.solve_ivp`` (RK45,
rtol 1e-8) on the full ``n^3`` ODE built from the same ``D2`` by
``scipy.sparse`` Kronecker sums, started from the same rank-``r`` initial
condition -- so the reported error is the integrator + fixed-rank manifold
error, not the truncation of ``u0``.  The acceptance version in
``tests/test_examples.py`` runs ``n = 8`` on the *full* manifold (rank 8),
where the whole gap to the dense solve is the first-order time discretization,
and asserts the energy monotonicity on top.

References
----------
1. A. Dektor, "Collocation methods for nonlinear differential equations on
   low-rank manifolds", Linear Algebra and its Applications 705 (2025)
   143-184, arXiv:2402.18721 -- sec. 7.2 is this exact problem (Fourier
   pseudospectral, n = 64) and the source of the initial condition.
2. oseledets/ttpy PR #102 (A. Dektor) -- the original ``tt.ksl_deim``.
"""

import sys
import time
import warnings

import numpy as np

import tt
from tt.algs.ksl_deim import ksl_deim
from tt.algs.multifuncrs import multifuncrs

ALPHA = 0.1
CSC_CAP = 30.0          # exp cap for the csc singularities of u0 (docstring)


def grid(n):
    """The paper's periodic grid: ``x_i = 2 pi i / n``, spacing ``h``."""
    h = 2.0 * np.pi / n
    return h * np.arange(n), h


def spectral_d2(n):
    """Dense 1D pseudospectral second derivative ``F^{-1} diag(-k^2) F``.

    Real symmetric circulant (integer wavenumbers ``fftfreq(n) * n``); the
    self-check pins it on the grid eigenfunction ``sin``.
    """
    k = np.fft.fftfreq(n) * n
    D2 = np.real(np.fft.ifft(np.fft.fft(np.eye(n), axis=0)
                             * (-(k ** 2))[:, None], axis=0))
    x, _ = grid(n)
    assert np.abs(D2 @ np.sin(x) + np.sin(x)).max() < 1e-11 * n, \
        "spectral D2 fails on sin(x)"
    return D2


def _mat3(ms):
    """Kronecker product of three 1D matrices as a TT-matrix (mode 1 = x1)."""
    return tt.matrix.from_list(
        [np.ascontiguousarray(m[None, :, :, None]) for m in ms])


def laplacian3d(n, alpha=ALPHA):
    """``A = alpha Laplace`` as a rank-2 TT-matrix from the Kronecker sum."""
    D2, I = spectral_d2(n), np.eye(n)
    A = (_mat3([alpha * D2, I, I]) + _mat3([I, alpha * D2, I])
         + _mat3([I, I, alpha * D2]))
    return A.round(1e-13)


def dense_operator(n, alpha=ALPHA):
    """The same Kronecker sum in scipy.sparse (mode 1 fastest, as tt.full)."""
    import scipy.sparse as sp
    D2 = sp.csr_matrix(spectral_d2(n))
    I = sp.identity(n, format="csr")

    def k3(a, b, c):
        return sp.kron(sp.kron(c, b, "csr"), a, "csr")

    return alpha * (k3(D2, I, I) + k3(I, D2, I) + k3(I, I, D2))


def _g(a, b, c):
    """The paper's ``g`` with the csc exponents capped at ``CSC_CAP``."""
    num = (np.exp(-np.tan(a) ** 2) + np.exp(-np.tan(b) ** 2)
           + np.exp(-np.tan(c) ** 2)) * np.sin(a + b + c)
    with np.errstate(divide="ignore"):
        den = 1.0 + sum(
            np.exp(np.minimum(np.abs(1.0 / np.sin(-t / 2.0)), CSC_CAP))
            for t in (a, b, c))
    return num / den


def u0_values(v):
    """Pointwise ``u0`` of eq. 7.2; ``v`` is ``(batch, 3)`` of coordinates."""
    x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2]
    return (_g(x1, x2, x3) - _g(2 * x1, x2, x3)
            + _g(x1, 2 * x2, x3) - _g(x1, x2, 2 * x3))


def initial_condition(n, eps=1e-8):
    """``u0`` in TT by cross approximation from rank-1 coordinate tensors."""
    x, _ = grid(n)
    c, o = x.reshape(1, n, 1), np.ones((1, n, 1))
    X = [tt.vector.from_list(cores) for cores in
         ([c, o, o], [o, c, o], [o, o, c])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return multifuncrs(X, u0_values, eps, verb=0)


def pad_to_rank(u0, r, seed=0):
    """Rank exactly ``r``: 1e-8 noise padding, then cut (fixed-rank start).

    ``ksl_deim`` keeps the ranks of its input and rejects interior ranks of 1,
    so a low-rank ``u0`` must be padded up (the noise pattern of
    ``henon_heiles_spectrum.py``) and a rich one rounded down.
    """
    rng = np.random.default_rng(seed)
    noise = tt.rand([int(m) for m in u0.n], r=r,
                    samplefunc=rng.standard_normal)
    noise = noise * (float(u0.norm()) * 1e-8 / float(noise.norm()))
    return (u0 + noise).round(0.0, rmax=r)


def nonlinearity(v):
    """``Nf(u) = u - u^3`` on sampled entries -- the ``ksl_deim`` contract."""
    return v - v ** 3


def energy(U, h, alpha=ALPHA):
    """Discrete Ginzburg-Landau free energy of a dense ``(n,n,n)`` state.

    The Dirichlet term is the quadratic form of the same spectral Laplacian
    (``-alpha/2 <u, Lap u>``), so the semidiscrete flow decreases this ``E``
    exactly -- the monotonicity check is against machine noise, not against a
    quadrature mismatch.
    """
    n = U.shape[0]
    k2 = (np.fft.fftfreq(n) * n) ** 2
    sym = -(k2[:, None, None] + k2[None, :, None] + k2[None, None, :])
    lap = np.real(np.fft.ifftn(np.fft.fftn(U) * sym))
    return float(np.sum(-0.5 * alpha * U * lap
                        + 0.25 * (1.0 - U ** 2) ** 2) * h ** 3)


def dense_oracle(u0_flat, n, T, rtol=1e-8):
    """Full ``n^3`` ODE by RK45 from the same ``D2`` -- no TT anywhere."""
    from scipy.integrate import solve_ivp
    Ad = dense_operator(n)
    sol = solve_ivp(lambda t, u: Ad @ u + u - u ** 3, [0.0, T], u0_flat,
                    method="RK45", rtol=rtol, atol=1e-10, t_eval=[T])
    return sol.y[:, -1]


def run(n=32, r=16, tau=0.005, T=10.0, gif=None, nout=40, seed=0):
    A = laplacian3d(n)
    _, h = grid(n)
    u0 = initial_condition(n)
    y = pad_to_rank(u0, r, seed)
    u0_flat = np.asarray(y.full()).flatten("F")   # the oracle's start
    nsteps = int(round(T / tau))
    every = max(1, nsteps // nout)

    print(f"Allen-Cahn on [0,2pi]^3, alpha={ALPHA}: n={n}, TT rank {r} "
          f"(u0 cross rank {max(u0.r)}), {nsteps} steps of tau={tau} "
          f"(T={T:g})")
    U = np.asarray(y.full())
    E = energy(U, h)
    frames = [(0.0, E, float(np.abs(U).max()), U[:, :, n // 2].copy())]
    print(f"{'t':>8} {'E':>12} {'max|u|':>10}")
    print(f"{0.0:8.3f} {E:12.6f} {frames[0][2]:10.6f}")

    t0 = time.perf_counter()
    for k in range(nsteps):
        y = ksl_deim(A, nonlinearity, y, tau)
        if (k + 1) % every == 0 or k + 1 == nsteps:
            t = (k + 1) * tau
            U = np.asarray(y.full())
            E_new = energy(U, h)
            mono = "" if E_new <= E + 1e-10 else "   <-- E increased!"
            E = E_new
            print(f"{t:8.3f} {E:12.6f} {np.abs(U).max():10.6f}{mono}")
            frames.append((t, E, float(np.abs(U).max()),
                           U[:, :, n // 2].copy()))
    elapsed = time.perf_counter() - t0
    print(f"\n{elapsed:.1f} s ({elapsed / nsteps * 1e3:.1f} ms/step), "
          f"final TT ranks {[int(v) for v in y.r]} (fixed by the method)")

    if n <= 16:
        ref = dense_oracle(u0_flat, n, T)
        got = np.asarray(y.full()).flatten("F")
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"vs dense solve_ivp oracle (same D2, same start): "
              f"relative error {err:.2e}")
    if gif:
        render_gif(frames, gif, n, r)
    return y, frames


def render_gif(frames, path, n, r):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pngs = []
    for t, E, umax, S in frames:
        fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=100)
        im = ax.imshow(S.T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[0, 2 * np.pi, 0, 2 * np.pi])
        ax.set_title(f"t = {t:.2f}    E = {E:.4f}    TT rank {r}",
                     fontsize=10)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(r"Allen-Cahn, ksl_deim: slice $u(x_1,x_2,\pi,t)$, "
                     f"${n}^3$ grid", fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pngs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    pngs[0].save(path, save_all=True, append_images=pngs[1:],
                 duration=[160] * (len(pngs) - 1) + [2500], loop=0)
    print(f"saved {path} ({len(pngs)} frames)")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    gif = None
    if "--gif" in args:
        i = args.index("--gif")
        gif = args[i + 1]
        del args[i:i + 2]
    n = int(args[0]) if len(args) > 0 else 32
    r = int(args[1]) if len(args) > 1 else 16
    tau = float(args[2]) if len(args) > 2 else 0.005
    T = float(args[3]) if len(args) > 3 else 10.0
    run(n, r, tau, T, gif)
