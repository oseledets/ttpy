#!/usr/bin/env python
"""``-div(k grad u) = 1`` in QTT, with the coefficient built by TT-cross.

    python examples/qtt_divgrad_cross.py            # solve + dense oracle
    python examples/qtt_divgrad_cross.py 8          # bits per axis
    python examples/qtt_divgrad_cross.py 8 --gif docs/media/divgrad_amen.gif

The README opener, in full.  A conservative finite-difference discretization
of a variable-coefficient diffusion on ``[0,1]^2`` is assembled *entirely in
QTT* from the package's public pieces -- no dense matrix is ever formed:

* the coefficient ``k(x, y) = 10^(sin 3 pi x * sin 3 pi y)`` (smooth,
  contrast 100) is sampled at the flux faces by :func:`tt.multifuncrs` --
  the TT-cross of Oseledets-Tyrtyshnikov applied to pointwise calls;
* the difference operator is ``D = (I - S)/h`` with ``S = tt.qshift`` (the
  QTT shift matrix), and the operator is the sum of two Kronecker products
  ``Dx^T diag(kx) Dx + Dy^T diag(ky) Dy``;
* ``tt.amen_solve`` solves the system without any preconditioner; the
  README animation is nothing but its iterate after each sweep.

Boundary conditions, stated rather than hidden: with ``D`` alone the flux
through the right and top faces is dropped, which makes the problem
``u = 0`` on the left/bottom sides and ``k du/dn = 0`` (natural) on the
right/top.  The ``--dirichlet`` variant adds the rank-1 corner corrections
``(k_face/h^2) e_n e_n^T`` per axis and is pinned against a scipy.sparse
assembly of the same scheme in ``tests/test_examples.py``.

References: the QTT shift/Laplacian representations are Kazeev-Khoromskij
(SIAM J. Matrix Anal. Appl. 33(3), 2012); TT-cross is
Oseledets-Tyrtyshnikov (Linear Algebra Appl. 432, 2010); AMEn is
Dolgov-Savostyanov (SIAM J. Sci. Comput. 36(5), 2014).
"""

import sys
import warnings

import numpy as np

import tt
from tt.algs.amen import amen_solve
from tt.algs.multifuncrs import multifuncrs


def coefficient(v):
    """Smooth checkerboard lens, contrast 100, low TT rank."""
    return 10.0 ** (np.sin(3 * np.pi * v[:, 0]) * np.sin(3 * np.pi * v[:, 1]))


def assemble(bits, dirichlet=False, eps=1e-10):
    """The QTT operator; ``dirichlet=True`` adds the right/top face terms."""
    h = 1.0 / (2 ** bits + 1)
    one = tt.ones(2, bits)
    x = (tt.xfun(2, bits) + one) * h            # interior points, QTT rank 2
    I = tt.eye(2, bits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kx = multifuncrs([tt.kron(x - one * (h / 2), one), tt.kron(one, x)],
                         coefficient, eps, verb=0)
        ky = multifuncrs([tt.kron(x, one), tt.kron(one, x - one * (h / 2))],
                         coefficient, eps, verb=0)

    D = (I - tt.qshift(bits)) * (1.0 / h)       # d/dx with u(0) = 0
    Dx, Dy = tt.kron(D, I), tt.kron(I, D)
    A = Dx.T @ tt.diag(kx) @ Dx + Dy.T @ tt.diag(ky) @ Dy
    if dirichlet:
        # the flux through the right/top faces: (k_face/h^2) e_n e_n^T
        corner = tt.matrix.from_list(
            [np.array([[0., 0.], [0., 1.]])[None, :, :, None]] * bits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kxb = multifuncrs([tt.kron(one * (1 - h / 2), one),
                               tt.kron(one, x)], coefficient, eps, verb=0)
            kyb = multifuncrs([tt.kron(x, one),
                               tt.kron(one, one * (1 - h / 2))],
                              coefficient, eps, verb=0)
        Cx, Cy = tt.kron(corner, I), tt.kron(I, corner)
        A = (A + (Cx @ tt.diag(kxb) @ Cx) * (1 / h ** 2)
             + (Cy @ tt.diag(kyb) @ Cy) * (1 / h ** 2))
    return A.round(1e-12), h


def dense_oracle(bits, dirichlet=False):
    """The same scheme in scipy.sparse -- no tensor format anywhere."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = 2 ** bits
    h = 1.0 / (n + 1)
    xg = (np.arange(n) + 1) * h
    kf = lambda X, Y: 10.0 ** (np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y))
    Dd = (sp.identity(n) - sp.diags(np.ones(n - 1), -1)) / h
    k2 = lambda a, b: sp.kron(b, a, "csr")      # mode 1 (x) fastest
    KX = sp.diags(kf(*np.meshgrid(xg - h / 2, xg, indexing="ij")).flatten("F"))
    KY = sp.diags(kf(*np.meshgrid(xg, xg - h / 2, indexing="ij")).flatten("F"))
    DX, DY = k2(Dd, sp.identity(n)), k2(sp.identity(n), Dd)
    Ad = DX.T @ KX @ DX + DY.T @ KY @ DY
    if dirichlet:
        corn = sp.csr_matrix(([1.0], ([n - 1], [n - 1])), (n, n))
        CX, CY = k2(corn, sp.identity(n)), k2(sp.identity(n), corn)
        KXB = sp.diags(kf(*np.meshgrid(np.full(n, 1 - h / 2), xg,
                                       indexing="ij")).flatten("F"))
        KYB = sp.diags(kf(*np.meshgrid(xg, np.full(n, 1 - h / 2),
                                       indexing="ij")).flatten("F"))
        Ad = Ad + CX @ KXB @ CX / h ** 2 + CY @ KYB @ CY / h ** 2
    return spla.spsolve(Ad.tocsc(), np.ones(n * n))


def sweep_frames(bits, eps=1e-9, nswp=8):
    """AMEn iterates sweep by sweep -- the material of the README animation.

    The wall clock shown in the frames is cumulative over the sweeps of
    *this* solve; the one-time numba/import warm-up is spent on a throwaway
    small problem first, so the first frame does not carry it.
    """
    import time
    A, h = assemble(bits, dirichlet=True)   # the symmetric, fully-clamped one
    f = tt.ones(2, 2 * bits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Aw, _ = assemble(4)
        amen_solve(Aw, tt.ones(2, 8), tt.ones(2, 8), 1e-6, verb=0)  # warm-up
    x, out, elapsed = f, [], 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for swp in range(1, nswp + 1):
            t0 = time.perf_counter()
            x = amen_solve(A, f, x, eps, nswp=1, verb=0)
            elapsed += time.perf_counter() - t0
            res = float((tt.matvec(A, x) - f).norm() / f.norm())
            n = 2 ** bits
            U = np.asarray(x.full()).reshape(-1, order="F").reshape(
                n, n, order="F")
            out.append((swp, res, int(max(x.r)), elapsed, U))
    return out


def render_gif(frames, path):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from PIL import Image

    n = frames[0][4].shape[0]
    g = (np.arange(n) + 1) / (n + 1)
    K = 10.0 ** (np.sin(3 * np.pi * g)[:, None] * np.sin(3 * np.pi * g)[None, :])
    vmax = frames[-1][4].max()
    pngs = []
    for swp, res, rank, elapsed, U in frames:
        fig, (axk, axu) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)
        imk = axk.imshow(K.T, origin="lower", cmap="viridis",
                         norm=colors.LogNorm(vmin=0.1, vmax=10),
                         extent=[0, 1, 0, 1])
        axk.set_title(r"coefficient  $k = 10^{\,\sin 3\pi x\,\sin 3\pi y}$",
                      fontsize=10)
        fig.colorbar(imk, ax=axk, fraction=0.046)
        imu = axu.imshow(U.T, origin="lower", cmap="magma", vmin=0, vmax=vmax,
                         extent=[0, 1, 0, 1])
        axu.set_title(f"sweep {swp}:  residual {res:.1e},  TT rank {rank},"
                      f"  t = {elapsed:.2f} s", fontsize=10)
        fig.colorbar(imu, ax=axu, fraction=0.046)
        fig.suptitle(r"amen_solve on $-\nabla\cdot(k\nabla u)=1$, "
                     r"QTT $%d^2$ (no preconditioner)" % n, fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pngs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    pngs[0].save(path, save_all=True, append_images=pngs[1:],
                 duration=[1400] * (len(pngs) - 1) + [3000], loop=0)
    print(f"saved {path}")


if __name__ == "__main__":
    bits = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    A, h = assemble(bits, dirichlet=True)
    f = tt.ones(2, 2 * bits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = amen_solve(A, f, f, 1e-8, verb=0)
    res = float((tt.matvec(A, u) - f).norm() / f.norm())
    print(f"bits={bits} ({2**bits}^2 grid): residual {res:.2e}, "
          f"solution TT ranks <= {max(u.r)}, operator ranks <= {max(A.tt.r)}")
    if bits <= 6:
        ud = dense_oracle(bits, dirichlet=True)
        utt = np.asarray(u.full()).flatten("F")
        print(f"vs scipy.sparse oracle: "
              f"{np.linalg.norm(utt - ud) / np.linalg.norm(ud):.2e}")
    if "--gif" in sys.argv:
        out = sys.argv[sys.argv.index("--gif") + 1]
        render_gif(sweep_frames(bits), out)
