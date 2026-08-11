"""tAMEn: spectral-in-time integration of ``dx/dt = A x`` in the TT format.

One *time interval* is treated as a single TT tensor with an extra time mode
and solved as one global linear system (Dolgov, CMAM 19(1):23-38, 2019,
arXiv:1403.8085):

    B y = f,    B = I (x) S - A (x) I_J,    f = x0 (x) (S e),

where ``S`` is the interior block of the Chebyshev-Lobatto spectral
differentiation matrix on ``J + 1`` nodes of ``[0, h]`` (so the scheme is
exponentially accurate in ``J`` for trajectories analytic near the interval)
and the last TT mode of the unknown ``y`` enumerates the collocation nodes.
The global system is solved by :func:`tt.amen_solve` -- the alternating
solver with residual enrichment; its local GMRES handles the nonsymmetric
``B``.

Conservation, the point of the method
-------------------------------------
After the alternating solve, the spatial basis is *enriched with the
invariants* and the last step is redone exactly: the spatial cores of
``y + sum_m c_m (x) e`` are left-orthogonalized into a frame ``X`` (which
therefore spans every ``c_m`` with ``c_m^* A = 0``), and the reduced
``r J x r J`` collocation system

    (S (x) I_r - I_J (x) X^* A X) v = (S e) (x) (X^* x0)

is solved densely.  Because ``c_m`` is in the span of ``X``,

    d/dt (c_m^* X v) = c_m^* X X^* A X v = (X X^* c_m)^* A X v
                     = c_m^* A X v = 0,

and ``c_m^* X X^* x0 = c_m^* x0`` exactly -- so the invariants are conserved
to the accuracy of a small dense solve, **independently of the TT truncation
threshold**.  This is the argument of the paper's section 3.4; the one
deviation from its Algorithm 1 is that the co-kernel enrichment happens once
before the final reduced solve rather than inside every sweep, which changes
the iteration but not the conservation property, since the property only
needs ``span(X)`` to contain the invariants at the *last* step.

Step control
------------
The embedded estimate costs nothing: the reduced system is also solved on
the coarser ``ceil(J/2)``-node grid (both grids contain the endpoint
``t = h``), and ``E = |v_J(h) - v_{J/2}(h)| / |v_J(h)|``.  An interval with
``E > eps`` is rejected and shrunk; an accepted one suggests
``h <- h (eps/E)^{1/J}``, clipped to ``[h/3, 3h]`` (the paper's eq. 23 with
its ``q = J`` for the spectral scheme).

Not implemented (recorded, not hidden): time-dependent ``A(t)``; the 2-norm
conservation for skew-symmetric ``A`` (the paper's theta rescaling) -- the
complex Schroedinger side of this package is served by the KSL integrator.

References
----------
* S. V. Dolgov, "A tensor decomposition algorithm for large ODEs with
  conservation laws", CMAM 19(1):23-38, 2019, arXiv:1403.8085.  Reference
  implementation ``github.com/dolgov/tamen`` (MATLAB; the algorithm was
  ported from the paper, the code was not copied).
* L. N. Trefethen, "Spectral Methods in MATLAB", SIAM 2000, Chapter 6 --
  the Chebyshev differentiation matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import kron as tt_kron
from ..core import _ops
from ..core.matrix import matrix
from ..core.vector import vector
from . import _localops as lo
from .amen import amen_solve

__all__ = ["tamen", "TamenHistory", "cheb_interior"]


def cheb_interior(J, h):
    """Interior block of the Chebyshev-Lobatto differentiation on ``[0, h]``.

    Nodes ``t_j = h/2 (1 - cos(pi j / J))``, ``j = 0..J`` (ascending,
    ``t_0 = 0`` and ``t_J = h`` included).  Returns ``(t, S, se)`` with
    ``t`` the ``J`` interior-plus-endpoint nodes ``t_1..t_J``, ``S`` the
    ``J x J`` block ``D[1:, 1:]`` of the differentiation matrix and
    ``se = -D[1:, 0]``, so that collocating ``x' = A x`` with the known
    ``x(0) = x0`` reads ``(S - A) x = se x0`` per node.  Note
    ``se = S @ ones`` exactly (the derivative of a constant is zero), which
    is the ``f = x0 (x) (S e)`` of the paper.
    """
    J = int(J)
    if J < 2:
        raise ValueError(f"J={J}: the scheme needs at least 2 nodes")
    j = np.arange(J + 1)
    # standard Trefethen matrix is on cos(pi j / J), descending in x; build
    # directly on the ascending mapped nodes instead
    x = 0.5 * h * (1.0 - np.cos(np.pi * j / J))
    c = np.ones(J + 1)
    c[0] = c[J] = 2.0
    c = c * (-1.0) ** j
    X = x[:, None] - x[None, :]
    D = (c[:, None] / c[None, :]) / (X + np.eye(J + 1))
    D = D - np.diag(D.sum(axis=1))
    return x[1:], D[1:, 1:], -D[1:, 0]


@dataclass
class TamenHistory:
    """Per-interval bookkeeping; the numbers the caller should not re-derive.

    Attributes:
        intervals: one dict per attempted interval: ``t0, h, accepted,
            err_est, ranks, amen_time, time``.
        invariant_drift: worst ``|c_m^* x(T) - c_m^* x(0)| / |c_m^* x(0)|``
            over the whole run -- the number the method exists for.
        rejections: how many intervals were redone.
        time: total wall-clock seconds.
    """

    intervals: list = field(default_factory=list)
    invariant_drift: float = 0.0
    rejections: int = 0
    time: float = 0.0

    def __repr__(self):
        acc = sum(1 for r in self.intervals if r["accepted"])
        return (f"TamenHistory({acc} intervals + {self.rejections} rejected, "
                f"invariant_drift={self.invariant_drift:.2e}, "
                f"time={self.time:.2f}s)")


def _project_state(frames, x0_cores):
    """``X^* x0`` through the left-orthogonal spatial frame."""
    m = np.ones((1, 1))
    for q, c in zip(frames, x0_cores):
        # m[a, p] -> m'[b, p'] = sum q*[a, i, b] m[a, p] c[p, i, p']
        t = np.einsum("ap,piq->aiq", m, np.asarray(c))
        m = np.einsum("aib,aiq->bq", np.asarray(q).conj(), t)
    return m.reshape(-1)


def _reduced_operator(frames, acores):
    """``X^* A X`` through the frame: the Galerkin matrix, ``(r, r)``."""
    phi = np.ones((1, 1, 1))
    for q, a in zip(frames, acores):
        phi = lo.phi_left(phi, np.asarray(a), np.asarray(q), np.asarray(q))
    return np.asarray(phi)[:, 0, :]


def _reduced_solve(Ahat, v0, J, h):
    """The dense collocation system on ``J`` nodes; returns ``v (J, r)``."""
    _, S, se = cheb_interior(J, h)
    r = Ahat.shape[0]
    M = np.kron(S, np.eye(r)) - np.kron(np.eye(J), Ahat)
    v = np.linalg.solve(M, np.kron(se, v0))
    return v.reshape(J, r)


def tamen(A, x0, T, eps, J=8, invariants=(), h0=None, nswp=20, kickrank=4,
          rmax=1000, max_intervals=10000, verbose=0, return_history=False,
          **amen_kwargs):
    """Integrate ``dx/dt = A x`` from ``x0`` to time ``T``.

    Args:
        A: :class:`tt.matrix`, square, time-independent.
        x0: :class:`tt.vector`, the initial state.
        T: Final time.
        eps: One threshold for everything, as in the paper: the AMEn
            residual, the TT rounding, and the per-interval time-error
            target of the step control.
        J: Chebyshev nodes per interval (the time order; error ~ C^-J).
        invariants: TT vectors ``c_m`` with ``c_m^* A = 0``; each is
            conserved to machine precision regardless of ``eps``.  For a
            master equation pass ``[tt.ones(...)]``.
        h0: Initial interval; default ``T / 8``.
        nswp, kickrank, rmax: forwarded to :func:`tt.amen_solve`.
        max_intervals: hard cap on attempts (rejections included).
        verbose: 0 silent, 1 one line per interval.
        return_history: also return :class:`TamenHistory`.
        **amen_kwargs: forwarded to :func:`tt.amen_solve`.

    Returns:
        ``x(T)`` as a :class:`tt.vector` (or ``(x, history)``).

    Raises:
        ValueError: on shape mismatches or an invariant with
            ``c^* A != 0``.
        RuntimeError: if ``max_intervals`` attempts do not reach ``T``.
    """
    if not isinstance(A, matrix):
        raise TypeError(f"tamen needs a tt.matrix, got {type(A)!r}")
    if not np.array_equal(np.asarray(A.n).ravel(), np.asarray(x0.n).ravel()):
        raise ValueError(f"mode mismatch: A.n={A.n}, x0.n={x0.n}")
    invariants = list(invariants)
    from ..core.tools import matvec as tt_matvec
    for c in invariants:
        # the conservation argument needs c* A = 0; a wrong c would be
        # conserved anyway (it is in the basis), silently faking physics
        gap = float(tt_matvec(A.T, c).norm())
        anorm = float(A.tt.norm()) if hasattr(A, "tt") else 1.0
        if gap > 1e-10 * anorm * float(c.norm()):
            raise ValueError(
                f"invariant check failed: ||A^T c|| = {gap:.3E} "
                f"(c is not a co-kernel vector of A)")

    hist = TamenHistory()
    t_start = time.time()
    d = int(x0.d)
    inv0 = [float(_dot_full(c, x0)) for c in invariants]

    x = x0
    tcur = 0.0
    h = float(h0) if h0 else T / 8.0
    J2 = max(2, (J + 1) // 2)
    attempts = 0
    while tcur < T * (1 - 1e-14):
        attempts += 1
        if attempts > max_intervals:
            raise RuntimeError(
                f"tamen: {max_intervals} interval attempts did not reach "
                f"T={T} (stuck at t={tcur:.6g} with h={h:.3E}); the "
                f"dynamics is stiffer than J={J} nodes can carry -- "
                f"increase J or loosen eps")
        h = min(h, T - tcur)
        _, S, se = cheb_interior(J, h)
        t0 = time.time()

        # the global space-time system on this interval
        Sm = matrix.from_list([np.ascontiguousarray(S[None, :, :, None])])
        IJ = matrix.from_list([np.ascontiguousarray(np.eye(J)[None, :, :, None])])
        Isp = _ops_eye_like(x)
        B = (tt_kron(Isp, Sm) - tt_kron(A, IJ)).round(1e-14)
        f = tt_kron(x, vector.from_list([se.reshape(1, J, 1)]))
        guess = tt_kron(x, vector.from_list([np.ones((1, J, 1))]))
        t_amen0 = time.time()
        y = amen_solve(B, f, guess, eps, nswp=nswp, kickrank=kickrank,
                       rmax=rmax, verb=0, **amen_kwargs)
        t_amen = time.time() - t_amen0

        # enrich the spatial basis with the invariants, then left-orth
        aug = y
        ynorm = float(y.norm())
        for c in invariants:
            cn = float(c.norm())
            aug = aug + tt_kron(c * (ynorm / cn),
                                vector.from_list([np.ones((1, J, 1))]))
        cores = _ops.orthogonalize(list(aug.cores), center=d)
        frames = cores[:d]

        # the exact reduced solve, plus the embedded coarse one
        acores = lo.operator_cores(A, np.asarray(frames[0]), "float64")
        Ahat = _reduced_operator(frames, acores)
        v0 = _project_state(frames, list(x.cores))
        vJ = _reduced_solve(Ahat, v0, J, h)
        vC = _reduced_solve(Ahat, v0, J2, h)
        vend = vJ[-1]
        err = float(np.linalg.norm(vend - vC[-1])
                    / max(np.linalg.norm(vend), 1e-300))

        # The Galerkin re-solve conserves the invariants by construction, but
        # the projection of a nonnormal operator is not stability-preserving:
        # X* A X can have eigenvalues in the right half-plane even when A has
        # none (measured on the SIR master equation: a re-solve accepted
        # without this check compounded to ||p|| ~ 1e19 over 6 intervals,
        # while the embedded time estimate stayed quiet -- both J and J/2
        # solves share the same bad projection).  So the re-solved iterate is
        # accepted only if its residual in the FULL system is small; else the
        # endpoint comes from the amen iterate and the invariants are
        # restored by an explicit O(eps) shift along the c_m.
        yhat = vector.from_list(
            [np.asarray(c) for c in frames]
            + [np.ascontiguousarray(vJ.T[:, :, None])])
        fnorm = float(f.norm())
        res_hat = float((_tt_matvec(B, yhat) - f).norm()) / fnorm
        res_y = float((_tt_matvec(B, y) - f).norm()) / fnorm
        # "no worse than the amen iterate itself": a fixed multiple of eps
        # is the wrong yardstick at crude eps (10 eps = 0.1 admitted the
        # unstable projections this guard exists to catch)
        galerkin_ok = np.isfinite(res_hat) and res_hat <= 2.0 * res_y + eps

        accepted = err <= eps
        rec = dict(t0=tcur, h=h, accepted=accepted, err_est=err,
                   res_hat=res_hat, path="galerkin" if galerkin_ok
                   else "corrected", ranks=[int(r) for r in aug.r],
                   amen_time=t_amen, time=time.time() - t0)
        hist.intervals.append(rec)
        if verbose:
            mark = "ok " if accepted else "REJ"
            print(f"  tamen [{mark}] t={tcur:9.4g} h={h:9.3E} "
                  f"E={err:9.3E} rank={max(rec['ranks'])} "
                  f"({rec['time']:.2f}s)", flush=True)
        if not accepted:
            hist.rejections += 1
            h = max(h * max(0.1, 0.5 * (eps / err) ** (1.0 / J)), 1e-12 * T)
            continue

        if galerkin_ok:
            # x(T_l + h) = X v(h): absorb the endpoint coefficients
            last = np.einsum("anb,b->an", np.asarray(frames[d - 1]), vend)
            newcores = [np.asarray(c) for c in frames[:d - 1]]
            newcores.append(np.ascontiguousarray(last[:, :, None]))
            x = vector.from_list(newcores).round(0.0, rmax=rmax)
        else:
            # endpoint of the amen iterate: t_J = h is a Chebyshev node
            ycores = [np.asarray(c) for c in y.cores]
            tend = ycores[d][:, -1, 0]
            last = np.einsum("anb,b->an", ycores[d - 1], tend)
            x = vector.from_list(
                ycores[:d - 1] + [np.ascontiguousarray(last[:, :, None])]
            ).round(0.0, rmax=rmax)
            if invariants:
                # restore c_m^* x = c_m^* x(0) exactly: a Gram-solved shift
                # along the c_m, of size O(the eps-level solver error)
                G = np.array([[float(_dot_full(ci, cj))
                               for cj in invariants] for ci in invariants])
                gap = np.array([v_init - float(_dot_full(c, x))
                                for c, v_init in zip(invariants, inv0)])
                coef = np.linalg.solve(G, gap)
                for c, a in zip(invariants, coef):
                    x = x + c * float(a)
                x = x.round(0.0, rmax=rmax)
        tcur += h
        # aim at eps/2, not eps: without the safety factor the controller
        # oscillates on the acceptance boundary (grow to E ~ 1.0002 eps,
        # reject, halve, regrow), wasting one amen solve per cycle
        h = h * min(3.0, max(1.0 / 3.0,
                             (0.5 * eps / max(err, 1e-300)) ** (1.0 / J)))

        for c, v_init in zip(invariants, inv0):
            drift = abs(float(_dot_full(c, x)) - v_init) / max(abs(v_init),
                                                               1e-300)
            hist.invariant_drift = max(hist.invariant_drift, drift)

    hist.time = time.time() - t_start
    return (x, hist) if return_history else x


def _dot_full(a, b):
    from ..core.tools import dot
    return dot(a, b)


def _tt_matvec(A, v):
    from ..core.tools import matvec
    return matvec(A, v)


def _ops_eye_like(x):
    from ..core.tools import eye
    return eye([int(v) for v in x.n])
