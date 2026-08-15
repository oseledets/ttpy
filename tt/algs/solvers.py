"""Flexible TT-GMRES: solve ``A x = b`` when every operation is inexact.

Classical GMRES assumes exact arithmetic in the Krylov recurrence.  In the TT
format nothing is exact: a matrix-vector product multiplies the ranks, so it
must be followed by a truncation, and so must every orthogonalization.  The
saving observation (Bouras & Fraysse; Simoncini & Szyld; Dolgov for the TT case)
is that the accuracy of the ``j``-th matvec only needs to be

    delta_j = eps * ||r_0|| / ||r_j||

-- the further the iteration has come, the *less* accurate the operator
application has to be.  That is what makes the method usable: the ranks stay
small exactly when the residual is small, which is when the vectors would
otherwise be most expensive.  ``A`` is therefore not a matrix here but a closure
``A(x, eps)`` that promises ``||A(x, eps) - A x|| <= eps ||A x||``.

"Flexible" means more than inexact matvecs: with the ``prec`` argument the
right preconditioner ``M_j`` may change from one Krylov step to the next
(FGMRES, Saad 1993).  That requires keeping the preconditioned basis
``z_j = M_j^{-1} v_j`` next to the orthonormal ``v_j`` and expanding the
correction in the ``z``'s -- expanding it in the ``v``'s is only correct when
``M`` never changes.  The flexible variant was contributed to ttpy by Larisa
Markeeva (develop branch, ``new_gmres``, 2018), part of the work on solving
equations on complicated domains in the QTT format via z-order curves; it is
ported here with the ``Z``-basis stored explicitly.

The small least squares problem
-------------------------------
The Hessenberg system is solved by a dense least squares solve of the
``(j+2) x (j+1)`` matrix at every step, rather than by incrementally applied
Givens rotations.  It costs ``O(m^3)`` scalar flops per restart cycle -- nothing
next to a single TT rounding -- and it removes a whole class of bugs (the legacy
code applied *real* rotations to what may be a complex Hessenberg matrix, and
mixed up the conjugation in the inner product that fills it).  The residual
estimate ``||beta e_1 - H y||`` comes out of the same solve.

Reference:
    S. V. Dolgov, "TT-GMRES: solution to a linear system in the structured
    tensor format", Russ. J. Numer. Anal. Math. Modelling 28(2), 2013,
    arXiv:1206.5512.  Replaces ``tt/solvers.py`` of legacy ttpy.

Differences from the legacy implementation
------------------------------------------
* Restarts are a loop, not recursion.  The legacy version called itself for
  every restart, so ``maxit=1000, m=20`` piled up 50 stack frames (and a deep
  enough run simply crashed).
* ``u_0`` is not modified.  The legacy version did ``u_0 += ...`` on the
  caller's tensor, so the "initial guess" came back as the answer and a second
  call with the same starting vector solved a different problem.
* The returned residual is the true relative residual ``||b - A x|| / ||b||`` of
  the returned ``x``.  The legacy version returned the residual of the *first*
  iterate of the last restart cycle, which is the quantity least related to the
  answer, and it declared convergence on a criterion relative to that same
  stale number.
* Non-convergence warns, with the residual actually reached.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import backend as bk
from ..core.tools import dot as tt_dot
from ..core.vector import vector

__all__ = ["GMRES", "GmresHistory"]


@dataclass
class GmresHistory:
    """What the run knows about itself; recorded in full even with ``verbose=0``.

    Attributes:
        eps: Requested relative residual (and the base truncation accuracy).
        cycles: One dict per restart cycle, with keys ``cycle``, ``iterations``
            (Krylov steps used in this cycle), ``res_start`` and ``res_end``
            (true relative residuals before and after the cycle), ``res_est``
            (the Krylov estimate at the end of the cycle), ``max_rank`` of the
            iterate, ``breakdown`` (the Krylov space became invariant) and
            ``time``.
        iterations: Total number of Krylov steps, i.e. of calls to ``A``.
        residuals: True relative residual at the start of every cycle and at the
            end of the run.
        converged: Whether ``||b - A x|| / ||b|| < eps`` was reached.
        true_res: The final true relative residual.
        ranks: TT ranks of the returned solution.
        time: Wall-clock seconds.
        message: Human-readable outcome; identical to the warning text on failure.
    """

    eps: float = 0.0
    cycles: list = field(default_factory=list)
    iterations: int = 0
    residuals: list = field(default_factory=list)
    converged: bool = False
    true_res: float = float("nan")
    ranks: list = field(default_factory=list)
    time: float = 0.0
    message: str = ""

    def __repr__(self):
        return (f"GmresHistory(cycles={len(self.cycles)}, "
                f"iterations={self.iterations}, converged={self.converged}, "
                f"true_res={self.true_res:.2e}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


def _scalar(v):
    """A python/numpy scalar out of whatever the backend returned."""
    return v.item() if hasattr(v, "item") else v


def GMRES(A, u_0, b, eps=1e-6, maxit=100, m=20, callback=None, verbose=0, *,
          prec=None, return_history=False):
    """Flexible restarted GMRES in the TT format.

    Flexible variant contributed by Larisa Markeeva to ttpy (develop branch,
    2018); ported here.  Passing ``prec`` turns the method into FGMRES
    (Saad 1993): the preconditioned vectors ``z_j = prec(v_j)`` are stored as
    their own basis and the correction is expanded in them, so ``prec`` is
    free to be a different operator at every Krylov step -- an inner iterative
    solve, an alternating sweep, anything.

    Args:
        A: The operator, as a closure ``A(x, eps)`` returning an approximation
            of ``A x`` accurate to relative ``eps``.  It is always called with
            ``eps`` as a keyword, so ``def matvec(x, eps): ...`` works.
        u_0: Initial guess, a :class:`tt.vector`.  **Not** modified.
        b: Right-hand side, a :class:`tt.vector`.
        eps: Target relative residual, and the base accuracy of the TT
            truncations.
        maxit: Maximum total number of Krylov steps (calls to ``A``) over all
            restart cycles.
        m: Krylov dimension per cycle; the method restarts after ``m`` steps.
        callback: Called as ``callback(x)`` with the iterate at the end of every
            restart cycle.
        verbose: ``0`` silent, ``1`` one line per cycle, ``2`` one line per step.
            The history is recorded in full regardless.
        prec: Optional right preconditioner, a closure ``prec(x, eps)``
            returning an approximation of ``M^{-1} x`` accurate to relative
            ``eps`` (called with ``eps`` as a keyword, like ``A``).  It may
            change between calls -- this is flexible GMRES, so the correction
            is built from the stored ``z_j = prec(v_j)``, not from the
            orthonormal basis.  The returned ``x`` solves ``A x = b`` directly;
            no un-preconditioning step is left to the caller.
        return_history: Also return the :class:`GmresHistory`.

    Returns:
        tuple: ``(x, res)`` -- the solution and its *true* relative residual
        ``||b - A x|| / ||b||``, both computed in the TT format.  With
        ``return_history`` the triple ``(x, res, history)``.

    Raises:
        ValueError: ``b`` is zero (the problem has no scale to be relative to),
            or ``m < 1``.

    Warns:
        RuntimeWarning: the iteration stopped at ``maxit`` without reaching
            ``eps``.  The returned ``x`` and ``res`` are still the honest state
            of the iteration -- ``res`` is measured, not estimated.

    Example:
        >>> import tt
        >>> A = tt.qlaplace_dd([4])
        >>> b = tt.ones(2, 4)
        >>> matvec = lambda x, eps: tt.matvec(A, x).round(eps)
        >>> x, res = GMRES(matvec, tt.rand(b.n, r=1), b, eps=1e-8, maxit=200)
    """
    if not isinstance(u_0, vector) or not isinstance(b, vector):
        raise TypeError("GMRES expects tt.vectors for u_0 and b, got "
                        f"{type(u_0)!r} and {type(b)!r}")
    m = int(m)
    if m < 1:
        raise ValueError(f"the Krylov dimension m must be at least 1, got {m}")
    eps = float(eps)
    if eps < 0.0:
        # A negative target can never be met, and it is passed straight on to
        # every ``round``, where "eps <= 0" means "truncate nothing".  The run
        # would burn ``maxit`` iterations at full rank and then warn about a
        # threshold that was never reachable.
        raise ValueError(f"eps must be non-negative, got {eps}; use eps = 0 to "
                         "iterate to maxit without a residual target")

    t0 = time.perf_counter()
    hist = GmresHistory(eps=eps)
    bnorm = float(b.norm())
    if bnorm == 0.0:
        raise ValueError(
            "the right-hand side is zero: a relative residual is undefined. "
            "The solution is zero; there is nothing to iterate on.")

    x = u_0.copy()                       # never touch the caller's tensor
    dtype = bk.result_dtype(x.dtype, b.dtype)
    cdtype = np.complex128 if dtype.startswith("complex") else np.float64
    # relative scale at which a subdiagonal entry means "the Krylov space is
    # invariant": below it the next basis vector would be pure roundoff.  It is
    # compared against the norm of the whole Hessenberg column, so it does not
    # care how the operator is scaled.
    breakdown = np.sqrt(bk.eps_of(dtype))

    cycle = 0
    while True:
        r = (b - A(x, eps=eps)).round(eps)
        resnorm = float(r.norm())
        rel = resnorm / bnorm
        hist.residuals.append(rel)
        if rel < eps:
            hist.converged = True
            break
        if hist.iterations >= maxit:
            break

        t_cycle = time.perf_counter()
        basis = [(1.0 / resnorm) * r]
        zbasis = [] if prec is not None else None    # FGMRES: z_j = M_j^{-1} v_j
        hess = np.zeros((m + 1, m), dtype=cdtype)
        curr_beta = resnorm
        used, broke = 0, False
        y = np.zeros(0, dtype=cdtype)

        for j in range(m):
            # inexact-Krylov relaxation: the further along, the coarser the
            # operator may be.  Capped at 1 -- a relative error of 1 means the
            # result carries no information, and rounding to it would silently
            # replace the vector by noise.
            # curr_beta == 0 happens when the projected problem is solved
            # exactly (an invariant Krylov space reached with eps = 0); the
            # ratio is then +inf, which is what the cap is for -- but as python
            # floats it would be a ZeroDivisionError, so it is spelled out.
            delta = 1.0 if curr_beta == 0.0 else min(eps * resnorm / curr_beta, 1.0)
            if verbose > 1:
                print(f"it = {hist.iterations + 1} delta = {delta:.3e}")

            if prec is None:
                z = basis[j]
            else:
                z = prec(basis[j], eps=delta)
                zbasis.append(z)
            w = A(z, eps=delta)
            hist.iterations += 1
            used = j + 1
            for i in range(j + 1):
                hess[i, j] = _scalar(tt_dot(basis[i], w))   # <v_i, w>, v_i conjugated
                w = w - hess[i, j] * basis[i]
            w = w.round(delta)
            hess[j + 1, j] = float(w.norm())

            rhs = np.zeros(j + 2, dtype=cdtype)
            rhs[0] = resnorm
            y = np.linalg.lstsq(hess[:j + 2, :j + 1], rhs, rcond=None)[0]
            curr_beta = float(np.linalg.norm(hess[:j + 2, :j + 1] @ y - rhs))
            if verbose > 1:
                print(f"it = {hist.iterations}, ||r||/||b|| = {curr_beta / bnorm:.6e}")

            if curr_beta / bnorm < eps or hist.iterations >= maxit:
                break
            if hess[j + 1, j].real <= breakdown * np.linalg.norm(hess[:j + 2, j]):
                # invariant Krylov space: y already solves the projected problem
                # exactly, another basis vector would be division by noise
                broke = True
                break
            basis.append((1.0 / hess[j + 1, j].real) * w)

        coefs = y[:used].real if cdtype is np.float64 else y[:used]
        # FGMRES expands the correction in the preconditioned basis: the
        # Arnoldi relation is  A Z_used = V_{used+1} H,  so the projected
        # solution lives in span(Z), not span(V).
        span = basis if zbasis is None else zbasis
        upd = coefs[0] * span[0]
        for i in range(1, used):
            upd = upd + coefs[i] * span[i]
        x = (x + upd).round(eps)

        cycle += 1
        hist.cycles.append(dict(cycle=cycle, iterations=used, res_start=rel,
                                res_end=float("nan"), res_est=curr_beta / bnorm,
                                max_rank=int(max(x.r)), breakdown=broke,
                                time=time.perf_counter() - t_cycle))
        if verbose:
            print(f"cycle {cycle}: {used} steps, estimated ||r||/||b|| = "
                  f"{curr_beta / bnorm:.6e}, max rank {max(x.r)}")
        if callback is not None:
            callback(x)

    if hist.cycles:
        hist.cycles[-1]["res_end"] = hist.residuals[-1]
    hist.true_res = hist.residuals[-1]
    hist.ranks = [int(v) for v in x.r]
    hist.time = time.perf_counter() - t0
    hist.message = (
        f"converged in {hist.iterations} iterations, "
        f"||b - A x|| / ||b|| = {hist.true_res:.3e}" if hist.converged else
        f"stopped after {hist.iterations} iterations (maxit = {maxit}) with "
        f"||b - A x|| / ||b|| = {hist.true_res:.3e} > eps = {eps:.3e}")
    if not hist.converged:
        warnings.warn("TT-GMRES " + hist.message, RuntimeWarning, stacklevel=2)
    if verbose:
        print("TT-GMRES " + hist.message)

    if return_history:
        return x, hist.true_res, hist
    return x, hist.true_res
