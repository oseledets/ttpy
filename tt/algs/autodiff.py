"""Riemannian gradients by automatic differentiation, and gradient descent.

The Riemannian gradient of a smooth ``f`` at a point ``X`` of the fixed-rank
TT manifold is ``P_{T_X M} grad_E f(X)``.  Forming the Euclidean gradient as a
TT tensor first is correct and can be ruinous -- for a sampled functional its
rank is the number of samples.  [NRO22] observe that the *ordinary* reverse-mode
gradient of the auxiliary function ``g = f o T_X`` with respect to the tangent
delta cores, evaluated at ``(S_1, 0, ..., 0)`` and projected onto the gauge
complement, IS the Riemannian gradient -- the Euclidean gradient is never
formed, and the cost is ``O(F + d n r^3)`` where ``F`` is one evaluation of
``f`` ([NRO22] Prop. 5.2).

One sign in the paper is wrong somewhere: eq. (5.11) writes the gauge step with
a minus, Alg. 5.2 line 9 with a plus.  Only the minus is the projection onto
the gauge complement and only the minus reproduces :func:`tt.algs.riemannian.
project`; t3f's ``_enforce_gauge_conditions`` also subtracts.  This
implementation uses the minus, and a test pins it.

This is the niche where alternating least squares does not apply at all: ALS
needs a per-core *linear* local problem, i.e. a quadratic outer functional.
``rgd`` below asks only for a differentiable one -- robust losses, likelihoods,
anything torch can differentiate.

References
----------
* A. Novikov, M. Rakhuba, I. Oseledets, "Automatic differentiation for
  Riemannian optimization on low-rank matrix and tensor-train manifolds",
  SIAM J. Sci. Comput. 44(2):A843-A869, 2022, arXiv:2103.14974.
* M. Rakhuba, A. Novikov, I. Oseledets, "Low-rank Riemannian eigensolver for
  high-dimensional Hamiltonians", J. Comput. Phys. 396:718-737, 2019 -- the
  tangent parametrization and the cheap inner product.
* Reference implementation surveyed: ``github.com/Bihaqo/t3f`` (``autodiff.py``;
  read, not copied).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import backend as bk
from ..core.vector import vector
from .riemannian import (frames, project_delta, retract, tangent_inner,
                         tangent_to_tt)

__all__ = ["riemannian_grad", "rgd", "RgdHistory"]

#: Relative tolerance of the representation-invariance check: ``f`` evaluated
#: on two different core lists of the same tensor must agree to this.
_INVARIANCE_RTOL = 1e-8


def _torch_of(X):
    backend = bk.backend_of(X.cores[0])
    if backend.name != "torch":
        raise ValueError(
            "riemannian_grad differentiates with torch autograd and needs the "
            f"torch backend; X lives on {backend.name!r}.  Move it with "
            "x.to('torch', 'cpu', 'float64'), or supply an explicit Euclidean "
            "gradient to rgd(grad=...) instead")
    return backend.torch


def _assemble_stack(t, U, V, R, rx, n):
    """The rank-2r tangent stack from torch cores, differentiably.

    Built with ``torch.cat`` rather than index assignment so that autograd
    flows through ``R`` without in-place bookkeeping.
    """
    d = len(U)
    if d == 1:
        return [R[0]]
    cores = []
    for k in range(d):
        if k == 0:
            cores.append(t.cat([R[0], U[0]], dim=2))
        elif k < d - 1:
            zero = t.zeros((rx[k], n[k], rx[k + 1]),
                           dtype=V[k].dtype, device=V[k].device)
            top = t.cat([V[k], zero], dim=2)
            bottom = t.cat([R[k], U[k]], dim=2)
            cores.append(t.cat([top, bottom], dim=0))
        else:
            cores.append(t.cat([V[k], R[k]], dim=0))
    return cores


def riemannian_grad(f, X, *, runtime_check=True, frames_=None):
    """Delta cores of ``grad f(X) = P_{T_X M} grad_E f(X)``, by reverse-mode AD.

    Args:
        f: Takes a LIST OF CORES (backend tensors) and returns a backend
            scalar.  It must depend only on the tensor those cores represent,
            not on the representation; ``runtime_check`` verifies this by
            evaluating ``f`` on two different core lists of the same tensor
            and raises on disagreement.
        X: The point, on the torch backend (raises otherwise, naming the
            backend and the ``rgd(grad=...)`` alternative).
        runtime_check: See above; switch off only in an inner loop that has
            already passed it once.
        frames_: Precomputed :class:`tt.algs.riemannian.Frames` of ``X``.

    Returns:
        ``(value, deltas, frames)``: ``f(X)`` as a float (the evaluation comes
        free with the forward pass), the gauge delta cores of the Riemannian
        gradient (detached), and the frames of ``X``.
    """
    t = _torch_of(X)
    fr = frames_ if frames_ is not None else frames(X)
    d = len(fr.U)
    rx = fr.r
    n = [int(v) for v in X.n]

    if runtime_check:
        a = f(list(X.cores))
        cores2 = [c.clone() for c in X.cores]
        cores2[0] = cores2[0] * 2.0
        cores2[-1] = cores2[-1] * 0.5
        b = f(cores2)
        if abs(float(a) - float(b)) > _INVARIANCE_RTOL * (abs(float(a)) + 1.0):
            raise ValueError(
                "f is not a function of the tensor: two core representations "
                f"of the same X gave {float(a)!r} and {float(b)!r}.  A "
                "Riemannian gradient of such an f is meaningless")

    R = []
    for k in range(d):
        if k == 0:
            r0 = fr.V[0].clone().detach().requires_grad_(True)
        else:
            r1 = rx[k]
            r2 = rx[k + 1] if k < d - 1 else 1
            r0 = t.zeros((r1, n[k], r2), dtype=fr.V[0].dtype,
                         device=fr.V[0].device, requires_grad=True)
        R.append(r0)
    stack = _assemble_stack(t, fr.U, fr.V, R, rx, n)
    g = f(stack)
    grads = t.autograd.grad(g, R)

    if not bool(t.isfinite(g.detach())):
        raise ValueError(
            "f evaluated non-finite at the current point; a gradient of that "
            "is meaningless.  If the loss can overflow (exp, cosh, ...), use "
            "a numerically stable form -- e.g. log cosh(z) = |z| + "
            "log1p(exp(-2|z|)) - log 2")
    deltas = []
    for k in range(d):
        dk = grads[k]
        if not bool(t.isfinite(dk).all()):
            raise ValueError(
                "the gradient of f is non-finite while f itself is finite -- "
                "an intermediate of f overflowed on the backward pass; use a "
                "numerically stable form of the loss")
        if k < d - 1:
            u = fr.U[k].reshape(-1, rx[k + 1])
            gm = dk.reshape(-1, dk.shape[2])
            # the minus: projection onto the gauge complement (see module
            # docstring for the sign discrepancy in the paper)
            gm = gm - u @ (u.conj().T @ gm)
            dk = gm.reshape(dk.shape)
        deltas.append(dk.detach())
    return float(g.detach()), deltas, fr


@dataclass
class RgdHistory:
    """What the run knows about itself; recorded in full even with ``verbose=False``.

    Attributes:
        tol: Requested tangent-gradient norm (the stopping rule reads it
            relative to the first gradient norm).
        iterations: One dict per iteration with keys ``it, f, gnorm, step,
            backtracks, discarded, time``.
        converged: ``||grad||`` fell below ``tol * max(1, ||grad_0||)``.
        stop_reason: ``'gtol'``, ``'maxit'`` or ``'linesearch'`` (Armijo could
            not find a decrease -- the iterate is returned as is, and on this
            non-convex landscape that is a *report*, not a failure to hide:
            long plateaus are a measured property of the problem class).
        f: Final value.
        gnorm: Final tangent gradient norm.
        ranks: TT ranks of the iterate (constant: the retraction keeps them).
        fun_calls: Evaluations of ``f`` (line search included).
        grad_calls: Riemannian gradient evaluations.
        time: Wall-clock seconds.
    """

    tol: float = 0.0
    iterations: list = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""
    f: float = float("nan")
    gnorm: float = float("nan")
    ranks: list = field(default_factory=list)
    fun_calls: int = 0
    grad_calls: int = 0
    time: float = 0.0

    def __repr__(self):
        return (f"RgdHistory(iterations={len(self.iterations)}, "
                f"converged={self.converged}, stop={self.stop_reason!r}, "
                f"f={self.f:.6e}, gnorm={self.gnorm:.2e}, "
                f"time={self.time:.2f}s)")


def rgd(f, x0, *, grad=None, maxit=200, tol=1e-8, step0=1.0, shrink=0.5,
        armijo=1e-4, max_backtracks=40, method="svd", verbose=False):
    """Riemannian gradient descent on the fixed-rank TT manifold.

    Minimises a smooth ``f`` over tensors of the ranks of ``x0`` -- **any**
    smooth ``f``, which is the point: alternating least squares needs a
    quadratic functional to have local problems at all, this needs only a
    gradient.  Armijo backtracking line search, TT-SVD retraction, monotone by
    construction.

    Args:
        f: Takes a list of cores, returns a backend scalar (the
            :func:`riemannian_grad` contract).
        x0: Starting point; its ranks define the manifold and are kept.  On
            the torch backend the gradient comes from autodiff; otherwise
            ``grad`` must be given.
        grad: Optional callback ``grad(x: tt.vector) -> tt.vector`` returning
            the *Euclidean* gradient as a TT tensor, projected here with
            :func:`project_delta`.  The escape hatch for the numpy backend
            and for hand-written adjoints.
        maxit: Iteration cap; hitting it is reported, not hidden.
        tol: Stop when the tangent gradient norm falls below
            ``tol * max(1, gnorm_0)``.
        step0: Initial step of every line search (the accepted step of the
            previous iteration is tried first, doubled).
        shrink: Backtracking factor.
        armijo: Sufficient-decrease constant ``c`` in
            ``f(x_t) <= f(x) - c t ||grad||^2``.
        max_backtracks: Line-search budget per iteration.
        method: Retraction, ``'svd'`` or ``'psa'`` (see
            :func:`tt.algs.riemannian.retract`).
        verbose: One line per iteration; the history is recorded regardless.

    Returns:
        ``(x, history)``.

    Note:
        A rank chosen too *high* is not forgiving: the iterate stalls on a
        plateau with a small gradient step budget and nothing crashes -- the
        honest signal is ``history.f`` not decreasing, which is why it is
        recorded per iteration (measured in
        ``docs/plans/riemannian-autodiff.md`` 2.3 and 5.2).
    """
    if not isinstance(x0, vector):
        raise TypeError(f"rgd expects a tt.vector start, got {type(x0)!r}")
    if grad is None:
        _torch_of(x0)                      # raises with the actionable message
    t0 = time.time()
    hist = RgdHistory(tol=float(tol))
    x = x0
    step = float(step0)
    gnorm0 = None
    checked = False

    for it in range(int(maxit)):
        t_it = time.time()
        if grad is None:
            fval, deltas, fr = riemannian_grad(f, x, runtime_check=not checked)
            checked = True
            hist.fun_calls += 1
        else:
            fval = float(f(list(x.cores)))
            hist.fun_calls += 1
            deltas, fr = project_delta(x, grad(x))
        hist.grad_calls += 1
        gsq = float(abs(tangent_inner(deltas, deltas)))
        gnorm = gsq ** 0.5
        if gnorm0 is None:
            gnorm0 = gnorm
        if verbose:
            print(f"{it:4d}: f {fval:.6e}  |grad| {gnorm:.3e}  step {step:.2e}")

        if gnorm <= tol * max(1.0, gnorm0):
            hist.converged, hist.stop_reason = True, "gtol"
            hist.f, hist.gnorm = fval, gnorm
            break

        xi = tangent_to_tt(x, [-dl for dl in deltas], frames_=fr)
        step = min(step * 2.0, 1e6)        # optimistic warm start
        accepted = False
        backtracks = 0
        for _ in range(int(max_backtracks)):
            cand = retract(x, step * xi, method=method)
            fc = float(f(list(cand.cores)))
            hist.fun_calls += 1
            if fc <= fval - armijo * step * gsq:
                accepted = True
                break
            step *= shrink
            backtracks += 1
        hist.iterations.append({
            "it": it, "f": fval, "gnorm": gnorm, "step": step,
            "backtracks": backtracks, "time": time.time() - t_it})
        if not accepted:
            hist.stop_reason = "linesearch"
            hist.f, hist.gnorm = fval, gnorm
            warnings.warn(
                f"rgd: Armijo found no decrease in {max_backtracks} "
                f"backtracks at iteration {it} (f={fval:.3e}, "
                f"|grad|={gnorm:.3e}); returning the current iterate",
                RuntimeWarning, stacklevel=2)
            break
        x = cand
        hist.f, hist.gnorm = fc, gnorm
    else:
        hist.stop_reason = "maxit"
        warnings.warn(
            f"rgd spent {maxit} iterations without the gradient criterion "
            f"firing (|grad| {hist.gnorm:.3e} vs tol*|grad_0| "
            f"{tol * max(1.0, gnorm0 or 1.0):.3e}); returning the current "
            "iterate", RuntimeWarning, stacklevel=2)

    hist.ranks = [int(v) for v in x.r]
    hist.time = time.time() - t0
    return x, hist
