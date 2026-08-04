"""KSL: the projector-splitting integrator for ``dy/dt = A y`` on the TT manifold.

One call performs one step of size ``tau`` and returns ``y(tau)`` approximated on
the manifold of TT tensors of *fixed* rank ``r(y0)``.  The exact flow is replaced
by the projected flow ``y' = P_{T_y M} (A y)``, and the tangent-space projector
splits into ``2d - 1`` elementary projections, each of which can be integrated
exactly by a matrix exponential of a small (projected) operator.

The sweep
---------
With the frames orthonormal on both sides of the current site, the elementary
sub-flows are

* ``K_k``: the whole core at site ``k`` evolves with ``exp(+tau B_k)``,
  ``B_k`` the local projected operator of size ``r_k n_k r_{k+1}``;
* ``S_k``: the interface between sites ``k`` and ``k+1`` evolves *backwards*,
  ``exp(-tau M_k)``, ``M_k`` the projected operator of size ``r_{k+1}^2``.

The left-to-right sweep is ``K_1 S_1 K_2 ... S_{d-1} K_d`` and the right-to-left
sweep is its exact reverse ``K_d S_{d-1} K_{d-1} ... S_1 K_1``.  ``scheme='first'``
does the right-to-left sweep with the full ``tau`` (Lie-Trotter, order 1);
``scheme='symm'`` does right-to-left with ``tau/2`` followed by left-to-right
with ``tau/2``, a palindromic composition (Strang, order 2).  The order is
verified numerically against the dense solution of the *projected* ODE
``y' = P_{T_y M} A y`` -- the equation this integrator discretizes, and the only
reference against which a splitting order is visible at all
(``tests/test_verify_eigb_ksl.py::test_ksl_order_against_the_dense_projected_flow``,
observed 1.00 and 2.00).  Measured against ``expm(tau A) y0`` instead, the two
schemes are indistinguishable: either the manifold contains the trajectory and
both are exact, or it does not and the tau-independent modelling error hides the
splitting error.

The local exponentials
----------------------
``exp(t B) v`` is computed by Arnoldi in a Krylov space of dimension ``space``
with EXPOKIT-style adaptive substepping (Sidje, ACM TOMS 24(1), 1998): the
classical corrected error estimate ``beta |[exp(h H_aug)]_{m+1,1}|`` drives the
substep, and -- the point of that estimate -- rejecting a substep costs only one
``expm`` of an ``(m+1)x(m+1)`` matrix, because the Krylov basis does not depend
on ``h``.  A step that cannot reach the requested tolerance within
``max_substeps`` raises; it never returns a quietly wrong vector.

The estimate is normalized by the norm of the *input*, and the substep control
compares it with ``tol`` times the norm of the current iterate.  When the flow
grows or decays by orders of magnitude over one call, that normalization and the
error of the *answer* part company: on a strongly non-normal operator with
``||exp(A) x|| / ||x|| ~ 1e5``, asking for ``tol = 1e-10`` delivers 5.5e-8
relative to the result and reports ``err_est = 6.4e-2``
(``tests/test_verify_eigb_ksl.py::test_expmv_krylov_on_a_strongly_non_normal_operator``).
Inside KSL, where ``tau ||B||`` is small, the two agree.

What the fixed rank cannot see
------------------------------
The integrator moves on the manifold, so the part of ``A y`` that points off the
manifold, ``(I - P_{T_y M}) A y``, is invisible to it: with too small a rank the
answer is wrong and nothing in the sweep notices.  ``check_rank=True`` (the
default) therefore measures that defect explicitly after the step and records it
in the history; ``history.step_error_est = tau ||(I-P) A y|| / ||y||`` is an
estimate of the error the manifold restriction introduced in this step.  It
costs one TT matvec (ranks ``r_A r_y``) plus one sweep -- switch it off only
when the rank is known to be adequate.

References
----------
* C. Lubich, I. V. Oseledets, "A projector-splitting integrator for dynamical
  low-rank approximation", BIT 54(1):171-188, 2014, arXiv:1301.1058.
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor
  trains", SIAM J. Numer. Anal. 53(2):917-941, 2015, arXiv:1407.2042.
* Replaces ``tt/tt-fort/tt_ksl.f90`` and ``tt_diag_ksl.f90`` of legacy ttpy
  (same public signatures).  The Fortran ``tt_ksl`` (real path) applies ``K``
  and ``S`` in the wrong order in the backward sweep, which breaks the
  palindromic structure; the complex path ``ztt_ksl`` has it right and is what
  is reproduced here.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from einops import rearrange
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from ..core import _ops
from ..core.matrix import matrix
from ..core.vector import vector
from . import _localops as lo

__all__ = ["ksl", "diag_ksl", "expmv_krylov", "tangent_defect", "KslHistory"]


@dataclass
class KslHistory:
    """What the step knows about itself; recorded even with ``verb=0``.

    Attributes:
        tau: The step size, exactly as passed (complex for a Schroedinger step
            -- recording only its real part would make the history lie about
            what was integrated).
        scheme: ``'symm'`` (order 2) or ``'first'`` (order 1).
        steps: One dict per local exponential, with keys ``sweep, site, kind
            ('K' or 'S'), size, substeps, krylov, err_est, time``.
        max_local_err: Largest local Krylov error estimate, in units of the norm
            of the vector the exponential was applied to.
        max_growth: Largest factor by which a single local exponential grew its
            argument, and ``max_growth_kind`` the step ('K' or 'S') that did it.
            The S-steps integrate backwards, so a dissipative ``A`` amplifies
            there; this is the number that says how much accuracy that cost.
        roundoff_floor: ``eps_machine * max_growth ** KSL_GROWTH_EXPONENT`` --
            the relative accuracy this step cannot go below, whatever the local
            tolerances.  Reported always; warned about when it exceeds
            ``local_tol``; refused in :func:`_step_exp` when it exceeds 1.
        total_substeps: Number of Krylov substeps over the whole sweep.
        ranks: TT ranks of the result.
        defect_abs: ``||(I - P_{T_y M}) A y||`` at the end of the step, or NaN
            if ``check_rank=False``.  This is the part of the dynamics the fixed
            rank cannot represent.
        defect_rel: ``defect_abs / ||A y||``.
        step_error_est: ``tau * defect_abs / ||y||`` -- the relative error the
            manifold restriction introduces per step.
        time: Wall-clock seconds.
    """

    tau: complex
    scheme: str
    steps: list = field(default_factory=list)
    max_local_err: float = 0.0
    max_growth: float = 1.0
    max_growth_kind: str = ""
    roundoff_floor: float = 0.0
    total_substeps: int = 0
    ranks: list = field(default_factory=list)
    defect_abs: float = float("nan")
    defect_rel: float = float("nan")
    step_error_est: float = float("nan")
    time: float = 0.0

    def __repr__(self):
        return (f"KslHistory(scheme={self.scheme}, tau={self.tau}, "
                f"substeps={self.total_substeps}, "
                f"max_local_err={self.max_local_err:.2e}, "
                f"defect_rel={self.defect_rel:.2e}, "
                f"step_error_est={self.step_error_est:.2e}, "
                f"max_rank={max(self.ranks) if self.ranks else 0}, "
                f"time={self.time:.2f}s)")


# --- Krylov exponential ------------------------------------------------------

def _vdot(a, b):
    """``<a, b> = sum conj(a) b`` for a backend array of any shape."""
    return (a.conj() * b).sum()


def _arnoldi(op, v, space):
    """Arnoldi with one reorthogonalization pass.

    Returns ``(basis, H, beta, happy)``: ``basis`` is a list of ``k`` orthonormal
    arrays shaped like ``v``, ``H`` is ``(k+1, k+1)`` with the extra row carrying
    ``h_{k+1,k}`` (the augmented matrix of the EXPOKIT error estimate), ``beta``
    is ``||v||`` and ``happy`` flags an exact (breakdown) Krylov space.

    The basis is kept as one contiguous ``(space+1, size)`` array so that
    orthogonalisation is two matrix products, not a python loop of ``j`` inner
    products: on the tiny blocks of a KSL sweep that loop was 128 numpy calls
    per exponential, each on a hundred numbers.
    """
    dt = bk.dtype_of(v)
    shape = v.shape
    size = int(np.prod(shape))
    probe = op(v)
    if type(probe) is not type(v) or probe.shape != shape:
        # A mixed-backend operator (a numpy A applied to a torch iterate) does
        # not necessarily return the shape it was given.  The contiguous-basis
        # path below cannot express that, so such a run keeps the original
        # list-of-blocks Arnoldi: correctness first, speed only where it is free.
        return _arnoldi_blocks(op, v, space)
    beta = float(bk.norm(v))
    hmat = bk.zeros((space + 1, space + 1), dtype=dt, like=v)
    V = bk.zeros((space + 1, size), dtype=dt, like=v)
    V[0] = v.reshape((size,)) / beta
    cplx = bk.is_complex(v)
    happy = False
    k = space
    for j in range(space):
        w = op(V[j].reshape(shape))
        if type(w) is not type(V):
            # the operator may live on another backend than the iterate (a numpy
            # A applied to a torch y is legitimate); the basis is one array, so
            # bring the result to where the basis is
            w = bk.asarray(bk.to_numpy(w), dt, backend=bk.backend_of(V))
        w = w.reshape((size,))
        # the breakdown test must compare like with like: `hn` below is a
        # residual of `A v_j`, so it scales with ||A||, not with ||v||.  Testing
        # it against ||v|| declares a breakdown at the first step for any
        # badly-scaled input and silently degrades the exponential to a
        # one-dimensional Krylov space, with a reported error of exactly zero.
        wn0 = float(bk.norm(w))
        Vj = V[:j + 1]
        Vc = Vj.conj() if cplx else Vj
        for _ in range(2):  # classical Gram-Schmidt, repeated -> stable
            h = Vc @ w
            w = w - h @ Vj
            hmat[:j + 1, j] = hmat[:j + 1, j] + h
        hn = float(bk.norm(w))
        if wn0 == 0.0 or hn <= 1e-13 * wn0:
            k, happy = j + 1, True
            break
        if j + 1 <= space:
            hmat[j + 1, j] = hn
        if j + 1 < space:
            V[j + 1] = w / hn
        else:
            k = space
    basis = [V[i].reshape(shape) for i in range(k)]
    return basis, hmat[:k + 1, :k + 1], beta, happy


def _arnoldi_blocks(op, v, space):
    """Arnoldi with the basis as a list of blocks (the general-shape fallback)."""
    dt = bk.dtype_of(v)
    beta = float(bk.norm(v))
    hmat = bk.zeros((space + 1, space + 1), dtype=dt, like=v)
    basis = [v / beta]
    happy = False
    k = space
    for j in range(space):
        w = op(basis[j])
        wn0 = float(bk.norm(w))
        for _ in range(2):
            for i in range(j + 1):
                h = _vdot(basis[i], w)
                hmat[i, j] = hmat[i, j] + h
                w = w - h * basis[i]
        hn = float(bk.norm(w))
        if wn0 == 0.0 or hn <= 1e-13 * wn0:
            k, happy = j + 1, True
            break
        if j + 1 <= space:
            hmat[j + 1, j] = hn
        if j + 1 < space:
            basis.append(w / hn)
        else:
            k = space
    return basis, hmat[:k + 1, :k + 1], beta, happy


def expmv_krylov(op, x, t, space=8, tol=1e-8, anorm=None, max_substeps=4096):
    """``w ~ exp(t A) x`` by Arnoldi with adaptive substepping.

    Args:
        op: Callable applying ``A`` to an array shaped like ``x``.
        x: Backend array of any shape (treated as a vector).
        t: Step (may be negative; complex is allowed if ``x`` is complex).
        space: Krylov dimension per substep.
        tol: Relative accuracy requested for the result.
        anorm: Estimate of ``||A||`` used only to pick the *first* substep.
            Rejections are cheap (the Krylov basis does not depend on the step),
            so a bad estimate costs little.
        max_substeps: Cap; exceeding it raises rather than returning garbage.

    Returns:
        ``(w, info)`` with ``info`` a dict ``{substeps, err_est, krylov}``.
        ``err_est`` is the accumulated local error estimate relative to
        ``||x||``.

    Raises:
        RuntimeError: if ``max_substeps`` substeps do not cover ``t``.
    """
    beta0 = float(bk.norm(x))
    if beta0 == 0.0 or t == 0:
        # ``underflow`` distinguishes "you handed me a zero vector" from "the
        # norm of a subnormal input underflowed on the way in"; both give an
        # exact answer, only one is worth knowing about.
        return x, dict(substeps=0, err_est=0.0, krylov=0,
                       underflow=beta0 == 0.0 and t != 0)

    tmag = abs(t)
    phase = t / tmag                      # +-1, or a complex phase
    v = x
    done = 0.0
    h = tmag if not anorm else min(tmag, float(space) / (2.0 * anorm))
    h = max(h, tmag * 1e-12)
    err_total, nsteps, kmax = 0.0, 0, 0

    underflowed = False
    while done < tmag * (1 - 1e-14):
        if float(bk.norm(v)) == 0.0:
            # A strongly contracting flow can drive the iterate to exactly zero
            # part-way through the substeps.  exp(tA) 0 = 0, so the rest of the
            # step is done and the answer is exact -- but the Arnoldi below
            # would divide by ||v|| = 0, produce NaN, and then report a failure
            # blaming the caller's operator for it.  Say what happened instead.
            underflowed = True
            break
        basis, hmat, beta, happy = _arnoldi(op, v, space)
        k = len(basis)
        kmax = max(kmax, k)
        step = min(h, tmag - done)
        err = 0.0
        for _ in range(40):
            f = bk.expm((phase * step) * hmat)
            err = 0.0 if happy else float(abs(f[k, 0])) * beta
            target = tol * beta * (step / tmag)
            if err <= target or happy:
                break
            # local error ~ step^k: shrink with the classical EXPOKIT rule
            shrink = 0.9 * (target / err) ** (1.0 / max(k, 1))
            step = step * min(0.9, max(0.1, shrink))
        else:
            raise RuntimeError(
                f"expmv_krylov: the substep could not be reduced enough "
                f"(estimated local error {err:.3E} vs target {target:.3E}); "
                f"the local operator is likely not what the caller thinks")
        w = f[0, 0] * basis[0]
        for j in range(1, k):
            w = w + f[j, 0] * basis[j]
        v = beta * w
        done += step
        err_total += err
        nsteps += 1
        h = step * 2.0     # try to grow again; a rejection is cheap
        if nsteps > max_substeps:
            raise RuntimeError(
                f"expmv_krylov: more than {max_substeps} substeps for t={t}; "
                f"covered {done / tmag:.3%} of the step. Increase `space`, "
                "loosen `tol`, or take a smaller time step.")
    return v, dict(substeps=nsteps, err_est=err_total / beta0, krylov=kmax,
                   underflow=underflowed)


def _norm_estimate(op, v, iters=4):
    """Lower estimate of ``||A||_2`` by a few power iterations from ``v``.

    Only used to pick the first substep of :func:`expmv_krylov`; a lower
    estimate makes the first step too long, which the step control then fixes at
    the cost of one small ``expm``.
    """
    nrm = float(bk.norm(v))
    if nrm == 0:
        return 1.0
    u = v / nrm
    est = 0.0
    for _ in range(iters):
        w = op(u)
        est = float(bk.norm(w))
        if est == 0:
            return 0.0
        u = w / est
    return est


# --- the manifold defect -----------------------------------------------------

def tangent_defect(A, y):
    """``(||(I - P_{T_y M}) A y||, ||A y||)`` -- what the fixed rank cannot follow.

    ``P_{T_y M}`` is the orthogonal projector onto the tangent space of the
    fixed-rank TT manifold at ``y``.  Using the orthogonal decomposition of that
    tangent space (Lubich/Oseledets/Vandereycken 2015, Sec. 2), with the frames
    orthonormal and ``Q_k`` the left unfolding of the ``k``-th core,

        ||P Z||^2 = sum_{k<d} ||Z_k - Q_k (Q_k^H Z_k)||^2 + ||Z_d||^2,
        Z_k = Y_{<k}^H Z Y_{>k}.

    The defect is then ``sqrt(||Z||^2 - ||P Z||^2)``.  That difference cancels
    when the defect is small, so the value returned is clamped at the resolution
    floor ``||Z|| sqrt(eps)``: a saturated bound is honest, a confident tiny
    number would not be.

    Args:
        A: TT-matrix.
        y: TT-vector (the current point on the manifold).

    Returns:
        ``(defect, ||A y||)``, both floats.
    """
    yc = _ops.orthogonalize(y.cores, center=0)   # cores 1..d-1 right-orthogonal
    d = len(yc)
    dt = bk.result_dtype(A.dtype, y.dtype)
    yc = _ops.to_dtype(yc, dt)
    zc = _ops.matvec_cores(lo.operator_cores(A, yc[0], dt), yc)

    mr = [None] * (d + 1)
    mr[d] = bk.eye(1, 1, dtype=dt, like=yc[0])
    for k in range(d - 1, -1, -1):
        t = einsum(yc[k].conj(), mr[k + 1], "a i b, b e -> a i e")
        mr[k] = einsum(t, zc[k], "a i e, c i e -> a c")

    ml = bk.eye(1, 1, dtype=dt, like=yc[0])
    proj2 = 0.0
    for k in range(d):
        zk = einsum(ml, zc[k], "a c, c i e -> a i e")
        zk = einsum(zk, mr[k + 1], "a i e, b e -> a i b")
        if k < d - 1:
            q, s = lo.left_orthogonalize(yc[k])
            yc[k] = q
            if k + 1 < d:
                r0, nk, r1 = yc[k + 1].shape
                yc[k + 1] = (s @ yc[k + 1].reshape((r0, nk * r1))).reshape(
                    (s.shape[0], nk, r1))
            qm = rearrange(q, "a i c -> (a i) c")
            zm = rearrange(zk, "a i b -> (a i) b")
            proj2 += float(bk.norm(zm - qm @ (qm.conj().T @ zm))) ** 2
            ml = einsum(q.conj(), einsum(ml, zc[k], "a c, c i e -> a i e"),
                        "a i b, a i e -> b e")
        else:
            proj2 += float(bk.norm(zk)) ** 2

    znorm = float(_ops.norm(zc))
    gap = float(np.sqrt(max(znorm ** 2 - proj2, 0.0)))
    floor = znorm * np.sqrt(bk.eps_of(dt))
    return max(gap, floor), znorm



DENSE_LOCAL_LIMIT = 256
"""Below this local size the operator is formed densely for the Krylov step.

The blocks of a KSL sweep are tiny -- ``r n r`` with the fixed rank of the
manifold -- and applying them as three contractions costs about 25 us of numpy
dispatch against 2 us of arithmetic.  Forming the matrix once per exponential
(``size^2 R`` flops) and using ``B @ v`` for all ``space`` Krylov steps turns
that around.  Above the limit the contraction wins again and is used.
"""


def _dense_or_contract_local(left, acore, right, block):
    """The K-step operator: dense when the block is small, contracted otherwise."""
    shape = block.shape
    size = int(np.prod(shape))
    if (size <= DENSE_LOCAL_LIMIT and type(left) is np.ndarray
            and type(block) is np.ndarray and type(acore) is np.ndarray):
        mat = np.asarray(lo.local_matrix(left, acore, right)).reshape((size, size))
        return lambda x: (mat @ x.reshape((size,))).reshape(shape)
    return lambda x: lo.local_matvec(left, acore, right, x)


def _dense_or_contract_interface(left, right, block):
    """The S-step operator, same rule."""
    shape = block.shape
    size = int(np.prod(shape))
    if (size <= DENSE_LOCAL_LIMIT and type(left) is np.ndarray
            and type(block) is np.ndarray):
        mat = np.asarray(lo.interface_matrix(left, right)).reshape((size, size))
        return lambda x: (mat @ x.reshape((size,))).reshape(shape)
    return lambda x: lo.interface_matvec(left, right, x)


# --- the integrator ----------------------------------------------------------

def _sweep_backward(cores, acores, left, right, tau0, space, tol, use_normest,
                    hist, sweep_id):
    """``K_d S_{d-1} K_{d-1} ... S_1 K_1`` with steps ``+tau0`` / ``-tau0``."""
    d = len(cores)
    for i in range(d - 1, -1, -1):
        k = _step_exp(_dense_or_contract_local(left[i], acores[i], right[i + 1],
                                               cores[i]),
                      cores[i], tau0, space, tol, use_normest, hist, sweep_id, i, "K")
        if i == 0:
            cores[0] = k
            continue
        s, q = lo.right_orthogonalize(k)
        cores[i] = q
        right[i] = lo.phi_right(right[i + 1], acores[i], q, q)
        s = _step_exp(_dense_or_contract_interface(left[i], right[i], s),
                      s, -tau0, space, tol, use_normest, hist, sweep_id, i, "S")
        r0, nk, _ = cores[i - 1].shape
        cores[i - 1] = (cores[i - 1].reshape((r0 * nk, -1)) @ s).reshape(
            (r0, nk, s.shape[1]))
    return cores


def _sweep_forward(cores, acores, left, right, tau0, space, tol, use_normest,
                   hist, sweep_id):
    """``K_1 S_1 K_2 ... S_{d-1} K_d`` with steps ``+tau0`` / ``-tau0``."""
    d = len(cores)
    for i in range(d):
        k = _step_exp(_dense_or_contract_local(left[i], acores[i], right[i + 1],
                                               cores[i]),
                      cores[i], tau0, space, tol, use_normest, hist, sweep_id, i, "K")
        if i == d - 1:
            cores[i] = k
            continue
        q, s = lo.left_orthogonalize(k)
        cores[i] = q
        left[i + 1] = lo.phi_left(left[i], acores[i], q, q)
        s = _step_exp(_dense_or_contract_interface(left[i + 1], right[i + 1], s),
                      s, -tau0, space, tol, use_normest, hist, sweep_id, i, "S")
        _, nk, r2 = cores[i + 1].shape
        cores[i + 1] = (s @ cores[i + 1].reshape((s.shape[1], nk * r2))).reshape(
            (s.shape[0], nk, r2))
    return cores


#: The S-steps of the projector splitting run *backwards* in time, so for a
#: dissipative ``A`` they amplify.  The following K-step shrinks the data back
#: but not the rounding error the amplification carried with it, so a local
#: exponential that grew its argument by ``g`` leaves the answer with a relative
#: roundoff floor of about ``eps_machine * g**KSL_GROWTH_EXPONENT``.
#:
#: The exponent is *fitted*, not derived.  Measured on the QTT heat equation
#: ``dy/dt = -(2^L+1)^2 Laplace y`` at ``L = 6``, fixed rank 4, ``local_tol=1e-8``
#: (relative error against a dense ``expm``, the modelling error of the fixed
#: rank being ~2e-2 throughout):
#:
#: ===========  ==========  ==========  ================
#: ``tau|A|``   ``g``       error       ``eps g**2.5``
#: ===========  ==========  ==========  ================
#: 15.3         7.7e+00     3.1e-02     6.5e-14
#: 20.6         6.9e+01     5.5e-02     8.6e-12
#: 27.7         1.5e+03     1.0e-01     2.0e-08
#: 37.3         5.8e+04     2.9e-01     1.7e-04
#: 50.2         1.3e+07     1.2e+02     1.4e+02
#: 67.6         2.6e+10     3.5e+09     7.3e+10
#: ===========  ==========  ==========  ================
#:
#: The last two rows are the ones that matter: the fit tracks the observed error
#: to within an order of magnitude where the answer is destroyed, and stays far
#: below the modelling error where it is not.
KSL_GROWTH_EXPONENT = 2.5

def _step_exp(op, x, t, space, tol, use_normest, hist, sweep_id, site, kind):
    """One local exponential, with its bookkeeping."""
    t0 = time.time()
    if use_normest == 0:
        anorm = None      # ask for no hint: the first substep is the whole step
    elif use_normest == 2:
        anorm = 1.0
    else:
        anorm = _norm_estimate(op, x)
    w, info = expmv_krylov(op, x, t, space=space, tol=tol, anorm=anorm)
    # How much this local exponential *grew* its argument.  The S-steps run
    # backwards in time, so for a dissipative A they amplify, and whatever
    # rounding error the K-step before them left is amplified with the data.
    # The following K-step shrinks the data back but not the error, so this
    # number is the factor by which the answer's relative accuracy degrades.
    nx = float(bk.norm(x))
    growth = float(bk.norm(w)) / nx if nx > 0 else 1.0
    # Refuse as soon as a single local exponential has amplified enough to
    # destroy every digit.  Left to run, this returns a number: measured on a
    # dissipative QTT heat equation at tau|A| = 169, ||y|| = 3.3e+106 where the
    # exact solution has norm 0.307, with nothing in the history to say so.
    # Raising here rather than at the end of the sweep also keeps the failure
    # legible: a few steps later the iterate overflows and expmv_krylov reports
    # "estimated local error NAN vs target NAN", which blames the wrong thing.
    floor = float(np.finfo(np.float64).eps) * growth ** KSL_GROWTH_EXPONENT
    if floor > 1.0:
        raise RuntimeError(
            f"ksl: the {kind}-step at site {site} amplified its argument by "
            f"{growth:.3E}, which leaves no correct digits (roundoff floor "
            f"{floor:.3E}). The S-steps integrate backwards, so a dissipative A "
            f"amplifies there by exp(tau |lambda_min|); the projector splitting "
            f"is not stiff-stable and this step size is past what it can carry. "
            f"Reduce tau (the floor falls off fast), or use an integrator that "
            f"never forms the growing factor -- see docs/plans/bug-integrator.md.")
    hist.steps.append(dict(sweep=sweep_id, site=site, kind=kind,
                           size=int(np.prod(x.shape)),
                           substeps=info["substeps"], krylov=info["krylov"],
                           err_est=info["err_est"], growth=growth,
                           time=time.time() - t0))
    hist.max_local_err = max(hist.max_local_err, info["err_est"])
    if growth > hist.max_growth:
        hist.max_growth, hist.max_growth_kind = growth, kind
    hist.total_substeps += info["substeps"]
    return w


def ksl(A, y0, tau, verb=1, scheme="symm", space=8, rmax=2000, use_normest=1,
        local_tol=1e-8, check_rank=True, defect_warn=1e-2, return_history=False):
    """One KSL step: ``y(tau)`` for ``dy/dt = A y``, ``y(0) = y0``, at fixed rank.

    Args:
        A: :class:`tt.matrix` with ``A.n == A.m``.
        y0: :class:`tt.vector`; its TT ranks define the manifold and are kept.
        tau: Time step (may be negative; complex ``tau`` promotes the dtype).
        verb: 0 silent, 1 one summary line, 2 one line per local exponential.
        scheme: ``'symm'`` for the second-order (Strang) composition, anything
            else for the first-order (Lie-Trotter) one -- as in legacy ttpy.
        space: Krylov dimension of the local exponentials.
        rmax: Guard only: KSL does not change ranks, so ``max(y0.r) > rmax`` is
            an error rather than something to truncate.
        use_normest: How the first Krylov substep is guessed: 0 no estimate,
            1 power iteration, 2 fixed ``||A|| = 1``.  It cannot change the
            result, only the number of substeps.
        local_tol: Relative accuracy of every local exponential.
        check_rank: Measure ``(I - P_{T_y M}) A y`` after the step (see
            :func:`tangent_defect`).  Costs one TT matvec plus one sweep.
        defect_warn: Warn when ``tau ||(I-P) A y|| / ||y||`` exceeds this.  Pure
            reporting: it never changes the computation, and the number is in
            ``history.step_error_est`` whatever the threshold.

            That number is a *first-order, one-point* estimate -- the tangent
            defect at the end point, times ``tau`` -- so it is an indicator, not
            a bound, and it is optimistic on large steps.  Measured on
            ``diag_ksl`` with a rank-3 guess against the exact elementwise
            exponential: 4.73e-02 predicted against 4.95e-02 actual at
            ``tau = 0.01``, 1.13e-01 against 2.16e-01 at 0.1, and 6.64e-02
            against 5.39e-01 at 0.3 -- eight times optimistic at the largest
            step, and *non-monotone*, because the defect is read at one point of
            a trajectory that has already left the manifold.  It warns in every
            one of those cases, so nothing is silent; just do not read it as a
            certificate.
        return_history: also return the :class:`KslHistory`.

    Returns:
        The TT-vector ``y(tau)``, or ``(y, history)`` if ``return_history``.

    Raises:
        ValueError: on shape mismatches or ``max(y0.r) > rmax``.
        RuntimeError: if a local exponential cannot reach ``local_tol``.
    """
    if not isinstance(A, matrix):
        raise TypeError(f"ksl needs a tt.matrix, got {type(A)!r}")
    if not np.array_equal(A.n, A.m):
        raise ValueError(f"ksl needs a square TT-matrix, got n={A.n}, m={A.m}")
    if A.d != y0.d:
        raise ValueError(f"dimension mismatch: A.d={A.d}, y0.d={y0.d}")
    if not np.array_equal(np.asarray(A.n).ravel(), np.asarray(y0.n).ravel()):
        raise ValueError(f"mode mismatch: A.n={A.n}, y0.n={y0.n}")
    if max(y0.r) > rmax:
        raise ValueError(
            f"y0 already has rank {max(y0.r)} > rmax={rmax}; KSL keeps the rank "
            "fixed, it cannot truncate for you -- round y0 first")

    t_start = time.time()
    d = int(y0.d)
    dt = bk.result_dtype(A.dtype, y0.dtype)
    if isinstance(tau, complex) and tau.imag != 0:
        dt = bk.complex_dtype(dt)
    cores = [bk.asarray(c, dt) for c in y0.cores]
    acores = lo.operator_cores(A, cores[0], dt)
    hist = KslHistory(tau=tau, scheme=scheme)
    if verb > 0:
        kind = "complex" if bk.is_complex(cores[0]) else "real"
        tau_str = str(tau) if isinstance(tau, complex) else f"{tau:.1E}"
        print(f"Solving a {kind}-valued dynamical problem with tau={tau_str}")

    if d == 1:
        left = lo.ones_interface(cores[0], dt)
        right = lo.ones_interface(cores[0], dt)
        cores[0] = _step_exp(
            _dense_or_contract_local(left, acores[0], right, cores[0]),
            cores[0], tau,
            space, local_tol, use_normest, hist, 0, 0, "K")
        y = vector.from_list(cores)
        hist.ranks = [int(v) for v in y.r]
        hist.time = time.time() - t_start
        return (y, hist) if return_history else y

    # left-orthogonalize everything but the last core; build the left interfaces
    left = [None] * (d + 1)
    right = [None] * (d + 1)
    left[0] = lo.ones_interface(cores[0], dt)
    right[d] = lo.ones_interface(cores[0], dt)
    for k in range(d - 1):
        q, s = lo.left_orthogonalize(cores[k])
        cores[k] = q
        _, nk, r2 = cores[k + 1].shape
        cores[k + 1] = (s @ cores[k + 1].reshape((s.shape[1], nk * r2))).reshape(
            (s.shape[0], nk, r2))
        left[k + 1] = lo.phi_left(left[k], acores[k], q, q)

    symm = (scheme == "symm")
    tau0 = tau / 2.0 if symm else tau
    cores = _sweep_backward(cores, acores, left, right, tau0, space, local_tol,
                            use_normest, hist, 0)
    if symm:
        # the backward sweep left the frames right-orthogonal with the centre on
        # core 0, which is exactly the entry state of the forward sweep
        cores = _sweep_forward(cores, acores, left, right, tau0, space, local_tol,
                               use_normest, hist, 1)

    y = vector.from_list(cores)
    hist.ranks = [int(v) for v in y.r]
    if check_rank:
        defect, znorm = tangent_defect(A, y)
        hist.defect_abs = defect
        hist.defect_rel = defect / znorm if znorm > 0 else 0.0
        ynorm = y.norm()
        hist.step_error_est = abs(tau) * defect / ynorm if ynorm > 0 else float("inf")
        if hist.step_error_est > defect_warn:
            warnings.warn(
                f"KSL: the fixed rank {max(hist.ranks)} cannot follow the "
                f"dynamics -- the off-manifold part of A y is "
                f"{hist.defect_rel:.3E} of ||A y||, which over tau={tau:.3E} "
                f"amounts to a relative error of about "
                f"{hist.step_error_est:.3E} in this step. Increase the rank of "
                "y0 (this is a modelling error, not a solver failure).",
                RuntimeWarning, stacklevel=2)
    # Below the hard refusal in _step_exp there is still a band where the
    # amplification costs real digits without destroying them.  Report it
    # against what the caller asked of the local solves: roundoff above
    # local_tol means the step is no longer delivering the accuracy requested,
    # whatever the Krylov error estimates say.
    roundoff = float(np.finfo(np.float64).eps) * hist.max_growth ** KSL_GROWTH_EXPONENT
    hist.roundoff_floor = roundoff
    if roundoff > local_tol:
        warnings.warn(
            f"KSL: a local {hist.max_growth_kind}-step amplified its argument "
            f"by {hist.max_growth:.3E}, so this step carries a roundoff floor "
            f"of about {roundoff:.3E} -- above the local_tol={local_tol:.1E} it "
            f"was asked for. The S-steps run backwards in time and amplify for "
            f"a dissipative A; the splitting is not stiff-stable. Reduce tau. "
            f"(history.max_growth and history.roundoff_floor carry these "
            f"numbers whatever the threshold.)",
            RuntimeWarning, stacklevel=2)
    hist.time = time.time() - t_start
    if verb > 0:
        print(f"KSL done: {hist.total_substeps} Krylov substeps, "
              f"max local error estimate {hist.max_local_err:.2E}, "
              f"rank {max(hist.ranks)}")
    if verb > 1:
        for s in hist.steps:
            print(f"  sweep {s['sweep']} site {s['site']} {s['kind']}: "
                  f"size {s['size']} substeps {s['substeps']} "
                  f"err {s['err_est']:.2E}")
    return (y, hist) if return_history else y


def diag_ksl(A, y0, tau, verb=1, scheme="symm", space=8, rmax=2000,
             use_normest=1, **kwargs):
    """One KSL step for ``dy/dt = diag(V) y`` with ``V`` a TT-*vector*.

    Legacy ttpy had a separate Fortran kernel for the diagonal case; here it is
    the same integrator applied to ``tt.diag(V)``, which keeps one owner for the
    projector splitting.  The price is that the diagonal matrix cores are
    ``(r, n, n, r)`` instead of ``(r, n, r)``: ``n`` times more memory and work
    in the *local* operator, nothing in the TT ranks.  For QTT (``n = 2``) that
    is irrelevant; for large mode sizes prefer building the diagonal matrix once
    and calling :func:`ksl` repeatedly.

    Args:
        A: :class:`tt.vector` holding the diagonal, or a :class:`tt.matrix`
            (then it is used as is).
        y0, tau, verb, scheme, space, rmax, use_normest: see :func:`ksl`.
        **kwargs: forwarded to :func:`ksl`.

    Returns:
        The TT-vector ``y(tau)`` (or ``(y, history)`` with ``return_history``).
    """
    from ..core.tools import diag
    amat = A if isinstance(A, matrix) else diag(A)
    return ksl(amat, y0, tau, verb=verb, scheme=scheme, space=space, rmax=rmax,
               use_normest=use_normest, **kwargs)
