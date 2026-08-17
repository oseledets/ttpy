"""The paper's variational step: incompressibility by penalty, at fixed rank.

``qtt_ns.py`` enforces ``div V = 0`` by a Chorin projection -- a pressure Poisson
solve per stage.  Gourianov et al. instead advance the velocity by *minimizing*
(their eq. 8)

    Theta(V*) = mu ||div V*||^2 + || (V* - V)/dt + (V.grad)V - nu Lap V ||^2

over the MPS manifold.  Written out, ``Theta`` is a quadratic form: with
``R = V/dt - (V.grad)V + nu Lap V`` its stationarity condition is the SPD system

    (mu D^T D + I/dt^2) V* = R/dt,

so a step is one SPD solve, with no Poisson equation and no projection at all.
The penalty couples the velocity components, so the unknown is the whole field:
one tensor train with a trailing component mode of size ``K`` (2 in 2D, 3 in 3D)
appended to the z-order spatial modes, and the divergence as a non-square
``K -> 1`` TT-matrix.  Being SPD, at a **fixed** bond dimension the system is
exactly the fixed-rank energy minimization of ``tt.lobpcg_solve``, so the flow
never leaves the bounded-rank manifold -- the paper's premise.  ``solver="amen"``
lets the rank adapt instead, for reference runs.

Choosing ``mu``
---------------
``mu`` must be scaled, not guessed: the penalty term ``mu ||D||^2`` competes with
the data term ``1/dt^2``, so the balance sits near ``mu ~ (1/dt^2) / ||D||^2`` and
the useful window is one or two decades above it.  Larger ``mu`` presses the
divergence down but conditions the system worse, and past the window the
fixed-rank solve stops converging inside its sweep budget.

Do not answer that cost by capping the sweeps.  An under-solved ill-conditioned
step does not fail -- it returns a plausible wrong field (measured on the 1024^2
jet: one sweep at an over-large ``mu`` collapsed the enstrophy to zero while
running eight times faster).  Lower ``mu`` until the solve converges on its own.

Which route wins depends on whether the rank cap binds.  Where the flow is
genuinely low rank the cap never binds, no truncation happens, and the
projection is both more accurate and faster.  Where the cap binds -- the regime
the paper is about -- the projection solves and *then* truncates, while this
step searches for the best field on the rank-chi manifold, and it reproduces the
vortex-stretching peak several times more faithfully.
"""

import numpy as np

import tt
from tt import backend as bk
from tt.algs.amen import amen_solve

import qtt_ns as q


def _like(x, ref):
    """Move the TT vector ``x`` onto the backend/device that ``ref`` lives on."""
    core = tt.vector.to_list(ref)[0]
    backend = bk.backend_of(core)
    if backend.name == "numpy":
        return x
    return x.to(backend.name, device=bk.device_of(core))


def _fold(r, core):
    """``r @ core`` over the core's left bond, on any backend."""
    r0, n, r1 = (int(s) for s in core.shape)
    return (r @ core.reshape(r0, n * r1)).reshape(-1, n, r1)


# --- the extended (component x space) representation -------------------------

def stack(components):
    """Pack ``K`` TT fields into one train with a trailing mode-``K`` core."""
    K = len(components)
    out = None
    for i, comp in enumerate(components):
        sel = np.zeros((1, K, 1))
        sel[0, i, 0] = 1.0
        unit = _like(tt.vector.from_list([sel]), comp)
        term = tt.kron(comp, unit)
        out = term if out is None else out + term
    return out.round(1e-14)


def unstack(V, K, eps=1e-14):
    """Inverse of :func:`stack`: slice the component mode, fold the tail in."""
    cores = tt.vector.to_list(V)
    spatial, last = list(cores[:-1]), cores[-1]
    out = []
    for i in range(K):
        tail = last[:, i, :]                       # (r, r_last)
        comp = list(spatial)
        r0, n, r1 = (int(s) for s in comp[-1].shape)
        comp[-1] = (comp[-1].reshape(r0 * n, r1) @ tail).reshape(r0, n, -1)
        out.append(tt.vector.from_list(comp).round(eps))
    return out


def divergence_operator(ops, derivs):
    """``D`` on the extended space: ``K`` components in, one scalar field out."""
    D = None
    K = len(derivs)
    for i, Di in enumerate(derivs):
        row = np.zeros((1, 1, K, 1))
        row[0, 0, i, 0] = 1.0
        term = tt.kron(Di, tt.matrix.from_list([row]))
        D = term if D is None else D + term
    return D.round(1e-13)


def penalty_system(ops, derivs, dt, mu, eps=1e-12):
    """``A = mu D^T D + I/dt^2`` on the extended space (SPD)."""
    D = divergence_operator(ops, derivs)
    K = len(derivs)
    d = ops.d
    n = int(ops.Dx.n[0])
    eye_sp = tt.matrix.from_list([np.eye(n).reshape(1, n, n, 1)
                                  for _ in range(d)])
    eye = tt.kron(eye_sp, tt.matrix.from_list([np.eye(K).reshape(1, K, K, 1)]))
    A = ((D.T @ D) * mu + eye * (1.0 / dt ** 2)).round(eps)
    return A, D


# --- the variational step ----------------------------------------------------

def _feasible_start(d, n, K, chi, seed=0):
    modes = [n] * d + [K]
    prof = [1] * (len(modes) + 1)
    for k in range(len(modes)):                      # left feasibility
        prof[k + 1] = min(chi, prof[k] * modes[k])
    prof[len(modes)] = 1                             # the train must close
    for k in range(len(modes) - 1, 0, -1):           # right feasibility
        prof[k] = min(prof[k], prof[k + 1] * modes[k])
    cores = [np.random.default_rng(seed + k).standard_normal(
        (prof[k], modes[k], prof[k + 1])) for k in range(len(modes))]
    x = tt.vector.from_list(cores)
    return x * (1.0 / x.norm())


def _start_like(ref, d, n, K, chi, seed=0):
    """A feasible fixed-rank start on the backend/device of ``ref``."""
    return _like(_feasible_start(d, n, K, chi, seed=seed), ref)


def step_penalty(ops, comps, dt, nu, A, chi, mu=1e5, eps=1e-8,
                 solver="lobpcg", guess=None, derivs=None, nswp=5):
    """One variational step: minimize the paper's cost over the TT manifold.

    Returns ``(components, V_extended)``; pass the latter back as ``guess`` so
    the fixed-rank solve warm-starts from the previous step.
    """
    K = len(comps)
    derivs = derivs or ([ops.Dx, ops.Dy] if K == 2 else [ops.Dx, ops.Dy, ops.Dz])
    # R = V/dt - (V.grad)V + nu Lap V, componentwise
    rhs_comps = []
    for a in comps:
        terms = [[comps[j], tt.matvec(derivs[j], a)] for j in range(K)]
        adv = tt.hadamard_sum(terms, eps=eps, rmax=chi)
        r = (a * (1.0 / dt) - adv + tt.matvec(ops.Lap, a) * nu)
        rhs_comps.append(r.round(eps, rmax=chi))
    b = (stack(rhs_comps) * (1.0 / dt)).round(eps, rmax=chi)

    if solver == "lobpcg":
        x0 = guess if guess is not None else _start_like(
            comps[0], ops.d, int(ops.Dx.n[0]), K, chi)
        # with a warm start the field barely moves in a step, so a handful of
        # sweeps suffices; 40 was paying for convergence already in hand (the
        # same lesson the amen path taught: a warm start cut it to one sweep).
        V = tt.lobpcg_solve(A, b, x0, eps, nswp=nswp, verb=0, local_prec="c")
    else:
        V = amen_solve(A, b, guess, eps, nswp=20, verb=0, rmax=chi,
                       local_prec="c")
    return unstack(V, K, eps), V
