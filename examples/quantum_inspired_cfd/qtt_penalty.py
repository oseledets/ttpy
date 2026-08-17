"""The paper's variational step: incompressibility by penalty, at fixed rank.

``qtt_ns.py`` enforces ``div V = 0`` by a Chorin projection -- a pressure Poisson
solve per stage.  Gourianov et al. instead advance the velocity by *minimizing*
(their eq. 8)

    Theta(V*) = mu ||div V*||^2 + || (V* - V)/dt + (V.grad)V - nu Lap V ||^2

over the MPS manifold.  Written out, ``Theta`` is a quadratic form: with

    R = V/dt - (V.grad)V + nu Lap V,

its stationarity condition is the SPD linear system

    (mu D^T D + I/dt^2) V* = R/dt,

so a step is one SPD solve and no Poisson equation and no projection appear at
all.  Two things make this the natural formulation here:

* the penalty couples the velocity components, so the unknown is the whole
  field -- carried as one tensor train with an extra *component* mode of size
  ``K`` (2 in 2D, 3 in 3D) appended to the z-order spatial modes;
* the system is SPD, so at a **fixed** bond dimension it is exactly the
  fixed-rank energy minimization of ``tt.lobpcg_solve`` -- the flow never leaves
  the bounded-rank manifold, which is the paper's whole premise.  Passing
  ``solver="amen"`` instead lets the rank adapt, for reference runs.

Which route wins depends on whether the rank cap **binds**, and that is the
distinction that matters.

*When the flow is genuinely low rank* (Taylor-Green is rank 4, ABC rank 8) the
cap never binds, no truncation happens, and the comparison is just "which solver
is more accurate": the projection wins outright.  Taylor-Green, 64^2, rank cap
30, 30 steps::

    penalty, mu=1e7    |E - analytic| 4.7e-05   div 1.5e-04   0.32 s/step
    Chorin projection  |E - analytic| 8.7e-09   div 2.8e-07   0.02 s/step

*When the cap binds* -- the regime the paper is about -- the two do something
structurally different: the projection solves and *then* truncates, so the
truncation sits outside the scheme, while the penalty searches for the best
field **on** the rank-chi manifold.  3D Taylor-Green at Re=400, 32^3, run to
t=3 (full rank is 64), measured against the full-rank reference whose enstrophy
peaks at ``Z/Z0 = 2.081``::

    chi=16  projection + truncation    Zpeak/Z0 2.380   (+14%)
    chi=16  penalty / fixed-rank ALS   Zpeak/Z0 2.024   (-2.7%)
    chi= 8  projection + truncation    Zpeak/Z0 2.409   (+16%)

So at a binding cap the variational step reproduces the vortex-stretching peak
several times more faithfully -- optimizing on the manifold beats truncating
after the fact, which is the paper's premise.

``mu`` must be scaled, not guessed.  The penalty term ``mu ||D||^2`` competes
with the data term ``1/dt^2``; with ``h=0.196, dt=0.0295`` those are ``mu*78``
and ``1153``, so the balance sits near ``mu ~ 15`` and the useful window is a
decade or two above it.  The sweep at ``chi=16`` (reference 2.081)::

    mu=1e1   Zpeak/Z0 1.334   div 3.4e+01
    mu=1e2   Zpeak/Z0 1.900   div 6.1e+00
    mu=1e3   Zpeak/Z0 2.024   div 1.5e+00
    mu=1e4   Zpeak/Z0 1.867   div 5.1e-01
    mu=1e5   Zpeak/Z0 1.014   div  --      (over-damped: the solve minimizes
                                            divergence and ignores momentum)

Raising ``mu`` buys incompressibility and costs conditioning: too small and the
field drifts off the divergence-free manifold, too large and the momentum term
is swamped, the fixed-rank solve stalls, and the flow freezes.
"""

import numpy as np

import tt
from tt.algs.amen import amen_solve

import qtt_ns as q


# --- the extended (component x space) representation -------------------------

def stack(components):
    """Pack ``K`` TT fields into one train with a trailing mode-``K`` core."""
    K = len(components)
    out = None
    for i, comp in enumerate(components):
        sel = np.zeros((1, K, 1))
        sel[0, i, 0] = 1.0
        term = tt.kron(comp, tt.vector.from_list([sel]))
        out = term if out is None else out + term
    return out.round(1e-14)


def unstack(V, K, eps=1e-14):
    """Inverse of :func:`stack`: slice the component mode, fold the tail in."""
    cores = tt.vector.to_list(V)
    spatial, last = [c.copy() for c in cores[:-1]], cores[-1]
    out = []
    for i in range(K):
        tail = last[:, i, :]                       # (r, r_last)
        comp = [c.copy() for c in spatial]
        comp[-1] = np.tensordot(comp[-1], tail, axes=(2, 0))
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


def step_penalty(ops, comps, dt, nu, A, chi, mu=1e5, eps=1e-8,
                 solver="lobpcg", guess=None, derivs=None):
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
        x0 = guess if guess is not None else _feasible_start(
            ops.d, int(ops.Dx.n[0]), K, chi)
        V = tt.lobpcg_solve(A, b, x0, eps, nswp=40, verb=0, local_prec="c")
    else:
        V = amen_solve(A, b, guess, eps, nswp=20, verb=0, rmax=chi,
                       local_prec="c")
    return unstack(V, K, eps), V
