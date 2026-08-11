#!/usr/bin/env python
"""SIR epidemic on a network: the chemical master equation in TT [DS24].

    python examples/sir_network_cme.py            # chain of N=9, dense oracle
    python examples/sir_network_cme.py 32         # chain of N=32, TT only
    python examples/sir_network_cme.py 9 7        # ... and the threshold I*

An SIR epidemic on a contact network: person ``n`` is susceptible, infected
or recovered, an infected neighbour infects at rate ``beta`` per edge, an
infected person recovers at rate ``gamma``.  The probability of every one of
the ``3^N`` network states solves the master equation ``p' = A p``, which is
hopeless as a vector but short as a tensor: the generator is the exact sum
of ``2|E| + N`` Kronecker products of 3x3 matrices ([DS24] eq. 12),

    A = sum_{n, m~n} beta (J^T - I) diag(s)_n  diag(i)_m
      + sum_n       gamma (J^T - I) diag(i)_n,

with ``J`` the 3x3 shift and ``s = (1,0,0)``, ``i = (0,1,0)`` the indicator
vectors.  The observables never leave TT either: the infected count ``I(x)``
is an explicit rank-2 tensor ([DS24] eq. 22) and the exceedance indicator
``1[I(x) > I*]`` an explicit rank-(N - I* + 1) one (eq. 28), so
``P(I(t) > I*)`` -- the probability the epidemic exceeds a hospital-capacity
threshold -- is one TT dot product.  That is the point of the method: for
rare events SSA needs ``~ 1/P`` trajectories to see the event at all, while
the TT solution carries the whole distribution and reads the tail off
directly ([DS24] Fig. 4 makes this comparison at P ~ 1e-9).

Parameters follow the paper's experiments: ``beta = 1``, ``gamma = 0.3``,
the first node infected and the rest susceptible at ``t = 0``.

The propagator here is Crank-Nicolson with a warm-started ``amen_solve``
per step, with the probability renormalized after each step: a
solve-and-round scheme conserves ``sum p = 1`` only to the solver accuracy.
(The integrator that conserves it to machine precision *by construction* is
tAMEn -- ``docs/plans/tamen.md``; this example is its acceptance case.)

Oracles:

* at ``N <= 10`` the same Crank-Nicolson runs densely on the sparse
  ``3^N x 3^N`` generator assembled by ``scipy.sparse`` Kronecker products
  -- an independent path to the same distribution
  (``tests/test_examples.py`` pins the TT path against it);
* a vectorized Gillespie SSA gives the Monte-Carlo view of ``E[I(t)]`` and
  shows its noise floor on the exceedance tail.

References
----------
* S. Dolgov, D. Savostyanov, "Tensor product approach to modelling
  epidemics on networks", Applied Mathematics and Computation 460:128290,
  2024, arXiv:2209.03756 [DS24].  Reference implementation:
  github.com/savostyanov/ttsir (MATLAB; read for the operator and
  observable factorizations, eqs. 12/22/28, not copied).
* D. T. Gillespie, J. Phys. Chem. 81:2340-2361, 1977 -- the SSA.
"""

import sys
import time

import numpy as np

import tt
from tt.algs.amen import amen_solve

BETA = 1.0
GAMMA = 0.3

E3 = np.eye(3)
JT = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
DS = np.diag([1., 0., 0.])          # diag(s): susceptible
DI = np.diag([0., 1., 0.])          # diag(i): infected
VI = np.array([0., 1., 0.])         # the indicator vector i
VE = np.ones(3)


def chain_edges(N):
    """The linear chain: person n talks to n+1."""
    return [(k, k + 1) for k in range(N - 1)]


def _site_mpo(N, factors):
    """Kronecker product with given 3x3 ``factors`` at given sites."""
    cores = []
    for k in range(N):
        m = factors.get(k, E3)
        cores.append(np.ascontiguousarray(m[None, :, :, None]))
    return tt.matrix.from_list(cores)


def generator(N, edges, beta=BETA, gamma=GAMMA):
    """The CME generator, [DS24] eq. (12): exact, ``2|E| + N`` terms."""
    terms = []
    for (m, n) in edges:
        # m infects n and n infects m (undirected contact)
        terms.append(beta * _site_mpo(N, {n: (JT - E3) @ DS, m: DI}))
        terms.append(beta * _site_mpo(N, {m: (JT - E3) @ DS, n: DI}))
    for n_ in range(N):
        terms.append(gamma * _site_mpo(N, {n_: (JT - E3) @ DI}))
    A = terms[0]
    for m_ in terms[1:]:
        A = A + m_
    return A.round(1e-13)


def infected_count(N):
    """``[I(x)]`` as the explicit rank-2 TT of [DS24] eq. (22)."""
    cores = []
    for k in range(N):
        if k == 0:
            c = np.zeros((1, 3, 2))
            c[0, :, 0] = VE
            c[0, :, 1] = VI
        elif k == N - 1:
            c = np.zeros((2, 3, 1))
            c[0, :, 0] = VI
            c[1, :, 0] = VE
        else:
            c = np.zeros((2, 3, 2))
            c[0, :, 0] = VE
            c[0, :, 1] = VI
            c[1, :, 1] = VE
        cores.append(np.ascontiguousarray(c))
    return tt.vector.from_list(cores)


def exceedance(N, istar):
    """``[1(I(x) > I*)]`` as the explicit TT of [DS24] eq. (28), rank H+1.

    The cores count *non-infected* people: the automaton state is how many
    of ``e - i`` have been seen so far, capped at ``H = N - I*``; a state
    that stays below the cap means more than ``I*`` infected.
    """
    H = N - istar
    if H < 1:
        raise ValueError(f"threshold I*={istar} needs I* < N={N}")
    r = H + 1
    ei = VE - VI
    cores = []
    first = np.zeros((1, 3, r))
    first[0, :, 0] = VI
    if r > 1:
        first[0, :, 1] = ei
    cores.append(first)
    for _ in range(1, N - 1):
        c = np.zeros((r, 3, r))
        for a in range(r):
            c[a, :, a] = VI
            if a + 1 < r:
                c[a, :, a + 1] = ei
        cores.append(np.ascontiguousarray(c))
    # accept iff the total count of non-infected people is <= H - 1
    last = np.zeros((r, 3, 1))
    for a in range(r - 2):
        last[a, :, 0] = VE      # <= H-2 seen: the last symbol is free
    last[r - 2, :, 0] = VI      # H-1 seen: the last person must be infected
    cores.append(last)          # H seen: rejected (zero row)
    return tt.vector.from_list(cores)


def initial_state(N):
    """Node 1 infected, the rest susceptible: a deterministic rank-1 p(0)."""
    s0 = np.array([1., 0., 0.])
    cores = [VI if k == 0 else s0 for k in range(N)]
    return tt.vector.from_list([c.reshape(1, 3, 1) for c in cores])


def ssa(N, edges, T, nsamples, seed=0, beta=BETA, gamma=GAMMA, tgrid=None):
    """Plain per-trajectory Gillespie.

    Returns ``E[I]`` on ``tgrid`` (piecewise-constant trajectories sampled
    exactly) and the peak count ``max_t I`` per trajectory.
    """
    rng = np.random.default_rng(seed)
    nbr = [[] for _ in range(N)]
    for (m, n) in edges:
        nbr[m].append(n)
        nbr[n].append(m)
    tgrid = np.asarray(tgrid)
    total_counts = np.zeros(len(tgrid))
    peak = np.zeros(nsamples)
    for s_ in range(nsamples):
        state = np.zeros(N, dtype=np.int8)      # 0=s, 1=i, 2=r
        state[0] = 1
        t, jgrid, pk = 0.0, 0, 1
        while True:
            icount = int((state == 1).sum())
            pk = max(pk, icount)
            rates = np.empty(2 * N)
            for n_ in range(N):
                rates[n_] = (beta * sum(state[m_] == 1 for m_ in nbr[n_])
                             if state[n_] == 0 else 0.0)
                rates[N + n_] = gamma if state[n_] == 1 else 0.0
            tot = rates.sum()
            t_next = t + rng.exponential(1.0 / tot) if tot > 0 else np.inf
            # the state was `state` on [t, t_next): record grid points inside
            while jgrid < len(tgrid) and tgrid[jgrid] < min(t_next, T):
                total_counts[jgrid] += icount
                jgrid += 1
            if t_next >= T or tot == 0:
                break
            t = t_next
            choice = int(np.searchsorted(np.cumsum(rates),
                                         rng.random() * tot, side="right"))
            state[choice % N] = 1 if choice < N else 2
        peak[s_] = pk
    return total_counts / nsamples, peak


def run(N=9, istar=None, T=30.0, nsteps=300, eps_amen=1e-8, verbose=True):
    istar = istar if istar is not None else max(2, N - 2)
    edges = chain_edges(N)
    A = generator(N, edges)
    p = initial_state(N)
    Iw = infected_count(N)
    Xw = exceedance(N, istar)
    ones = tt.ones(3, N)

    tau = T / nsteps
    IN = tt.eye(3, N)
    Mp = (IN - (tau / 2.0) * A).round(1e-13)   # p' = +A p
    Mm = (IN + (tau / 2.0) * A).round(1e-13)

    if verbose:
        print(f"[DS24] SIR on a chain of N={N} (3^{N} = {3**N} states), "
              f"beta={BETA}, gamma={GAMMA}: {nsteps} CN steps to T={T}\n")
    checkpoints = []
    t0 = time.perf_counter()
    for k in range(nsteps):
        rhs = tt.matvec(Mm, p).round(1e-12)
        p = amen_solve(Mp, rhs, p, eps_amen, verb=0)
        p = p * (1.0 / float(tt.dot(ones, p)))
        if (k + 1) % max(1, nsteps // 30) == 0:
            checkpoints.append((
                (k + 1) * tau,
                float(tt.dot(Iw, p)),
                float(tt.dot(Xw, p)),
                int(max(p.r))))
    t_run = time.perf_counter() - t0
    if verbose:
        print(f"  propagation: {t_run:.1f} s "
              f"({t_run / nsteps * 1e3:.0f} ms/step), "
              f"final TT rank {max(p.r)}")
        tmid = checkpoints[len(checkpoints) // 3]
        print(f"  E[I] at t={tmid[0]:.1f}: {tmid[1]:.6f};  "
              f"P(I > {istar}) peak over t: "
              f"{max(c[2] for c in checkpoints):.3e}")
    return p, checkpoints, (A, Iw, Xw, ones)


def rare_event_ladder(N, thresholds, T=30.0, nsteps=300, eps_amen=1e-10):
    """``max_t P(I(t) > I*)`` for a ladder of thresholds, one propagation.

    The part SSA cannot do: each rung further down the ladder would need
    ``~ 1/P`` trajectories just to see one event, while here every rung is
    one dot product with the explicit indicator TT of [DS24] eq. (28)
    against the same propagated distribution.

    A rung is only as good as the solver tolerance lets it be: the tail is
    a linear functional of ``p``, and a global error of ``eps`` in the
    distribution can contaminate any probability near or below ``eps``.
    Measured on the N=32 chain: refining ``eps_amen`` 1e-8 -> 1e-10 moved
    the ~1e-8 rung by 7% and the ~1e-12 rung by 37%.  Hence the 1e-10
    default here, and the rule: trust a rung when it is stable under an
    ``eps`` refinement, not because it printed.  ([DS24] hits the same wall
    and propagates ``sqrt(p)`` to square the reachable depth; that variant
    is not implemented here.)
    """
    edges = chain_edges(N)
    A = generator(N, edges)
    p = initial_state(N)
    ones = tt.ones(3, N)
    ws = {i: exceedance(N, i) for i in thresholds}
    peaks = {i: 0.0 for i in thresholds}
    tau = T / nsteps
    IN = tt.eye(3, N)
    Mp = (IN - (tau / 2.0) * A).round(1e-13)
    Mm = (IN + (tau / 2.0) * A).round(1e-13)
    for k in range(nsteps):
        rhs = tt.matvec(Mm, p).round(1e-12)
        p = amen_solve(Mp, rhs, p, eps_amen, verb=0)
        p = p * (1.0 / float(tt.dot(ones, p)))
        for i in thresholds:
            peaks[i] = max(peaks[i], float(tt.dot(ws[i], p)))
    print(f"\n  rare-event ladder (chain N={N}, max_t P(I(t) > I*)):")
    for i in thresholds:
        need = int(np.ceil(100.0 / max(peaks[i], 1e-300)))
        print(f"    I* = {i:3d}:  P = {peaks[i]:.3e}   "
              f"(SSA would need ~{need:.0e} trajectories for 10% accuracy)")
    return peaks


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    istar = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(N, istar)
    if N >= 16:
        rare_event_ladder(N, [N // 4, N // 3, N // 2, 2 * N // 3])
