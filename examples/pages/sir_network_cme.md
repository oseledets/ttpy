# SIR epidemic on a network: the chemical master equation in TT ([DS24])

The full $3^N$-state probability distribution of an SIR epidemic on a
contact network, propagated as a tensor train, with rare-event tail
probabilities down to $\sim 10^{-12}$ read off as single dot products —
where Gillespie SSA would need $\sim 5 \cdot 10^{13}$ trajectories to see
one event.

![SIR mean and tail on a chain of N=9, and the N=32 rare-event ladder](../../docs/media/sir_rare_events.png)

## The problem

Person $n$ on a contact network is susceptible, infected or recovered; an
infected neighbour infects at rate $\beta$ per edge, an infected person
recovers at rate $\gamma$.  The probability $p(x, t)$ of every one of the
$3^N$ network states solves the chemical master equation

$$
\frac{dp}{dt} = A  p,
$$

which is hopeless as a vector but short as a tensor: the generator is the
*exact* sum of $2|E| + N$ Kronecker products of $3 \times 3$ matrices
([DS24] eq. (12)),

$$
A \quad =\quad  \sum_{n,\quad  m \sim n} \beta  \bigl(J^{T} - I\bigr) \mathrm{diag}(s)\big|_n  \otimes  \mathrm{diag}(i)\big|_m
\quad +\quad  \sum_{n} \gamma  \bigl(J^{T} - I\bigr) \mathrm{diag}(i)\big|_n,
$$

with $J$ the $3 \times 3$ shift and $s = (1, 0, 0)$, $i = (0, 1, 0)$ the
indicator vectors of the susceptible and infected states (all sites not
named carry the identity).  The observables never leave TT either:

* the infected count $I(x) = \sum_n \mathbf{1}[x_n = \mathrm{i}]$ is an
  explicit **rank-2** tensor train ([DS24] eq. (22));
* the exceedance indicator $\mathbf{1}[I(x) \gt I^*]$ is an explicit
  **rank-$(N - I^* + 1)$** one ([DS24] eq. (28)) — a counting automaton
  written directly in TT cores.

So $P(I(t) \gt I^*)$ — the probability the epidemic exceeds a
hospital-capacity threshold — is one TT dot product.  That is the point of
the method: for rare events SSA needs $\sim 1/P$ trajectories to see the
event at all, while the TT solution carries the whole distribution and
reads the tail off directly ([DS24] Fig. 4 makes this comparison at
$P \sim 10^{-9}$).

Parameters follow the paper's experiments: $\beta = 1$, $\gamma = 0.3$, the
first node infected and the rest susceptible at $t = 0$, on a linear chain.

## The code, walked through

**Every Kronecker term is one rank-1 TT-matrix.**  `_site_mpo` places given
$3 \times 3$ factors at given sites and the identity everywhere else; the
generator is then a literal transcription of eq. (12), rounded losslessly:

```python
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
```

**The exceedance indicator is a counting automaton written in cores.**  The
cores count *non-infected* people: the automaton state is how many of
$e - i$ have been seen so far, capped at $H = N - I^*$; a state that stays
below the cap means more than $I^*$ infected.  The delicate part is the
acceptance condition in the last core — exactly the place where an
off-by-one at $I = I^*$ hides (this one was caught by the brute-force
oracle in the tests and fixed; the commit records it):

```python
    # accept iff the total count of non-infected people is <= H - 1
    last = np.zeros((r, 3, 1))
    for a in range(r - 2):
        last[a, :, 0] = VE      # <= H-2 seen: the last symbol is free
    last[r - 2, :, 0] = VI      # H-1 seen: the last person must be infected
    cores.append(last)          # H seen: rejected (zero row)
```

**Propagation is Crank–Nicolson with a warm-started `amen_solve` per
step**, with the probability renormalized after each step — a
solve-and-round scheme conserves $\sum p = 1$ only to the solver accuracy
(the integrator that conserves it to machine precision *by construction* is
tAMEn; see "Why believe it"):

```python
    for k in range(nsteps):
        rhs = tt.matvec(Mm, p).round(1e-12)
        p = amen_solve(Mp, rhs, p, eps_amen, verb=0)
        p = p * (1.0 / float(tt.dot(ones, p)))
```

**The rare-event ladder** propagates once and evaluates a whole ladder of
thresholds against the same distribution — each rung one dot product with
the explicit indicator TT.  Its docstring carries the honesty rule about
`eps`, measured, not assumed:

> A rung is only as good as the solver tolerance lets it be: the tail is a
> linear functional of $p$, and a global error of $\varepsilon$ in the
> distribution can contaminate any probability near or below
> $\varepsilon$.  Measured on the $N = 32$ chain: refining `eps_amen`
> $10^{-8} \to 10^{-10}$ moved the $\sim 10^{-8}$ rung by 7% and the
> $\sim 10^{-12}$ rung by 37%.  Hence the $10^{-10}$ default here, and the
> rule: trust a rung when it is stable under an `eps` refinement, not
> because it printed.  ([DS24] hits the same wall and propagates
> $\sqrt{p}$ to square the reachable depth; that variant is not
> implemented here.)

## What comes out

* **Chain of $N = 9$** ($3^9 = 19683$ states): 300 CN steps to $T = 30$ in
  **3.3 s**, final TT rank **7** — small enough that the dense oracle runs
  alongside.
* **Chain of $N = 32$** ($3^{32} \approx 1.9 \cdot 10^{15}$ states — the
  vector does not fit in any memory): the same 300 steps in **13 s**, final
  TT rank **11**.
* **The rare-event ladder at $N = 32$** (thresholds
  $I^* = 8, 10, 16, 21$): $\max_t P(I(t) \gt I^*)$ spans $\sim 10^{-3}$
  down to $\sim 10^{-12}$ — the deepest rung is where SSA would need
  $\sim 5 \cdot 10^{13}$ trajectories for 10% accuracy, and it is reported
  under the stability rule above ($10^{-8} \to 10^{-10}$ refinement moved
  it by 37%, so the default `eps_amen` for the ladder is $10^{-10}$).

## Why believe it

* **Brute force over every state.**
  `tests/test_examples.py::test_sir_cme_matches_brute_force_at_small_n`
  checks the $N = 6$ machinery against all $3^6$ states: the TT generator
  applied to a random vector against a state-by-state loop over every
  transition of [DS24] eq. (2) (parity $10^{-12}$); the explicit indicator
  TTs against direct counting — `If[flat_f(s)] == ninf` and
  `Xf[flat_f(s)] == (1.0 if ninf > istar else 0.0)` for *every* state $s$,
  which is the check that caught the off-by-one in the exceedance
  automaton; and the Crank–Nicolson TT run against the same scheme on the
  `scipy.sparse` generator (distribution parity $10^{-6}$).
* **Gillespie SSA.**  A plain per-trajectory SSA in the example gives the
  Monte-Carlo view of $E[I(t)]$ — agreement within Monte-Carlo error — and
  shows its noise floor on the exceedance tail, which is precisely what the
  TT solution does not have.
* **A probability-conserving integrator as the end state.**  This example
  is the acceptance case of tAMEn (`docs/plans/tamen.md`), the
  spectral-in-time AMEn integrator that conserves linear invariants — here
  $\sum_x p(x) = 1$ — to machine precision *independently of the TT
  truncation accuracy*, which no solve-and-round scheme does.  Measured on
  this very problem against a dense-`expm` oracle ($N = 7$ chain, $T = 30$),
  tAMEn holds the invariant to a drift of $\sim 3\times 10^{-14}$ at error
  $\sim 10^{-6}$, where a fixed-rank KSL step drifts by $\sim 10^{-4}$ and
  stalls at the $\sim 10^{-3}$ of its fixed-rank modelling error — tAMEn is
  the integrator suited to this class of conservation-law problems.

## Run it

```
python examples/sir_network_cme.py            # chain of N=9, dense oracle
python examples/sir_network_cme.py 32         # chain of N=32, TT only (+ the ladder)
python examples/sir_network_cme.py 9 7        # ... and the threshold I*
```

The acceptance version is
`tests/test_examples.py::test_sir_cme_matches_brute_force_at_small_n`.

## References

* S. Dolgov, D. Savostyanov, "Tensor product approach to modelling
  epidemics on networks", Applied Mathematics and Computation 460:128290,
  2024, arXiv:2209.03756 [DS24].  Reference implementation:
  github.com/savostyanov/ttsir (MATLAB; read for the operator and
  observable factorizations, eqs. 12/22/28, not copied).
* D. T. Gillespie, "Exact stochastic simulation of coupled chemical
  reactions", J. Phys. Chem. 81:2340–2361, 1977 — the SSA.
