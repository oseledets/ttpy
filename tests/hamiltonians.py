"""Hamiltonians in TT form, with references that do not come from us.

These are the "hard test problems" of ``docs/plans/eigenvalues.md``: operators
whose ground-state energy is known in closed form, from a dense ``eigh``, or
from an independent sparse eigensolver, so an eigensolver can be checked
against something that is not itself an eigensolver of ours.

Conventions, as everywhere in ttpy2: a TT-matrix core is ``(R, i, j, R)`` with
the **row** index first, and mode 1 is the fastest index.  An MPO is written as
a matrix ``W`` of ``n x n`` blocks; with open boundary conditions the first core
takes the last row of ``W`` and the last core its first column.  Concretely the
row index ``R - 1`` is the "nothing placed yet" state and ``0`` is the "term
finished" state, so a two-site term ``a_i b_j`` is a path
``R-1 --a--> s --...--> s --b--> 0``.

Two families live here:

* **spin chains** (:func:`heisenberg`, :func:`tfim`) on ``n = 2``;
* **vibrational Hamiltonians** in a harmonic-oscillator product basis
  (:func:`coupled_oscillator`, :func:`henon_heiles`), the standard TT
  eigenvalue benchmarks.  Units are dimensionless throughout: ``hbar = 1``,
  mass-weighted normal coordinates, and the reference oscillator has unit
  frequency, so ``(1/2)(-d^2/dq^2 + q^2)`` is exactly ``diag(n + 1/2)``.
  Any published number in other units (cm^-1, hartree) has to be converted
  before it can be compared -- see ``docs/BENCHMARKS.md``.
"""

from __future__ import annotations

import numpy as np

import tt

I2 = np.eye(2)
SP = np.array([[0.0, 1.0], [0.0, 0.0]])       # S^+
SM = np.array([[0.0, 0.0], [1.0, 0.0]])       # S^-
SZ = np.array([[0.5, 0.0], [0.0, -0.5]])      # S^z, spin-1/2
PX = np.array([[0.0, 1.0], [1.0, 0.0]])       # Pauli sigma^x
PZ = np.array([[1.0, 0.0], [0.0, -1.0]])      # Pauli sigma^z


def _mpo(W, d):
    """``W`` of shape ``(R, R, 2, 2)`` -> a ``tt.matrix`` on ``d`` sites."""
    R = W.shape[0]
    cores = []
    for k in range(d):
        c = np.transpose(W, (0, 2, 3, 1))     # (R, i, j, R)
        if k == 0:
            c = c[R - 1:R]
        if k == d - 1:
            c = c[:, :, :, 0:1]
        cores.append(np.ascontiguousarray(c))
    return tt.matrix.from_list(cores)


def heisenberg(d, jz=1.0, jxy=1.0):
    """``H = sum_i [jxy/2 (S+_i S-_{i+1} + h.c.) + jz S^z_i S^z_{i+1}]``.

    Open boundary conditions, TT-rank 5, real.  No closed form for the open
    chain -- use :func:`dense` and ``numpy.linalg.eigh`` as the oracle.
    """
    W = np.zeros((5, 5, 2, 2))
    W[0, 0] = I2
    W[1, 0] = 0.5 * jxy * SM
    W[2, 0] = 0.5 * jxy * SP
    W[3, 0] = jz * SZ
    W[4, 1] = SP
    W[4, 2] = SM
    W[4, 3] = SZ
    W[4, 4] = I2
    return _mpo(W, d)


def tfim(d, g=1.0):
    """``H = -sum_i sz_i sz_{i+1} - g sum_i sx_i`` (Pauli), open bc, TT-rank 3."""
    W = np.zeros((3, 3, 2, 2))
    W[0, 0] = I2
    W[1, 0] = -PZ
    W[2, 0] = -g * PX
    W[2, 1] = PZ
    W[2, 2] = I2
    return _mpo(W, d)


def tfim_critical_ground_energy(d):
    """Exact ground-state energy of :func:`tfim` at ``g = 1``, open chain.

    ``E_0(L) = 1 - 1 / sin(pi / (2(2L+1)))``.  Checked against dense ``eigh``
    to 5e-15 at ``L = 4, 8, 10, 12``; :func:`heisenberg` has no such formula,
    which is why both are here -- one gives an oracle independent of LAPACK,
    the other an oracle independent of a closed form.
    """
    return 1.0 - 1.0 / np.sin(np.pi / (2.0 * (2.0 * d + 1.0)))


def heisenberg_bulk_energy_per_site():
    """``1/4 - ln 2``: the Bethe-ansatz energy per site of the infinite chain.

    Hulthen 1938, for exactly the normalization of :func:`heisenberg` with
    ``jz = jxy = 1`` (spin-1/2 operators, ``H = sum_i S_i . S_{i+1}``).  This is
    an **asymptotic** reference: it is not the energy of any finite chain, and
    the approach of the open chain to it carries a surface term of order ``1/L``
    and a logarithmic correction on top.  The honest use is to take the
    *difference* of two chain lengths, which cancels the surface term.
    """
    return 0.25 - np.log(2.0)


def dense(A):
    """The TT-matrix as a dense square array (only for small ``d``)."""
    n = int(np.prod(A.n))
    return np.asarray(A.full()).reshape(n, n)


# --- vibrational Hamiltonians in a harmonic-oscillator product basis ---------

def _mpo_sites(Ws):
    """MPO from a per-site list of blocks ``W[k]`` of shape ``(R, R, n, n)``."""
    cores, d = [], len(Ws)
    for k, W in enumerate(Ws):
        R = W.shape[0]
        c = np.transpose(np.asarray(W, float), (0, 2, 3, 1))   # (R, i, j, R)
        if k == 0:
            c = c[R - 1:R]
        if k == d - 1:
            c = c[:, :, :, 0:1]
        cores.append(np.ascontiguousarray(c))
    return tt.matrix.from_list(cores)


def ho_operators(n, pad=6):
    """``(N, Q, Q2, Q3)`` in the first ``n`` harmonic-oscillator eigenstates.

    ``N = diag(k + 1/2)`` is the matrix of ``(1/2)(-d^2/dq^2 + q^2)``, exactly
    diagonal.  ``Q[k, k+1] = sqrt((k+1)/2)`` is the matrix of ``q``.

    ``Q2`` and ``Q3`` are the **Galerkin** matrices ``<i|q^2|j>``, ``<i|q^3|j>``,
    not the powers of the truncated ``Q``: the exact matrix elements need
    intermediate states above the basis, so the products are formed at size
    ``n + pad`` and then cut.  Squaring the truncated ``Q`` instead loses
    ``<n-1|q^2|n-1>`` by ``n/2``, which is a percent-level error on the top
    basis functions and silently changes the operator being diagonalized.
    """
    n, m = int(n), int(n) + int(pad)
    k = np.arange(m, dtype=float)
    q = np.diag(np.sqrt((k[:-1] + 1.0) / 2.0), 1)
    q = q + q.T
    q2, q3 = q @ q, q @ q @ q
    return (np.diag(k[:n] + 0.5), q[:n, :n], q2[:n, :n], q3[:n, :n])


def coupled_oscillator(d, n=15, alpha=0.1, omega=None):
    """Bilinearly coupled oscillators, the benchmark of [RO16] section V.1.

    ``H = sum_i (w_i/2)(-d^2/dq_i^2 + q_i^2) + alpha sum_{i<j} q_i q_j``
    with ``w_j = sqrt(j/2)`` (``j = 1..d``) and ``alpha = 0.1``, in the
    ``n``-function harmonic-oscillator product basis.  **All** pairs are
    coupled, not only neighbours, which is why the MPO needs a carry state; its
    TT rank is 3 regardless of ``d``.

    Rakhuba & Oseledets, *Calculating vibrational spectra of molecules using
    tensor train decomposition*, J. Chem. Phys. 145:124101 (2016),
    arXiv:1605.08422, section V.1.  Its ground-state energy is analytic --
    see :func:`coupled_oscillator_exact_levels`.
    """
    d = int(d)
    w = np.sqrt(np.arange(1, d + 1) / 2.0) if omega is None else np.asarray(omega, float)
    N, Q, _, _ = ho_operators(n)
    I = np.eye(int(n))
    Ws = []
    for k in range(d):
        W = np.zeros((3, 3, int(n), int(n)))
        W[2, 2] = I                 # nothing placed yet
        W[0, 0] = I                 # term finished
        W[1, 1] = I                 # one q placed, waiting for its partner
        W[2, 0] = w[k] * N          # the on-site harmonic term
        W[2, 1] = Q                 # place q_i
        W[1, 0] = alpha * Q         # place q_j, j > i
        Ws.append(W)
    return _mpo_sites(Ws)


def coupled_oscillator_exact_levels(d, alpha=0.1, omega=None):
    """Normal-mode frequencies of :func:`coupled_oscillator`, in closed form.

    The Hamiltonian is quadratic: ``H = (1/2) p^T A p + (1/2) q^T B q`` with
    ``A = diag(w)`` and ``B = diag(w) + alpha (J - I)``.  Its exact spectrum is
    ``sum_k (m_k + 1/2) Omega_k`` with ``Omega_k^2`` the eigenvalues of ``A B``,
    computed here from the symmetric ``A^{1/2} B A^{1/2}``.  The ground state is
    ``(1/2) sum_k Omega_k``.

    This is a ``d x d`` dense eigenproblem: an oracle for a ``n**d`` quantum
    problem that never touches a tensor format.  It is the reference for the
    **continuous** operator; a run in a finite ``n``-function basis carries a
    basis-truncation error on top, which is why ``alpha = 0`` (where the basis
    is exact) is measured as well.

    Returns:
        ``(Omega, E0)`` -- the ``d`` normal-mode frequencies, ascending, and the
        zero-point energy.
    """
    d = int(d)
    w = np.sqrt(np.arange(1, d + 1) / 2.0) if omega is None else np.asarray(omega, float)
    B = np.diag(w) + alpha * (np.ones((d, d)) - np.eye(d))
    s = np.diag(np.sqrt(w))
    ev = np.linalg.eigvalsh(s @ B @ s)
    if np.min(ev) <= 0:
        raise ValueError(f"the quadratic form is not positive definite: "
                         f"smallest eigenvalue {np.min(ev):.3E}")
    Omega = np.sqrt(np.sort(ev))
    return Omega, 0.5 * float(np.sum(Omega))


HENON_HEILES_LAMBDA = 0.111803


def henon_heiles(d, n=15, lam=HENON_HEILES_LAMBDA):
    """The Henon-Heiles vibrational Hamiltonian, MPO of TT rank 3.

    ``H = sum_{i=1}^{d} (1/2)(-d^2/dq_i^2 + q_i^2)
          + lam * sum_{i=1}^{d-1} (q_i^2 q_{i+1} - q_{i+1}^3 / 3)``

    with ``lam = 0.111803`` -- the value used throughout the MCTDH / DVR /
    tensor-train literature (``docs/plans/eigenvalues.md`` section 9.4).  The
    coupling is **nearest-neighbour**, the cubic on-site term sits on sites
    ``2..d``, and everything is in dimensionless oscillator units.

    Beware: other papers write the same model with an overall ``lam`` in front
    of both terms, or with ``sum_{i<j}``.  The convention above is the one this
    function implements and the only one its references are valid for.
    """
    d = int(d)
    N, Q, Q2, Q3 = ho_operators(n)
    I = np.eye(int(n))
    Ws = []
    for k in range(d):
        W = np.zeros((3, 3, int(n), int(n)))
        W[2, 2] = I
        W[0, 0] = I
        W[2, 0] = N - (lam / 3.0) * Q3 if k > 0 else N
        W[2, 1] = lam * Q2            # place q_i^2 ...
        W[1, 0] = Q                   # ... and q_{i+1} right after it
        Ws.append(W)
    return _mpo_sites(Ws)


def henon_heiles_sparse(d, n=15, lam=HENON_HEILES_LAMBDA):
    """The same operator as a ``scipy.sparse`` matrix -- the independent oracle.

    Assembled from Kronecker products of the one-mode matrices, with no tensor
    format anywhere, so ``scipy.sparse.linalg.eigsh`` on it is a reference that
    shares only :func:`ho_operators` with the TT path.  Costs ``n**d`` states:
    usable to ``d = 5`` at ``n = 15``, not beyond.

    The index order matches ``tt.matrix.full()``: mode 1 is the fastest index,
    so the Kronecker product is taken from the last mode to the first.
    """
    import scipy.sparse as sp

    d, n = int(d), int(n)
    N, Q, Q2, Q3 = (sp.csr_matrix(m) for m in ho_operators(n))
    I = sp.identity(n, format="csr")

    def embed(ops):
        """``ops[k]`` acting on mode ``k``, mode 1 fastest."""
        out = ops[d - 1]
        for k in range(d - 2, -1, -1):
            out = sp.kron(out, ops[k], format="csr")
        return out

    H = sp.csr_matrix((n ** d, n ** d))
    for k in range(d):
        ops = [I] * d
        ops[k] = N - (lam / 3.0) * Q3 if k > 0 else N
        H = H + embed(ops)
    for k in range(d - 1):
        ops = [I] * d
        ops[k], ops[k + 1] = lam * Q2, Q
        H = H + embed(ops)
    return H.tocsr()
