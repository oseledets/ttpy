"""Spin-chain Hamiltonians in TT form, with exact references.

These are the first of the "hard test problems" of ``docs/plans/eigenvalues.md``:
operators whose ground-state energy is known in closed form or from a dense
``eigh``, so an eigensolver can be checked against something that is not itself
an eigensolver of ours.

Conventions, as everywhere in ttpy2: a TT-matrix core is ``(R, i, j, R)`` with
the **row** index first, and mode 1 is the fastest index.  An MPO is written as
a matrix ``W`` of 2x2 blocks; with open boundary conditions the first core takes
the last row of ``W`` and the last core its first column.
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


def dense(A):
    """The TT-matrix as a dense square array (only for small ``d``)."""
    n = int(np.prod(A.n))
    return np.asarray(A.full()).reshape(n, n)
