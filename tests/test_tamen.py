"""tAMEn against external truth: dense propagators, invariants, convergence.

Every oracle here is independent of the solver: ``scipy.linalg.expm`` on the
full discrete operator, the machine-precision conservation contract, and the
spectral convergence law of the Chebyshev collocation.
"""

import warnings

import numpy as np
import pytest
import scipy.linalg as sla

import tt
from tt.algs.tamen import cheb_interior, tamen

E3 = np.eye(3)
JT = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
DS = np.diag([1., 0., 0.])
DI = np.diag([0., 1., 0.])


def _sir_chain(N, beta=1.0, gamma=0.3):
    """The [DS24] SIR generator on a chain (see examples/sir_network_cme.py)."""
    def site(factors):
        cores = []
        for k in range(N):
            m = factors.get(k, E3)
            cores.append(np.ascontiguousarray(m[None, :, :, None]))
        return tt.matrix.from_list(cores)

    terms = []
    for k in range(N - 1):
        terms.append(beta * site({k + 1: (JT - E3) @ DS, k: DI}))
        terms.append(beta * site({k: (JT - E3) @ DS, k + 1: DI}))
    for k in range(N):
        terms.append(gamma * site({k: (JT - E3) @ DI}))
    A = terms[0]
    for m in terms[1:]:
        A = A + m
    return A.round(1e-13)


def _delta_state(N):
    s0 = np.array([1., 0., 0.])
    i0 = np.array([0., 1., 0.])
    return tt.vector.from_list(
        [(i0 if k == 0 else s0).reshape(1, 3, 1) for k in range(N)])


def test_chebyshev_block_differentiates_polynomials_exactly():
    """The interior block on J+1 nodes is exact for degree <= J."""
    t, S, se = cheb_interior(9, 2.5)
    for k in (1, 3, 7, 9):
        p = t ** k
        # collocation of p' with p(0) = 0: S p - se * 0
        assert np.abs(S @ p - k * t ** (k - 1)).max() < 1e-10 * (2.5 ** k)
    # se = S @ ones exactly (the derivative of a constant)
    assert np.abs(S @ np.ones(len(t)) - se).max() < 1e-12


def test_tamen_matches_dense_expm_on_a_generic_ode():
    """d=3 random dissipative operator: the answer is expm, to eps."""
    rng = np.random.default_rng(0)
    n, d = 4, 3
    cores = [rng.standard_normal((1 if k == 0 else 2, n, n,
                                  1 if k == d - 1 else 2)) for k in range(d)]
    B = tt.matrix.from_list(cores)
    A = ((B + B.T) * 0.3 - 1.2 * tt.eye(n, d)).round(1e-13)
    x0 = tt.rand([n] * d, r=2)
    x0 = x0 * (1.0 / x0.norm())
    T = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = tamen(A, x0, T, 1e-8, J=10)
    ref = sla.expm(T * np.asarray(A.full())) @ np.asarray(
        x0.full()).flatten("F")
    got = np.asarray(x.full()).flatten("F")
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6


def test_tamen_matches_dense_expm_on_periodic_convection():
    """The paper's convection example (its eq. 24) at oracle size.

    2D periodic transport with central differences is skew-symmetric; the
    oracle is ``expm`` of the full 1024x1024 discrete operator, so this
    measures the *time* integrator alone, with no spatial-dispersion
    excuses.
    """
    n = 32
    h = 20.0 / n
    grad = np.zeros((n, n))
    for i in range(n):
        grad[i, (i + 1) % n] = 1.0
        grad[i, (i - 1) % n] = -1.0
    grad /= 2.0 * h
    I = np.eye(n)
    A = (tt.matrix.from_list([grad[None, :, :, None], I[None, :, :, None]])
         + tt.matrix.from_list([I[None, :, :, None],
                                grad[None, :, :, None]])).round(1e-13)
    q = -10.0 + h * np.arange(n)
    g = np.exp(-q ** 2)
    x0 = tt.vector.from_list([g.reshape(1, n, 1), g.reshape(1, n, 1)])
    T = 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = tamen(A, x0, T, 1e-8, J=12)
    ref = sla.expm(T * np.asarray(A.full())) @ np.asarray(
        x0.full()).flatten("F")
    got = np.asarray(x.full()).flatten("F")
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-6
    # skew-symmetric flow: the 2-norm may drift only at the eps level
    assert abs(float(x.norm()) - float(x0.norm())) < 1e-6 * float(x0.norm())


def test_tamen_conserves_probability_at_crude_accuracy():
    """The property the method exists for ([Dolgov 2019] Sec. 3.4).

    On the SIR master equation at eps = 1e-2 the *solution* is only
    2-digits accurate, but sum(p) must still hold to machine precision:
    conservation does not degrade with the TT truncation.
    """
    N = 7
    A = _sir_chain(N)
    p0 = _delta_state(N)
    ones = tt.ones(3, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, hist = tamen(A, p0, 20.0, 1e-2, J=8, invariants=[ones],
                        return_history=True)
    assert hist.invariant_drift < 1e-12
    assert abs(float(tt.dot(ones, p)) - 1.0) < 1e-12
    # and the same at a tight accuracy, against the dense answer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2, hist2 = tamen(A, p0, 20.0, 1e-6, J=10, invariants=[ones],
                          return_history=True)
    assert hist2.invariant_drift < 1e-12
    ref = sla.expm(20.0 * np.asarray(A.full())) @ np.asarray(
        p0.full()).flatten("F")
    got = np.asarray(p2.full()).flatten("F")
    assert np.linalg.norm(got - ref) < 1e-4 * np.linalg.norm(ref)


def test_tamen_rejects_a_non_invariant():
    """A c with c^T A != 0 must be refused, not silently 'conserved'."""
    N = 5
    A = _sir_chain(N)
    bad = tt.rand([3] * N, r=2)
    with pytest.raises(ValueError, match="invariant"):
        tamen(A, _delta_state(N), 1.0, 1e-4, invariants=[bad])


def test_chebyshev_convergence_is_spectral_in_j():
    """One interval, full basis: the collocation error falls spectrally in J.

    ``_reduced_solve`` on the dense operator itself (the frame is the whole
    space), so nothing but the time scheme is measured -- inside ``tamen``
    the same solve runs on the Galerkin-projected operator, whose basis
    quality is a separate concern with its own tests above.
    """
    from tt.algs.tamen import _reduced_solve
    rng = np.random.default_rng(3)
    m = 24
    B = rng.standard_normal((m, m))
    A = 0.3 * (B + B.T) - 0.8 * np.eye(m)
    x0 = rng.standard_normal(m)
    T = 1.0
    ref = sla.expm(T * A) @ x0
    errs = []
    for J in (3, 5, 8, 12):
        v = _reduced_solve(A, x0, J, T)
        errs.append(np.linalg.norm(v[-1] - ref) / np.linalg.norm(ref))
    errs = np.array(errs)
    assert errs[-1] < 1e-9, errs
    # each J refinement gains at least a factor 30 until the floor
    for a, b in zip(errs[:-1], errs[1:]):
        assert b < a / 30.0 or b < 1e-11, errs
