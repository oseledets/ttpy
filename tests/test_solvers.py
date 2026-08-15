"""Flexible preconditioning (FGMRES) in ``tt.algs.solvers.GMRES``.

The ``prec`` argument is the port of Larisa Markeeva's flexible GMRES
(ttpy develop branch, ``new_gmres``, 2018).  What makes it *flexible* is that
the preconditioner may change from one Krylov step to the next, which is only
correct if the preconditioned vectors ``z_j = M_j^{-1} v_j`` are stored as
their own basis and the correction is expanded in them (Saad 1993).  These
tests check exactly the three claims that matter:

1. a genuinely *varying* preconditioner converges where the unpreconditioned
   iteration with the same budget does not (so the Z-basis bookkeeping is
   right -- expanding in the V-basis instead makes the iteration diverge);
2. with a *fixed* preconditioner nothing changes with respect to ordinary
   GMRES semantics (identity preconditioner reproduces the plain run, a
   nontrivial fixed one still solves the original system);
3. the returned residual is honest: it matches ``||b - A x|| / ||b||``
   recomputed outside the solver, in dense arithmetic.

The dense cross-checks use the package's F-order flattening convention:
``x.full(asvector=True)`` pairs with ``A.full()`` (see ``matrix.full``).
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.solvers import GMRES


def flat(x):
    """F-order flattening of a TT vector, the convention of the package."""
    return np.asarray(x.full(asvector=True))


def as_ttmatrix(dense, d):
    """A dense ``2**d x 2**d`` matrix as a TT-matrix, F-order convention."""
    return tt.matrix(np.asarray(dense).reshape([2] * (2 * d), order="F"),
                     eps=1e-12)


def matvec_of(A):
    return lambda x, eps: tt.matvec(A, x).round(eps)


def spd_problem(d, seed=0):
    """``A x = b`` with ``A`` the (SPD) QTT Laplacian on ``2**d`` points.

    Returns the TT operator, its dense image, and a normalized random
    right-hand side in both formats.
    """
    A = tt.qlaplace_dd([d])
    Ad = np.asarray(A.full())
    rng = np.random.default_rng(seed)
    cores = [rng.standard_normal((1 if i == 0 else 2, 2, 1 if i == d - 1 else 2))
             for i in range(d)]
    b = tt.vector.from_list(cores)
    b = (1.0 / b.norm()) * b
    return A, Ad, b, flat(b)


def shifted_inverse_precs(Ad, d, shifts):
    """TT preconditioners ``(A + s I)^{-1}`` for each shift, plus their dense images."""
    n = Ad.shape[0]
    dense = [np.linalg.inv(Ad + s * np.eye(n)) for s in shifts]
    return [as_ttmatrix(M, d) for M in dense], dense


def alternating_prec(Ms):
    """A preconditioner closure that cycles through ``Ms``, counting its calls."""
    state = {"calls": 0}

    def prec(x, eps):
        M = Ms[state["calls"] % len(Ms)]
        state["calls"] += 1
        return tt.matvec(M, x).round(eps)

    return prec, state


def test_fgmres_variable_preconditioner_beats_plain_gmres_on_spd():
    """Alternating shifted inverses must converge where plain GMRES stalls.

    The two preconditioners are different matrices, so this run exercises the
    genuinely flexible path: the correction has to come from the stored
    ``z_j``, and getting that wrong is not a small error -- it produces an
    iterate outside the right search space, and this test fails loudly.
    """
    d, eps = 6, 1e-8
    A, Ad, b, bd = spd_problem(d)
    Ms, _ = shifted_inverse_precs(Ad, d, shifts=(0.5, 2.0))
    prec, state = alternating_prec(Ms)
    budget = dict(maxit=200, m=20)

    x, res = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps, prec=prec,
                   **budget)
    assert res < eps
    assert state["calls"] >= 2          # both preconditioners actually ran

    # the same budget without preconditioning must do no better (on this
    # ill-conditioned Laplacian it does not even converge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, res_plain = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps,
                             **budget)
    assert res <= res_plain


def test_fgmres_identity_preconditioner_matches_plain_gmres():
    """``prec = identity`` is a fixed preconditioner; the run must reproduce
    the unpreconditioned one exactly (same Krylov space, same small problems)."""
    d, eps = 5, 1e-8
    A, _, b, _ = spd_problem(d, seed=1)
    args = dict(eps=eps, maxit=200, m=20)

    x_prec, res_prec = GMRES(matvec_of(A), tt.zeros([2] * d), b,
                             prec=lambda v, eps: v, **args)
    x_plain, res_plain = GMRES(matvec_of(A), tt.zeros([2] * d), b, **args)

    assert res_prec == pytest.approx(res_plain, rel=1e-12)
    assert float((x_prec - x_plain).norm()) <= 1e-12 * float(x_plain.norm())


def test_fgmres_fixed_preconditioner_solves_the_original_system():
    """With one fixed ``(A + 0.5 I)^{-1}`` the method is ordinary
    right-preconditioned GMRES, and the answer it returns must solve
    ``A x = b`` itself -- no un-preconditioning left to the caller."""
    d, eps = 5, 1e-9
    A, Ad, b, bd = spd_problem(d, seed=2)
    Ms, _ = shifted_inverse_precs(Ad, d, shifts=(0.5,))
    prec, state = alternating_prec(Ms)

    x, res = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps, prec=prec,
                   maxit=100, m=20)
    assert res < eps
    x_dense = np.linalg.solve(Ad, bd)
    assert np.linalg.norm(flat(x) - x_dense) < 1e-6 * np.linalg.norm(x_dense)


def test_fgmres_returned_residual_is_recomputed_not_estimated():
    """The residual that comes back must be the measured
    ``||b - A x|| / ||b||`` of the returned ``x``, verified here in dense
    arithmetic outside the solver -- both at convergence and at a budget cut."""
    d, eps = 6, 1e-8
    A, Ad, b, bd = spd_problem(d, seed=3)
    Ms, _ = shifted_inverse_precs(Ad, d, shifts=(0.5, 2.0))

    # converged run
    prec, _ = alternating_prec(Ms)
    # the solver recomputes the residual through the eps-truncated closure, so
    # the agreement with the untruncated dense number is up to O(eps) absolute
    # (near convergence the two are the same size; an *estimated* residual
    # would be off by orders of magnitude, not by eps)
    x, res = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps, prec=prec,
                   maxit=200, m=20)
    true_res = np.linalg.norm(bd - Ad @ flat(x)) / np.linalg.norm(bd)
    assert res == pytest.approx(true_res, rel=1e-4, abs=5 * eps)

    # truncated run: the estimate inside the last cycle is optimistic; the
    # returned number must still be the recomputed truth
    prec, _ = alternating_prec(Ms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, res = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps, prec=prec,
                       maxit=4, m=4)
    true_res = np.linalg.norm(bd - Ad @ flat(x)) / np.linalg.norm(bd)
    assert res == pytest.approx(true_res, rel=1e-4, abs=5 * eps)
