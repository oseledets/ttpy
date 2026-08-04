"""Block eigensolver (``tt.algs.eigb``) and KSL integrator (``tt.algs.ksl``).

Every check is against dense truth (``numpy.linalg.eigh``, ``scipy.linalg.expm``,
the analytic spectrum of the discrete Laplacian) or against a mathematical
invariant (residual, orthonormality, observed convergence order).  Nothing here
compares against the legacy Fortran output.

A note on the KSL order test.  Comparing one step against ``expm(tau A) y0``
cannot show the temporal order of the integrator: on a manifold that contains
the exact trajectory (full ranks) the projector splitting is *exact* -- see
:func:`test_ksl_is_exact_when_the_manifold_is_full`, error 1e-14 for every step
size -- while on a manifold that does not contain it, the tau-independent
modelling error swamps the splitting error (that error is what
:func:`test_ksl_reports_the_rank_it_cannot_follow` measures, and the integrator
reports it).  The order below is therefore measured by Richardson
self-convergence, which is a statement about the discretization alone.

That is not the last word, and this file used to claim it was.  The right
reference is neither ``expm(tau A) y0`` nor the integrator itself but the
solution of the ODE KSL actually discretizes, ``y' = P_{T_y M} A y``; integrated
densely it is an independent oracle, and against it the observed orders are 1.00
and 2.00 with the modelling error 25x larger than the splitting error being
fitted.  See
``tests/test_verify_eigb_ksl.py::test_ksl_order_against_the_dense_projected_flow``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg as sla

import hamiltonians as ham
import tt
from tt.algs.eigb import eigb, spectral_norm_estimate
from tt.algs.ksl import diag_ksl, expmv_krylov, ksl, tangent_defect
from tt.core import _ops


# --- helpers -----------------------------------------------------------------

def sym_tt_matrix(d, n, r=2, seed=0):
    """Random symmetric TT-matrix, spectrally normalized to ``||A||_2 = 1``."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [rng.standard_normal((ranks[k], n, n, ranks[k + 1])) for k in range(d)]
    m = tt.matrix.from_list(cores)
    m = (m + m.T).round(1e-14)
    dense = m.full()
    return (1.0 / np.linalg.norm(dense, 2)) * m


def rand_tt(n, r, seed):
    """Seeded random TT-vector: the tests must not depend on the draw."""
    rng = np.random.default_rng(seed)
    return tt.rand(n, r=r, samplefunc=rng.standard_normal)


def block_columns(y):
    """The ``B`` eigenvectors of a block TT-vector as dense columns.

    ``full()`` squeezes a boundary rank of one, so ``B == 1`` has no block axis.
    """
    nblock = int(y.r[-1])
    full = y.full()
    if nblock == 1:
        return full.reshape(-1, order="F")[:, None]
    return np.stack([full[..., b].reshape(-1, order="F") for b in range(nblock)], 1)


def laplace_1d_eigs(nsize):
    """Spectrum of ``tridiag(-1, 2, -1)`` of size ``nsize`` (Dirichlet)."""
    k = np.arange(1, nsize + 1)
    return 4.0 * np.sin(np.pi * k / (2.0 * (nsize + 1))) ** 2


# --- eigb: eigenvalues against dense eigh ------------------------------------

@pytest.mark.parametrize("d, nblock", [(6, 1), (6, 3), (8, 2), (10, 4)])
def test_eigb_matches_dense_eigh(d, nblock):
    """The B smallest Ritz values match ``numpy.linalg.eigh`` of the dense matrix."""
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(d * 100 + nblock)
    x = tt.rand([2] * d, r=[1] + [max(4, nblock + 1)] * (d - 1) + [nblock],
                samplefunc=rng.standard_normal)
    y, lam = eigb(A, x, 1e-8, verb=0)

    exact = np.linalg.eigh(A.full())[0][:nblock]
    assert y.r[-1] == nblock
    assert np.allclose(lam, exact, atol=1e-6, rtol=0), f"{lam} vs {exact}"
    # the QTT Laplacian eigenvectors are low-rank: the sweep must not blow up
    assert max(y.r) <= 4 * nblock + 4


@pytest.mark.parametrize("d, nblock", [(6, 3), (10, 2)])
def test_eigb_eigenvector_residuals(d, nblock):
    """``||A y_i - lam_i y_i||`` is small and the block is orthonormal."""
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(7)
    x = tt.rand([2] * d, r=[1] + [5] * (d - 1) + [nblock],
                samplefunc=rng.standard_normal)
    y, lam = eigb(A, x, 1e-10, verb=0)

    dense = A.full()
    cols = block_columns(y)
    assert np.abs(cols.T @ cols - np.eye(nblock)).max() < 1e-10
    for b in range(nblock):
        res = np.linalg.norm(dense @ cols[:, b] - lam[b] * cols[:, b])
        assert res < 1e-8, f"eigenvector {b}: residual {res:.3E}"


def test_eigb_matrix_free_path_agrees_with_dense_path():
    """LOBPCG on the un-formed local operator gives the same eigenvalues."""
    d, nblock = 10, 4
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(11)
    r = [1] + [6] * (d - 1) + [nblock]
    x = tt.rand([2] * d, r=r, samplefunc=rng.standard_normal)

    _, lam_dense = eigb(A, x, 1e-8, verb=0, max_full_size=10 ** 6)
    _, lam_free, hist = eigb(A, x, 1e-8, verb=0, max_full_size=30,
                             return_history=True)

    assert "lobpcg" in {s["solver"] for s in hist.steps}, "the dense path was used"
    assert np.allclose(lam_free, lam_dense, atol=1e-8, rtol=0)
    # the local residual of the iterative solver is measured, not assumed
    assert hist.max_local_res > 0.0


def test_eigb_qlaplace_3d_against_the_analytic_spectrum():
    """A 3D QTT Laplacian is far too large to build densely; use the spectrum.

    ``qlaplace_dd([d, d, d])`` is the Kronecker sum of three 1D Dirichlet
    Laplacians on ``2^d`` points, so its eigenvalues are the sums of the 1D ones.
    """
    d = 5
    nsize = 2 ** d
    A = tt.qlaplace_dd([d, d, d])
    nblock = 4
    rng = np.random.default_rng(3)
    x = tt.rand([2] * (3 * d), r=[1] + [8] * (3 * d - 1) + [nblock],
                samplefunc=rng.standard_normal)
    y, lam = eigb(A, x, 1e-8, nswp=40, verb=0)

    e1 = laplace_1d_eigs(nsize)
    sums = (e1[:6, None, None] + e1[None, :6, None] + e1[None, None, :6]).ravel()
    exact = np.sort(sums)[:nblock]
    assert np.allclose(lam, exact, atol=1e-9, rtol=1e-6), f"{lam} vs {exact}"


def test_eigb_records_history_and_reports_non_convergence():
    """``verb=0`` prints nothing but records everything; a short run says so."""
    d = 10
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(5)
    x = tt.rand([2] * d, r=[1] + [4] * (d - 1) + [3], samplefunc=rng.standard_normal)

    with pytest.warns(RuntimeWarning, match="did not converge"):
        y, lam, hist = eigb(A, x, 1e-12, nswp=1, verb=0, return_history=True)
    assert hist.converged is False
    assert hist.nswp_done == 1
    assert len(hist.steps) == 2 * d - 2      # one full sweep: d -> 1 -> d
    assert len(hist.sweeps) == 1
    assert np.isfinite(hist.ermax) and hist.ermax > 1e-12
    assert list(hist.ranks) == [int(v) for v in y.r]


def test_eigb_refuses_what_it_cannot_do():
    d = 6
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=49)

    with pytest.raises(TypeError):
        eigb(x, x, 1e-8, verb=0)                       # not a TT-matrix
    with pytest.raises(ValueError):
        eigb(tt.qlaplace_dd([d - 1]), x, 1e-8, verb=0)  # dimension mismatch
    with pytest.raises(ValueError):
        eigb(A, x, 1e-8, nswp=0, verb=0)               # nothing to do

    # a non-symmetric operator must be refused, not silently symmetrized
    rng = np.random.default_rng(2)
    cores = [rng.standard_normal((1 if k == 0 else 2, 2, 2, 1 if k == d - 1 else 2))
             for k in range(d)]
    nonsym = tt.matrix.from_list(cores)
    with pytest.raises(ValueError, match="Hermitian"):
        eigb(nonsym, x, 1e-8, verb=0)


def test_eigb_warns_when_a_too_small_guess_rank_stalls_it():
    """A converged-looking run that is wrong must still say so.

    ``eigb`` cannot grow the TT rank past ``B * r_guess``, and at ``B == 1`` not
    at all: both local SVD groupings bound the new rank by the old one.  Handed
    a guess of rank 8 for a Heisenberg ground state that needs more, it happily
    converges *inside* that manifold -- the Ritz value stops moving to 2.2e-09
    while the eigenvalue is wrong in the 6th digit.  The sweep indicator cannot
    see this; the eigenresidual can, and the threshold has to be tied to ``eps``
    for it to fire.  A fixed 1e-2 (what this used to be) leaves six silent
    decades exactly where such a run comes to rest.
    """
    d = 10
    H = ham.heisenberg(d)
    exact = np.linalg.eigvalsh(ham.dense(H))[0]
    rng = np.random.default_rng(0)

    y0 = tt.rand(2, d, r=[1] + [8] * (d - 1) + [1], samplefunc=rng.standard_normal)
    with pytest.warns(RuntimeWarning, match="backward error"):
        _, lam, hist = eigb(H, y0, 1e-8, nswp=60, verb=0, return_history=True)

    # the run looks converged and is not: this is the pair the warning is about
    assert hist.converged is True
    assert hist.ermax < 1e-8
    assert abs(lam[0] - exact) / abs(exact) > 1e-7      # requested 1e-8
    assert max(hist.ranks) == 8                         # never grew past the guess

    # and the other side of it: enough rank, no warning, no false alarm
    y0 = tt.rand(2, d, r=[1] + [32] * (d - 1) + [1],
                 samplefunc=np.random.default_rng(0).standard_normal)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, lam = eigb(H, y0, 1e-8, nswp=60, verb=0)
    assert abs(lam[0] - exact) < 1e-10 * abs(exact)


def test_spectral_norm_estimate_is_a_close_lower_bound():
    """The warning's denominator, against a dense ``||A||_2``.

    It must be a *lower* bound (a warning that goes quiet is worse than one
    that is eager) and close enough that the threshold means something.  Both
    operators below have a clustered top of the spectrum, the slow case for
    power iteration, so this is near the worst it does.
    """
    for A in (tt.qlaplace_dd([10]), ham.heisenberg(10)):
        exact = np.linalg.norm(ham.dense(A), 2)
        est = spectral_norm_estimate(A)
        assert est <= exact * (1 + 1e-10)          # a lower bound
        assert est >= 0.9 * exact, f"{est} vs {exact}"
        assert est == spectral_norm_estimate(A)    # deterministic

    # the degenerate operator has no scale to divide by, and must not blow up
    assert spectral_norm_estimate(0.0 * tt.qlaplace_dd([6])) == 0.0


def test_eigb_does_not_cry_wolf_near_the_bottom_of_the_spectrum():
    """Correct eigenpairs of a near-singular operator must not warn.

    ``qlaplace_dd([10])`` has ``lam_1 = 9.4e-06`` against ``||A||_2 = 4``, so
    the residual relative to ``||A y_1||`` is 3.0e-04 -- above any threshold
    tied to ``eps=1e-8`` -- while the eigenvalues are right to 1e-9 absolute.
    Normalizing by ``||A||`` instead of ``||A y_i||`` is what separates a wrong
    answer from a small one; this test is the reason the denominator changed.
    """
    d, nblock = 10, 4
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(d * 100 + nblock)
    x = tt.rand([2] * d, r=[1] + [5] * (d - 1) + [nblock],
                samplefunc=rng.standard_normal)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, lam, hist = eigb(A, x, 1e-8, verb=0, return_history=True)

    exact = np.linalg.eigh(A.full())[0][:nblock]
    assert np.allclose(lam, exact, atol=1e-6, rtol=0)
    # the two measures disagree by four decades here -- that is the whole point
    assert np.max(hist.res_rel) > 1e-5
    assert np.max(hist.res_back) < 1e-8


def test_tfim_ground_energy_matches_the_closed_form():
    """The critical open TFIM has an exact ground-state energy; check the fixture.

    An oracle that does not go through LAPACK, so it stays valid at chain
    lengths where forming the dense matrix is impossible.
    """
    for d in (4, 8, 10):
        dense_e0 = np.linalg.eigvalsh(ham.dense(ham.tfim(d, g=1.0)))[0]
        assert abs(dense_e0 - ham.tfim_critical_ground_energy(d)) < 1e-13 * abs(dense_e0)


def test_eigb_largest_eigenvalues_via_minus_a():
    """``eigb(-A, ...)`` gives the largest eigenvalues, as documented."""
    d, nblock = 8, 2
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(13)
    x = tt.rand([2] * d, r=[1] + [4] * (d - 1) + [nblock],
                samplefunc=rng.standard_normal)
    _, lam = eigb((-1.0) * A, x, 1e-8, verb=0)
    exact = np.linalg.eigh(A.full())[0][::-1][:nblock]
    assert np.allclose(-lam, exact, atol=1e-6, rtol=0)


# --- the Krylov exponential, against scipy ------------------------------------

@pytest.mark.parametrize("t", [0.1, 1.0, -2.5])
@pytest.mark.parametrize("symmetric", [True, False])
def test_expmv_krylov_matches_dense_expm(t, symmetric):
    rng = np.random.default_rng(int(abs(t) * 10) + symmetric)
    n = 40
    a = rng.standard_normal((n, n)) / np.sqrt(n)
    if symmetric:
        a = a + a.T
    x = rng.standard_normal(n)
    w, info = expmv_krylov(lambda v: a @ v, x, t, space=8, tol=1e-12)
    exact = sla.expm(t * a) @ x
    err = np.linalg.norm(w - exact) / np.linalg.norm(exact)
    assert err < 1e-10, f"err {err:.3E}, info {info}"
    assert info["substeps"] >= 1
    assert info["err_est"] < 1e-9


def test_expmv_krylov_fails_loudly_when_it_cannot_finish():
    """A step budget that cannot cover ``t`` raises instead of returning garbage."""
    rng = np.random.default_rng(0)
    n = 30
    a = rng.standard_normal((n, n)) * 40.0
    x = rng.standard_normal(n)
    with pytest.raises(RuntimeError, match="substeps"):
        expmv_krylov(lambda v: a @ v, x, 1.0, space=4, tol=1e-14, max_substeps=3)


# --- KSL against dense truth --------------------------------------------------

@pytest.mark.parametrize("scheme", ["first", "symm"])
@pytest.mark.parametrize("tau", [0.5, 0.1])
def test_ksl_is_exact_when_the_manifold_is_full(scheme, tau):
    """With full TT ranks the exact trajectory stays on the manifold.

    The projector splitting is then exact (Lubich/Oseledets exactness property),
    so this compares the whole machinery -- interfaces, orthogonalizations, the
    K/S order and signs, the Krylov exponential -- against
    ``scipy.linalg.expm``, with no splitting error to hide behind.
    """
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    dense = A.full()
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=41)
    y0 = (1.0 / y0.norm()) * y0

    y = ksl(A, y0, tau, verb=0, scheme=scheme, local_tol=1e-13, check_rank=False)
    exact = sla.expm(tau * dense) @ y0.full(asvector=True)
    err = np.linalg.norm(y.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert err < 1e-11, f"scheme={scheme} tau={tau}: err {err:.3E}"
    assert list(y.r) == list(y0.r)      # KSL keeps the rank


@pytest.mark.parametrize("scheme, expected", [("first", 1.0), ("symm", 2.0)])
def test_ksl_convergence_order(scheme, expected):
    """The observed temporal order is 1 (Lie-Trotter) and 2 (Strang).

    Reference: the same integrator with a 16x smaller step (Richardson).  See
    the module docstring for why the dense ``expm`` cannot serve as the
    reference here.
    """
    d, n = 6, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, 3, seed=42)
    y0 = (1.0 / y0.norm()) * y0
    total = 0.5

    def integrate(nsteps):
        y = y0
        for _ in range(nsteps):
            y = ksl(A, y, total / nsteps, verb=0, scheme=scheme, local_tol=1e-13,
                    check_rank=False)
        return y.full(asvector=True)

    ref = integrate(256)
    steps = [2, 4, 8, 16]
    errs = np.array([np.linalg.norm(integrate(m) - ref) / np.linalg.norm(ref)
                     for m in steps])
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(errs[:-1] > errs[1:]), f"errors do not decrease: {errs}"
    assert np.abs(orders - expected).max() < 0.25, f"orders {orders}, errs {errs}"


def test_ksl_reports_the_rank_it_cannot_follow():
    """A rank too small is a *reported* modelling error, not a hidden one."""
    d, n = 6, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    dense = A.full()
    y0 = rand_tt([n] * d, 3, seed=43)   # far from enough for this operator
    y0 = (1.0 / y0.norm()) * y0
    v0 = y0.full(asvector=True)

    tau = 0.2
    with pytest.warns(RuntimeWarning, match="fixed rank"):
        y, hist = ksl(A, y0, tau, verb=0, return_history=True, local_tol=1e-13)

    exact = sla.expm(tau * dense) @ v0
    err = np.linalg.norm(y.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert hist.defect_rel > 0.1
    # the reported estimate is the error, not a vague flag: within 20% of it
    assert 0.8 < err / hist.step_error_est < 1.2, (err, hist.step_error_est)

    # `defect_warn` is a reporting threshold only: the same computation, the
    # same recorded numbers, no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y2, hist2 = ksl(A, y0, tau, verb=0, return_history=True, local_tol=1e-13,
                        defect_warn=1.0)
    assert hist2.step_error_est == pytest.approx(hist.step_error_est, rel=1e-12)
    assert (y2 - y).norm() / y.norm() < 1e-14

    # and with enough rank the same measurement collapses
    y0_full = rand_tt([n] * d, [1, 2, 4, 8, 4, 2, 1], seed=44)
    _, hist_full = ksl(A, y0_full, tau, verb=0, return_history=True,
                       local_tol=1e-13)
    assert hist_full.defect_rel < 1e-6
    assert hist_full.step_error_est < 1e-6


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_ksl_is_scale_invariant(scale):
    """``ksl(A, c y0, tau) = c ksl(A, y0, tau)``: the flow is linear.

    A regression test for the Krylov breakdown criterion: comparing the Arnoldi
    residual (which scales with ``||A||``) against ``||v||`` declares a
    breakdown at the first step for a badly scaled input, degrades the local
    exponential to a one-dimensional Krylov space and reports an error estimate
    of exactly zero.  ``tt.rand`` on 24 cores produces exactly such norms.
    """
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=51)
    y0 = (1.0 / y0.norm()) * y0
    tau = 0.3

    base = ksl(A, y0, tau, verb=0, check_rank=False, local_tol=1e-13)
    scaled, hist = ksl(A, scale * y0, tau, verb=0, check_rank=False,
                       local_tol=1e-13, return_history=True)
    assert (scaled - scale * base).norm() / scaled.norm() < 1e-12
    # A Krylov space that collapses to one dimension reports err_est = 0 while
    # being wrong; a step taken exactly by a dense expm reports the same pair
    # truthfully.  The property is "no step was silently degraded", so the
    # exact ones are allowed and the Krylov ones must be real.
    assert all(s.get("exact") or s["krylov"] > 1 for s in hist.steps), \
        "a Krylov space collapsed to one dimension"
    # Same invariant from the other side: a Krylov step that reports zero local
    # error is lying (it collapsed); a step taken exactly by a dense expm has no
    # local error to report.  So zero is admissible only if nothing approximated.
    if not all(s.get("exact") for s in hist.steps):
        assert hist.max_local_err > 0.0, (
            "a Krylov step reported an error estimate of exactly 0")

    exact = scale * (sla.expm(tau * A.full()) @ y0.full(asvector=True))
    err = np.linalg.norm(scaled.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert err < 1e-11, err


@pytest.mark.parametrize("use_normest", [0, 1, 2])
@pytest.mark.parametrize("space", [4, 8, 16])
def test_ksl_knobs_do_not_change_the_answer(use_normest, space):
    """``space`` and ``use_normest`` buy speed, not accuracy.

    The Krylov dimension and the norm estimate only decide how the requested
    ``local_tol`` is reached (how many substeps); the answer must be the same to
    within that tolerance.
    """
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=52)
    y0 = (1.0 / y0.norm()) * y0

    y = ksl(A, y0, 0.4, verb=0, check_rank=False, local_tol=1e-12,
            space=space, use_normest=use_normest)
    exact = sla.expm(0.4 * A.full()) @ y0.full(asvector=True)
    err = np.linalg.norm(y.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert err < 1e-10, err


def test_eigb_is_scale_invariant():
    """The initial guess is orthonormalized, so its scale cannot matter."""
    d, nblock = 6, 3
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [5] * (d - 1) + [nblock], seed=53)
    _, lam = eigb(A, x, 1e-10, verb=0)
    _, lam_scaled = eigb(A, 1e14 * x, 1e-10, verb=0)
    assert np.allclose(lam, lam_scaled, rtol=1e-12, atol=0)


def test_tangent_defect_is_zero_on_a_full_rank_point():
    """``(I - P) A y = 0`` when the tangent space is everything."""
    d, n = 4, 2
    A = sym_tt_matrix(d, n, 2, seed=1)
    y = rand_tt([n] * d, [1, 2, 4, 2, 1], seed=45)
    defect, znorm = tangent_defect(A, y)
    assert defect / znorm < 1e-7      # the floor is ||Z|| * sqrt(eps)


def test_ksl_preserves_an_eigenvector():
    """An eigenvector of ``A`` is a fixed point of the flow up to ``exp(-lam t)``.

    This is also a cross-check between the two modules of this task: the
    eigenvector comes from :func:`eigb`.
    """
    d = 8
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(21)
    x = tt.rand([2] * d, r=[1] + [4] * (d - 1) + [1], samplefunc=rng.standard_normal)
    y0, lam = eigb(A, x, 1e-10, verb=0)

    tau = 1e-2
    y1 = ksl((-1.0) * A, y0, tau, verb=0, check_rank=False, local_tol=1e-13)
    cos = tt.dot(y1, y0) / (y1.norm() * y0.norm())
    assert abs(cos - 1.0) < 1e-12
    assert abs(y1.norm() / y0.norm() - np.exp(-lam[0] * tau)) < 1e-12


def test_diag_ksl_matches_dense():
    """``dy/dt = diag(V) y`` against the dense exponential of the diagonal."""
    d, n = 5, 2
    rng = np.random.default_rng(4)
    v = tt.rand([n] * d, r=2, samplefunc=rng.standard_normal)
    y0 = tt.rand([n] * d, r=[1, 2, 4, 4, 2, 1], samplefunc=rng.standard_normal)
    tau = 0.1

    y = diag_ksl(v, y0, tau, verb=0, check_rank=False, local_tol=1e-13)
    exact = sla.expm(tau * np.diag(v.full(asvector=True))) @ y0.full(asvector=True)
    err = np.linalg.norm(y.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert err < 1e-11, err


def test_ksl_complex_step_matches_dense():
    """``exp(-i tau H) y0``: the Schroedinger case, complex arithmetic."""
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=11)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=46)
    tau = -0.3j

    y = ksl(A, y0, tau, verb=0, check_rank=False, local_tol=1e-13)
    exact = sla.expm(tau * A.full()) @ y0.full(asvector=True)
    err = np.linalg.norm(y.full(asvector=True) - exact) / np.linalg.norm(exact)
    assert err < 1e-11, err
    assert abs(y.norm() / y0.norm() - 1.0) < 1e-10   # unitary flow


def test_ksl_records_history_with_verb_zero(capsys):
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=47)
    y, hist = ksl(A, y0, 0.1, verb=0, return_history=True, check_rank=True)

    assert capsys.readouterr().out == ""
    assert len(hist.steps) == 2 * (2 * d - 1)      # two half sweeps, K and S
    assert {s["kind"] for s in hist.steps} == {"K", "S"}
    assert hist.total_substeps >= len(hist.steps)
    assert np.isfinite(hist.defect_rel)
    assert list(hist.ranks) == [int(v) for v in y.r]


def test_ksl_refuses_what_it_cannot_do():
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=48)

    with pytest.raises(TypeError):
        ksl(y0, y0, 0.1, verb=0)
    with pytest.raises(ValueError):
        ksl(sym_tt_matrix(d - 1, n, 2), y0, 0.1, verb=0)
    with pytest.raises(ValueError, match="rank"):
        ksl(A, y0, 0.1, verb=0, rmax=2)      # KSL cannot truncate for you


# --- legacy namespaces --------------------------------------------------------

def test_legacy_imports():
    import tt.eigb
    import tt.ksl

    assert tt.eigb.eigb is eigb
    assert tt.ksl.ksl is ksl
    assert tt.ksl.diag_ksl is diag_ksl
    assert tt.eigb_solve is eigb
    assert tt.ksl_step is ksl


# --- a core invariant the two modules rely on ---------------------------------

def test_block_vector_columns_are_orthonormal_after_eigb():
    """``tt.dot`` on a block TT-vector returns the Gram matrix of the columns."""
    d, nblock = 6, 3
    A = tt.qlaplace_dd([d])
    rng = np.random.default_rng(31)
    x = tt.rand([2] * d, r=[1] + [4] * (d - 1) + [nblock],
                samplefunc=rng.standard_normal)
    y, _ = eigb(A, x, 1e-10, verb=0)
    gram = np.asarray(_ops.dot(y.cores, y.cores)).reshape((nblock, nblock))
    assert np.abs(gram - np.eye(nblock)).max() < 1e-10


def test_krylov_exponential_survives_an_iterate_that_underflows_to_zero():
    """A contracting flow can zero the iterate mid-substep; exp(tA) 0 = 0.

    Without a guard the next Arnoldi divides by ||v|| = 0, returns NaN and then
    raises a message blaming the caller's operator. Found while specifying the
    BUG integrator (docs/plans/bug-integrator.md).
    """
    from tt.algs.ksl import expmv_krylov

    n = 4
    a = -400.0 * np.eye(n)     # contracts by exp(-400) = 2e-174 over the step
    x = np.zeros(n)
    x[0] = 1e-160              # the norm survives the entry check; the flow does not
    w, info = expmv_krylov(lambda v: a @ v, x, 1.0, space=4, tol=1e-8)
    assert np.all(np.isfinite(w)), "the exponential must not return NaN"
    assert float(np.linalg.norm(w)) == 0.0
    assert info["underflow"] is True          # and it says so
    assert info["substeps"] >= 1              # it really entered the loop

    # a genuinely zero input is exact and unremarkable, not an underflow
    _, info0 = expmv_krylov(lambda v: a @ v, np.zeros(n), 1.0, space=4)
    assert info0["underflow"] is True and info0["substeps"] == 0


def stiff_heat(levels, rank=4, seed=0):
    """``dy/dt = -(2^L+1)^2 Laplace y``: a contraction semigroup in QTT."""
    n = 2 ** levels
    a = (-1.0 * (n + 1) ** 2) * tt.qlaplace_dd([levels])
    rng = np.random.default_rng(seed)
    y0 = tt.rand([2] * levels, r=rank, samplefunc=rng.standard_normal)
    return a, y0 * (1.0 / y0.norm())


def test_ksl_refuses_a_step_that_amplifies_past_every_digit():
    """The projector splitting is not stiff-stable, and must say so.

    Its S-steps integrate backwards, so for a dissipative ``A`` they amplify by
    ``exp(tau |lambda_min|)``; the following K-step shrinks the data back but
    not the rounding error that rode along. Left unguarded this *returns a
    number*: measured at ``tau ||A|| = 169``, ``||y|| = 3.3e+106`` for a flow
    whose exact solution has norm 0.307, with nothing in the history to say so.
    """
    a, y0 = stiff_heat(6)
    with pytest.raises(RuntimeError, match="amplified its argument"):
        ksl(a, y0, 1e-2, verb=0, check_rank=False, local_tol=1e-8)

    # and the message points at the mechanism, not at a symptom downstream
    try:
        ksl(a, y0, 1e-2, verb=0, check_rank=False, local_tol=1e-8)
    except RuntimeError as exc:
        text = str(exc)
    assert "S-step" in text and "stiff-stable" in text and "Reduce tau" in text


def test_ksl_is_silent_and_accurate_where_the_amplification_is_harmless():
    """The guard must not fire on the steps the integrator handles well."""
    a, y0 = stiff_heat(6)
    dense = np.asarray(a.full())
    v0 = np.asarray(y0.full(asvector=True))

    for tau in (1e-4, 1e-3):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            y, hist = ksl(a, y0, tau, verb=0, check_rank=False, local_tol=1e-8,
                          return_history=True)
        exact = sla.expm(tau * dense) @ v0
        got = np.asarray(y.full(asvector=True))
        err = np.linalg.norm(got - exact) / np.linalg.norm(exact)
        assert err < 0.1, f"tau={tau}: {err:.3E}"          # modelling error only
        assert hist.roundoff_floor < 1e-8
        assert np.linalg.norm(got) <= np.linalg.norm(v0) * 1.05   # a contraction


def test_ksl_history_carries_the_amplification_whatever_the_threshold():
    """The numbers are reported even when nothing warns -- no hidden unknown."""
    a, y0 = stiff_heat(6)
    _, hist = ksl(a, y0, 1e-5, verb=0, check_rank=False, return_history=True)
    assert hist.max_growth >= 1.0
    assert hist.max_growth_kind in ("K", "S")
    assert hist.roundoff_floor == pytest.approx(
        np.finfo(float).eps * hist.max_growth ** 2.5, rel=1e-12)
    assert all("growth" in s for s in hist.steps)
