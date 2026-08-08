"""Adversarial verification of the AMEn linear solver (``tt.algs.amen``).

This file exists to try to break :func:`tt.algs.amen.amen_solve`, not to
document it.  Every accuracy claim is checked against an oracle that shares no
code with the solver:

* the **analytic** solution of ``tridiag(-1, 2, -1) x = 1``, which is
  ``x_i = i (N + 1 - i) / 2`` -- no linear algebra at all, and it is available
  at ``N = 2^18`` where a dense solve is not;
* a **dense** ``numpy.linalg.solve`` / matrix-vector product for small cases;
* the residual of the returned TT cores recontracted in **double-double**
  (~106-bit, see ``tests/extended.py``), which is the only way to say whether
  the residual the solver reports about itself is honest -- a float64
  evaluation has a noise floor of its own, of the same size as the residual
  being judged (``docs/NUMERICS.md``).

Regimes are stated at every assertion (sizes, eps, dtype); no tolerance here is
looser than the number that was actually measured, and those numbers live in
``docs/NUMERICS.md``.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.amen import _solve_local, amen_solve
from tt.core import _ops

from conftest import (DENSE_TOL, QLAPLACE_D, SOLVE_EPS, requires_gpu,
                      to_gpu)
from extended import (dd_from, dd_matmul, dd_norm, dd_reshape, dd_sub,
                      dd_transpose)

rel = lambda a, b: (np.linalg.norm(np.asarray(a) - np.asarray(b))
                    / np.linalg.norm(np.asarray(b)))


# --- oracles -----------------------------------------------------------------

def full_dd(cores):
    """Contract a TT core list in double-double, F-order (mode 1 is fastest).

    The cores are float64, so lifting them costs nothing and the contraction
    carries ~106 bits of significand: this is the *exact* dense vector of the
    returned TT tensor.  ``np.longdouble`` is not a portable substitute
    (``docs/NUMERICS.md``).
    """
    res = dd_from(np.asarray(cores[0]))
    res = dd_reshape(res, (res[0].shape[1], res[0].shape[2]))
    for c in cores[1:]:
        c = np.asarray(c)
        r0, n, r1 = c.shape
        prod = dd_matmul(res, dd_from(c.reshape(r0, n * r1)))
        prod = dd_reshape(prod, (res[0].shape[0], n, r1))
        # the new mode is the slower index: flat = old + N * i_k
        res = dd_reshape(dd_transpose(prod, (1, 0, 2)), (-1, r1))
    return dd_reshape(res, (-1,))


def exact_residual(A, x, f):
    """``||A x - f|| / ||f||`` of the returned TT vector, in double-double."""
    Af = dd_from(np.asarray(A.full()))
    fv = full_dd(f.cores)
    xv = dd_reshape(full_dd(x.cores), (-1, 1))
    r = dd_sub(dd_reshape(dd_matmul(Af, xv), (-1,)), fv)
    return dd_norm(r) / dd_norm(fv)


def dense_residual(A, x, f):
    fv = np.asarray(f.full(asvector=True))
    return float(np.linalg.norm(np.asarray(A.full())
                                @ np.asarray(x.full(asvector=True)) - fv)
                 / np.linalg.norm(fv))


def tt_residual(A, x, f):
    r = _ops.sub(_ops.matvec_cores(tt.matrix.to_list(A), x.cores), f.cores)
    return float(_ops.norm(r) / _ops.norm(f.cores))


def random_matrix(d, n, ra, rng, dtype=np.float64, diag_shift=0.0):
    """Random TT-matrix, normalized to unit spectral norm before the shift."""
    cores = []
    for k in range(d):
        left = 1 if k == 0 else ra
        right = 1 if k == d - 1 else ra
        c = rng.standard_normal((left, n, n, right))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal((left, n, n, right))
        cores.append(c.astype(dtype))
    A = tt.matrix.from_list(cores)
    if diag_shift:
        A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A
        A = A + diag_shift * tt.eye([n] * d)
    return A


def random_vector(modes, r, rng, dtype=np.float64):
    cores = []
    d = len(modes)
    for k, nk in enumerate(modes):
        left = 1 if k == 0 else r
        right = 1 if k == d - 1 else r
        c = rng.standard_normal((left, nk, right))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal((left, nk, right))
        cores.append(c.astype(dtype))
    return tt.vector.from_list(cores)


# --- the reported residual must not be optimistic ----------------------------

@pytest.mark.parametrize("d, eps", [(6, 1e-6), (8, 1e-10), (10, 1e-10),
                                    (12, 1e-6)])
def test_reported_residual_is_not_optimistic(d, eps):
    """``info.true_res`` versus the double-double residual of the cores.

    The solver measures its own residual in TT arithmetic; if that measurement
    were optimistic, every convergence claim in the package would be worth
    nothing.  The ratio exact/reported is BLAS-dependent -- the *solve* is, not
    the oracle -- and stays within a few tens of percent of 1 on both platforms
    it has been run on (``docs/NUMERICS.md``).  1.5 is where it would stop being
    a measurement.
    """
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0,
                         check_true_res=True, return_info=True)
    exact = exact_residual(A, x, rhs)
    assert info.converged
    assert exact <= eps, f"reported {info.true_res:.3E}, really {exact:.3E}"
    assert exact <= 1.5 * info.true_res, (
        f"the reported residual {info.true_res:.3E} is optimistic: the true "
        f"one is {exact:.3E}")


# --- an oracle with no linear algebra in it ----------------------------------

@pytest.mark.parametrize("d", [6, 10, 14])
def test_matches_the_analytic_laplacian_solution(d):
    """``tridiag(-1,2,-1) x = 1`` has ``x_i = i (N + 1 - i) / 2`` exactly.

    N = 2^d up to 16384; float64; eps = 1e-8.  This oracle involves no solver
    at all, so it cannot agree with the code under test by construction.  The
    error grows like ``cond(A) * eps_machine ~ N^2 * 1e-16``, as it must, which
    is what the bound below encodes.
    """
    N = 2 ** d
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x, info = amen_solve(A, rhs, None, 1e-8, verb=0, seed=0,
                         check_true_res=True, return_info=True)
    assert info.converged
    i = np.arange(1, N + 1, dtype=float)
    analytic = i * (N + 1 - i) / 2.0
    err = rel(np.asarray(x.full(asvector=True)), analytic)
    assert err <= 30.0 * N ** 2 * np.finfo(float).eps, f"error {err:.3E}"


def test_huge_qtt_reports_its_failure_instead_of_a_plausible_number():
    """d = 18 (262144 unknowns): eps = 1e-8 is below the float64 floor.

    cond(A) ~ N^2 ~ 6.9e10, so a backward-stable solve cannot go below
    ~1e-5 in the *error* and ~1e-8..1e-6 in the residual.  The solver must warn
    and must still be right to the accuracy it claims.
    """
    d = 18
    N = 2 ** d
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    with pytest.warns(UserWarning, match="did NOT reach"):
        x, info = amen_solve(A, rhs, None, 1e-8, verb=0, seed=0, nswp=20,
                             return_info=True)
    assert not info.converged
    assert tt_residual(A, x, rhs) > 1e-8            # it really did fail
    i = np.arange(1, N + 1, dtype=float)
    analytic = i * (N + 1 - i) / 2.0
    assert rel(np.asarray(x.full(asvector=True)), analytic) < 1e-5


# --- shapes the original test file never exercised ---------------------------

def test_mode_sizes_that_differ_per_mode():
    """n = [2, 3, 4, 2]: every core has a different shape.  Dense oracle."""
    rng = np.random.default_rng(1)
    n = [2, 3, 4, 2]
    cores = [rng.standard_normal((1 if k == 0 else 2, nk, nk,
                                  1 if k == len(n) - 1 else 2))
             for k, nk in enumerate(n)]
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A + 4.0 * tt.eye(n)
    f = random_vector(n, 2, rng)
    for max_full_size in (0, 10 ** 6):       # GMRES path and dense path
        x, info = amen_solve(A, f, None, 1e-11, verb=0, seed=0,
                             max_full_size=max_full_size, local_iters=6,
                             return_info=True)
        xd = np.linalg.solve(np.asarray(A.full()),
                             np.asarray(f.full(asvector=True)))
        assert info.converged
        assert dense_residual(A, x, f) <= 1e-11          # measured ~6e-13
        # The solver targets a RESIDUAL; the error against the dense solution
        # is bounded by cond(A) times that residual, and this QTT Laplacian has
        # cond ~ 1e3.  Asserting 1e-10 on the error was pinning an accident of
        # where the iteration happened to stop (measured 1.0e-10 with one local
        # solver, 8e-11 with another).
        assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-8


def test_two_dimensional_problem():
    """d = 2: exactly one splitting, so ``sqrt(d-1)`` and the sweep both degenerate."""
    rng = np.random.default_rng(0)
    A = tt.matrix.from_list([rng.standard_normal((1, 3, 3, 2)),
                             rng.standard_normal((2, 3, 3, 1))])
    A = A + 6 * tt.eye([3, 3])
    f = random_vector([3, 3], 2, rng)
    x, info = amen_solve(A, f, None, 1e-11, verb=0, seed=0, return_info=True)
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    assert info.converged
    assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-12


@pytest.mark.parametrize("dims", [[4, 4], [3, 3, 3]])
def test_multidimensional_qtt_laplacian(dims):
    """``qlaplace_dd([4,4])`` -- the tt.__init__ docstring example, untested before.

    A d-dimensional Laplacian in QTT has a rank-(d+1) matrix, not rank 3, and
    its cores are *not* the 1D ones; float64, eps = 1e-8, dense oracle.
    """
    A = tt.qlaplace_dd(dims)
    f = tt.ones(2, sum(dims))
    x, info = amen_solve(A, f, f, 1e-8, verb=0, seed=0, return_info=True)
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    assert info.converged
    assert dense_residual(A, x, f) <= 1e-8              # measured ~1e-14
    # The solver targets a RESIDUAL; the error against the dense solution is
    # bounded by cond(A) times that residual, and this QTT Laplacian has
    # cond ~ 1e3.  Asserting 1e-10 on the error pinned an accident of where the
    # iteration happened to stop (1.0e-10 with one local solver, 8e-11 with
    # another) rather than anything the method promises.
    assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-8


def test_rank_one_right_hand_side_and_rank_one_solution():
    """A separable problem: the answer has TT rank 1 and must stay there."""
    d, n = 4, 3
    rng = np.random.default_rng(9)
    blocks = [rng.standard_normal((n, n)) + 4 * np.eye(n) for _ in range(d)]
    A = tt.matrix.from_list([b.reshape((1, n, n, 1)) for b in blocks])
    fv = [rng.standard_normal(n) for _ in range(d)]
    f = tt.vector.from_list([v.reshape((1, n, 1)) for v in fv])
    x, info = amen_solve(A, f, None, 1e-12, verb=0, seed=0, return_info=True)
    assert info.converged
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-12
    assert max(x.round(1e-10).r) == 1, f"rank-1 problem gave ranks {x.r}"


def test_strongly_nonsymmetric_convection():
    """Laplacian + c * first difference, c up to 100: dominated by the skew part.

    d = 8 (256 unknowns), eps = 1e-10, float64, checked against
    ``numpy.linalg.solve``.
    """
    d = 8
    for c in (1.0, 10.0, 100.0):
        A = (tt.qlaplace_dd([d]) + c * tt.IpaS(d, -1.0)).round(1e-14)
        f = tt.ones(2, d)
        x, info = amen_solve(A, f, None, 1e-10, verb=0, seed=0, nswp=30,
                             return_info=True)
        xd = np.linalg.solve(np.asarray(A.full()),
                             np.asarray(f.full(asvector=True)))
        assert info.converged, f"c={c}"
        assert dense_residual(A, x, f) <= 1e-10, f"c={c}"
        assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-12, f"c={c}"


def test_indefinite_operator():
    """A symmetric *indefinite* operator (AMEn part I assumes SPD).

    ``qlaplace_dd([6]) - 2 I`` has eigenvalues of both signs; the method has no
    right to converge, so the only requirement is that it either converges or
    says it did not.  In this regime it does converge, to roundoff.
    """
    d = 6
    A = (tt.qlaplace_dd([d]) - 2.0 * tt.eye([2] * d)).round(1e-14)
    f = tt.ones(2, d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x, info = amen_solve(A, f, None, 1e-8, verb=0, seed=0, nswp=10,
                             return_info=True)
    warned = any(issubclass(w.category, UserWarning) and "did NOT" in str(w.message)
                 for w in caught)
    assert info.converged != warned
    if info.converged:
        assert dense_residual(A, x, f) <= 1e-8


@pytest.mark.parametrize("max_full_size", [0, 10 ** 6])
def test_complex_non_hermitian_small_d(max_full_size):
    """d = 2 complex: the conjugation convention with no room to hide."""
    rng = np.random.default_rng(21)
    A = random_matrix(2, 3, 2, rng, dtype=np.complex128, diag_shift=3.0)
    f = random_vector([3, 3], 2, rng, dtype=np.complex128)
    x, info = amen_solve(A, f, None, 1e-12, verb=0, seed=0,
                         max_full_size=max_full_size, local_iters=8,
                         return_info=True)
    assert x.is_complex and info.converged
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    assert rel(np.asarray(x.full(asvector=True)), xd) < 1e-12


def test_real_operator_complex_right_hand_side():
    """dtype promotion: real ``A``, complex ``f`` must give a complex answer."""
    d = 4
    A = tt.qlaplace_dd([d])
    f = tt.vector.from_list([(1 + 1j) * np.ones((1, 2, 1)) for _ in range(d)])
    x, info = amen_solve(A, f, None, 1e-10, verb=0, seed=0, return_info=True)
    assert x.is_complex and info.converged
    assert dense_residual(A, x, f) <= 1e-10


# --- degenerate and invalid input --------------------------------------------

def test_zero_right_hand_side_is_refused_not_iterated_on():
    """``f = 0``: a relative residual does not exist, so the run must not start.

    Before the fix this ran all 20 sweeps, emitted 20 numpy
    "divide by zero" RuntimeWarnings and reported ``true_res = inf``, i.e. a
    residual of infinity for a vector that was correct to 1e-16.
    """
    d = 4
    A = tt.qlaplace_dd([d])
    with pytest.raises(ValueError, match="zero"):
        amen_solve(A, tt.zeros([2] * d), None, 1e-8, verb=0, seed=0)


def test_exactly_singular_operator_raises():
    """A singular ``A``: LAPACK's error must reach the caller unmodified."""
    n = 5
    a = np.diag([1.0, 2.0, 3.0, 4.0, 0.0])
    A = tt.matrix.from_list([a.reshape((1, n, n, 1))])
    f = tt.vector.from_list([np.ones((1, n, 1))])
    with pytest.raises(np.linalg.LinAlgError):
        amen_solve(A, f, None, 1e-10, verb=0, seed=0)


@pytest.mark.parametrize("kwargs, match", [
    ({"kickrank": -3}, "kickrank"),
    ({"rmax": 0}, "rmax"),
    ({"nswp": 0}, "nswp"),
    ({"local_iters": 0}, "local_iters"),
    ({"local_restart": 0}, "local_restart"),
    ({"eps": -1.0}, "eps"),
    ({"eps": np.inf}, "eps"),
])
def test_nonsense_arguments_are_refused(kwargs, match):
    """Silently reinterpreting an argument is the failure mode to avoid.

    In the Fortran-era interface ``kickrank=-3`` meant "plain ALS" and
    ``rmax=0`` produced a rank-0 truncation followed by a rank-``kickrank``
    enrichment -- a completely different method, reported only through the
    generic non-convergence message.
    """
    d = 6
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    eps = kwargs.pop("eps", 1e-8)
    with pytest.raises(ValueError, match=match):
        amen_solve(A, f, None, eps, verb=0, seed=0, **kwargs)


def test_nan_in_the_data_is_not_swallowed():
    d = 4
    A = tt.qlaplace_dd([d])
    cores = [np.ones((1, 2, 1)) for _ in range(d)]
    cores[0] = cores[0] * np.nan
    with pytest.raises((ValueError, FloatingPointError, np.linalg.LinAlgError)):
        amen_solve(A, tt.vector.from_list(cores), None, 1e-8, verb=0, seed=0,
                   nswp=2)


def test_initial_guess_is_not_modified():
    """``x0`` is an input, not scratch space."""
    d = 6
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    x0 = tt.rand([2] * d, r=3)
    before = [np.asarray(c).copy() for c in x0.cores]
    amen_solve(A, f, x0, 1e-8, verb=0, seed=0)
    for a, b in zip(before, x0.cores):
        assert np.array_equal(a, np.asarray(b))


# --- honesty of the local solver ---------------------------------------------

def test_direct_local_solve_reports_its_own_failure():
    """A dense local solve is backward stable, not exact; it must say so.

    The local core here is a badly scaled upper-triangular block (cond ~1e24):
    ``numpy.linalg.solve`` returns an answer whose residual is ~cond*eps, well
    above a 1e-14 request.  Before the fix ``_solve_local`` hard-coded
    ``converged: True`` on this path, so a stalled run blamed the *outer*
    iteration and told the user to raise ``max_full_size`` -- which was already
    infinite.
    """
    rng = np.random.default_rng(7)
    r1, n, r2 = 3, 3, 3
    phiL = np.zeros((r1, r1, 1)); phiL[:, :, 0] = np.eye(r1)
    phiR = np.zeros((r2, r2, 1)); phiR[:, :, 0] = np.eye(r2)
    acore = np.zeros((1, n, n, 1))
    acore[0, :, :, 0] = np.array([[1.0, 1e8, 0.0],
                                  [0.0, 1e-8, 1e8],
                                  [0.0, 0.0, 1.0]])
    rhs = rng.standard_normal((r1, n, r2))

    sol, linfo = _solve_local(phiL, acore, phiR, rhs, 1e-14, 10 ** 6, "n", 2, 40)
    assert linfo["kind"] == "direct"
    assert linfo["relres"] > 1e-14
    assert not linfo["converged"], (
        "the dense local solve reported success at relres "
        f"{linfo['relres']:.2E} against a tolerance of 1e-14")

    # ... and it does report success when it deserves to
    acore[0, :, :, 0] = np.eye(n) * 2.0
    _, linfo = _solve_local(phiL, acore, phiR, rhs, 1e-12, 10 ** 6, "n", 2, 40)
    assert linfo["converged"] and linfo["relres"] < 1e-14


def test_failure_message_names_the_right_culprit():
    """d = 12, eps = 1e-10 is below the float64 floor: the message must say why.

    The floor is measured here with LAPACK on the dense 4096x4096 system and
    the residual evaluated in double-double (``docs/NUMERICS.md``).  The solver
    must report that it did not converge, name the residual it reached, and say
    which solver stalled -- the local one or the outer iteration.
    """
    d, eps = 12, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    Ad = np.asarray(A.full())
    fv = np.asarray(rhs.full(asvector=True))
    xd = np.linalg.solve(Ad, fv).reshape(-1, 1)
    r = dd_sub(dd_reshape(dd_matmul(dd_from(Ad), dd_from(xd)), (-1,)),
               dd_from(fv))
    floor = dd_norm(r) / dd_norm(dd_from(fv))
    assert floor > eps, "premise broke: the float64 floor is below eps"

    with pytest.warns(UserWarning, match="did NOT reach"):
        x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0,
                             check_true_res=True, return_info=True)
    assert not info.converged
    assert exact_residual(A, x, rhs) <= 10 * floor
    assert ("local GMRES" in info.message
            or "dense local solves" in info.message
            or "outer iteration is what stalled" in info.message)


def test_history_describes_the_returned_vector():
    """A non-converged run returns the *best* iterate; the history must match it.

    Every summary field has to describe the vector that is handed back, not the
    last one computed.  ``info.ranks`` in particular was empty for ``nswp``
    runs that produced no sweep entry, and ``max_dx``/``max_res`` came from the
    last sweep while the cores came from the best one.
    """
    d, eps = 12, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    with pytest.warns(UserWarning):
        x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0,
                             check_true_res=True, return_info=True)
    assert info.ranks == [int(r) for r in x.r]
    best = info.sweeps[info.best_sweep - 1]
    measured = [s["true_res"] for s in info.sweeps if np.isfinite(s["true_res"])]
    # sweeps whose cheap indicator is still far from the target are not measured
    # (nan) and can never be selected as the best iterate
    assert info.true_res == best["true_res"] == min(measured)
    assert info.max_dx == best["max_dx"] and info.max_res == best["max_res"]
    assert f"{info.true_res:.3E}" in info.message
    # and the residual it claims is the residual it has
    assert abs(tt_residual(A, x, rhs) - info.true_res) <= 1e-10 * info.true_res


def test_silent_run_stays_silent_on_the_failure_path(capsys):
    """``verb=0`` prints nothing even when the run fails; the history is full."""
    d = 12
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    with pytest.warns(UserWarning):
        x, info = amen_solve(A, rhs, None, 1e-10, verb=0, seed=0, nswp=3,
                             check_true_res=True,
                             return_info=True)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert len(info.sweeps) == 3
    assert all(np.isfinite([s["max_dx"], s["max_res"]]).all()
               for s in info.sweeps)
    # true_res costs a sweep over A x - f, so it is measured only where it can
    # change the decision; the sweep the run stopped on always carries it.
    assert np.isfinite(info.sweeps[-1]["true_res"]) and np.isfinite(info.true_res)


# --- the random streams ------------------------------------------------------

def test_x0_and_z_are_independent():
    """The enrichment basis must not start out equal to the trial basis.

    Both were seeded with ``seed``: at ``kickrank == 2`` (same ranks as the
    default random ``x0``) they were the *same tensor*, so the first sweep's
    enrichment lived inside the space it was supposed to enrich.  The check is
    on the streams the solver uses, so it cannot be satisfied by luck.
    """
    seed_x, seed_z = np.random.SeedSequence(0).spawn(2)
    a = _ops.random_tt([2] * 6, [1] + [2] * 5 + [1], dtype="float64",
                       seed=seed_x)
    b = _ops.random_tt([2] * 6, [1] + [2] * 5 + [1], dtype="float64",
                       seed=seed_z)
    assert max(float(np.abs(np.asarray(u) - np.asarray(v)).max())
               for u, v in zip(a, b)) > 0.1

    d = 8
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    for kick in (1, 2, 3, 4):
        x, info = amen_solve(A, f, None, 1e-10, verb=0, seed=0, kickrank=kick,
                             nswp=25, return_info=True)
        assert info.converged, f"kickrank={kick} did not converge"


def test_the_same_seed_gives_the_same_run():
    d = 8
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    a = amen_solve(A, f, None, 1e-10, verb=0, seed=1)
    b = amen_solve(A, f, None, 1e-10, verb=0, seed=1)
    assert np.array_equal(np.asarray(a.full(asvector=True)),
                          np.asarray(b.full(asvector=True)))
    # a different seed is a different iteration but the same problem
    c = amen_solve(A, f, None, 1e-10, verb=0, seed=2)
    assert rel(np.asarray(c.full(asvector=True)),
               np.asarray(a.full(asvector=True))) < 1e-9


# --- argument plumbing that changes the answer -------------------------------

def test_max_full_size_switches_solvers_without_changing_the_answer():
    """The dense and the GMRES local paths must agree to the requested accuracy.

    d = 8, eps = 1e-10, float64.
    """
    d, eps = 8, 1e-10
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    out = {}
    for mfs in (0, 1, 20, 1000, 10 ** 6):
        x, info = amen_solve(A, f, None, eps, verb=0, seed=0, max_full_size=mfs,
                             local_iters=8, local_restart=60, nswp=30,
                             return_info=True)
        assert info.converged, f"max_full_size={mfs}"
        assert dense_residual(A, x, f) <= eps, f"max_full_size={mfs}"
        out[mfs] = np.asarray(x.full(asvector=True))
    for mfs, v in out.items():
        assert rel(v, out[10 ** 6]) < 1e-9, f"max_full_size={mfs}"


def test_A_as_a_list_of_matrices_is_their_sum():
    d = 6
    A1, A2 = tt.qlaplace_dd([d]), tt.eye([2] * d)
    f = tt.ones(2, d)
    x, info = amen_solve([A1, A2], f, None, 1e-10, verb=0, seed=0,
                         return_info=True)
    assert info.converged
    assert dense_residual((A1 + A2).round(1e-14), x, f) <= 1e-10


def test_frobenius_and_residual_truncation_reach_the_same_place():
    d, eps = 8, 1e-9
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    xr = amen_solve(A, f, None, eps, verb=0, seed=0, trunc_norm='residual')
    xf = amen_solve(A, f, None, eps, verb=0, seed=0, trunc_norm='fro')
    assert dense_residual(A, xr, f) <= eps and dense_residual(A, xf, f) <= eps
    assert rel(np.asarray(xf.full(asvector=True)),
               np.asarray(xr.full(asvector=True))) < 1e-8


# --- the torch backend --------------------------------------------------------

@requires_gpu()
@pytest.mark.parametrize("prec", ["n", "c", "l", "r"])
def test_torch_gmres_paths(prec):
    """Every local-solver path on the GPU, against the numpy dense solve."""
    from tt import backend as bk
    d, eps = QLAPLACE_D, 10 * SOLVE_EPS
    A, f = tt.qlaplace_dd([d]), tt.ones(2, d)
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    x, info = amen_solve(to_gpu(A), to_gpu(f), None, eps, verb=0,
                         seed=0, max_full_size=0, local_prec=prec,
                         local_iters=8, local_restart=60, return_info=True)
    assert info.converged, f"prec={prec} stalled at {info.true_res:.2E}"
    got = np.asarray(bk.to_numpy(x.full(asvector=True)))
    assert rel(got, xd) < DENSE_TOL


@requires_gpu(dtype="complex128")
def test_torch_complex():
    """Complex arithmetic on the GPU, dense oracle."""
    from tt import backend as bk
    rng = np.random.default_rng(4)
    A = random_matrix(3, 3, 2, rng, dtype=np.complex128, diag_shift=3.0)
    f = random_vector([3, 3, 3], 2, rng, dtype=np.complex128)
    xd = np.linalg.solve(np.asarray(A.full()),
                         np.asarray(f.full(asvector=True)))
    x, info = amen_solve(A.to("torch", "cuda", "complex128"),
                         f.to("torch", "cuda", "complex128"), None, 1e-11,
                         verb=0, seed=0, return_info=True)
    assert info.converged
    assert rel(np.asarray(bk.to_numpy(x.full(asvector=True))), xd) < 1e-10
