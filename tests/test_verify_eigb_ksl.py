"""Adversarial verification of ``tt.algs.eigb`` and ``tt.algs.ksl``.

These are the tests the module's own suite does not have.  Two of them replace
an oracle rather than add a case:

* **the convergence order.**  ``tests/test_eigb_ksl.py`` measures it by
  Richardson extrapolation *of the integrator against itself*, on the argument
  that no independent reference exists: with full ranks the splitting is exact
  and shows no order, and with deficient ranks the tau-independent modelling
  error swamps the splitting error.  The second half of that argument is wrong,
  and :func:`test_ksl_order_against_the_dense_projected_flow` shows why: the
  right reference is not ``expm(tau A) y0`` but the solution of the *projected*
  ODE ``y' = P_{T_y M} A y``, which is what KSL discretizes.  Integrated densely
  (a projector built here from scratch, numpy only) it is an oracle the module
  never touches, and against it the observed orders are 1.00 and 2.00 while the
  modelling error is 25x larger than the largest splitting error being measured.

* **the eigenresidual.**  ``eigb`` used to report only ``ermax``, the movement
  of the Ritz values.  A stalled alternating iteration has ``ermax = 0`` at a
  point that is not an eigenvector, and the module returned that answer with
  ``converged=True`` and no residual anywhere in the API
  (:func:`test_eigb_reports_a_stalled_iteration`).

Everything is checked against dense numpy/scipy truth, an independent dense
construction, or a mathematical invariant.  Nothing is compared with the legacy
Fortran, and nothing is compared with the module's own output except where the
test is explicitly about two paths of the module agreeing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg as sla

import tt
from tt.algs.eigb import block_residuals, eigb
from tt.algs.ksl import diag_ksl, expmv_krylov, ksl, tangent_defect

try:                       # optional second backend: those tests skip without it
    import torch as _torch
except ImportError:        # pragma: no cover - depends on the environment
    _torch = None
needs_torch = pytest.mark.skipif(_torch is None, reason="torch is not installed")


# --- input builders (seeded: a test must not depend on the draw) -------------

def rand_tt(n, r, seed):
    rng = np.random.default_rng(seed)
    return tt.rand(n, r=r, samplefunc=rng.standard_normal)


def sym_tt_matrix(d, n, r=2, seed=0):
    """Random symmetric TT-matrix normalized to ``||A||_2 = 1``."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [rng.standard_normal((ranks[k], n, n, ranks[k + 1])) for k in range(d)]
    m = tt.matrix.from_list(cores)
    m = (m + m.T).round(1e-14)
    return (1.0 / np.linalg.norm(m.full(), 2)) * m


def herm_tt_matrix(d, n, r=2, seed=0):
    """Complex Hermitian TT-matrix: every core slice is Hermitian, links real."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = np.zeros((ranks[k], n, n, ranks[k + 1]), dtype=complex)
        for a in range(ranks[k]):
            for b in range(ranks[k + 1]):
                h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
                c[a, :, :, b] = h + h.conj().T
        cores.append(c)
    m = tt.matrix.from_list(cores)
    return (1.0 / np.linalg.norm(m.full(), 2)) * m


def block_columns(y):
    """The ``B`` columns of a block TT-vector, densely (``full`` squeezes B=1)."""
    nblock = int(y.r[-1])
    full = np.asarray(y.full())
    if nblock == 1:
        return full.reshape(-1, order="F")[:, None]
    return np.stack([full[..., b].reshape(-1, order="F") for b in range(nblock)], 1)


# --- an independent dense implementation of the TT manifold ------------------
# Only numpy, only the index convention from the README (mode 1 fastest).  This
# is the oracle for tangent_defect and for the order of the integrator, so it
# gets its own self-check first.

def _left_frames(cores):
    """``L[k]``: ``(prod n_{<k}, r_k)``, columns orthonormal if the cores are."""
    out = [np.ones((1, 1))]
    for c in cores:
        t = np.einsum("ia,ajb->ijb", out[-1], c)
        out.append(t.reshape((-1, c.shape[2]), order="F"))
    return out


def _right_frames(cores):
    """``R[k]``: ``(prod n_{>=k}, r_k)``."""
    d = len(cores)
    out = [None] * (d + 1)
    out[d] = np.ones((1, 1))
    for k in range(d - 1, -1, -1):
        t = np.einsum("ajb,ib->aji", cores[k], out[k + 1])
        out[k] = t.reshape((cores[k].shape[0], -1), order="F").T
    return out


def dense_tangent_projector(cores):
    """Orthogonal projector onto the tangent space of the fixed-rank TT manifold.

    ``P = sum_{k<d} U_k [(I - Q_k Q_k^H) (x) I] U_k^H + U_d U_d^H`` with ``U_k``
    the orthonormal frame ``Y_{<k} (x) I_{n_k} (x) Y_{>k}`` and ``Q_k`` the left
    unfolding of the ``k``-th core in the left-orthogonal gauge (Lubich,
    Oseledets, Vandereycken 2015, Sec. 2).
    """
    cores = [np.array(c, dtype=float) for c in cores]
    d = len(cores)
    left = [c.copy() for c in cores]                    # left-orthogonal gauge
    for k in range(d - 1):
        r0, n, r1 = left[k].shape
        q, s = np.linalg.qr(left[k].reshape((r0 * n, r1)))
        left[k] = q.reshape((r0, n, -1))
        left[k + 1] = np.einsum("ab,bjc->ajc", s, left[k + 1])
    right = [c.copy() for c in cores]                   # right-orthogonal gauge
    for k in range(d - 1, 0, -1):
        r0, n, r1 = right[k].shape
        q, s = np.linalg.qr(right[k].reshape((r0, n * r1)).T)
        right[k] = q.T.reshape((-1, n, r1))
        right[k - 1] = np.einsum("ajb,cb->ajc", right[k - 1], s)
    L, R = _left_frames(left), _right_frames(right)

    N = int(np.prod([c.shape[1] for c in cores]))
    P = np.zeros((N, N))
    for k in range(d):
        r0, n, r1 = left[k].shape
        Lk, Rk = L[k], R[k + 1]
        U = np.zeros((N, r0 * n * r1))
        col = 0
        for a in range(r0):          # C order (a, i, b), matching the unfolding
            for i in range(n):
                for b in range(r1):
                    v = np.zeros((Lk.shape[0], n, Rk.shape[0]))
                    v[:, i, :] = np.outer(Lk[:, a], Rk[:, b])
                    U[:, col] = v.reshape(-1, order="F")
                    col += 1
        if k < d - 1:
            Q = left[k].reshape((r0 * n, r1))
            gauge = np.kron(np.eye(r0 * n) - Q @ Q.T, np.eye(r1))
            P += U @ gauge @ U.T
        else:
            P += U @ U.T
    return P


def tt_svd_fixed_rank(v, ns, ranks):
    """Dense vector -> TT cores at prescribed ranks (numpy only)."""
    cores, r = [], 1
    c = v.reshape(ns, order="F")
    for k in range(len(ns) - 1):
        u, s, vh = np.linalg.svd(c.reshape((r * ns[k], -1), order="F"),
                                 full_matrices=False)
        rk = min(ranks[k + 1], len(s))
        cores.append(u[:, :rk].reshape((r, ns[k], rk), order="F"))
        c = np.diag(s[:rk]) @ vh[:rk, :]
        r = rk
    cores.append(c.reshape((r, ns[-1], 1), order="F"))
    return cores


def test_the_dense_oracle_is_a_tangent_projector():
    """Self-check of the oracle used by the two tests below.

    A wrong projector would silently turn them into decoration: it must be an
    orthogonal projector, of the dimension of the manifold's tangent space, it
    must fix ``y`` itself (the manifold is a cone) and it must fix the velocity
    of a curve that moves one core.
    """
    d, n = 4, 2
    ranks = [1, 2, 2, 2, 1]
    y = rand_tt([n] * d, ranks, seed=21)
    P = dense_tangent_projector(y.cores)
    v = np.asarray(y.full(asvector=True))

    assert np.abs(P - P.T).max() < 1e-14
    assert np.abs(P @ P - P).max() < 1e-13
    dim = sum((ranks[k] * n - ranks[k + 1]) * ranks[k + 1] for k in range(d - 1))
    dim += ranks[d - 1] * n * ranks[d]
    assert abs(np.trace(P) - dim) < 1e-10
    assert np.linalg.norm(P @ v - v) / np.linalg.norm(v) < 1e-13

    rng = np.random.default_rng(0)
    moved = [c.copy() for c in y.cores]
    moved[1] = moved[1] + 1e-6 * rng.standard_normal(moved[1].shape)
    vel = (np.asarray(tt.vector.from_list(moved).full(asvector=True)) - v) / 1e-6
    assert np.linalg.norm(vel - P @ vel) / np.linalg.norm(vel) < 1e-8


@pytest.mark.parametrize("ranks, seed", [([1, 2, 2, 2, 1], 21),
                                         ([1, 2, 3, 2, 1], 22),
                                         ([1, 1, 1, 1, 1], 24)])
def test_tangent_defect_matches_the_dense_projector(ranks, seed):
    """``||(I - P) A y||`` from the TT sweep equals the dense computation.

    The module's own test only checks that the defect *vanishes* when the
    tangent space is everything, which any function returning something small
    would pass.  Here the defect is O(1) and the two numbers must agree.
    """
    d, n = 4, 2
    A = sym_tt_matrix(d, n, 2, seed=3)
    y = rand_tt([n] * d, ranks, seed=seed)
    y = (1.0 / y.norm()) * y

    got, znorm = tangent_defect(A, y)
    P = dense_tangent_projector(y.cores)
    z = np.asarray(A.full()) @ np.asarray(y.full(asvector=True))
    ref = np.linalg.norm(z - P @ z)

    assert abs(znorm - np.linalg.norm(z)) < 1e-12 * np.linalg.norm(z)
    assert ref > 1e-3, "the case must have a defect, otherwise it proves nothing"
    assert abs(got - ref) < 1e-9 * ref, f"tt {got:.6e} vs dense {ref:.6e}"


@pytest.mark.parametrize("scheme, expected", [("first", 1.0), ("symm", 2.0)])
def test_ksl_order_against_the_dense_projected_flow(scheme, expected):
    """The temporal order, measured against an oracle the module never touches.

    KSL discretizes ``y' = P_{T_y M} A y``; that ODE is integrated here densely
    with DOP853 at rtol 1e-11, using the projector built above.  The modelling
    error (the distance between the projected flow and ``expm(T A) y0``) is
    checked to be much larger than the splitting errors being fitted, which is
    exactly the regime in which the module's docstring claims no order can be
    measured -- it can, against the right reference.
    """
    from scipy.integrate import solve_ivp

    d, n = 4, 2
    ranks = [1, 2, 2, 2, 1]
    A = sym_tt_matrix(d, n, 2, seed=3)
    dense = np.asarray(A.full())
    y0 = rand_tt([n] * d, ranks, seed=25)
    y0 = (1.0 / y0.norm()) * y0
    v0 = np.asarray(y0.full(asvector=True))

    def rhs(_t, v):
        return dense_tangent_projector(tt_svd_fixed_rank(v, [n] * d, ranks)) @ (dense @ v)

    total = 0.5
    ref = solve_ivp(rhs, [0.0, total], v0, rtol=1e-11, atol=1e-13,
                    method="DOP853").y[:, -1]
    modelling = np.linalg.norm(ref - sla.expm(total * dense) @ v0)

    errs = []
    for nsteps in (2, 4, 8, 16):
        y = y0
        for _ in range(nsteps):
            y = ksl(A, y, total / nsteps, verb=0, scheme=scheme, local_tol=1e-13,
                    check_rank=False)
        errs.append(np.linalg.norm(np.asarray(y.full(asvector=True)) - ref))
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])

    assert modelling > 10 * errs[0], (
        "the regime is wrong: the modelling error must dominate the splitting "
        f"error, got {modelling:.3e} vs {errs[0]:.3e}")
    assert np.abs(orders - expected).max() < 0.2, f"orders {orders}, errs {errs}"


# --- eigb: the answer must come with evidence --------------------------------

def test_eigb_reports_a_stalled_iteration():
    """A degenerate initial guess makes ALS stall; that must not read as success.

    ``tt.zeros`` is the extreme case: every QR in the sweep returns an arbitrary
    orthonormal frame, the Ritz values never move, ``ermax`` goes to 1e-16 and
    the iteration reports convergence -- to 0.0625 where the true smallest
    eigenvalue of ``qlaplace_dd([5])`` is 0.00906, with a residual of 0.24.
    Before the fix nothing in the API could tell the two apart.
    """
    d = 5
    A = tt.qlaplace_dd([d])
    with pytest.warns(RuntimeWarning, match="eigenresidual"):
        y, lam, hist = eigb(A, tt.zeros([2] * d), 1e-8, verb=0, return_history=True)

    exact = np.linalg.eigh(np.asarray(A.full()))[0][0]
    assert abs(lam[0] - exact) > 1e-2, "the guess was supposed to be degenerate"
    assert hist.ermax < 1e-10 and hist.converged      # the old indicator says fine
    assert hist.res_rel[0] > 0.5                      # the new one does not
    col = block_columns(y)[:, 0]
    dense_res = np.linalg.norm(np.asarray(A.full()) @ col - lam[0] * col)
    assert abs(hist.res[0] - dense_res) < 1e-10 * dense_res


def test_eigb_reports_a_local_solver_that_did_not_converge():
    """LOBPCG capped at one iteration returns garbage; the run must say so.

    ``lam`` comes out 100% wrong (7.6e-05 for a true 9.4e-06).  The sweep
    indicator alone reports "did not converge in 20 sweeps", which a user may
    read as slow convergence; the residual says the returned pairs are not
    eigenpairs at all.
    """
    d, nblock = 10, 4
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [6] * (d - 1) + [nblock], seed=11)
    with pytest.warns(RuntimeWarning, match="eigenresidual"):
        _, lam, hist = eigb(A, x, 1e-8, verb=0, max_full_size=30, lobpcg_maxiter=1,
                            return_history=True)
    exact = np.linalg.eigh(np.asarray(A.full()))[0][:nblock]
    assert np.abs(lam - exact).max() > 1e-5
    assert hist.max_local_res > 1e-3
    assert np.min(hist.res_rel) > 0.5


@pytest.mark.parametrize("d, nblock", [(6, 1), (6, 3), (8, 2), (10, 4)])
def test_eigb_residuals_match_a_dense_computation(d, nblock):
    """``history.res`` is the true residual, not a formula that cancels.

    Computed in TT (matvec, sum, QR sweep) and compared with
    ``||A y_i - lam_i y_i||`` on the dense vectors.  The naive expansion
    ``||Ay||^2 - 2 lam <Ay, y> + lam^2`` would saturate near 1e-9 here and pass
    a loose test while being meaningless.
    """
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [max(4, nblock + 1)] * (d - 1) + [nblock],
                seed=d * 100 + nblock)
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # a well posed run warns about nothing
        y, lam, hist = eigb(A, x, 1e-8, verb=0, return_history=True)

    dense = np.asarray(A.full())
    cols = block_columns(y)
    ref = np.array([np.linalg.norm(dense @ cols[:, b] - lam[b] * cols[:, b])
                    for b in range(nblock)])
    # A converged eigenpair here has a residual of ~1e-16, i.e. pure rounding
    # noise, and two ways of computing noise agree only in order of magnitude.
    # The absolute floor is what that noise costs: eps * ||A||_2 * ||y||, with a
    # factor for the O(d) operations that build the residual in TT.  Above the
    # floor the relative check still bites.
    floor = 50 * np.finfo(float).eps * np.linalg.norm(dense, 2)
    assert np.allclose(hist.res, ref, rtol=1e-6, atol=floor), (
        f"{hist.res} vs {ref} (floor {floor:.2e})")
    assert np.max(hist.res_rel) < 1e-3
    # and block_residuals is callable on its own, on any block vector
    res2, znorm = block_residuals(A, y, lam)
    assert np.allclose(res2, ref, rtol=1e-6, atol=floor)   # same noise floor
    assert np.all(znorm > 0)


def test_eigb_verb_zero_is_silent(capsys):
    """``verb=0`` prints nothing, yet the history is complete."""
    d = 6
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=15)
    _, _, hist = eigb(A, x, 1e-10, verb=0, return_history=True)
    assert capsys.readouterr().out == ""
    assert hist.steps and hist.sweeps and hist.res is not None
    assert hist.lam is not None and hist.ranks


def test_eigb_verb_one_prints_the_residual(capsys):
    d = 6
    A = tt.qlaplace_dd([d])
    x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=15)
    eigb(A, x, 1e-10, verb=1)
    out = capsys.readouterr().out
    assert "Eigenresiduals" in out and "swp" in out


# --- eigb: dtypes and fields the suite never exercised -----------------------

def test_eigb_in_float32():
    """float32 must work: a fixed 1e-8 symmetry tolerance rejected every input.

    The projected local matrix of a float32 problem is asymmetric at the 1e-7
    level from rounding alone, so ``sym_tol=1e-8`` raised "not Hermitian" on a
    perfectly symmetric operator.  The tolerance now follows the dtype.
    """
    d = 6
    A = tt.qlaplace_dd([d])
    A32 = tt.matrix.from_list([np.ascontiguousarray(c).astype(np.float32)
                               for c in tt.matrix.to_list(A)])
    x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=35)
    x32 = tt.vector.from_list([c.astype(np.float32) for c in x.cores])

    y, lam, hist = eigb(A32, x32, 1e-5, verb=0, return_history=True)
    exact = np.linalg.eigh(np.asarray(A.full()))[0][:2]
    assert np.allclose(lam, exact, rtol=1e-4, atol=0), f"{lam} vs {exact}"
    assert np.max(hist.res_rel) < 1e-3
    assert tt.vector.from_list(y.cores).dtype == "float32"


@pytest.mark.parametrize("max_full_size", [10 ** 6, 30])
def test_eigb_on_a_complex_hermitian_matrix(max_full_size):
    """Complex Hermitian input, both local solvers, and no ComplexWarning.

    The LOBPCG path used to cast its (real) eigenvalues to the complex dtype of
    the eigenvectors and back, which raised a ``ComplexWarning`` per local solve
    -- an error for anyone running under ``-W error``.
    """
    d = 6
    A = herm_tt_matrix(d, 2, 2, seed=5)
    dense = np.asarray(A.full())
    assert np.abs(dense - dense.conj().T).max() == 0.0
    x = rand_tt([2] * d, [1] + [8] * (d - 1) + [2], seed=27)
    xc = tt.vector.from_list([c.astype(complex) for c in x.cores])

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        y, lam, hist = eigb(A, xc, 1e-9, verb=0, max_full_size=max_full_size,
                            nswp=8, return_history=True)

    exact = np.linalg.eigh(dense)[0][:2]
    assert np.allclose(lam, exact, atol=1e-8, rtol=0), f"{lam} vs {exact}"
    cols = block_columns(y)
    assert np.abs(cols.conj().T @ cols - np.eye(2)).max() < 1e-8
    for b in range(2):
        assert np.linalg.norm(dense @ cols[:, b] - lam[b] * cols[:, b]) < 1e-7
    assert np.max(hist.res_rel) < 1e-6


def test_eigb_with_mode_sizes_that_differ_per_mode():
    """Nothing in the sweep may assume a QTT-like constant mode size."""
    ns = [2, 3, 4, 2]
    rng = np.random.default_rng(4)
    ranks = [1, 2, 2, 2, 1]
    cores = [rng.standard_normal((ranks[k], ns[k], ns[k], ranks[k + 1]))
             for k in range(len(ns))]
    A = (tt.matrix.from_list(cores) + tt.matrix.from_list(cores).T).round(1e-14)
    dense = np.asarray(A.full())
    x = rand_tt(ns, [1, 4, 4, 4, 2], seed=8)

    y, lam, hist = eigb(A, x, 1e-9, verb=0, return_history=True)
    exact = np.linalg.eigh(dense)[0][:2]
    assert np.allclose(lam, exact, rtol=1e-8, atol=1e-10), f"{lam} vs {exact}"
    assert np.max(hist.res_rel) < 1e-7


@pytest.mark.parametrize("nblock", [1, 2])
def test_eigb_d1(nblock):
    """``d = 1``: there is no sweep, the local problem is the whole problem."""
    a = np.array([[2.0, -1.0], [-1.0, 2.0]])
    A = tt.matrix.from_list([a.reshape(1, 2, 2, 1)])
    x = tt.vector.from_list([np.eye(2)[:, :nblock].reshape(1, 2, nblock)])
    y, lam, hist = eigb(A, x, 1e-10, verb=0, return_history=True)
    assert np.allclose(lam, np.linalg.eigh(a)[0][:nblock])
    assert hist.converged and np.max(hist.res_rel) < 1e-12
    assert int(y.r[-1]) == nblock


def test_eigb_d2():
    A = tt.qlaplace_dd([2])
    x = rand_tt([2, 2], [1, 2, 2], seed=5)
    y, lam, hist = eigb(A, x, 1e-10, verb=0, return_history=True)
    exact = np.linalg.eigh(np.asarray(A.full()))[0][:2]
    assert np.allclose(lam, exact)
    assert len(hist.steps) == 2 * 2 - 2
    assert np.max(hist.res_rel) < 1e-10


def test_eigb_from_a_rank_one_guess_is_wrong_and_says_so():
    """The classical call -- ``B = 1``, rank-1 random start -- does not work.

    One-site ALS cannot grow a rank: the block index is what allows an interface
    to reach ``min(B r, n r')``, so with ``B = 1`` and ``r = 1`` the iterate is
    trapped on the rank-1 manifold for every one of the 20 sweeps.  On
    ``qlaplace_dd([8])`` it stops at 7.8e-03 where the smallest eigenvalue is
    1.5e-04, with ``ermax = 1e-14``: a converged-looking run, wrong by a factor
    of 50.  Every test in the module's own suite starts from rank >= 4, which is
    why this never showed up.

    The answer cannot be fixed inside a one-site method -- but it must not be
    handed back as if it were converged, and the measured residual (0.996 of
    ``||A y||``) is what says so.
    """
    d = 8
    A = tt.qlaplace_dd([d])
    with pytest.warns(RuntimeWarning, match="eigenresidual"):
        _, lam, hist = eigb(A, rand_tt([2] * d, 1, seed=61), 1e-10, verb=0,
                            return_history=True)
    exact = np.linalg.eigh(np.asarray(A.full()))[0][0]
    assert lam[0] > 10 * exact and hist.converged and hist.ermax < 1e-10
    assert hist.res_rel[0] > 0.5
    assert max(hist.ranks) == 1

    # the documented cure: more rank in the guess (still B = 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, lam4, hist4 = eigb(A, rand_tt([2] * d, [1] + [4] * (d - 1) + [1], seed=62),
                              1e-10, verb=0, return_history=True)
    assert abs(lam4[0] - exact) < 1e-10 * abs(exact)
    assert hist4.res_rel[0] < 1e-8


# --- ksl: cases the suite did not cover --------------------------------------

@pytest.mark.parametrize("d, ranks", [(1, [1, 1]), (2, [1, 2, 1])])
def test_ksl_small_d_against_dense_expm(d, ranks):
    """``d = 1`` (no sweep at all) and ``d = 2`` (one interface), at full rank."""
    n = 3 if d == 1 else 2
    A = sym_tt_matrix(d, n, 2, seed=2)
    y0 = rand_tt([n] * d, ranks, seed=6)
    y = ksl(A, y0, 0.3, verb=0, check_rank=False, local_tol=1e-13)
    exact = sla.expm(0.3 * np.asarray(A.full())) @ np.asarray(y0.full(asvector=True))
    err = np.linalg.norm(np.asarray(y.full(asvector=True)) - exact)
    assert err / np.linalg.norm(exact) < 1e-12


def test_ksl_with_mode_sizes_that_differ_per_mode():
    ns = [2, 3, 4, 2]
    rng = np.random.default_rng(4)
    ranks = [1, 2, 2, 2, 1]
    cores = [rng.standard_normal((ranks[k], ns[k], ns[k], ranks[k + 1]))
             for k in range(len(ns))]
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A
    y0 = rand_tt(ns, [1, 2, 6, 2, 1], seed=10)      # full ranks -> exact
    y = ksl(A, y0, 0.2, verb=0, check_rank=False, local_tol=1e-13)
    exact = sla.expm(0.2 * np.asarray(A.full())) @ np.asarray(y0.full(asvector=True))
    err = np.linalg.norm(np.asarray(y.full(asvector=True)) - exact)
    assert err / np.linalg.norm(exact) < 1e-11
    assert list(y.r) == list(y0.r)


def test_ksl_does_not_need_a_symmetric_operator():
    """The projector splitting is defined for any ``A``; only eigb needs symmetry."""
    d, n = 5, 2
    rng = np.random.default_rng(12)
    ranks = [1, 2, 2, 2, 2, 1]
    cores = [rng.standard_normal((ranks[k], n, n, ranks[k + 1])) for k in range(d)]
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=13)      # full ranks -> exact
    y = ksl(A, y0, 0.3, verb=0, check_rank=False, local_tol=1e-13)
    exact = sla.expm(0.3 * np.asarray(A.full())) @ np.asarray(y0.full(asvector=True))
    assert np.linalg.norm(np.asarray(y.full(asvector=True)) - exact) < 1e-12


def test_ksl_default_local_tol_delivers_what_it_promises():
    """Users get ``local_tol=1e-8``; the suite only ever measured 1e-13."""
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=41)
    y0 = (1.0 / y0.norm()) * y0
    y, hist = ksl(A, y0, 0.5, verb=0, check_rank=False, return_history=True)
    exact = sla.expm(0.5 * np.asarray(A.full())) @ np.asarray(y0.full(asvector=True))
    err = np.linalg.norm(np.asarray(y.full(asvector=True)) - exact) / np.linalg.norm(exact)
    assert err < 1e-8, err
    assert hist.max_local_err < 1e-8


def test_ksl_zero_tensor_is_a_fixed_point():
    """``y0 = 0``: the answer is 0 and the run says the estimate is meaningless."""
    d, n = 4, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    with pytest.warns(RuntimeWarning):
        y, hist = ksl(A, tt.zeros([n] * d), 0.1, verb=0, return_history=True)
    assert y.norm() == 0.0
    assert not np.isfinite(hist.step_error_est)   # 0/0 is reported, not invented


def test_ksl_tau_zero_and_reversibility():
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=17)
    same = ksl(A, y0, 0.0, verb=0, check_rank=False)
    assert (same - y0).norm() / y0.norm() < 1e-14
    back = ksl(A, ksl(A, y0, 0.4, verb=0, check_rank=False, local_tol=1e-13),
               -0.4, verb=0, check_rank=False, local_tol=1e-13)
    assert (back - y0).norm() / y0.norm() < 1e-11


def test_ksl_at_a_rank_deficient_point():
    """A point where the manifold is not a manifold must not produce NaN.

    Two of the four directions of a core are exactly zero, so the S-step
    operator is singular; the projector splitting is provably robust to this
    (Kieri, Lubich, Walach 2016) and the result must still match the dense
    exponential, since the ranks are full for this mode size.
    """
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    cores = [c.copy() for c in rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=28).cores]
    cores[2][:, :, 2:] = 0.0
    y0 = tt.vector.from_list(cores)
    y0 = (1.0 / y0.norm()) * y0

    y = ksl(A, y0, 0.3, verb=0, check_rank=False, local_tol=1e-13)
    got = np.asarray(y.full(asvector=True))
    exact = sla.expm(0.3 * np.asarray(A.full())) @ np.asarray(y0.full(asvector=True))
    assert np.all(np.isfinite(got))
    assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-11


def test_ksl_history_keeps_a_complex_step():
    """``tau = -0.3i`` is the Schroedinger case; the history must not drop the i."""
    d, n = 5, 2
    A = sym_tt_matrix(d, n, 2, seed=7)
    y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=19)
    y, hist = ksl(A, y0, -0.3j, verb=0, check_rank=False, return_history=True)
    assert hist.tau == -0.3j
    assert "0.3j" in repr(hist)


def test_diag_ksl_accepts_a_matrix_as_well_as_a_vector():
    """The legacy signature takes the diagonal as a TT-vector; a matrix must work."""
    d, n = 4, 2
    rng = np.random.default_rng(4)
    v = tt.rand([n] * d, r=2, samplefunc=rng.standard_normal)
    y0 = tt.rand([n] * d, r=[1, 2, 4, 2, 1], samplefunc=rng.standard_normal)
    exact = sla.expm(0.1 * np.diag(np.asarray(v.full(asvector=True)))) \
        @ np.asarray(y0.full(asvector=True))
    for arg in (v, tt.diag(v)):
        y = diag_ksl(arg, y0, 0.1, verb=0, check_rank=False, local_tol=1e-13)
        err = np.linalg.norm(np.asarray(y.full(asvector=True)) - exact)
        assert err / np.linalg.norm(exact) < 1e-11


def test_expmv_krylov_on_a_strongly_non_normal_operator():
    """What the Krylov solver actually delivers where the estimate is weakest.

    ``a = 5 triu(1) - I`` of size 60: ``||exp(a) x|| / ||x||`` is about 1e5, so
    the flow grows enormously and the error estimate -- which is normalized by
    ``||x||``, the input -- has no clean relation to the error of the answer.
    Measured: asked for ``tol = 1e-10``, delivered 5.5e-08 relative to the
    result, reported ``err_est = 6.4e-02``.  Both directions of that mismatch
    are the estimate's normalization, not a wrong answer, and the point of the
    test is to pin the delivered accuracy so a regression cannot hide behind the
    estimate.
    """
    n = 60
    a = np.triu(np.ones((n, n)), 1) * 5.0 - np.eye(n)
    x = np.random.default_rng(0).standard_normal(n)
    w, info = expmv_krylov(lambda v: a @ v, x, 1.0, space=8, tol=1e-10)
    exact = sla.expm(a) @ x
    assert np.linalg.norm(exact) / np.linalg.norm(x) > 1e3        # the regime
    assert np.linalg.norm(w - exact) / np.linalg.norm(exact) < 1e-6
    assert info["err_est"] > 0.0 and info["krylov"] > 1
    # documented and measured: the requested tolerance is *not* what a growing
    # flow delivers, and the estimate is stated relative to the input norm
    assert info["err_est"] > 1e-10


# --- backends ----------------------------------------------------------------

@needs_torch
@pytest.mark.parametrize("what", ["eigb", "ksl"])
def test_algorithms_run_on_the_torch_backend(what):
    """Neither module was ever executed on the second backend."""
    tt.set_backend("torch", device="cpu", dtype="float64")
    try:
        d, n = 5, 2
        if what == "eigb":
            A = tt.qlaplace_dd([d])
            x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=31)
            _, lam, hist = eigb(A, x, 1e-10, verb=0, return_history=True)
            exact = np.linalg.eigh(np.asarray(A.full()))[0][:2]
            assert np.allclose(lam, exact, rtol=1e-8, atol=1e-12)
            assert np.max(hist.res_rel) < 1e-7
        else:
            cores = [_torch.as_tensor(np.ascontiguousarray(c), dtype=_torch.float64)
                     for c in tt.matrix.to_list(sym_tt_matrix(d, n, 2, seed=7))]
            A = tt.matrix.from_list(cores)
            y0 = rand_tt([n] * d, [1, 2, 4, 4, 2, 1], seed=32)
            y, hist = ksl(A, y0, 0.3, verb=0, local_tol=1e-13, return_history=True)
            exact = sla.expm(0.3 * np.asarray(A.full())) \
                @ np.asarray(y0.full(asvector=True))
            got = np.asarray(y.full(asvector=True))
            assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-11
            assert hist.defect_rel < 1e-6
    finally:
        tt.set_backend("numpy")


@needs_torch
def test_a_numpy_operator_with_a_torch_iterate():
    """Mixing backends used to die inside einops with a torch/numpy TypeError.

    The operator follows the iterate (the convention of ``amen_mv``).
    """
    tt.set_backend("torch", device="cpu", dtype="float64")
    try:
        d = 5
        A = tt.qlaplace_dd([d])                       # built on torch
        A_np = tt.matrix.from_list([np.asarray(_torch.as_tensor(c).cpu())
                                    for c in tt.matrix.to_list(A)])
        x = rand_tt([2] * d, [1] + [4] * (d - 1) + [2], seed=33)   # torch
        _, lam = eigb(A_np, x, 1e-10, verb=0)
        exact = np.linalg.eigh(np.asarray(A.full()))[0][:2]
        assert np.allclose(lam, exact, rtol=1e-8, atol=1e-12)
        y = ksl(A_np, x, 0.05, verb=0, check_rank=True)
        assert np.all(np.isfinite(np.asarray(y.full())))
    finally:
        tt.set_backend("numpy")
