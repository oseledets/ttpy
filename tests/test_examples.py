"""End-to-end scenarios, taken from the legacy `examples/` directory.

The legacy examples only printed their results; here every one of them is
checked against an oracle that does not come from ttpy: an analytic formula, a
dense computation, or an exact identity.  These are the acceptance tests for
"the package still does what people used it for".

Modules that are not built yet are skipped, not silently passed.
"""

import numpy as np
import pytest

import tt


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


# --- examples/test_amen.py ---------------------------------------------------

def test_amen_solve_laplacian_residual():
    """Legacy: d=12 QTT Laplacian, right-hand side of ones. Oracle: the residual."""
    amen = pytest.importorskip("tt.algs.amen")
    d = 12
    A = tt.qlaplace_dd([d])
    f = tt.ones(2, d)
    x = amen.amen_solve(A, f, f, 1e-6, verb=0)
    res = (tt.matvec(A, x) - f).norm() / f.norm()
    assert res < 1e-6, f"AMEn returned a solution with residual {res:.3e}"


def test_amen_solve_matches_dense_for_small_d():
    amen = pytest.importorskip("tt.algs.amen")
    d = 6
    A = tt.qlaplace_dd([d])
    f = tt.ones(2, d)
    x = amen.amen_solve(A, f, f, 1e-10, verb=0)
    ref = np.linalg.solve(A.full(), np.asarray(f.full(asvector=True)))
    assert rel(x.full(asvector=True), ref) < 1e-8


# --- examples/test_cross.py --------------------------------------------------

def test_multifuncrs_sinc_integral_is_pi_over_two():
    """Legacy: sin(x)/x on 2^d QTT points; the sum must approach pi/2.

    Oracle: the analytic value of the improper integral, plus the tail estimate.
    """
    mf = pytest.importorskip("tt.algs.multifuncrs")
    d = 30
    n = 2 ** d
    b = 1e3
    h = b / (n + 1)
    x = (tt.xfun(2, d) + tt.ones(2, d)) * h
    y = mf.multifuncrs([x], lambda v: np.sin(v) / v, 1e-6,
                       y0=tt.ones(2, d), verb=0)
    approx = float(tt.dot(y, tt.ones(2, d)) * h)
    # int_0^b sin(x)/x dx = pi/2 - cos(b)/b + O(1/b^2); the midpoint sum adds O(h^2)
    assert abs(approx - np.pi / 2) < 2e-3, approx


# --- examples/test_multifuncrs.py --------------------------------------------

def test_multifuncrs2_sum_reproduces_exact_addition():
    """funs = sum over the arguments, so the answer is exactly a + b."""
    mf = pytest.importorskip("tt.algs.multifuncrs")
    a = tt.rand([3, 5, 7, 11], 4, [1, 4, 6, 5, 1])
    b = tt.rand([3, 5, 7, 11], 4, [1, 2, 4, 3, 1])
    c = mf.multifuncrs2([a, b], lambda v: np.sum(v, axis=1), eps=1e-8, verb=0)
    assert rel(c.full(), (a + b).full()) < 1e-6


# --- examples/test_eigb.py ---------------------------------------------------

def test_eigb_matches_analytic_laplacian_eigenvalues():
    """Legacy: 8 smallest eigenvalues of an 8-dimensional QTT Laplacian.

    Oracle: for the 1D discrete Laplacian with Dirichlet conditions on N points
    the eigenvalues are 4 sin^2(pi k / (2(N+1))); in d dimensions they are sums.
    """
    eigb_mod = pytest.importorskip("tt.algs.eigb")
    d, f = 4, 3                      # 2^4 points per axis, 3 axes: N = 16, 4096 total
    nblock = 4
    A = tt.qlaplace_dd([d] * f)
    N = 2 ** d
    lam1 = 4 * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1))) ** 2
    grid = sum(np.ix_(*([lam1] * f)))          # all sums lam1[k1]+lam1[k2]+lam1[k3]
    exact = np.sort(grid.ravel())[:nblock]

    ranks = [1] + [8] * (d * f - 1) + [nblock]
    x0 = tt.rand([2] * (d * f), d * f, ranks)
    y, lam = eigb_mod.eigb(A, x0, 1e-6, verb=0)
    assert np.max(np.abs(np.sort(np.asarray(lam)) - exact)) < 1e-4 * exact[-1], (
        f"eigenvalues {np.sort(np.asarray(lam))} vs analytic {exact}")


# --- examples/test_common.py -------------------------------------------------

def test_gmres_solves_a_well_conditioned_system():
    """Restarted GMRES without a preconditioner needs a decent condition number.

    The bare QTT Laplacian on 2^8 points has cond ~ 2.6e4, where GMRES(20)
    stagnates -- that is a property of the method, not a defect, so it is tested
    separately below. Here the operator is shifted to cond ~ 1.4.
    """
    solvers = pytest.importorskip("tt.algs.solvers")
    d = 8
    A = (tt.eye(2, d) + 0.1 * tt.qlaplace_dd([d])).round(1e-14)
    rhs = tt.ones(2, d)
    x0 = tt.ones(2, d) * 0.0
    x, res = solvers.GMRES(lambda v, eps: tt.matvec(A, v).round(eps),
                           x0, rhs, eps=1e-8, maxit=100, m=20)
    assert (tt.matvec(A, x) - rhs).norm() / rhs.norm() < 1e-6


def test_gmres_reports_stagnation_instead_of_claiming_success():
    """On the unpreconditioned Laplacian GMRES(20) stalls; it must say so."""
    solvers = pytest.importorskip("tt.algs.solvers")
    d = 8
    A = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d)
    with pytest.warns(RuntimeWarning, match="GMRES"):
        x, res = solvers.GMRES(lambda v, eps: tt.matvec(A, v).round(eps),
                               tt.ones(2, d) * 0.0, rhs, eps=1e-8, maxit=60, m=20)
    true_res = (tt.matvec(A, x) - rhs).norm() / rhs.norm()
    assert true_res > 1e-8                       # it really did not converge
    assert res == pytest.approx(true_res, rel=0.5)   # and the number it reports is real


def test_cross_recovers_a_low_rank_tensor():
    cross = pytest.importorskip("tt.algs.cross")
    d, n = 6, 4
    ref = tt.rand([n] * d, d, r=3)
    dense = np.asarray(ref.full())

    calls = {"n": 0}

    def fun(idx):
        idx = np.asarray(idx, dtype=int)
        calls["n"] += idx.shape[0]
        return dense[tuple(idx.T)]

    got = cross.cross(fun, [n] * d, eps=1e-10)
    assert rel(got.full(), dense) < 1e-8
    assert calls["n"] < n ** d, "cross must not evaluate the whole tensor"


def test_ksl_is_exact_when_the_manifold_is_the_whole_space():
    """d=6, n=2, ranks [1,2,4,8,4,2,1] IS the full space: no projection error.

    Whatever error survives is the local matrix exponential, so this pins the
    replacement of EXPOKIT (5803 lines of Fortran in the old package) against
    scipy.linalg.expm on the dense operator.
    """
    ksl_mod = pytest.importorskip("tt.algs.ksl")
    import scipy.linalg as sla

    d = 6
    ranks = [1, 2, 4, 8, 4, 2, 1]
    rng = np.random.default_rng(12345)
    y0 = tt.vector.from_list(
        [rng.standard_normal((ranks[k], 2, ranks[k + 1])) for k in range(d)])
    A = (-1.0) * tt.qlaplace_dd([d])
    dense = np.asarray(A.full())
    y0d = np.asarray(y0.full()).flatten("F")

    for tau in (1e-3, 1e-2, 1e-1):
        ref = sla.expm(tau * dense) @ y0d
        got = np.asarray(ksl_mod.ksl(A, y0, tau, verb=0).full()).flatten("F")
        err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert err < 1e-12, f"tau={tau}: relative error {err:.3e}"


def test_eigb_matches_a_dense_symmetric_eigensolver():
    """Replaces PRIMME (89 vendored C/Fortran files) with dense eigh / lobpcg.

    Oracle: the analytic eigenvalues of the 1D discrete Laplacian.
    """
    eigb_mod = pytest.importorskip("tt.algs.eigb")
    d, B = 8, 4
    A = tt.qlaplace_dd([d])
    n = 2 ** d
    exact = np.sort(4 * np.sin(np.pi * np.arange(1, n + 1) / (2 * (n + 1))) ** 2)[:B]
    ranks = [1] + [2 * B] * (d - 1) + [B]
    y, lam = eigb_mod.eigb(A, tt.rand([2] * d, d, ranks), 1e-8, verb=0)
    lam = np.sort(np.asarray(lam).ravel())[:B]
    assert np.max(np.abs(lam - exact) / exact) < 1e-9, (lam, exact)


# --- examples/henon_heiles_ksl_paper.py, examples/henon_heiles_spectrum.py ---

def _examples_path():
    import pathlib
    import sys
    p = str(pathlib.Path(__file__).resolve().parent.parent / "examples")
    if p not in sys.path:
        sys.path.insert(0, p)


def test_ksl_paper_setup_matches_dense_propagation():
    """The [LOV15] section 6.2 machinery at f=2 against a dense propagator.

    Oracle: ``expm(-1j h H)`` on the full 1024-dimensional state -- the DVR
    kinetic matrix, the CAP sign, the MPO assembly and the complex KSL step
    all have to be right at once for the *state* (not just a summary number)
    to track the dense flow.  The error at T=3 is the accumulated rank-12
    modelling error, an order below the acceptance bound; the error after the
    first 50 steps is the integrator's own, and is seven orders below it.
    """
    import warnings
    import scipy.linalg as sla
    ksl_mod = pytest.importorskip("tt.algs.ksl")
    _examples_path()
    from henon_heiles_ksl_paper import hamiltonian, packet

    f, n, r = 2, 32, 12
    A, x = hamiltonian(f, n)
    psi0 = packet(f, x)
    Hd = np.asarray(A.full())
    yd = np.asarray(psi0.full()).flatten("F")

    rng = np.random.default_rng(0)
    noise = tt.rand([n] * f, r=r, samplefunc=rng.standard_normal)
    noise = noise * (1e-8 / noise.norm())
    y = (psi0 + noise).round(0.0, rmax=r)

    h, nsteps = 0.01, 300
    P = sla.expm(-1j * h * Hd)
    early = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(nsteps):
            y = ksl_mod.ksl(A, y, -1j * h, verb=0, check_rank=False,
                            use_normest=2)
            yd = P @ yd
            if k + 1 == 50:
                got = np.asarray(y.full()).flatten("F")
                early = np.linalg.norm(got - yd) / np.linalg.norm(yd)
    got = np.asarray(y.full()).flatten("F")
    late = np.linalg.norm(got - yd) / np.linalg.norm(yd)
    assert early < 1e-5, f"integrator error after 50 steps: {early:.3e}"
    assert late < 2e-3, f"accumulated rank-12 error at T=3: {late:.3e}"
    # the CAP must have started absorbing by T=3 (at f=2 the packet barely
    # reaches +-6 this early, so the drain is small but must be nonzero --
    # a Hermitian-by-mistake H would conserve the norm to 1e-12)
    assert float(y.norm()) < 1.0 - 1e-6


def test_spectrum_transform_recovers_planted_frequencies():
    """``spectrum`` + ``find_peaks`` of the autocorrelation example.

    Oracle: a synthetic ``a(t) = sum w_l exp(-i lam_l t)`` with planted
    frequencies; the windowed transform must peak at each ``lam_l`` to far
    better than the grid resolution (parabolic refinement), and must find
    nothing else above threshold.
    """
    _examples_path()
    from henon_heiles_spectrum import find_peaks, spectrum

    lam = np.array([1.0, 2.3, 4.7])
    w = np.array([1.0, 0.6, 0.3])
    h, nsteps = 0.05, 4096
    t = np.arange(nsteps + 1) * h
    acorr = (w[None, :] * np.exp(-1j * np.outer(t, lam))).sum(axis=1)
    omega, S = spectrum(acorr, h)
    peaks = find_peaks(omega, S, rel=0.05)
    got = np.array(sorted(p[0] for p in peaks))
    assert len(got) == 3, f"expected 3 peaks, found {len(got)}: {got}"
    assert np.max(np.abs(got - lam)) < 1e-3, (got, lam)
