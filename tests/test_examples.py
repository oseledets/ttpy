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


# --- examples/fokker_planck_dumbbell.py --------------------------------------

def test_fokker_planck_dumbbell_matches_a_sparse_propagator():
    """The [DKO12] 4.2 machinery at n=32 against an independent sparse CN run.

    Oracle: the same Crank-Nicolson scheme assembled from ``scipy.sparse``
    Kronecker products with no tensor format anywhere, LU-factorized once.
    The TT operator is exact (a short sum of Kronecker terms), so the state
    parity is governed by the amen_solve tolerance, and the Kramers stress
    -- the physical output -- must agree to the same accuracy.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    amen = pytest.importorskip("tt.algs.amen")
    _examples_path()
    from fokker_planck_dumbbell import _vec3, grid, kramers_weights, operator

    n, nsteps, beta, T = 32, 16, 1.0, 1.0
    alpha, p = 0.1, 0.5
    x, h = grid(n)
    I = sp.identity(n, format="csr")
    lap1 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], (n, n)) / h ** 2
    C = sp.diags([-0.5, 0.5], [-1, 1], (n, n)) / h
    g = np.exp(-x ** 2 / (2 * p ** 2))
    c = alpha / (2 * p ** 5)
    dg, dx_ = sp.diags(g), sp.diags(x)

    def k3(a, b, cm):
        # mode 1 fastest, matching tt.full's Fortran order
        return sp.kron(sp.kron(cm, b, "csr"), a, "csr")

    Ad = (k3(-0.5 * lap1, I, I) + k3(I, -0.5 * lap1, I)
          + k3(I, I, -0.5 * lap1)
          + k3(C @ (-0.5 * dx_), I, I) + k3(I, C @ (-0.5 * dx_), I)
          + k3(I, I, C @ (-0.5 * dx_))
          + c * k3(C @ (dx_ @ dg), dg, dg) + c * k3(dg, C @ (dx_ @ dg), dg)
          + c * k3(dg, dg, C @ (dx_ @ dg))
          + beta * k3(C, dx_, I))

    A, _, _ = operator(n, beta)
    v = tt.rand([n] * 3, r=5)
    parity = np.linalg.norm(
        np.asarray(tt.matvec(A, v).full()).flatten("F")
        - Ad @ np.asarray(v.full()).flatten("F"))
    assert parity < 1e-10 * spla.norm(Ad) # the TT operator is exact

    g0 = np.exp(-x ** 2 / 2)
    psi = _vec3([g0] * 3)
    psi = psi * (1.0 / (tt.sum(psi) * h ** 3))
    pd = np.asarray(psi.full()).flatten("F")
    tau = T / nsteps
    lu = spla.splu((sp.identity(n ** 3, format="csc") + tau / 2 * Ad).tocsc())
    Mm_d = sp.identity(n ** 3, format="csc") - tau / 2 * Ad
    I3 = tt.eye(n, 3)
    Mp = (I3 + (tau / 2) * A).round(1e-13)
    Mm = (I3 - (tau / 2) * A).round(1e-13)
    for _ in range(nsteps):
        pd = lu.solve(Mm_d @ pd)
        pd /= pd.sum() * h ** 3
        rhs = tt.matvec(Mm, psi).round(1e-10)
        psi = amen.amen_solve(Mp, rhs, psi, 1e-8, verb=0)
        psi = psi * (1.0 / (tt.sum(psi) * h ** 3))
    ptt = np.asarray(psi.full()).flatten("F")
    assert np.linalg.norm(ptt - pd) / np.linalg.norm(pd) < 1e-6

    w, _ = kramers_weights(n)
    t12_tt = float(tt.dot(w[(1, 2)], psi)) * h ** 3
    t12_d = float(np.asarray(w[(1, 2)].full()).flatten("F") @ pd) * h ** 3
    assert abs(t12_tt - t12_d) < 1e-6 * abs(t12_d)


# --- examples/sir_network_cme.py ---------------------------------------------

def test_sir_cme_matches_brute_force_at_small_n():
    """The [DS24] machinery at N=6 against brute force over all 3^6 states.

    Oracles: the generator applied to a random vector against a
    state-by-state loop over every transition of eq. (2); the explicit
    indicator TTs against direct counting; the Crank-Nicolson TT run
    against the same scheme on the scipy.sparse generator.
    """
    import itertools
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    amen = pytest.importorskip("tt.algs.amen")
    _examples_path()
    from sir_network_cme import (BETA, GAMMA, chain_edges, exceedance,
                                 generator, infected_count, initial_state)

    N, istar, T, nsteps = 6, 3, 10.0, 50
    edges = chain_edges(N)
    states = list(itertools.product([0, 1, 2], repeat=N))
    index = {s: i for i, s in enumerate(states)}
    nbr = [[] for _ in range(N)]
    for (m, n) in edges:
        nbr[m].append(n)
        nbr[n].append(m)
    rows, cols, vals = [], [], []
    diag = np.zeros(len(states))
    for s in states:
        i0 = index[s]
        for n_ in range(N):
            if s[n_] == 0:
                rate = BETA * sum(1 for m_ in nbr[n_] if s[m_] == 1)
                if rate:
                    s2 = list(s)
                    s2[n_] = 1
                    rows.append(index[tuple(s2)])
                    cols.append(i0)
                    vals.append(rate)
                    diag[i0] -= rate
            elif s[n_] == 1:
                s2 = list(s)
                s2[n_] = 2
                rows.append(index[tuple(s2)])
                cols.append(i0)
                vals.append(GAMMA)
                diag[i0] -= GAMMA
    M = len(states)
    Ad = sp.csr_matrix((vals, (rows, cols)), (M, M)) + sp.diags(diag)

    def flat_f(s):                       # mode 1 fastest, as tt.full
        r_ = 0
        for k in reversed(range(N)):
            r_ = r_ * 3 + s[k]
        return r_

    A = generator(N, edges)
    v = tt.rand([3] * N, r=4)
    vf = np.asarray(v.full()).flatten("F")
    got = np.asarray(tt.matvec(A, v).full()).flatten("F")
    ref = Ad @ np.array([vf[flat_f(s)] for s in states])
    got_d = np.array([got[flat_f(s)] for s in states])
    assert np.linalg.norm(got_d - ref) < 1e-12 * np.linalg.norm(ref)

    If = np.asarray(infected_count(N).full()).flatten("F")
    Xf = np.asarray(exceedance(N, istar).full()).flatten("F")
    for s in states:
        ninf = sum(1 for x in s if x == 1)
        assert If[flat_f(s)] == ninf
        assert Xf[flat_f(s)] == (1.0 if ninf > istar else 0.0)

    # CN parity, TT vs sparse
    p = initial_state(N)
    pd = np.zeros(M)
    pd[index[tuple([1] + [0] * (N - 1))]] = 1.0
    tau = T / nsteps
    lu = spla.splu((sp.identity(M, format="csc") - tau / 2 * Ad).tocsc())
    Mm_d = sp.identity(M, format="csc") + tau / 2 * Ad
    IN = tt.eye(3, N)
    Mp = (IN - (tau / 2) * A).round(1e-13)
    Mm = (IN + (tau / 2) * A).round(1e-13)
    ones = tt.ones(3, N)
    for _ in range(nsteps):
        pd = lu.solve(Mm_d @ pd)
        pd /= pd.sum()
        rhs = tt.matvec(Mm, p).round(1e-12)
        p = amen.amen_solve(Mp, rhs, p, 1e-8, verb=0)
        p = p * (1.0 / float(tt.dot(ones, p)))
    pf = np.asarray(p.full()).flatten("F")
    pf_d = np.array([pf[flat_f(s)] for s in states])
    assert np.linalg.norm(pf_d - pd) / np.linalg.norm(pd) < 1e-6


# --- examples/qtt_divgrad_cross.py -------------------------------------------

def test_divgrad_cross_assembly_matches_scipy_sparse():
    """The README opener's QTT assembly against a dense-free sparse oracle.

    Both boundary variants: the two-term operator (natural BC on the far
    faces) and the Dirichlet one with the rank-1 corner corrections.  The
    coefficient goes through multifuncrs (TT-cross), so this also pins the
    cross on a smooth 2D function against direct sampling.
    """
    amen = pytest.importorskip("tt.algs.amen")
    _examples_path()
    from qtt_divgrad_cross import assemble, dense_oracle

    bits = 5
    for dirichlet in (False, True):
        A, h = assemble(bits, dirichlet=dirichlet)
        f = tt.ones(2, 2 * bits)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            u = amen.amen_solve(A, f, f, 1e-10, verb=0)
        ud = dense_oracle(bits, dirichlet=dirichlet)
        utt = np.asarray(u.full()).flatten("F")
        err = np.linalg.norm(utt - ud) / np.linalg.norm(ud)
        assert err < 1e-7, f"dirichlet={dirichlet}: {err:.2e}"
