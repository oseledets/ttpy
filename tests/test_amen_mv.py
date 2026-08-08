"""AMEn matrix-by-vector, checked against dense numpy and against invariants.

The oracles here are (a) the dense product ``A.full() @ x.full(asvector=True)``,
(b) the exact TT product ``tt.matvec(A, x).round(tol)``, which is a *different
algorithm* for the same quantity, and (c) mathematical invariants (rank bounds,
monotonicity in tol).  Never the legacy implementation.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs.amen_mv import amen_mv

from conftest import GPU_DEVICE, SOLVE_EPS, SOLVE_TOL, gpu_backend, requires_gpu


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def rand_matrix(n, m, d, r, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((rr[k], n, m, rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal((rr[k], n, m, rr[k + 1]))
        cores.append(c)
    return tt.matrix.from_list(cores)


def rand_vector(m, d, r, seed=1, complex_=False):
    rng = np.random.default_rng(seed)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((rr[k], m, rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal((rr[k], m, rr[k + 1]))
        cores.append(c)
    return tt.vector.from_list(cores)


def graded_vector(m, d, nterms, ratio, seed=2):
    """A TT vector whose singular values decay like ``ratio**s``.

    Built as a sum of ``nterms`` rank-one tensors with geometrically decaying
    weights, so the local blocks have condition number ``ratio**(1-nterms)``.
    """
    rng = np.random.default_rng(seed)
    vs = [[rng.standard_normal(m) for _ in range(d)] for _ in range(nterms)]
    for s in range(nterms):
        for k in range(d):
            vs[s][k] /= np.linalg.norm(vs[s][k])
    cores = []
    for k in range(d):
        left = 1 if k == 0 else nterms
        right = 1 if k == d - 1 else nterms
        c = np.zeros((left, m, right))
        for s in range(nterms):
            w = ratio ** s if k == 0 else 1.0
            c[0 if left == 1 else s, :, 0 if right == 1 else s] = w * vs[s][k]
        cores.append(c)
    return tt.vector.from_list(cores)


# --- against the exact TT product --------------------------------------------

@pytest.mark.parametrize("tol", [1e-4, 1e-8])
def test_matches_exact_rounded_matvec(tol):
    A = rand_matrix(4, 4, 6, 3, seed=10)
    x = rand_vector(4, 6, 4, seed=11)
    exact = tt.matvec(A, x).round(tol)
    y, z = amen_mv(A, x, tol, verb=0)
    assert rel(y.full(), exact.full()) <= 10 * tol
    # AMEn truncates at tol/sqrt(d), round() at tol/sqrt(d-1): never coarser,
    # so the adaptive ranks must not exceed the ones of the exact product.
    assert np.all(y.r <= exact.r), (list(y.r), list(exact.r))


def test_ranks_are_the_exact_ones_for_a_low_rank_product():
    # A x has exact TT ranks min(ra*rx, 4^k, 4^(d-k)); nothing to truncate at
    # 1e-10, so AMEn must discover exactly those ranks.
    A = rand_matrix(4, 4, 5, 2, seed=12)
    x = rand_vector(4, 5, 2, seed=13)
    exact = tt.matvec(A, x).round(1e-12)
    y, z = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert list(y.r) == list(exact.r)
    assert rel(y.full(), exact.full()) < 1e-9


# --- against dense numpy ------------------------------------------------------

def test_matches_dense_product():
    A = rand_matrix(3, 3, 4, 2, seed=20)
    x = rand_vector(3, 4, 3, seed=21)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    ref = A.full() @ x.full(asvector=True)
    assert rel(y.full(asvector=True), ref) < 1e-9


def test_matches_dense_product_rectangular_modes():
    A = rand_matrix(4, 2, 4, 2, seed=22)
    x = rand_vector(2, 4, 3, seed=23)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    ref = A.full() @ x.full(asvector=True)
    assert rel(y.full(asvector=True), ref) < 1e-9


def test_matches_dense_product_complex():
    """Complex data is the only test that can catch a wrong conjugation."""
    A = rand_matrix(3, 3, 4, 2, seed=24, complex_=True)
    x = rand_vector(3, 4, 2, seed=25, complex_=True)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    ref = A.full() @ x.full(asvector=True)
    assert rel(y.full(asvector=True), ref) < 1e-9


def test_single_core_tensor():
    A = rand_matrix(5, 5, 1, 1, seed=26)
    x = rand_vector(5, 1, 1, seed=27)
    y, _ = amen_mv(A, x, 1e-10, verb=0)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12


# --- the QTT Laplacian --------------------------------------------------------

def test_qlaplace_times_ones():
    d = 10
    A = tt.qlaplace_dd([d])
    x = tt.ones(2, d)
    exact = tt.matvec(A, x).round(1e-12)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(), exact.full()) < 1e-9
    assert np.all(y.r <= exact.r), (list(y.r), list(exact.r))


def test_qlaplace_times_ones_against_dense():
    d = 6
    A = tt.qlaplace_dd([d])
    x = tt.ones(2, d)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    ref = A.full() @ x.full(asvector=True)
    assert rel(y.full(asvector=True), ref) < 1e-9


def test_qlaplace_3d_times_random():
    A = tt.qlaplace_dd([4, 4, 4])
    x = rand_vector(2, 12, 3, seed=30)
    exact = tt.matvec(A, x).round(1e-10)
    y, _ = amen_mv(A, x, 1e-8, verb=0, nswp=30)
    assert rel(y.full(), exact.full()) < 1e-7


# --- accuracy control ---------------------------------------------------------

def test_error_decreases_with_tol():
    A = rand_matrix(4, 4, 6, 3, seed=40)
    x = rand_vector(4, 6, 3, seed=41)
    ref = tt.matvec(A, x).round(1e-14).full()
    errs = []
    for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        y, _ = amen_mv(A, x, tol, verb=0, nswp=30)
        errs.append(rel(y.full(), ref))
        assert errs[-1] <= 10 * tol
    # loosely monotone: a tighter tol never costs more than a factor 2 of error
    for a, b in zip(errs, errs[1:]):
        assert b <= max(2 * a, 1e-14)


def test_error_decreases_with_tol_on_a_graded_tensor():
    A = tt.eye([6] * 5)
    x = graded_vector(6, 5, 5, 0.1, seed=42)
    ref = x.full()
    for tol in [1e-2, 1e-6, 1e-10]:
        y, _ = amen_mv(A, x, tol, verb=0, nswp=40)
        assert rel(y.full(), ref) <= 10 * tol


# --- reporting ----------------------------------------------------------------

def test_non_convergence_is_reported_not_hidden():
    A = rand_matrix(4, 4, 8, 4, seed=50)
    x = rand_vector(4, 8, 4, seed=51)
    with pytest.warns(UserWarning, match="did not converge"):
        y, z, hist = amen_mv(A, x, 1e-12, verb=0, nswp=1, return_history=True)
    assert hist.converged is False
    assert hist.nswp_done == 1
    assert hist.max_dx > 1e-12


def test_history_is_recorded_at_verb_zero(capsys):
    A = rand_matrix(3, 3, 5, 2, seed=52)
    x = rand_vector(3, 5, 2, seed=53)
    y, z, hist = amen_mv(A, x, 1e-8, verb=0, return_history=True)
    assert capsys.readouterr().out == ""
    assert hist.converged is True
    assert len(hist.sweeps) >= 2
    assert set(hist.sweeps[0]) == {"sweep", "direction", "max_dx", "max_rank",
                                   "res_est", "time"}
    assert hist.time > 0
    assert list(hist.ranks) == list(y.r)
    assert hist.max_dx < 1e-8


def test_verbose_prints(capsys):
    A = rand_matrix(3, 3, 4, 2, seed=54)
    x = rand_vector(3, 4, 2, seed=55)
    amen_mv(A, x, 1e-8, verb=2)
    assert "amen_mv" in capsys.readouterr().out


def test_residual_tensor_is_valid_and_small():
    A = rand_matrix(4, 4, 6, 2, seed=56)
    x = rand_vector(4, 6, 2, seed=57)
    y, z, hist = amen_mv(A, x, 1e-10, verb=0, nswp=30, return_history=True)
    assert isinstance(z, tt.vector)
    assert z.d == y.d                       # a consistent rank chain: from_list
    assert list(z.n) == list(y.n)           # would have raised otherwise
    # z estimates (A x - y) / ||y||: it must be small once converged
    assert hist.res_est < 1e-6
    assert abs(z.norm() - hist.res_est) < 1e-12 * max(1.0, hist.res_est)


# --- options ------------------------------------------------------------------

def test_kickrank_zero_is_plain_als():
    A = rand_matrix(4, 4, 5, 2, seed=60)
    x = rand_vector(4, 5, 2, seed=61)
    exact = tt.matvec(A, x).round(1e-12)
    # ALS without enrichment cannot grow ranks, so it needs a guess of the
    # right rank; given one, it must converge to the exact answer.
    y0 = tt.rand(exact.n, exact.d, r=list(exact.r))
    y, z = amen_mv(A, x, 1e-10, y=y0, kickrank=0, verb=0, nswp=30)
    assert z is None
    assert rel(y.full(), exact.full()) < 1e-8


def test_kickrank2_random_enrichment():
    A = rand_matrix(4, 4, 6, 3, seed=62)
    x = rand_vector(4, 6, 3, seed=63)
    exact = tt.matvec(A, x).round(1e-10)
    y, z = amen_mv(A, x, 1e-8, kickrank=4, kickrank2=2, verb=0, nswp=30,
                   seed=7)
    assert rel(y.full(), exact.full()) < 1e-7


def test_fkick_forward_enrichment():
    A = rand_matrix(4, 4, 6, 3, seed=64)
    x = rand_vector(4, 6, 3, seed=65)
    exact = tt.matvec(A, x).round(1e-10)
    y, z = amen_mv(A, x, 1e-8, fkick=True, verb=0, nswp=30)
    assert rel(y.full(), exact.full()) < 1e-7


def test_seed_makes_the_run_reproducible():
    A = rand_matrix(4, 4, 5, 2, seed=66)
    x = rand_vector(4, 5, 2, seed=67)
    y1, _ = amen_mv(A, x, 1e-6, verb=0, seed=123)
    y2, _ = amen_mv(A, x, 1e-6, verb=0, seed=123)
    assert rel(y1.full(), y2.full()) < 1e-14


def test_warm_start_from_a_previous_run():
    A = rand_matrix(4, 4, 6, 2, seed=68)
    x = rand_vector(4, 6, 2, seed=69)
    exact = tt.matvec(A, x).round(1e-12)
    y0, z0, h0 = amen_mv(A, x, 1e-6, verb=0, return_history=True)
    y1, z1, h1 = amen_mv(A, x, 1e-10, y=y0, z=z0, verb=0, nswp=30,
                         return_history=True)
    assert rel(y1.full(), exact.full()) < 1e-9
    assert h1.nswp_done <= h0.nswp_done + 3


def test_init_qr_false_accepts_an_orthogonal_guess():
    A = rand_matrix(4, 4, 5, 2, seed=70)
    x = rand_vector(4, 5, 2, seed=71)
    exact = tt.matvec(A, x).round(1e-12)
    y0 = tt.rand(exact.n, exact.d, r=list(exact.r))
    # orthogonalize(center=d-1) makes cores 0..d-2 left-orthogonal
    y0 = tt.vector.from_list(
        [c for c in _left_orthogonalize(tt.vector.to_list(y0))])
    y, _ = amen_mv(A, x, 1e-10, y=y0, init_qr=False, verb=0, nswp=30)
    assert rel(y.full(), exact.full()) < 1e-8


def _left_orthogonalize(cores):
    """Left-to-right QR sweep (the mirror of _ops.orthogonalize(center=0))."""
    out = [np.asarray(c).copy() for c in cores]
    for k in range(len(out) - 1):
        r0, n, r1 = out[k].shape
        q, r = np.linalg.qr(out[k].reshape((r0 * n, r1)))
        out[k] = q.reshape((r0, n, q.shape[1]))
        out[k + 1] = np.einsum("ab,bnc->anc", r, out[k + 1])
    return out


def test_init_qr_false_rejects_a_non_orthogonal_guess():
    A = rand_matrix(4, 4, 5, 2, seed=72)
    x = rand_vector(4, 5, 2, seed=73)
    y0 = rand_vector(4, 5, 3, seed=74)
    with pytest.raises(ValueError, match="left-orthogonal"):
        amen_mv(A, x, 1e-8, y=y0, init_qr=False, verb=0)


# --- the renorm='gram' path ---------------------------------------------------

def test_gram_matches_direct_on_a_well_conditioned_problem():
    A = rand_matrix(8, 8, 6, 3, seed=80)
    x = rand_vector(8, 6, 3, seed=81)
    ref = tt.matvec(A, x).round(1e-14).full()
    yd, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30, renorm='direct')
    yg, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30, renorm='gram')
    ed, eg = rel(yd.full(), ref), rel(yg.full(), ref)
    print(f"\n[gram] well conditioned: direct={ed:.2e}  gram={eg:.2e}")
    # the docstring quotes ~3e-15 for both; guard that with two orders of slack
    assert ed < 1e-12
    assert eg < 1e-12


def test_gram_is_less_accurate_on_an_ill_conditioned_problem():
    """The documented honest difference between the two renorm paths."""
    A = tt.eye([8] * 6)
    x = graded_vector(8, 6, 6, 1e-2, seed=82)     # cond(blocks) ~ 1e10
    ref = x.full()
    yd, _ = amen_mv(A, x, 1e-12, verb=0, nswp=40, renorm='direct')
    yg, _ = amen_mv(A, x, 1e-12, verb=0, nswp=40, renorm='gram')
    ed, eg = rel(yd.full(), ref), rel(yg.full(), ref)
    print(f"\n[gram] ill conditioned: direct={ed:.2e}  gram={eg:.2e}")
    assert ed < 1e-11
    assert eg > 10 * ed          # the loss is real; the docstring says so
    # and it is the specific loss the docstring quotes: a stall around 1e-8
    assert 1e-10 < eg < 1e-6


def test_gram_orthogonality_degrades_with_the_condition_number():
    """Owns the table quoted in the ``_gram_svd`` docstring.

    The Gram path squares the condition number: singular directions whose Gram
    eigenvalue ``s^2`` sinks to machine epsilon come out as noise, so the
    orthogonality defect grows like ``eps * cond^2``, not ``eps * cond``.  The
    direct QR path is measured alongside it, because the claim in the docstring
    is a *comparison* -- a bound on the Gram path alone would not support it.
    """
    from tt.algs.amen_mv import _gram_svd
    rng = np.random.default_rng(90)
    # (cond, lower bound on the gram defect, upper bound on the gram defect)
    table = [(1.0, 0.0, 1e-13), (1e3, 1e-13, 1e-9), (1e7, 1e-5, 1e-1)]
    for cond, lo, hi in table:
        u0, _ = np.linalg.qr(rng.standard_normal((4096, 64)))
        v0, _ = np.linalg.qr(rng.standard_normal((64, 64)))
        s = np.logspace(0, -np.log10(cond) if cond > 1 else 0, 64)
        a = (u0 * s) @ v0
        u, sv, vh = _gram_svd(a)
        err = np.linalg.norm(u.T.conj() @ u - np.eye(u.shape[1]))
        q, _ = np.linalg.qr(a)
        err_direct = np.linalg.norm(q.T.conj() @ q - np.eye(q.shape[1]))
        recon = rel((u * sv) @ vh, a)
        print(f"\n[gram] cond={cond:.0e}: gram ||u^H u - I||={err:.2e}, "
              f"direct={err_direct:.2e}, recon={recon:.2e}")
        assert lo <= err < hi
        # the direct path is flat in cond(a) -- that is the whole point
        assert err_direct < 1e-13
        # only the frame is lost, never the factorization itself
        assert recon < 1e-13


# --- rank growth per sweep ----------------------------------------------------

def test_rank_growth_per_sweep_is_bounded_by_kickrank():
    """Owns the ``nswp`` docstring claim: one sweep adds at most ``kickrank``.

    This is *the* reason amen_mv fails to converge in practice, so it is pinned
    rather than left as folklore.
    """
    A = rand_matrix(4, 4, 6, 4, seed=120)
    x = rand_vector(4, 6, 4, seed=121)
    prev = None
    for nswp in [1, 2, 3, 4]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y, _ = amen_mv(A, x, 1e-12, verb=0, nswp=nswp, kickrank=3)
        r = np.asarray(y.r, dtype=int)
        if prev is not None:
            # interior ranks grow by at most kickrank per extra sweep
            assert np.all(r[1:-1] - prev[1:-1] <= 3), (list(prev), list(r))
        prev = r


def test_a_starved_kickrank_underestimates_the_rank_and_says_so():
    """The d=16 regime quoted in the ``nswp`` docstring, scaled down to run fast.

    Too few sweeps for the rank the product actually needs: the answer is wrong
    by orders of magnitude, and the point is that this is *reported*, never
    returned quietly.
    """
    A = rand_matrix(4, 4, 8, 6, seed=122)
    x = rand_vector(4, 8, 6, seed=123)
    exact = tt.matvec(A, x).round(1e-8)
    with pytest.warns(UserWarning, match="did not converge"):
        y_bad, _, h_bad = amen_mv(A, x, 1e-8, verb=0, nswp=3, kickrank=2,
                                  return_history=True)
    assert h_bad.converged is False
    assert max(y_bad.r) < max(exact.r)
    # raising kickrank instead of nswp fixes it: same target rank, few sweeps
    y_ok, _, h_ok = amen_mv(A, x, 1e-8, verb=0, nswp=30, kickrank=20,
                            return_history=True)
    assert h_ok.converged is True
    assert list(y_ok.r) == list(exact.r)
    assert rel(y_ok.full(), exact.full()) < 1e-7
    print(f"\n[rank] exact max r={max(exact.r)}; starved run reached "
          f"{max(y_bad.r)} (err {rel(y_bad.full(), exact.full()):.2e}), "
          f"kickrank=20 reached {max(y_ok.r)} in {h_ok.nswp_done} sweeps")


# --- the A argument -----------------------------------------------------------

def test_list_of_matrices_is_their_sum():
    A1 = rand_matrix(3, 3, 5, 2, seed=100)
    A2 = rand_matrix(3, 3, 5, 2, seed=101)
    x = rand_vector(3, 5, 2, seed=102)
    ref = (A1.full() + A2.full()) @ x.full(asvector=True)
    y, _ = amen_mv([A1, A2], x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(asvector=True), ref) < 1e-9


def test_canonical_format():
    rng = np.random.default_rng(103)
    d, n, nterms = 4, 3, 2
    blocks = [[rng.standard_normal((n, n)) for _ in range(nterms)]
              for _ in range(d)]
    dense = sum(_mkron([blocks[k][s] for k in range(d)]) for s in range(nterms))
    x = rand_vector(n, d, 2, seed=104)
    y, _ = amen_mv(blocks, x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(asvector=True), dense @ x.full(asvector=True)) < 1e-9


def _mkron(mats):
    """Kronecker product in the ttpy index order (mode 1 is the fastest)."""
    out = np.array([[1.0]])
    for a in mats[::-1]:
        out = np.kron(out, a)
    return out


def test_core_lists_in_and_out():
    A = rand_matrix(3, 3, 4, 2, seed=105)
    x = rand_vector(3, 4, 2, seed=106)
    ref = tt.matvec(A, x).round(1e-12)
    y, z = amen_mv(tt.matrix.to_list(A), tt.vector.to_list(x), 1e-10, verb=0,
                   nswp=30)
    assert isinstance(y, list)
    assert rel(tt.vector.from_list(y).full(), ref.full()) < 1e-9


# --- loud failures ------------------------------------------------------------

def test_mode_mismatch_raises():
    A = rand_matrix(3, 3, 4, 2, seed=110)
    x = rand_vector(4, 4, 2, seed=111)
    with pytest.raises(ValueError, match="mode mismatch"):
        amen_mv(A, x, 1e-8, verb=0)


def test_dimension_mismatch_raises():
    A = rand_matrix(3, 3, 4, 2, seed=112)
    x = rand_vector(3, 5, 2, seed=113)
    with pytest.raises(ValueError, match="cores"):
        amen_mv(A, x, 1e-8, verb=0)


def test_bad_renorm_raises():
    A = rand_matrix(3, 3, 4, 2, seed=114)
    x = rand_vector(3, 4, 2, seed=115)
    with pytest.raises(ValueError, match="renorm"):
        amen_mv(A, x, 1e-8, verb=0, renorm='fast')


def test_bad_A_type_raises():
    x = rand_vector(3, 4, 2, seed=116)
    with pytest.raises(TypeError, match="A:"):
        amen_mv("not a matrix", x, 1e-8, verb=0)


def test_tt_namespace_exposes_amen_mv():
    assert tt.amen_mv is amen_mv


# --- the backend-agnostic claim -----------------------------------------------

@requires_gpu()
def test_runs_on_the_torch_backend_and_agrees_with_numpy():
    """The module docstring claims backend agnosticism; this is the check.

    Skipped without torch or a GPU -- the package must work without either.
    Both backends solve the *same* problem from the *same* seed, so they must
    agree to the requested accuracy, not merely both terminate.  The accuracy
    asked for follows the device: MPS has no float64 (see conftest).
    """
    from tt import backend as bk

    A = rand_matrix(4, 4, 6, 3, seed=130)
    x = rand_vector(4, 6, 3, seed=131)
    ref = tt.matvec(A, x).round(1e-12).full()
    y_np, _ = amen_mv(A, x, SOLVE_EPS, verb=0, nswp=30, seed=5)

    gpu = gpu_backend()
    At = tt.matrix.from_list([bk.asarray(c, backend=gpu)
                              for c in tt.matrix.to_list(A)])
    xt = tt.vector.from_list([bk.asarray(c, backend=gpu)
                              for c in tt.vector.to_list(x)])
    y_t, z_t, h_t = amen_mv(At, xt, SOLVE_EPS, verb=0, nswp=30, seed=5,
                            return_history=True)
    assert bk.device_of(y_t.cores[0]).startswith(GPU_DEVICE)
    assert h_t.converged is True
    got = np.asarray(bk.to_numpy(y_t.full()))
    assert rel(got, ref) < SOLVE_TOL
    assert rel(got, y_np.full()) < SOLVE_TOL
    print(f"\n[torch] rel err vs dense={rel(got, ref):.2e}, "
          f"vs numpy backend={rel(got, y_np.full()):.2e}, ranks={list(y_t.r)}")


@requires_gpu(ops=("eigh",))
def test_gram_path_also_runs_on_torch():
    """The Gram path uses eigh/clip/sqrt; those are the easiest to get wrong."""
    from tt import backend as bk

    gpu = gpu_backend()
    A = rand_matrix(8, 8, 5, 3, seed=132)
    x = rand_vector(8, 5, 3, seed=133)
    ref = tt.matvec(A, x).round(1e-12).full()
    At = tt.matrix.from_list([bk.asarray(c, backend=gpu)
                              for c in tt.matrix.to_list(A)])
    xt = tt.vector.from_list([bk.asarray(c, backend=gpu)
                              for c in tt.vector.to_list(x)])
    y, _ = amen_mv(At, xt, SOLVE_EPS, verb=0, nswp=30, renorm='gram', seed=6)
    assert rel(np.asarray(bk.to_numpy(y.full())), ref) < 10 * SOLVE_TOL
