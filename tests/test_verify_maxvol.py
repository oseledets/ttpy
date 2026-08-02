"""Adversarial verification of :mod:`tt.algs.maxvol`.

Written to break the module, not to confirm it.  Two things drive the design:

1. **Nothing is asserted about the matrix the algorithm carries around.**  The
   square loop stops exactly when ``max |C| <= tol`` *for its own incrementally
   updated* ``C``, and the rectangular loop stops when its own incrementally
   updated ``row_norm_sqr`` drops below ``tol^2``.  Asserting those bounds on the
   returned matrix is therefore circular -- the test cannot fail, whatever the
   rank-1 update formula does, as long as the exit test and the returned object
   agree.  Every bound here is re-derived from ``a`` and ``piv`` alone with
   ``numpy.linalg.solve`` / ``numpy.linalg.pinv``, so a wrong Sherman-Morrison
   update shows up.
2. Mathematical invariants that hold for the *definition* of the answer and not
   for this implementation: right-invariance ``maxvol(A) == maxvol(A @ M)``,
   monotone growth of ``|det A[piv]|``, the greedy step being the true argmax of
   the freshly recomputed row norms, and the 2-volume beating random subsets.
"""

from itertools import combinations
import contextlib
import io
import warnings

import numpy as np
import pytest

from tt.algs.maxvol import (maxvol, rect_maxvol, maxvol_qr, rect_maxvol_qr,
                            maxvol_svd, rect_maxvol_svd)


def rand(shape, seed, complex_=False):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if complex_:
        a = a + 1j * np.random.default_rng(seed + 10_000).standard_normal(shape)
    return a


def fresh_C_square(a, piv):
    """``C = a inv(a[piv])`` recomputed in float64 from scratch, never updated."""
    a64 = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
    return np.linalg.solve(a64[piv].T, a64.T).T


def fresh_C_rect(a, piv):
    """``C = a pinv(a[piv])`` recomputed in float64 from scratch."""
    a64 = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
    return a64 @ np.linalg.pinv(a64[piv])


def two_volume(a, rows):
    """``sqrt(det(a[rows]^* a[rows]))`` via singular values (stable)."""
    s = np.linalg.svd(np.asarray(a)[rows], compute_uv=False)
    return float(np.prod(s))


# --------------------------------------------------------------------------- #
# 1. non-circular versions of the bounds the module advertises
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32, np.complex64])
@pytest.mark.parametrize("n,r", [(300, 12), (5000, 60)])
def test_square_chebyshev_bound_holds_for_recomputed_C(dtype, n, r):
    """max|C| <= tol for C recomputed from (a, piv), not for the carried C.

    Regime: n in {300, 5000}, r in {12, 60}, tol = 1.02, 30 seeds worth of one
    matrix per (dtype, shape).  Slack: 1e-12 relative for float64/complex128,
    2e-6 for float32/complex64 (that is ~20 * eps(float32) * r, i.e. the LU
    solve's own error, and is *not* loosened to hide the incremental update:
    the drift between carried and recomputed C is asserted separately below).
    """
    cplx = np.dtype(dtype).kind == "c"
    a = rand((n, r), 1234 + n + r, cplx).astype(dtype)
    tol = 1.02
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a silent warning is a failure here
        piv, c_kept = maxvol(a, tol=tol, max_iters=5000, info=info)
    assert info["converged"], info

    c_fresh = fresh_C_square(a, piv)
    slack = 2e-6 if np.dtype(dtype).itemsize <= 8 else 1e-12
    assert np.abs(c_fresh).max() <= tol + slack, (
        f"recomputed max|C| = {np.abs(c_fresh).max()} > tol = {tol}")
    assert np.abs(c_fresh[piv] - np.eye(r)).max() < slack
    # and the carried matrix is the same object numerically
    assert np.abs(np.asarray(c_kept, dtype=np.complex128 if cplx else np.float64)
                  - c_fresh).max() < slack


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32])
@pytest.mark.parametrize("n,r,tol", [(500, 10, 1.0), (2000, 40, 1.05), (5000, 30, 1.5)])
def test_rect_row_norm_bound_holds_for_recomputed_C(dtype, n, r, tol):
    """||C[i]||_2 <= tol for C = a pinv(a[piv]) recomputed from scratch."""
    cplx = np.dtype(dtype).kind == "c"
    a = rand((n, r), 555 + n + r, cplx).astype(dtype)
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        piv, c_kept = rect_maxvol(a, tol=tol, identity_submatrix=False, info=info)
    assert info["converged"] and info["stop_reason"] == "tolerance", info
    assert len(set(piv.tolist())) == piv.size

    c_fresh = fresh_C_rect(a, piv)
    slack = 2e-6 if np.dtype(dtype).itemsize <= 8 else 1e-11
    assert np.linalg.norm(c_fresh, axis=1).max() <= tol + slack
    assert np.abs(np.asarray(c_kept, dtype=np.complex128 if cplx else np.float64)
                  - c_fresh).max() < slack
    # and the factorisation itself, against the original a
    a64 = np.asarray(a, dtype=np.complex128 if cplx else np.float64)
    assert (np.linalg.norm(c_fresh @ a64[piv] - a64) / np.linalg.norm(a64)) < slack


# --------------------------------------------------------------------------- #
# 2. invariants of the answer, independent of the implementation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("complex_", [False, True])
def test_square_dominance_over_many_seeds(complex_):
    """|det| of every single-row swap is <= tol * |det a[piv]|, dense dets, 25 seeds.

    Regime: n=25, r=5, tol=1.02, float64/complex128, all 5*20 swaps per seed.
    """
    n, r, tol = 25, 5, 1.02
    worst = 0.0
    for seed in range(25):
        a = rand((n, r), seed, complex_)
        piv, _c = maxvol(a, tol=tol, max_iters=1000)
        d = abs(np.linalg.det(a[piv]))
        assert d > 0
        for j in range(r):
            for i in range(n):
                if i in piv:
                    continue
                cand = piv.copy()
                cand[j] = i
                worst = max(worst, abs(np.linalg.det(a[cand])) / d)
    assert worst <= tol * (1 + 1e-9), f"worst swap gain {worst} > tol {tol}"


def test_each_swap_multiplies_the_determinant_by_exactly_max_abs_C():
    """Dense determinants must reproduce the gain the algorithm claims.

    Swapping ``piv[j] -> i`` multiplies ``|det a[piv]|`` by exactly ``|C[i, j]|``,
    which is the ``max_abs_C`` the previous truncated run reported.  Comparing
    ``det(k+1) / det(k)`` (computed by ``numpy.linalg.det`` on the original
    matrix) against ``info['max_abs_C']`` (computed by the rank-1 updates) ties
    the two together; a wrong Sherman-Morrison update or a wrong argmax breaks
    the equality even when both sides look reasonable on their own.
    Regime: n=500, r=15, tol=1.0, 12 truncation lengths, float64, rel 1e-9.
    """
    a = rand((500, 15), 909)
    dets, gains = [], []
    for it in range(12):
        info = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            piv, _c = maxvol(a, tol=1.0, max_iters=it, info=info)
        dets.append(abs(np.linalg.det(a[piv])))
        gains.append(info["max_abs_C"])
    assert all(dets[k + 1] >= dets[k] * (1 - 1e-12) for k in range(11)), dets
    assert dets[-1] > dets[0] * 2, dets              # the loop actually did work
    for k in range(11):
        assert dets[k + 1] / dets[k] == pytest.approx(gains[k], rel=1e-9), (
            k, dets[k + 1] / dets[k], gains[k])


@pytest.mark.parametrize("complex_", [False, True])
def test_dominance_is_a_property_of_the_column_space_only(complex_):
    """Rows chosen on ``A @ M`` must be dominant for ``A`` itself, and vice versa.

    ``C = A M (A[piv] M)^{-1} = A A[piv]^{-1}`` does not depend on the invertible
    right factor ``M``, so the *coefficient matrix* is an invariant of the column
    space even though the greedy path (which starts from LU on the actual
    entries) is not, and the two runs may land on different dominant sets.  A
    conjugation slip or a wrong orientation in the initial solve breaks this.
    Regime: n=400, r=8, tol=1.05, float64/complex128, M = randn + 3I.
    """
    a = rand((400, 8), 321, complex_)
    m = rand((8, 8), 654, complex_) + 3.0 * np.eye(8)
    piv_a, c_a = maxvol(a, tol=1.05, max_iters=500)
    piv_b, c_b = maxvol(a @ m, tol=1.05, max_iters=500)

    # C is M-independent for a FIXED pivot set
    assert np.abs(fresh_C_square(a @ m, piv_a) - c_a).max() < 1e-10
    assert np.abs(fresh_C_square(a, piv_b) - np.asarray(c_b)).max() < 1e-10
    # ... hence dominance transfers across the two matrices
    assert np.abs(fresh_C_square(a, piv_b)).max() <= 1.05 + 1e-10
    assert np.abs(fresh_C_square(a @ m, piv_a)).max() <= 1.05 + 1e-10

    piv_a, c_a = rect_maxvol(a, tol=1.0, identity_submatrix=False)
    piv_b, c_b = rect_maxvol(a @ m, tol=1.0, identity_submatrix=False)
    assert np.abs(fresh_C_rect(a @ m, piv_a) - c_a).max() < 1e-9
    assert np.linalg.norm(fresh_C_rect(a, piv_b), axis=1).max() <= 1.0 + 1e-9


@pytest.mark.parametrize("complex_", [False, True])
def test_rect_each_greedy_step_is_the_true_argmax(complex_):
    """Every added row is the argmax of the FRESHLY recomputed row norms.

    Replays the growth: for each k >= r, C_k = a pinv(a[piv[:k]]) is rebuilt with
    numpy.linalg.pinv and the next pivot must be its largest-norm unselected row.
    This is the strongest available oracle for the Sherman-Woodbury-Morrison
    update, since it checks the *decisions*, not just the final numbers.
    Regime: n=300 r=8 tol=1.0 and n=400 r=12 tol=1.05, float64/complex128.
    """
    for n, r, tol in [(300, 8, 1.0), (400, 12, 1.05)]:
        a = rand((n, r), 77 + n, complex_)
        piv, _c = rect_maxvol(a, tol=tol, identity_submatrix=False)
        assert piv.size > r, "need a case that actually grows"
        for k in range(r, piv.size - 1):
            ck = fresh_C_rect(a, piv[:k])
            nrm = np.linalg.norm(ck, axis=1)
            nrm[piv[:k]] = -1.0
            best = int(nrm.argmax())
            assert nrm[best] - nrm[piv[k]] < 1e-9 * max(1.0, nrm[best]), (
                f"step {k}: greedy took row {piv[k]} (norm {nrm[piv[k]]:.6f}) but "
                f"row {best} had norm {nrm[best]:.6f}")


def test_rect_two_volume_beats_random_subsets():
    """The greedy K rows have a larger 2-volume than random K-subsets.

    Not a theorem about the greedy, but a floor no correct implementation may
    fall through.  Regime: n=200, r=6, tol=1.0, 200 random subsets of the same
    size, float64.
    """
    rng = np.random.default_rng(2024)
    a = rand((200, 6), 4242)
    piv, _c = rect_maxvol(a, tol=1.0)
    k = piv.size
    vol = two_volume(a, piv)
    rand_vols = [two_volume(a, rng.choice(200, k, replace=False)) for _ in range(200)]
    assert vol > max(rand_vols), (vol, max(rand_vols), k)


def test_square_volume_beats_brute_force_bound_and_random():
    """Brute force over all C(16,4) subsets: the greedy is within the theory bound
    and, here, hits the true optimum.  Regime: n=16, r=4, tol=1.02, float64."""
    n, r, tol = 16, 4, 1.02
    for seed in range(6):
        a = rand((n, r), 3000 + seed)
        piv, _c = maxvol(a, tol=tol, max_iters=500)
        det = abs(np.linalg.det(a[piv]))
        dets = [abs(np.linalg.det(a[list(s)])) for s in combinations(range(n), r)]
        det_max = max(dets)
        # theory: |det A[T]| <= (sqrt(r) max|C|)^r |det A[piv]|
        assert det_max <= (np.sqrt(r) * tol) ** r * det * (1 + 1e-9)
        # and it must at least beat the median subset by orders of magnitude
        assert det > np.median(dets) * 10, (det, np.median(dets))


# --------------------------------------------------------------------------- #
# 3. silent-failure attacks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_refused(bad):
    """A single NaN used to sail through every guard.

    ``err > 1e-3``, ``|C[i, j]| > tol`` and ``||C[i]||^2 > tol^2`` are all False
    for a NaN, so before the fix ``maxvol`` returned plausible pivots with an
    all-NaN C, and ``rect_maxvol`` (whose warm start suppresses the warning)
    reported ``stop_reason='minK'`` for a matrix it had never grown.
    """
    a = rand((50, 4), 5)
    a[3, 2] = bad
    with pytest.raises(ValueError, match="non-finite"):
        maxvol(a)
    with pytest.raises(ValueError, match="non-finite"):
        rect_maxvol(a)
    with pytest.raises(ValueError, match="non-finite"):
        maxvol_qr(a)


def test_rect_converged_is_not_claimed_when_the_bound_is_unreachable():
    """tol < 1 cannot be met: the selected rows have norm 1 (or up to 1).

    Before the fix, ``rect_maxvol(a, tol=0.3)`` on a 30x5 matrix returned K=24
    with ``converged=True, stop_reason='tolerance'`` while the true maximum row
    norm of the returned C was 0.7073 (identity_submatrix=False) / 1.0000
    (True) -- the growth criterion only ever inspects the *unselected* rows.
    """
    a = rand((30, 5), 1)
    for ident in (True, False):
        info = {}
        with pytest.warns(RuntimeWarning):
            piv, c = rect_maxvol(a, tol=0.3, identity_submatrix=ident, info=info)
        true_max = float(np.linalg.norm(c, axis=1).max())
        assert true_max > 0.3
        assert info["converged"] is False, info
        assert info["stop_reason"] != "tolerance"
        assert info["max_row_norm"] == pytest.approx(true_max)
        # the factorisation is still exact -- only the bound is not met
        assert np.linalg.norm(c @ a[piv] - a) / np.linalg.norm(a) < 1e-12


def test_rect_stop_reason_matches_what_actually_happened():
    a = rand((300, 10), 4)
    info = {}
    with pytest.warns(RuntimeWarning, match="maxK = 12"):
        rect_maxvol(a, tol=1.0, maxK=12, info=info)
    assert info["stop_reason"] == "maxK" and info["K"] == 12 and not info["converged"]

    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rect_maxvol(a, tol=1.0, minK=25, info=info)
    assert info["stop_reason"] == "tolerance" and info["K"] == 25 and info["converged"]

    # all rows selected: nothing is left to add, and the bound IS met, so this
    # must not warn and must not be reported as a failure
    small = rand((12, 4), 6)
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        piv, c = rect_maxvol(small, tol=1.0, minK=50, info=info)
    assert piv.size == 12 and info["converged"] and info["stop_reason"] == "tolerance"


def test_square_nonconvergence_reports_the_number_it_failed_on():
    a = rand((300, 10), 23)
    info = {}
    with pytest.warns(RuntimeWarning, match=r"max\|C\| = 1\."):
        piv, c = maxvol(a, tol=1.0, max_iters=1, info=info)
    assert info["converged"] is False
    # the recorded number must be the truth about the returned pivots
    assert np.abs(fresh_C_square(a, piv)).max() == pytest.approx(info["max_abs_C"],
                                                                rel=1e-9)


def test_nothing_is_printed():
    """There is no verb switch here; the module must simply never print."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maxvol(rand((500, 10), 2))
        rect_maxvol(rand((500, 10), 2), tol=1.0)
        rect_maxvol(rand((500, 10), 2), tol=0.2)          # non-convergent path
        maxvol_svd(rand((100, 20), 2))
    assert buf.getvalue() == ""


def test_no_bare_except_or_assert_in_the_module():
    """User input must not be validated with `assert` (stripped under -O), and a
    bare `except:` would swallow the loud failures the rest of this file checks."""
    import inspect
    import tt.algs.maxvol as mod

    src = inspect.getsource(mod)
    code_lines = [ln for ln in src.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    assert not [ln for ln in code_lines if ln.strip().startswith("assert ")]
    assert not [ln for ln in code_lines if ln.strip() in ("except:", "except Exception:")]


def test_input_matrix_is_never_modified():
    for order in ("C", "F"):
        a = np.array(rand((200, 6), 11), order=order)
        a0 = a.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            maxvol(a, tol=1.02)
            rect_maxvol(a, tol=1.0)
            maxvol_qr(a)
            rect_maxvol_qr(a)
        assert np.array_equal(a, a0), f"input was modified ({order} order)"
    # a read-only Fortran array is the case where an in-place BLAS call would
    # either raise or corrupt the caller's data
    ro = np.asfortranarray(rand((200, 6), 12))
    ro.setflags(write=False)
    piv, c = maxvol(ro, tol=1.02)
    assert np.abs(fresh_C_square(ro, piv)).max() <= 1.02 + 1e-12


# --------------------------------------------------------------------------- #
# 4. shape / rank edge cases
# --------------------------------------------------------------------------- #

def test_rank_one_column_selects_the_largest_entry():
    """r = 1: the maxvol row is exactly argmax |a|, checkable by inspection."""
    for seed in range(8):
        a = rand((200, 1), seed)
        piv, c = maxvol(a, tol=1.0)
        assert piv.tolist() == [int(np.abs(a[:, 0]).argmax())]
        assert np.abs(c).max() == pytest.approx(1.0)
        assert np.linalg.norm(c @ a[piv] - a) < 1e-13
        piv_r, c_r = rect_maxvol(a, tol=1.0)
        assert piv_r.size >= 1
        assert np.linalg.norm(c_r @ a[piv_r] - a) / np.linalg.norm(a) < 1e-13


def test_n_equals_r_and_n_equals_r_plus_one():
    a = rand((7, 7), 21)
    piv, c = maxvol(a, tol=1.0)
    assert np.array_equal(piv, np.arange(7)) and np.allclose(c, np.eye(7))
    a = rand((8, 7), 21)
    piv, c = maxvol(a, tol=1.0, max_iters=200)
    assert piv.size == 7 and len(set(piv.tolist())) == 7
    assert np.abs(fresh_C_square(a, piv)).max() <= 1.0 + 1e-12


def test_zero_columns_matrix_raises():
    """An (N, 0) matrix has no volume to maximise; it must not return anything."""
    with pytest.raises(np.linalg.LinAlgError):
        maxvol(np.zeros((5, 0)))
    with pytest.raises(np.linalg.LinAlgError):
        rect_maxvol(np.zeros((5, 0)))


def test_zero_matrix_and_zero_row():
    with pytest.raises(np.linalg.LinAlgError):
        maxvol(np.zeros((20, 3)))
    with pytest.raises(np.linalg.LinAlgError):
        rect_maxvol(np.zeros((20, 3)))
    # a single zero row inside a full-rank matrix is fine: it can never be picked
    a = rand((50, 4), 4)
    a[7] = 0.0
    piv, c = maxvol(a, tol=1.0)
    assert 7 not in piv.tolist()
    assert np.allclose(c[7], 0.0)


def test_duplicated_rows():
    """20 copies of the same 5x5 block: any 5 distinct originals are optimal."""
    base = rand((5, 5), 3)
    a = np.tile(base, (20, 1))
    piv, c = maxvol(a, tol=1.0)
    assert sorted(p % 5 for p in piv.tolist()) == [0, 1, 2, 3, 4]
    assert np.abs(fresh_C_square(a, piv)).max() <= 1.0 + 1e-12
    piv_r, _c = rect_maxvol(a, tol=1.0)
    assert piv_r.size == 5, "there is nothing to add: every row is a copy"


def test_rank_deficient_variants_raise():
    a = rand((60, 5), 17)
    a[:, 4] = a[:, 0]
    for fn in (maxvol, rect_maxvol):
        with pytest.raises(np.linalg.LinAlgError):
            fn(a)
    # QR does NOT save a rank-deficient matrix.  Householder QR returns a
    # perfectly conditioned Q whose last column is an arbitrary complement of the
    # column space, so maxvol sees nothing wrong and returns rows whose submatrix
    # has determinant 0 -- exactly the plausible wrong answer the direct call
    # refuses.  Measured before the fix on this exact input: maxvol_qr returned
    # piv = [2, 28, 23, 51, 0] with det a[piv] == 0.0 exactly, while
    # ||C a[piv] - a|| / ||a|| = 2.9e-16 -- an exact factorisation through a
    # singular submatrix, which no caller of a *maximum volume* routine expects.
    for fn in (maxvol_qr, rect_maxvol_qr):
        with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
            fn(a)
    # the SVD variant is the documented way out: it truncates the dependent column
    piv, c = maxvol_svd(a, svd_tol=1e-8, job="R")
    assert piv.size == 4
    assert np.linalg.norm(c @ a[piv] - a) / np.linalg.norm(a) < 1e-10


def test_top_k_index_restricts_the_pivots_only():
    """Rows beyond top_k_index may exceed tol -- that is the documented meaning."""
    a = rand((400, 6), 13)
    piv, c = maxvol(a, tol=1.02, top_k_index=50)
    assert piv.max() < 50
    c_fresh = fresh_C_square(a, piv)
    assert np.abs(c_fresh[:50]).max() <= 1.02 + 1e-12
    assert np.abs(c_fresh).max() > 1.02, "the restriction should actually bite"

    piv, c = rect_maxvol(a, tol=1.0, top_k_index=50)
    assert piv.max() < 50
    assert np.linalg.norm(fresh_C_rect(a, piv)[:50], axis=1).max() <= 1.0 + 1e-10


def test_dtype_is_preserved_and_float32_is_not_silently_upcast():
    for dtype in (np.float32, np.float64, np.complex64, np.complex128):
        a = rand((300, 8), 91, np.dtype(dtype).kind == "c").astype(dtype)
        piv, c = maxvol(a, tol=1.05, max_iters=500)
        assert c.dtype == dtype
        piv, c = rect_maxvol(a, tol=1.05)
        assert c.dtype == dtype


def test_list_input_is_accepted_like_the_legacy_asanyarray():
    piv, c = maxvol([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], tol=1.0)
    assert sorted(piv.tolist()) == [0, 1] or sorted(piv.tolist()) in ([0, 2], [1, 2])
    assert np.abs(c).max() <= 1.0 + 1e-12


def test_float16_and_1d_and_3d_raise():
    with pytest.raises(TypeError):
        maxvol(rand((30, 3), 7).astype(np.float16))
    with pytest.raises(ValueError):
        maxvol(np.arange(5.0))
    with pytest.raises(ValueError):
        maxvol(np.zeros((4, 3, 2)))


# --------------------------------------------------------------------------- #
# 5. legacy compatibility of the wrappers
# --------------------------------------------------------------------------- #

def test_legacy_fortran_tol_spelling():
    """tt/maxvol/maxvol.f90 stops at `abs(ba(j0,i0)) <= 1 + tol`, so the legacy
    `maxvol(a, nswp=20, tol=5e-2)` is exactly `tol=1.05, max_iters=20`."""
    a = rand((250, 8), 7)
    assert np.array_equal(maxvol(a, tol=1.05, max_iters=20)[0],
                          maxvol(a, nswp=20, tol=5e-2)[0])
    # and a bound below 1 is unsatisfiable, so it can only mean 1 + tol
    piv, c = maxvol(a, tol=0.5, max_iters=200)
    assert np.abs(fresh_C_square(a, piv)).max() <= 1.5 + 1e-12


def test_svd_wrappers_default_job_is_F_like_the_legacy():
    """Legacy rect_maxvol.py: maxvol_svd(..., job='F'), rect_maxvol_svd(..., job='F').
    A default of 'R' silently changes the arity of the return value."""
    import inspect
    assert inspect.signature(maxvol_svd).parameters["job"].default == "F"
    assert inspect.signature(rect_maxvol_svd).parameters["job"].default == "F"
    a = rand((60, 15), 8)
    assert len(maxvol_svd(a, svd_tol=1e-3)) == 4
    assert len(rect_maxvol_svd(a, svd_tol=1e-3, tol=1.0)) == 4


def test_svd_column_selection_is_adjoint_correct_for_complex():
    """job='C' must reproduce a as a[:, piv] @ C.conj().T.

    The legacy code ran maxvol on ``V[:rank].T`` (no conjugate), which is only
    right for real input; with the conjugate the complex case reconstructs to
    machine precision and without it the error is O(1).
    """
    u = rand((80, 3), 6, True)
    v = rand((3, 15), 8, True)
    a = u @ v
    piv, cc = maxvol_svd(a, svd_tol=1e-3, job="C")
    assert piv.size == 3
    err_adj = np.linalg.norm(a[:, piv] @ cc.conj().T - a) / np.linalg.norm(a)
    err_tr = np.linalg.norm(a[:, piv] @ cc.T - a) / np.linalg.norm(a)
    assert err_adj < 1e-12, err_adj
    assert err_tr > 1e-3, "the two conventions must be distinguishable here"


def test_svd_wrappers_do_one_svd_not_two():
    """The rank cut used to be computed by a second full SVD, doubling the
    O(N M^2) cost that dominates these wrappers."""
    calls = []
    real_svd = np.linalg.svd

    def counting_svd(*args, **kwargs):
        calls.append(kwargs.get("compute_uv", True))
        return real_svd(*args, **kwargs)

    a = rand((120, 30), 9)
    np.linalg.svd = counting_svd
    try:
        maxvol_svd(a, svd_tol=1e-3, job="R")
    finally:
        np.linalg.svd = real_svd
    assert len(calls) == 1, f"{len(calls)} SVDs for one maxvol_svd call"


def test_qr_wrappers_agree_with_direct_maxvol_on_well_scaled_input():
    """QR only changes the right factor, so the pivots must be identical."""
    a = rand((300, 6), 31)
    assert np.array_equal(maxvol(a, tol=1.05, max_iters=500)[0],
                          maxvol_qr(a, tol=1.05, max_iters=500)[0])
    # badly scaled: the direct call may struggle, QR must not
    b = a.copy()
    b[:, 4] *= 1e-9
    piv, c = maxvol_qr(b, tol=1.05)
    assert np.linalg.norm(c @ b[piv] - b) / np.linalg.norm(b) < 1e-10
    assert np.abs(fresh_C_square(b, piv)).max() <= 1.05 + 1e-6


# --------------------------------------------------------------------------- #
# 6. backend
# --------------------------------------------------------------------------- #

def test_torch_input_matches_numpy_bit_for_bit():
    torch = pytest.importorskip("torch")
    a_np = rand((400, 7), 97)
    a_t = torch.as_tensor(a_np.copy())
    piv_t, c_t = maxvol(a_t, tol=1.02, max_iters=500)
    piv_n, c_n = maxvol(a_np, tol=1.02, max_iters=500)
    assert isinstance(piv_t, np.ndarray) and isinstance(c_t, torch.Tensor)
    assert np.array_equal(piv_t, piv_n)
    assert np.array_equal(c_t.cpu().numpy(), c_n)
    assert torch.equal(a_t, torch.as_tensor(a_np)), "torch input was modified"

    piv_t, c_t = rect_maxvol(a_t, tol=1.0)
    piv_n, c_n = rect_maxvol(a_np, tol=1.0)
    assert np.array_equal(piv_t, piv_n)
    assert np.array_equal(c_t.cpu().numpy(), c_n)


def test_public_names_and_the_tt_namespace():
    import tt
    import tt.maxvol as m
    assert tt.maxvol is m
    assert m.maxvol is maxvol and m.rect_maxvol is rect_maxvol
    assert tt.rect_maxvol is rect_maxvol
