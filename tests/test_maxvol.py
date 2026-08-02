"""Tests for tt.algs.maxvol: square and rectangular maximum-volume row selection.

Everything is checked against dense linear algebra: brute force over all r-subsets,
determinants of the swapped submatrices, and ``numpy.linalg.pinv`` -- never against
the legacy implementation.
"""

from itertools import combinations
import time
import warnings

import numpy as np
import pytest

from tt.algs.maxvol import (maxvol, rect_maxvol, maxvol_qr, rect_maxvol_qr,
                            maxvol_svd, rect_maxvol_svd)


def rand(shape, seed, complex_=False):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if complex_:
        a = a + 1j * rng.standard_normal(shape)
    return a


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def row_norms(c):
    return np.linalg.norm(c, axis=1)


# --------------------------------------------------------------------------- #
# square maxvol
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("complex_", [False, True])
def test_maxvol_brute_force_and_local_optimality(complex_):
    """The selected submatrix is dominant, and its volume is within the
    Hadamard factor r^{r/2} tol^r of the brute-force optimum.

    For any other row set T:  A[T] = C[T] A[piv], so
    |det A[T]| = |det C[T]| |det A[piv]| <= (sqrt(r) max|C|)^r |det A[piv]|.
    """
    n, r, tol = 14, 4, 1.05
    a = rand((n, r), 11 + complex_, complex_)
    info = {}
    piv, c = maxvol(a, tol=tol, info=info)

    assert info["converged"], info
    assert piv.shape == (r,) and len(set(piv.tolist())) == r
    assert np.abs(c).max() <= tol + 1e-12
    assert np.allclose(c[piv], np.eye(r), atol=1e-12)
    assert rel_err(c @ a[piv], a) < 1e-12

    det = abs(np.linalg.det(a[piv]))
    assert det > 0

    # brute force over all C(14, 4) = 1001 subsets
    det_max = max(abs(np.linalg.det(a[list(s)])) for s in combinations(range(n), r))
    assert det_max <= r ** (r / 2) * tol ** r * det * (1 + 1e-9)
    print(f"\n[maxvol brute force complex={complex_}] |det|/|det_max| = {det / det_max:.4f}, "
          f"bound factor = {r ** (r / 2) * tol ** r:.1f}")

    # local optimality, checked with dense determinants only
    for j in range(r):
        for i in range(n):
            if i in piv:
                continue
            cand = piv.copy()
            cand[j] = i
            assert abs(np.linalg.det(a[cand])) <= tol * det * (1 + 1e-9)


@pytest.mark.parametrize("tol", [1.02, 1.05, 1.5, 3.0])
@pytest.mark.parametrize("complex_", [False, True])
def test_maxvol_chebyshev_bound(tol, complex_):
    # tol = 1.0 exactly is deliberately absent: max|C| >= 1 always (C[piv] = I), so
    # the stopping test is then satisfied only when the maximum is attained on a
    # selected row to the last bit -- a roundoff coin flip, not an algorithmic claim.
    a = rand((300, 12), 3, complex_)
    info = {}
    piv, c = maxvol(a, tol=tol, max_iters=500, info=info)
    assert info["converged"], info
    assert np.abs(c).max() <= tol + 1e-10
    assert np.allclose(c[piv], np.eye(12), atol=1e-10)
    assert rel_err(c @ a[piv], a) < 1e-12
    assert abs(np.linalg.det(a[piv])) > 0


def test_maxvol_dtypes():
    for dtype, atol in [(np.float32, 1e-4), (np.float64, 1e-11),
                        (np.complex64, 1e-4), (np.complex128, 1e-11)]:
        a = rand((120, 6), 5, np.dtype(dtype).kind == "c").astype(dtype)
        piv, c = maxvol(a, tol=1.05)
        assert c.dtype == dtype
        assert np.abs(c).max() <= 1.05 + 1e-5
        assert np.abs(c[piv] - np.eye(6, dtype=dtype)).max() < atol
        assert rel_err(c @ a[piv], a) < atol


def test_maxvol_legacy_spelling():
    """tt.maxvol.maxvol(a, nswp=20, tol=5e-2) must mean tol=1.05, max_iters=20."""
    a = rand((250, 8), 7)
    piv_new, c_new = maxvol(a, tol=1.05, max_iters=20)
    piv_old, c_old = maxvol(a, nswp=20, tol=5e-2)
    assert np.array_equal(piv_new, piv_old)
    assert np.allclose(c_new, c_old)


def test_maxvol_square_and_wide_input():
    a = rand((5, 5), 9)
    piv, c = maxvol(a)
    assert np.array_equal(piv, np.arange(5))
    assert np.allclose(c, np.eye(5))
    a = rand((3, 7), 9)
    piv, c = maxvol(a)
    assert np.array_equal(piv, np.arange(3))
    assert c.shape == (3, 3)


def test_maxvol_top_k_index():
    a = rand((400, 6), 13)
    piv, c = maxvol(a, tol=1.05, top_k_index=50)
    assert piv.max() < 50
    assert np.abs(c[:50]).max() <= 1.05 + 1e-10
    assert rel_err(c @ a[piv], a) < 1e-12


def test_maxvol_rank_deficient_raises():
    a = rand((60, 5), 17)
    a[:, 4] = a[:, 0]                      # exactly dependent column
    with pytest.raises(np.linalg.LinAlgError):
        maxvol(a)
    b = rand((60, 5), 19)
    b[:, 4] = b[:, 1] + 1e-17 * b[:, 2]    # numerically dependent
    with pytest.raises(np.linalg.LinAlgError):
        maxvol(b)
    with pytest.raises(np.linalg.LinAlgError):
        rect_maxvol(a)


def test_maxvol_ill_conditioned_but_full_rank_still_works():
    """cond ~ 1e8 is bad, not broken: return a result, do not refuse."""
    q, _ = np.linalg.qr(rand((200, 5), 29))
    a = q * np.array([1.0, 1e-2, 1e-4, 1e-6, 1e-8])          # cond = 1e8
    info = {}
    piv, c = maxvol(a, tol=1.05, info=info)
    assert np.abs(c).max() <= 1.05 + 1e-8
    assert info["identity_error"] < 1e-3
    assert rel_err(c @ a[piv], a) < 1e-8


def test_maxvol_zero_matrix_raises():
    with pytest.raises(np.linalg.LinAlgError):
        maxvol(np.zeros((20, 3)))


def test_maxvol_bad_input_raises():
    with pytest.raises(ValueError):
        maxvol(np.zeros((4, 3, 2)))
    with pytest.raises(ValueError):
        maxvol(rand((30, 3), 1), tol=0.0)


def test_maxvol_nonconvergence_is_loud():
    a = rand((300, 10), 23)
    info_full = {}
    maxvol(a, tol=1.0, max_iters=500, info=info_full)
    assert info_full["iters"] > 1, "need a case that takes more than one swap"

    info = {}
    with pytest.warns(RuntimeWarning, match="did not converge"):
        piv, c = maxvol(a, tol=1.0, max_iters=1, info=info)
    assert info["converged"] is False
    assert info["max_abs_C"] > 1.0
    # the factorisation is still exact, it is only not dominant
    assert rel_err(c @ a[piv], a) < 1e-12


def test_maxvol_integer_input_is_promoted():
    rng = np.random.default_rng(31)
    a = rng.integers(-9, 9, size=(40, 4))
    piv, c = maxvol(a)
    assert c.dtype == np.float64
    assert rel_err(c @ a[piv], a.astype(float)) < 1e-12


# --------------------------------------------------------------------------- #
# rectangular maxvol
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tol", [1.0, 1.05, 1.5, 2.0])
@pytest.mark.parametrize("complex_", [False, True])
def test_rect_maxvol_row_norm_bound(tol, complex_):
    n, r = 500, 10
    a = rand((n, r), 41 + complex_, complex_)
    info = {}
    piv, c = rect_maxvol(a, tol=tol, info=info)
    k = piv.size
    assert k >= r
    assert c.shape == (n, k)
    assert len(set(piv.tolist())) == k, "rows must not repeat"
    assert info["converged"] and info["stop_reason"] == "tolerance"
    assert row_norms(c).max() <= tol + 1e-10
    assert rel_err(c @ a[piv], a) < 1e-12
    assert np.allclose(c[piv], np.eye(k), atol=1e-11)
    # what the algorithm actually promises: every row 2-norm <= tol, hence
    # ||C||_2 <= ||C||_F <= sqrt(N) tol.  The sharper O(sqrt(N/K)) bound of the
    # paper is proved for the exact 2-volume maximiser, not for this greedy run,
    # so it is measured and printed rather than asserted.
    assert np.linalg.norm(c, 2) <= np.sqrt(n) * tol
    print(f"\n[rect_maxvol tol={tol} complex={complex_}] K={k} (r={r}), "
          f"max row norm={row_norms(c).max():.4f}, ||C||_2={np.linalg.norm(c, 2):.3f}")


@pytest.mark.parametrize("complex_", [False, True])
def test_rect_maxvol_matches_pinv(complex_):
    """Without the identity convention, C is exactly A pinv(A[piv])."""
    a = rand((200, 7), 43 + complex_, complex_)
    piv, c = rect_maxvol(a, tol=1.1, identity_submatrix=False)
    c_ref = a @ np.linalg.pinv(a[piv])
    assert np.abs(c - c_ref).max() < 1e-11 * max(1.0, np.abs(c_ref).max())
    assert rel_err(c @ a[piv], a) < 1e-12
    # C[piv] is the orthoprojector onto the row space: rank r, idempotent
    p = c[piv]
    assert np.abs(p @ p - p).max() < 1e-10
    assert np.linalg.matrix_rank(p) == 7


def test_rect_maxvol_row_norms_decrease():
    """Greedy 2-volume growth may only decrease the largest row norm."""
    a = rand((400, 8), 47)
    prev = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # maxK stop is the point here
        for k in range(8, 25):
            _piv, c = rect_maxvol(a, tol=1e-8, minK=k, maxK=k, identity_submatrix=False)
            cur = row_norms(c).max()
            assert cur <= prev + 1e-12, (k, cur, prev)
            prev = cur


def test_rect_maxvol_K_controls():
    a = rand((300, 9), 53)
    r = 9
    piv, _c = rect_maxvol(a, tol=1.0, min_add_K=5)
    assert piv.size >= r + 5
    piv, _c = rect_maxvol(a, tol=1.0, minK=20)
    assert piv.size >= 20
    with pytest.warns(RuntimeWarning, match="stopped at K"):
        piv, c = rect_maxvol(a, tol=0.05, maxK=r + 2)
    assert piv.size == r + 2
    assert rel_err(c @ a[piv], a) < 1e-12
    with pytest.raises(ValueError):
        rect_maxvol(a, minK=30, maxK=12)
    with pytest.raises(ValueError):
        rect_maxvol(a, tol=-1.0)


def test_rect_maxvol_clamps_to_available_rows():
    """minK above N is a physical bound, not a contradiction: clamp, do not raise."""
    a = rand((12, 4), 59)
    piv, c = rect_maxvol(a, tol=1.0, minK=50)
    assert piv.size == 12
    assert np.allclose(c[piv], np.eye(12), atol=1e-11)
    assert rel_err(c @ a[piv], a) < 1e-12


def test_rect_maxvol_top_k_index():
    a = rand((300, 6), 61)
    piv, c = rect_maxvol(a, tol=1.0, top_k_index=40)
    assert piv.max() < 40
    assert row_norms(c[:40]).max() <= 1.0 + 1e-10


def test_rect_maxvol_wide_input():
    a = rand((4, 9), 67)
    piv, c = rect_maxvol(a)
    assert np.array_equal(piv, np.arange(4))
    assert np.allclose(c, np.eye(4))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_rect_maxvol_dtypes(dtype):
    a = rand((200, 6), 71, np.dtype(dtype).kind == "c").astype(dtype)
    piv, c = rect_maxvol(a, tol=1.1)
    atol = 1e-4 if np.dtype(dtype).itemsize <= 8 else 1e-11
    assert c.dtype == dtype
    assert row_norms(c).max() <= 1.1 + 1e-4
    assert rel_err(c @ a[piv], a) < atol


# --------------------------------------------------------------------------- #
# wrappers
# --------------------------------------------------------------------------- #

def test_qr_wrappers_on_badly_scaled_input():
    """QR first: same row space, condition number 1, so the row choice is safe."""
    a = rand((150, 5), 73)
    a[:, 4] *= 1e-8                        # badly scaled but still full rank
    piv, c = maxvol_qr(a, tol=1.05)
    assert np.abs(c).max() <= 1.05 + 1e-10
    assert rel_err(c @ a[piv], a) < 1e-10
    piv, c = rect_maxvol_qr(a, tol=1.05)
    assert row_norms(c).max() <= 1.05 + 1e-10
    assert rel_err(c @ a[piv], a) < 1e-10


def test_svd_wrappers_rank_cut():
    """A rank-3 matrix plus noise: the SVD variants select 3 rows/columns."""
    u = rand((100, 3), 79)
    v = rand((3, 20), 83)
    a = u @ v + 1e-8 * rand((100, 20), 89)
    piv, c = maxvol_svd(a, svd_tol=1e-3, tol=1.05, job="R")
    assert piv.size == 3
    assert rel_err(c @ a[piv], a) < 1e-6
    piv_c, cc = maxvol_svd(a, svd_tol=1e-3, tol=1.05, job="C")
    assert piv_c.size == 3
    assert rel_err(a[:, piv_c] @ cc.conj().T, a) < 1e-6
    piv_r, _c_r = rect_maxvol_svd(a, svd_tol=1e-3, tol=1.0, job="R")
    assert piv_r.size >= 3
    out = maxvol_svd(a, svd_tol=1e-3, job="F")
    assert len(out) == 4
    with pytest.raises(ValueError):
        maxvol_svd(a, job="X")


# --------------------------------------------------------------------------- #
# TT-flavoured use and backends
# --------------------------------------------------------------------------- #

def test_public_names_resolve():
    """tt.__getattr__ owns the public export; do not duplicate it.

    In ttpy 1.x `tt.maxvol` was a MODULE (`from tt.maxvol import maxvol`), so
    that is what the name resolves to; the function is exported next to it.
    """
    import tt
    import tt.maxvol as maxvol_module

    assert tt.maxvol is maxvol_module
    assert tt.maxvol.maxvol is maxvol
    assert tt.rect_maxvol is rect_maxvol
    assert maxvol_module.rect_maxvol is rect_maxvol


def test_maxvol_on_tt_core_unfolding():
    """The real consumer: rows of a left unfolding of a TT core (Q from QR)."""
    import tt
    from einops import rearrange

    x = tt.rand([4, 5, 6, 7], r=5)
    unfolding = rearrange(x.cores[1], "a n b -> (a n) b")
    q, _r = np.linalg.qr(unfolding)
    piv, c = maxvol(q, tol=1.05)
    assert np.abs(c).max() <= 1.05 + 1e-10
    assert rel_err(c @ q[piv], q) < 1e-12
    piv_r, c_r = rect_maxvol(q, tol=1.0, min_add_K=2)
    assert piv_r.size >= q.shape[1] + 2
    assert rel_err(c_r @ q[piv_r], q) < 1e-12


def test_torch_backend_roundtrip():
    torch = pytest.importorskip("torch")
    a_np = rand((200, 6), 97)
    a_t = torch.as_tensor(a_np)
    piv, c = maxvol(a_t, tol=1.05)
    assert isinstance(piv, np.ndarray)
    assert isinstance(c, torch.Tensor)
    assert c.dtype == a_t.dtype and c.device == a_t.device
    piv_np, c_np = maxvol(a_np, tol=1.05)
    assert np.array_equal(piv, piv_np)
    assert np.allclose(c.cpu().numpy(), c_np)
    piv, c = rect_maxvol(a_t, tol=1.0)
    assert isinstance(c, torch.Tensor)


# --------------------------------------------------------------------------- #
# performance in the advertised regime
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,r", [(100_000, 500), (100_000, 50)])
def test_maxvol_large_is_fast(n, r):
    a = rand((n, r), 101)
    t0 = time.perf_counter()
    info = {}
    piv, c = maxvol(a, tol=1.05, max_iters=1000, info=info)
    dt = time.perf_counter() - t0
    print(f"\n[maxvol perf] N={n} r={r}: {dt:.2f}s, {info['iters']} swaps, "
          f"max|C|={info['max_abs_C']:.4f}, identity drift={info['identity_error']:.2e}")
    assert info["converged"]
    assert np.abs(c).max() <= 1.05 + 1e-9
    assert dt < 180.0
    # reconstruction on a random slice of rows (a full N x r gemm is not the point)
    sample = np.random.default_rng(5).choice(n, 2000, replace=False)
    assert rel_err(c[sample] @ a[piv], a[sample]) < 1e-12


def test_rect_maxvol_large_is_fast():
    n, r = 100_000, 200
    a = rand((n, r), 103)
    t0 = time.perf_counter()
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        piv, c = rect_maxvol(a, tol=1.0, maxK=r + 100, info=info)
    dt = time.perf_counter() - t0
    print(f"\n[rect_maxvol perf] N={n} r={r}: {dt:.2f}s, K={info['K']}, "
          f"max row norm={info['max_row_norm']:.4f}, stop={info['stop_reason']}")
    assert piv.size >= r
    assert dt < 180.0
