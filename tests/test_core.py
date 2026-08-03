"""Core TT algebra checked against dense ground truth.

The oracle is numpy on the full tensor, never the legacy implementation.
"""

import numpy as np
import pytest

import tt
from tt.core import _ops


def rand_dense(shape, seed=0, complex_=False):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape)
    if complex_:
        a = a + 1j * rng.standard_normal(shape)
    return a


def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), 1e-300)


# --- TT-SVD and full ---------------------------------------------------------

@pytest.mark.parametrize("shape", [(2,) * 6, (3, 4, 5), (2, 3, 2, 3), (5, 5)])
def test_tt_svd_exact(shape):
    a = rand_dense(shape, seed=len(shape))
    x = tt.vector(a, 1e-14)
    assert rel(x.full(), a) < 1e-12
    assert x.d == len(shape)
    assert list(x.n) == list(shape)


def test_tt_svd_ranks_are_exact_for_rank_one():
    a = np.outer(np.arange(1.0, 5.0), np.arange(1.0, 7.0)).reshape(4, 6)
    x = tt.vector(a, 1e-12)
    assert list(x.r) == [1, 1, 1]


def test_tt_svd_accuracy_respects_eps():
    a = rand_dense((4, 4, 4, 4), seed=7)
    for eps in (1e-1, 1e-2, 1e-4):
        x = tt.vector(a, eps)
        assert rel(x.full(), a) <= eps * 1.05


def test_rmax_caps_ranks():
    a = rand_dense((4, 4, 4, 4), seed=8)
    x = tt.vector(a, 1e-14, rmax=3)
    assert max(x.r) <= 3


# --- rounding ----------------------------------------------------------------

def test_round_removes_redundant_ranks():
    a = tt.ones(2, 10)
    b = a + a + a
    assert max(b.r) > 1
    c = b.round(1e-14)
    assert list(c.r) == [1] * 11
    assert rel(c.full(), 3 * a.full()) < 1e-13


def test_round_is_accurate():
    a = rand_dense((3,) * 5, seed=3)
    x = tt.vector(a, 1e-14)
    for eps in (1e-1, 1e-3):
        y = x.round(eps)
        assert rel(y.full(), a) <= eps * 1.05


def test_round_of_sum_matches_dense_sum():
    a, b = rand_dense((2,) * 8, 1), rand_dense((2,) * 8, 2)
    x, y = tt.vector(a, 1e-14), tt.vector(b, 1e-14)
    z = (x + y).round(1e-12)
    assert rel(z.full(), a + b) < 1e-11


def test_orthogonalize_gives_orthogonal_cores():
    x = tt.rand([3, 4, 5, 4], r=4)
    cores = _ops.orthogonalize(x.cores, center=0)
    for c in cores[1:]:
        r0, n, r1 = c.shape
        m = c.reshape((r0, n * r1))
        assert rel(m @ m.conj().T, np.eye(r0)) < 1e-12
    assert abs(_ops.norm(cores) - x.norm()) < 1e-10 * x.norm()


# --- arithmetic --------------------------------------------------------------

def test_add_sub_mul_against_dense():
    a, b = rand_dense((3, 2, 4), 11), rand_dense((3, 2, 4), 12)
    x, y = tt.vector(a, 1e-14), tt.vector(b, 1e-14)
    assert rel((x + y).full(), a + b) < 1e-12
    assert rel((x - y).full(), a - b) < 1e-12
    assert rel((2.5 * x).full(), 2.5 * a) < 1e-12
    assert rel((x * y).full(), a * b) < 1e-12   # Hadamard
    assert rel((-x).full(), -a) < 1e-12


def test_norm_and_dot_against_dense():
    a, b = rand_dense((3, 3, 3), 21), rand_dense((3, 3, 3), 22)
    x, y = tt.vector(a, 1e-14), tt.vector(b, 1e-14)
    assert abs(x.norm() - np.linalg.norm(a)) < 1e-11 * np.linalg.norm(a)
    assert abs(tt.dot(x, y) - np.sum(a * b)) < 1e-10 * abs(np.sum(a * b))


def test_dot_is_conjugate_linear_in_first_argument():
    a, b = rand_dense((2, 2, 2), 31, complex_=True), rand_dense((2, 2, 2), 32, complex_=True)
    x, y = tt.vector(a, 1e-14), tt.vector(b, 1e-14)
    assert abs(tt.dot(x, y) - np.sum(np.conj(a) * b)) < 1e-10


def test_kron_against_dense():
    a, b = rand_dense((2, 3), 41), rand_dense((4, 2), 42)
    x, y = tt.vector(a, 1e-14), tt.vector(b, 1e-14)
    z = tt.kron(x, y)
    assert rel(z.full(), np.multiply.outer(a, b)) < 1e-12


def test_getitem_element_and_slice():
    a = rand_dense((3, 4, 5), 51)
    x = tt.vector(a, 1e-14)
    assert abs(x[1, 2, 3] - a[1, 2, 3]) < 1e-12
    sub = x[1, :, :]
    assert rel(sub.full(), a[1]) < 1e-12


def test_sum_over_axes():
    a = rand_dense((3, 4, 5), 61)
    x = tt.vector(a, 1e-14)
    assert abs(tt.sum(x) - a.sum()) < 1e-10 * abs(a.sum())
    assert rel(tt.sum(x, axis=1).full(), a.sum(axis=1)) < 1e-11


def test_diag_roundtrip():
    a = rand_dense((3, 4), 71)
    x = tt.vector(a, 1e-14)
    m = tt.diag(x)
    assert rel(tt.diag(m).full(), a) < 1e-12


# --- legacy layout compatibility --------------------------------------------

def test_core_and_ps_match_legacy_layout():
    x = tt.rand([2, 3, 4], r=3)
    core, ps = x.core, x.ps
    cores = tt.vector.to_list(x)
    for k in range(x.d):
        block = core[ps[k] - 1:ps[k + 1] - 1].reshape(
            (x.r[k], x.n[k], x.r[k + 1]), order="F")
        assert rel(block, np.asarray(cores[k])) < 1e-14
    y = tt.vector.from_flat(core, x.n, x.r)
    assert rel(y.full(), x.full()) < 1e-13


def test_from_list_to_list_roundtrip():
    x = tt.rand([2, 3, 2], r=2)
    y = tt.vector.from_list(tt.vector.to_list(x))
    assert rel(y.full(), x.full()) < 1e-14


def test_full_asvector_is_fortran_flattening():
    x = tt.rand([2, 3, 4], r=2)
    assert rel(x.full(asvector=True), np.asarray(x.full()).flatten("F")) < 1e-14


# --- complex -----------------------------------------------------------------

def test_complex_arithmetic_and_real_imag():
    a = rand_dense((2, 3, 2), 81, complex_=True)
    x = tt.vector(a, 1e-14)
    assert x.is_complex
    assert rel(x.full(), a) < 1e-12
    assert rel(x.real().full(), a.real) < 1e-11
    assert rel(x.imag().full(), a.imag) < 1e-11


def test_c2r_r2c_roundtrip():
    a = rand_dense((2, 2, 2), 91, complex_=True)
    x = tt.vector(a, 1e-14)
    assert rel(x.c2r().r2c().full(), a) < 1e-11


# --- matrices ----------------------------------------------------------------

def test_matrix_full_and_matvec_against_dense():
    n = [2, 3, 2]
    dense = rand_dense((2, 3, 2, 2, 3, 2), 101)
    A = tt.matrix(dense, 1e-14)
    assert list(A.n) == n and list(A.m) == n
    full = A.full()
    assert full.shape == (12, 12)
    x = tt.rand(n, r=2)
    y = tt.matvec(A, x)
    assert rel(y.full(asvector=True), full @ np.asarray(x.full()).flatten("F")) < 1e-11


def test_matrix_matmul_against_dense():
    A = tt.matrix(rand_dense((2, 3, 2, 3), 111), 1e-14)   # (2x3) modes, square
    B = tt.matrix(rand_dense((2, 3, 2, 3), 112), 1e-14)
    C = A @ B
    assert rel(C.full(), A.full() @ B.full()) < 1e-11
    # rectangular: (n=[2,3]) x (m=[3,2]) needs A.m == B.n
    P = tt.matrix(rand_dense((2, 3, 4, 2), 113), 1e-14)   # n=[2,3], m=[4,2]
    Q = tt.matrix(rand_dense((4, 2, 3, 3), 114), 1e-14)   # n=[4,2], m=[3,3]
    assert rel((P @ Q).full(), P.full() @ Q.full()) < 1e-11


def test_eye_and_transpose():
    E = tt.eye([3, 4])
    assert rel(E.full(), np.eye(12)) < 1e-13
    A = tt.matrix(rand_dense((2, 3, 2, 3), 121), 1e-14)
    assert rel(A.T.full(), A.full().T) < 1e-12


def test_matrix_by_dense_vector():
    A = tt.matrix(rand_dense((2, 3, 2, 3), 131), 1e-14)
    v = rand_dense((6,), 132)
    assert rel(A * v, A.full() @ v) < 1e-11


def test_qlaplace_is_the_laplacian():
    A = tt.qlaplace_dd([3])
    n = 8
    ref = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    assert rel(A.full(), ref) < 1e-13


def test_qlaplace_2d():
    A = tt.qlaplace_dd([2, 2])
    n = 4
    lap = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    ref = np.kron(np.eye(n), lap) + np.kron(lap, np.eye(n))
    assert rel(A.full(), ref) < 1e-13


def test_ipas_matrix():
    d = 3
    M = tt.IpaS(d, 0.5)
    n = 2 ** d
    ref = np.eye(n) + 0.5 * np.eye(n, k=-1)
    assert rel(M.full(), ref) < 1e-13


def test_toeplitz_lower_triangular():
    d = 3
    x = tt.rand(2, d, r=2)
    T = tt.Toeplitz(x, kind="L")
    xv = np.asarray(x.full()).flatten("F")
    n = 2 ** d
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            ref[i, j] = xv[i - j]
    assert rel(T.full(), ref) < 1e-12


def test_toeplitz_circulant():
    d = 3
    x = tt.rand(2, d, r=2)
    T = tt.Toeplitz(x, kind="C")
    xv = np.asarray(x.full()).flatten("F")
    n = 2 ** d
    ref = np.array([[xv[(i - j) % n] for j in range(n)] for i in range(n)])
    assert rel(T.full(), ref) < 1e-12


def test_qshift():
    d = 3
    S = tt.qshift(d)
    n = 2 ** d
    assert rel(S.full(), np.eye(n, k=-1)) < 1e-12


# --- generators --------------------------------------------------------------

def test_xfun_ones_linspace_delta_stepfun():
    assert rel(tt.xfun(2, 4).full(asvector=True), np.arange(16)) < 1e-13
    assert rel(tt.ones(3, 2).full(), np.ones((3, 3))) < 1e-14
    assert rel(tt.linspace(2, 4, a=0.0, b=1.0).full(asvector=True),
               np.linspace(0, 1, 16)) < 1e-11
    e = np.zeros(16)
    e[5] = 1
    assert rel(tt.delta(2, 4, center=5).full(asvector=True), e) < 1e-14
    h = np.zeros(16)
    h[5:] = 1
    assert rel(tt.stepfun(2, 4, center=5).full(asvector=True), h) < 1e-13


def test_sin_cos():
    d, alpha, phase = 6, 0.3, 0.2
    k = np.arange(2 ** d)
    assert rel(tt.sin(d, alpha, phase).full(asvector=True),
               np.sin(alpha * k + phase)) < 1e-12
    assert rel(tt.cos(d, alpha, phase).full(asvector=True),
               np.cos(alpha * k + phase)) < 1e-12


def test_unit_and_concatenate():
    e = np.zeros(8)
    e[3] = 1
    assert rel(tt.unit([2, 2, 2], j=3).full(asvector=True), e) < 1e-14
    a, b = tt.rand([2, 3], r=2), tt.rand([2, 3], r=2)
    c = tt.concatenate(a, b)
    assert rel(c[0, :, :].full(), a.full()) < 1e-12
    assert rel(c[1, :, :].full(), b.full()) < 1e-12


def test_mkron_matches_kron_chain():
    a, b, c = tt.rand([2], r=1), tt.rand([3], r=1), tt.rand([2], r=1)
    z = tt.mkron([a, b, c])
    ref = np.multiply.outer(np.multiply.outer(a.full(), b.full()), c.full())
    assert rel(z.full(), ref) < 1e-12


# --- reshape / permute -------------------------------------------------------

def test_reshape_split_and_merge():
    a = rand_dense((4, 4), 141)
    x = tt.vector(a, 1e-14)
    y = tt.reshape(x, [2, 2, 2, 2], eps=1e-13)
    assert list(y.n) == [2, 2, 2, 2]
    assert rel(y.full(asvector=True), a.flatten("F")) < 1e-11
    z = tt.reshape(y, [4, 4], eps=1e-13)
    assert rel(z.full(), a) < 1e-11


def test_reshape_qtt_roundtrip():
    x = tt.rand(2, 8, r=3)
    y = tt.reshape(x, [4, 4, 4, 4], eps=1e-12)
    z = tt.reshape(y, [2] * 8, eps=1e-12)
    assert rel(z.full(asvector=True), np.asarray(x.full()).flatten("F")) < 1e-10


def test_permute_against_dense():
    a = rand_dense((2, 3, 4), 151)
    x = tt.vector(a, 1e-14)
    y = tt.permute(x, [2, 0, 1], eps=1e-12)
    assert rel(y.full(), np.transpose(a, (2, 0, 1))) < 1e-10


def test_reshape_matrix():
    A = tt.matrix(rand_dense((4, 4, 4, 4), 161), 1e-14)
    B = tt.reshape(A, np.array([[2, 2], [2, 2], [2, 2], [2, 2]]), eps=1e-12)
    assert rel(B.full(), A.full()) < 1e-10


# --- loud failures -----------------------------------------------------------

def test_mode_mismatch_raises():
    with pytest.raises(ValueError):
        tt.rand([2, 3], r=2) + tt.rand([2, 4], r=2)


def test_bad_core_shape_raises():
    with pytest.raises(ValueError):
        tt.vector.from_list([np.zeros((1, 2, 3)), np.zeros((2, 2, 1))])


def test_reshape_size_mismatch_raises():
    with pytest.raises(ValueError):
        tt.reshape(tt.rand([4, 4], r=2), [3, 3])


# --- randomized rounding -----------------------------------------------------

def test_randomized_round_matches_svd_accuracy():
    """A sketched rounding must be close to the quasi-optimal SVD rounding."""
    rng = np.random.default_rng(5)
    d, n, r = 12, 2, 20
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [rng.standard_normal((ranks[k], n, ranks[k + 1])) for k in range(d)]
    x = tt.vector.from_list(cores)
    for target in (4, 8, 12):
        svd = x.round(0.0, rmax=target)
        rnd, err = x.round(rmax=target, method="randomized", seed=0,
                           return_error=True)
        assert max(rnd.r) <= target, (target, list(rnd.r))
        e_svd = (x - svd).norm()
        e_rnd = (x - rnd).norm()
        assert e_rnd <= 3 * e_svd + 1e-12, (target, e_rnd, e_svd)
        # the reported number is an upper bound that saturates at ||x||*sqrt(eps)
        floor = x.norm() * np.sqrt(np.finfo(np.float64).eps)
        assert err >= e_rnd - 1e-9 * x.norm(), (err, e_rnd)
        assert err <= max(3 * e_rnd, 2 * floor), (err, e_rnd, floor)


def test_randomized_round_is_exact_for_exactly_low_rank():
    x = tt.rand([3, 4, 5, 4, 3], r=4)
    y = x.round(rmax=4, method="randomized", seed=1)
    assert rel(y.full(), x.full()) < 1e-10


def test_randomized_round_demands_a_rank():
    with pytest.raises(ValueError):
        tt.rand([2, 2, 2], r=2).round(1e-8, method="randomized")


# --- guards against a whole class of mistakes --------------------------------

def test_no_ternary_einsum_anywhere_in_the_package():
    """einops.einsum forwards to np.einsum WITHOUT optimize=True.

    A contraction of three or more operands is then evaluated by brute force:
    measured 5.5 s versus 0.017 s for the same contraction split into two binary
    ones (numpy 2.x, shapes (300,4,300)x(300,160)x(160,4,160)). Splitting is not
    a matter of taste, so the rule is enforced rather than remembered.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "tt"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "einsum" and len(node.args) > 3:
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert not offenders, (
        "einsum called with 3+ operands (split it into binary contractions): "
        + ", ".join(offenders))


def test_dot_block_boundary_ranks_matches_the_general_path():
    """dot() has a fast path for r0 = 1; the block case must agree with it.

    Boundary ranks > 1 (block TT, as produced by eigb) take the einsum branch,
    so both branches need an oracle: here the dense contraction.
    """
    rng = np.random.default_rng(77)
    n = [3, 4, 2]
    for ra0, rb0, rad, rbd in [(2, 3, 1, 1), (1, 1, 2, 3), (2, 2, 3, 3)]:
        acores = [rng.standard_normal((ra0, 3, 4)), rng.standard_normal((4, 4, 5)),
                  rng.standard_normal((5, 2, rad))]
        bcores = [rng.standard_normal((rb0, 3, 3)), rng.standard_normal((3, 4, 2)),
                  rng.standard_normal((2, 2, rbd))]
        x, y = tt.vector.from_list(acores), tt.vector.from_list(bcores)
        got = np.asarray(_ops.dot(x.cores, y.cores)).reshape(ra0, rb0, rad, rbd)
        xa = np.asarray(_ops.full(x.cores)).reshape((ra0,) + tuple(n) + (rad,))
        yb = np.asarray(_ops.full(y.cores)).reshape((rb0,) + tuple(n) + (rbd,))
        ref = np.einsum("aijkc,bijkd->abcd", xa.conj(), yb)
        assert rel(got, ref) < 1e-11, (ra0, rb0, rad, rbd)


def test_dot_fast_path_and_general_path_agree():
    x, y = tt.rand([3, 4, 3], r=3), tt.rand([3, 4, 3], r=3)
    fast = _ops.dot(x.cores, y.cores)
    ref = float(np.sum(np.asarray(x.full()) * np.asarray(y.full())))
    assert abs(fast - ref) < 1e-10 * abs(ref)


# --- einsum routing ----------------------------------------------------------

@pytest.mark.parametrize("pattern, shapes", [
    ("a n b, b m c -> a n m c", [(2, 3, 4), (4, 5, 6)]),
    ("a n m b, i m j -> a i n b j", [(2, 3, 4, 5), (6, 4, 7)]),
    ("i n j, k n l -> i k n j l", [(2, 3, 4), (5, 3, 6)]),
    ("c b p, b j f -> c p j f", [(3, 4, 2), (4, 5, 6)]),
    ("a b i j, i n p -> a b j n p", [(2, 3, 4, 5), (4, 6, 7)]),
    ("a n b, b c -> a n c", [(2, 3, 4), (4, 5)]),
])
def test_backend_einsum_matches_einops(pattern, shapes):
    """Our einsum must be a drop-in for einops.einsum, only faster.

    einops forwards to np.einsum without optimize=True, which keeps even binary
    contractions out of BLAS (measured 5.1x on an AMEn local matvec). The
    replacement rewrites the pattern into classic subscripts; this pins that the
    rewrite is faithful.
    """
    from einops import einsum as einops_einsum
    from tt import backend as bk

    rng = np.random.default_rng(0)
    ops = [rng.standard_normal(s) for s in shapes]
    assert rel(bk.einsum(*ops, pattern), einops_einsum(*ops, pattern)) < 1e-13


def test_backend_einsum_rejects_a_misplaced_pattern():
    from tt import backend as bk
    with pytest.raises(TypeError):
        bk.einsum("a b, b c -> a c", np.zeros((2, 2)), np.zeros((2, 2)))


@pytest.mark.parametrize("pattern, shapes", [
    ("a n b c, c d e -> a n b d e", [(2, 2, 2, 1), (1, 1, 1)]),   # size-1 broadcast
    ("a b, b c -> a c", [(3, 1), (4, 5)]),                        # broadcast on k
    ("a n b, a n b -> a", [(2, 3, 4), (2, 3, 4)]),                # everything summed
    ("a b, c d -> a b c d", [(2, 3), (4, 5)]),                    # outer product
])
def test_backend_einsum_handles_broadcast_and_degenerate_patterns(pattern, shapes):
    """numpy's einsum broadcasts size-1 axes; a matmul plan must not force them."""
    from einops import einsum as einops_einsum
    from tt import backend as bk

    rng = np.random.default_rng(1)
    ops = [rng.standard_normal(s) for s in shapes]
    try:
        ref = einops_einsum(*ops, pattern)
    except Exception:
        pytest.skip("einops itself rejects this pattern")
    assert rel(bk.einsum(*ops, pattern), ref) < 1e-13


def test_permute_returns_a_compressed_representation():
    """``permute`` must not leave rank slack behind.

    Each adjacent swap truncates what is negligible on its own two cores, and
    the intermediate orderings genuinely need higher ranks than the final one;
    without a recompression pass at the end nothing ever removes that slack.
    Measured before the fix, on a separable function interleaved into Morton
    order at d = 15: rank 1024 for a tensor whose own rank is 102, i.e. a
    10x rank for the same numbers -- quadratic in every later contraction.

    The property asserted here is the one that matters and does not depend on a
    magic number: rounding the result again must not shrink it much.
    """
    d = 6
    rng = np.random.default_rng(0)
    g = tt.vector(rng.standard_normal((2,) * d), 1e-12)
    h = tt.vector(rng.standard_normal((2,) * d), 1e-12)
    x = tt.kron(g, h)                      # 2d modes, dimension-major
    order = [j * d + k for k in range(d) for j in range(2)]   # Morton

    y = tt.permute(x, order, 1e-12)
    slack = max(y.r) / max(y.round(1e-12).r)
    assert slack < 1.3, f"permute left {slack:.1f}x rank slack"

    # and it is still the right tensor: check against the dense permutation
    dense = np.asarray(x.full())
    want = np.transpose(dense, order)
    assert rel(y.full(), want) < 1e-10
