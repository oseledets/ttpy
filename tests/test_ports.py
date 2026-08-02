"""Tests for the four ported modules: optimize, completion, riemannian, solvers.

Every accuracy claim is checked against dense truth (a full ``numpy`` argmin, a
full linear solve, a projector assembled from the SVDs of the unfoldings of the
dense tensor) or against a mathematical invariant of the method (idempotence of
a projector, orthogonality of a residual, exactness of the projector splitting,
monotonicity of an alternating minimization, ``||b - A x|| <= eps ||b||``).
Nothing is compared against the legacy implementation's output.
"""

import numpy as np
import pytest

import tt
from tt.algs.completion import completion_functional, ttSparseALS
from tt.algs.optimize import min_func, min_tens
from tt.algs.riemannian import project, projector_splitting_add, tt_qr
from tt.algs.solvers import GMRES

rel = lambda a, b: np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


# --- helpers -----------------------------------------------------------------

def flat(x):
    """F-order flattening of a TT vector, the convention of the package."""
    return np.asarray(x.full(asvector=True))


def kron_f(mats):
    """Kronecker product in the F-order convention (first factor runs fastest)."""
    out = np.ones((1, 1), dtype=mats[0].dtype)
    for m in mats:
        out = np.kron(m, out)
    return out


def dense_tangent_projector(X, tol=1e-10):
    """The tangent-space projector at ``X`` as a dense ``(N, N)`` matrix.

    Built from the dense tensor alone -- the SVDs of its unfoldings -- so it is
    an oracle independent of anything in ``tt.algs.riemannian``:

        P = sum_k  P_{<k} (x) I_{n_k} (x) P_{>k}
          - sum_k  P_{<=k} (x) P_{>k}

    (Lubich, Oseledets, Vandereycken 2015, Thm 3.1).
    """
    n = [int(v) for v in X.n]
    d = len(n)
    full = np.asarray(X.full())
    left = [np.ones((1, 1), dtype=full.dtype)] * (d + 1)   # left[k] acts on modes < k
    right = [np.ones((1, 1), dtype=full.dtype)] * (d + 1)  # right[k] acts on modes > k
    for k in range(1, d):
        unf = full.reshape((int(np.prod(n[:k])), -1), order="F")
        u, s, vh = np.linalg.svd(unf, full_matrices=False)
        r = max(1, int(np.sum(s > s[0] * tol)))
        left[k] = u[:, :r] @ u[:, :r].conj().T
        v = vh[:r].conj().T
        right[k - 1] = v @ v.conj().T
    proj = np.zeros((full.size, full.size), dtype=full.dtype)
    for k in range(d):
        eye = np.eye(n[k], dtype=full.dtype)
        proj = proj + kron_f([left[k], eye, right[k]])
    for k in range(d - 1):
        proj = proj - kron_f([left[k + 1], right[k]])
    return proj


def random_tangent(X, rng):
    """A tangent vector at ``X``: ``sum_k tau(C_1, ..., dC_k, ..., C_d)``."""
    cores = [np.asarray(c) for c in X.cores]
    out = None
    for k in range(X.d):
        curr = list(cores)
        curr[k] = rng.standard_normal(cores[k].shape).astype(cores[k].dtype)
        z = tt.vector.from_list(curr)
        out = z if out is None else out + z
    return out


def low_rank_tt(n, r, rng, dtype=np.float64):
    """Random TT with exactly the requested ranks (rounded to be safe)."""
    d = len(n)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((ranks[k], n[k], ranks[k + 1]))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c.astype(dtype))
    return tt.vector.from_list(cores).round(0.0)


# =============================================================================
# 1. tt.algs.optimize -- min_tens / min_func
# =============================================================================

def test_min_tens_matches_dense_argmin():
    """The minimum of a smooth tensor is found exactly; regime is in the asserts."""
    n = [7, 6, 5, 6]
    grid = [np.linspace(-1.0, 1.0, m) for m in n]
    centre = [0.3, -0.5, 0.1, 0.7]
    dense = np.zeros(n)
    for k in range(len(n)):
        shape = [1] * len(n)
        shape[k] = n[k]
        dense = dense + ((grid[k] - centre[k]) ** 2).reshape(shape)
    tens = tt.vector(dense, 1e-14)
    val, point = min_tens(tens, rmax=6, nswp=12, verb=False, seed=0)
    assert val == pytest.approx(dense.min(), rel=1e-12)
    assert dense[tuple(int(i) for i in point)] == pytest.approx(val, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_min_tens_random_tensor(seed):
    """The legacy demo problem: a random rank-3 tensor, exact minimum expected."""
    rng = np.random.default_rng(seed)
    tens = low_rank_tt([3, 4, 5, 4, 3], 3, rng)
    dense = np.asarray(tens.full())
    val, point, hist = min_tens(tens, rmax=10, nswp=20, verb=False, seed=seed,
                                return_history=True)
    assert val == pytest.approx(dense.min(), rel=1e-10)
    assert hist.consistency < 1e-10 * max(1.0, abs(val))
    assert dense[tuple(int(i) for i in point)] == pytest.approx(val, rel=1e-12)


def test_min_tens_self_consistent_and_subexhaustive():
    """The returned value is the tensor at the returned point, and the search
    looked at far fewer entries than the tensor has."""
    rng = np.random.default_rng(7)
    n = [4] * 8
    tens = low_rank_tt(n, 4, rng)
    val, point, hist = min_tens(tens, rmax=6, nswp=8, verb=False, seed=3,
                                return_history=True)
    assert tens[tuple(int(i) for i in point)] == pytest.approx(val, rel=1e-12)
    assert hist.consistency == pytest.approx(0.0, abs=1e-10)
    assert hist.evaluations < 0.5 * np.prod(n)


def test_min_func_shifted_quadratic():
    """A quadratic whose minimizer sits on a grid node: value and point exact."""
    d, n0 = 4, 65
    grid = np.linspace(-2.0, 2.0, n0)
    centre = grid[np.array([10, 30, 50, 20])]

    def fun(x):
        assert x.ndim == 2, "min_func must call fun on a (P, d) array"
        return ((x - centre) ** 2).sum(axis=1)

    val, point, hist = min_func(fun, -2.0, 2.0, d=d, rmax=8, nswp=16, n0=n0,
                                verb=False, seed=0, return_history=True)
    assert val == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(point, centre)
    assert hist.consistency == pytest.approx(0.0, abs=1e-14)
    assert hist.evaluations < 0.05 * n0 ** d


def test_min_func_rosenbrock():
    """Rosenbrock in 3d on a grid that contains the minimizer (1, 1, 1)."""
    from scipy.optimize import rosen

    d, n0 = 3, 65
    grid = np.linspace(-2.0, 2.0, n0)
    assert np.isclose(grid[48], 1.0)

    def fun(x):
        assert x.ndim == 2
        return rosen(x.T)

    val, point, hist = min_func(fun, -2.0, 2.0, d=d, rmax=10, nswp=30, n0=n0,
                                verb=False, seed=1, return_history=True)
    assert val == pytest.approx(fun(np.asarray(point).reshape(1, d))[0], rel=1e-12)
    assert val < 1e-2, f"Rosenbrock: got {val} at {point}"
    assert np.max(np.abs(np.asarray(point) - 1.0)) <= 4.0 / (n0 - 1) * 2


def test_min_func_vector_bounds_and_history():
    """Per-dimension bounds infer d; the history explains the run."""
    lo = np.array([-1.0, 0.0, 2.0])
    hi = np.array([1.0, 3.0, 5.0])
    target = np.array([0.0, 1.5, 3.5])

    def fun(x):
        return ((x - target) ** 2).sum(axis=1)

    val, point, hist = min_func(fun, lo, hi, rmax=6, nswp=12, n0=33, verb=False,
                                seed=2, return_history=True)
    assert val < 1e-2
    assert np.all(point >= lo) and np.all(point <= hi)
    assert hist.sweeps == 12
    assert hist.records and hist.records[-1]["value"] >= val - 1e-12
    assert hist.max_index_set >= 1


def test_min_func_rejects_bad_objective():
    with pytest.raises(ValueError, match="must map a"):
        min_func(lambda x: np.zeros(3), -1.0, 1.0, d=3, n0=8, nswp=1, verb=False)
    with pytest.raises(ValueError, match="empty search box"):
        min_func(lambda x: x.sum(axis=1), 1.0, -1.0, d=2, n0=8, nswp=1, verb=False)


def test_min_func_custom_smooth_and_rho():
    """A user-supplied smoothing map and a different ``rho`` both still work."""
    target = np.array([0.5, -0.5])

    def fun(x):
        return ((x - target) ** 2).sum(axis=1)

    base, _ = min_func(fun, -1.0, 1.0, d=2, rmax=6, nswp=10, n0=33, verb=False,
                       seed=0)
    sharp, _ = min_func(fun, -1.0, 1.0, d=2, rmax=6, nswp=10, n0=33, rho=0.05,
                        verb=False, seed=0)
    custom, _ = min_func(fun, -1.0, 1.0, d=2, rmax=6, nswp=10, n0=33,
                         smooth_fun=lambda p, lam: np.exp(-(p - lam)),
                         verb=False, seed=0)
    for v in (base, sharp, custom):
        assert v == pytest.approx(0.0, abs=1e-12)


def test_min_one_dimensional():
    """d = 1: there is no sweep, the whole vector is scanned."""
    vals = np.array([3.0, -1.0, 2.0, 0.5])
    x = tt.vector.from_list([vals.reshape(1, 4, 1)])
    val, point = min_tens(x, verb=False)
    assert val == pytest.approx(-1.0) and int(point[0]) == 1

    val, point = min_func(lambda z: (z[:, 0] - 0.25) ** 2, -1.0, 1.0, d=1,
                          n0=9, verb=False)
    assert val == pytest.approx(0.0, abs=1e-3)


def test_verbose_paths_do_not_crash(capsys):
    """``verb=True`` is the legacy default; it must print, not explode."""
    rng = np.random.default_rng(0)
    tens = low_rank_tt([3, 3, 3], 2, rng)
    min_tens(tens, rmax=4, nswp=4, verb=True, seed=0)
    min_func(lambda x: (x ** 2).sum(axis=1), -1.0, 1.0, d=2, n0=9, nswp=4,
             rmax=4, verb=True, seed=0)

    coo, _ = _sample(low_rank_tt([4, 4, 4], 2, rng), 200, rng)
    ttSparseALS(coo, [4, 4, 4], ttRank=2, maxnsweeps=3, verbose=True, seed=0)

    A, x_exact, b = laplace_problem(3, rank=2, seed=0)
    GMRES(matvec_of(A), tt.zeros([2] * 3), b, eps=1e-8, maxit=200, m=4, verbose=2)
    assert "New record" in capsys.readouterr().out


def test_min_tens_rejects_complex():
    x = tt.rand([3, 3, 3], r=2).astype("complex128")
    with pytest.raises(TypeError, match="complex"):
        min_tens(x, verb=False)


# =============================================================================
# 2. tt.algs.completion -- ttSparseALS
# =============================================================================

def _sample(x, m, rng):
    """``m`` distinct random positions of ``x`` with their values."""
    n = [int(v) for v in x.n]
    idx = np.stack([rng.integers(0, v, size=m) for v in n], axis=1)
    idx = np.unique(idx, axis=0)
    dense = np.asarray(x.full())
    vals = dense[tuple(idx[:, k] for k in range(len(n)))]
    return {"indices": idx, "values": vals}, dense


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_completion_recovers_unseen_entries(seed):
    """The real test: error on entries the fit never saw.

    Regime: rank-2 tensor of shape 8x8x8x8 (96 parameters), ~1330 distinct
    samples out of 4096 entries, recovered at the true rank with ``alpha = 0``.
    At that sampling density all six random starts converge (with ~820 samples
    two of six stall -- see ``test_completion_reports_a_stalled_run``).
    """
    rng = np.random.default_rng(seed)
    n = [8, 8, 8, 8]
    truth = low_rank_tt(n, 2, rng)
    dense = np.asarray(truth.full())
    coo, _ = _sample(truth, 1600, rng)

    x, info = ttSparseALS(coo, n, ttRank=2, tol=1e-13, maxnsweeps=100,
                          verbose=False, alpha=0.0, seed=seed)
    got = np.asarray(x.full())

    seen = np.zeros(n, dtype=bool)
    seen[tuple(coo["indices"][:, k] for k in range(len(n)))] = True
    err_unseen = (np.linalg.norm(got[~seen] - dense[~seen])
                  / np.linalg.norm(dense[~seen]))
    err_seen = (np.linalg.norm(got[seen] - dense[seen])
                / np.linalg.norm(dense[seen]))
    assert info.converged, info.stop_reason
    assert err_seen < 1e-6, f"training error {err_seen}"
    assert err_unseen < 1e-5, f"generalization error {err_unseen}"
    assert info["fit"][-1] < 1e-13
    assert info.ranks == [1, 2, 2, 2, 1]


def test_completion_reports_a_stalled_run():
    """A run that stops at a stationary point must not claim to have converged."""
    rng = np.random.default_rng(0)
    n = [8, 8, 8, 8]
    truth = low_rank_tt(n, 2, rng)
    coo, _ = _sample(truth, 900, rng)
    x, info = ttSparseALS(coo, n, ttRank=2, tol=1e-13, maxnsweeps=60,
                          verbose=False, alpha=0.0, seed=0)
    assert info.fit[-1] > 1e-3            # the documented stalling regime
    assert not info.converged
    assert info.stop_reason in ("stalled", "maxnsweeps")


def test_completion_from_the_exact_solution_is_a_fixed_point():
    """Started at the truth, ALS must stay there: the local systems are right."""
    rng = np.random.default_rng(0)
    n = [8, 8, 8, 8]
    truth = low_rank_tt(n, 2, rng)
    coo, _ = _sample(truth, 900, rng)
    x, info = ttSparseALS(coo, n, x0=truth, tol=0.0, maxnsweeps=3,
                          verbose=False, alpha=0.0)
    assert max(info.fit) < 1e-25
    assert rel(x.full(), truth.full()) < 1e-12


def test_completion_functional_decreases():
    """Exact local solves make the functional monotonically non-increasing."""
    rng = np.random.default_rng(3)
    n = [6, 6, 6, 6]
    truth = low_rank_tt(n, 3, rng)
    coo, _ = _sample(truth, 700, rng)
    x, info = ttSparseALS(coo, n, ttRank=3, tol=0.0, maxnsweeps=15,
                          verbose=False, alpha=0.0, seed=5)
    fit = np.asarray(info.fit)
    assert info.monotone, f"functional increased: {fit}"
    assert np.all(np.diff(fit) <= 1e-12 * max(fit[0], 1e-300))
    assert fit[-1] < fit[0]


def test_completion_functional_matches_definition():
    rng = np.random.default_rng(11)
    n = [5, 4, 6]
    truth = low_rank_tt(n, 2, rng)
    coo, dense = _sample(truth, 50, rng)
    x = low_rank_tt(n, 2, rng)
    got = completion_functional(x, coo)
    xd = np.asarray(x.full())
    want = 0.5 * sum((xd[tuple(i)] - v) ** 2
                     for i, v in zip(coo["indices"], coo["values"]))
    assert got == pytest.approx(want, rel=1e-12)


def test_completion_does_not_mutate_input():
    rng = np.random.default_rng(4)
    n = [5, 5, 5]
    truth = low_rank_tt(n, 2, rng)
    coo, _ = _sample(truth, 200, rng)
    idx_before = coo["indices"].copy()
    val_before = coo["values"].copy()
    x0 = low_rank_tt(n, 2, rng)
    cores_before = [np.asarray(c).copy() for c in x0.cores]

    ttSparseALS(coo, n, x0=x0, tol=1e-10, maxnsweeps=5, verbose=False, alpha=0.0)

    assert np.array_equal(coo["indices"], idx_before)
    assert np.array_equal(coo["values"], val_before)
    for before, after in zip(cores_before, x0.cores):
        assert np.array_equal(before, np.asarray(after))


def test_completion_x0_scaling_is_undone():
    """The result lives in the scale of the data, not of the normalized copy."""
    rng = np.random.default_rng(6)
    n = [6, 6, 6]
    truth = 1234.5 * low_rank_tt(n, 2, rng)
    coo, dense = _sample(truth, 600, rng)
    x, info = ttSparseALS(coo, n, ttRank=2, tol=1e-14, maxnsweeps=40,
                          verbose=False, alpha=0.0, seed=2)
    assert rel(x.full(), dense) < 1e-6


def test_completion_empty_slice_keeps_previous_value():
    """A slice no sample touches must not be zeroed (that destroys rank)."""
    rng = np.random.default_rng(8)
    n = [4, 4, 4]
    truth = low_rank_tt(n, 2, rng)
    dense = np.asarray(truth.full())
    idx = np.array([[i, j, k] for i in range(4) for j in range(4)
                    for k in range(3)])              # mode 2, slice 3 never used
    coo = {"indices": idx, "values": dense[tuple(idx[:, m] for m in range(3))]}
    x0 = low_rank_tt(n, 2, rng)
    before = np.asarray(x0.cores[2])[:, 3, :].copy()
    x, info = ttSparseALS(coo, n, x0=x0, tol=0.0, maxnsweeps=1, verbose=False,
                          alpha=0.0)
    assert info.empty_slices == 1
    got = np.asarray(x.cores[2])[:, 3, :]
    assert np.linalg.norm(got) > 0.0
    assert rel(got, before) < 1e-12       # the data scaling lives in core 0


def test_completion_rejects_bad_input():
    rng = np.random.default_rng(9)
    n = [4, 4, 4]
    truth = low_rank_tt(n, 2, rng)
    coo, _ = _sample(truth, 40, rng)
    with pytest.raises(IndexError, match="out of range"):
        ttSparseALS({"indices": coo["indices"], "values": coo["values"]},
                    [3, 4, 4], ttRank=2, maxnsweeps=1, verbose=False)
    with pytest.raises(ValueError, match="indices"):
        ttSparseALS({"values": coo["values"]}, n, maxnsweeps=1, verbose=False)


# =============================================================================
# 3. tt.algs.riemannian -- project / projector_splitting_add / tt_qr
# =============================================================================

@pytest.mark.parametrize("n,ranks", [([4, 4, 4], [1, 4, 4, 1]),
                                     ([2, 3, 4], [1, 2, 3, 1]),
                                     ([3, 3, 3, 3], [1, 3, 3, 3, 1])])
def test_project_matches_dense_projector(n, ranks):
    X = tt.rand(n, r=ranks).round(0.0)
    Z = tt.rand(n, r=2)
    got = flat(project(X, Z))
    want = dense_tangent_projector(X) @ flat(Z)
    assert rel(got, want) < 1e-10


def test_dense_projector_is_a_projector():
    """Sanity of the oracle itself, so a failure above is not its fault."""
    X = tt.rand([3, 4, 3], r=[1, 3, 3, 1]).round(0.0)
    p = dense_tangent_projector(X)
    assert rel(p @ p, p) < 1e-10
    assert rel(p.T, p) < 1e-10


def test_project_is_idempotent_and_fixes_tangent_vectors():
    rng = np.random.default_rng(2)
    X = tt.rand([4, 4, 4], r=[1, 4, 4, 1]).round(0.0)
    Z = tt.rand([4, 4, 4], r=3)
    pz = project(X, Z)
    ppz = project(X, pz)
    assert rel(ppz.full(), pz.full()) < 1e-10

    tangent = random_tangent(X, rng)
    assert rel(project(X, tangent).full(), tangent.full()) < 1e-10


def test_project_residual_is_orthogonal_to_the_tangent_space():
    rng = np.random.default_rng(3)
    X = tt.rand([3, 4, 5], r=[1, 3, 4, 1]).round(0.0)
    Z = tt.rand([3, 4, 5], r=3)
    resid = Z - project(X, Z)
    scale = float(Z.norm())
    for _ in range(5):
        t = project(X, tt.rand([3, 4, 5], r=2))
        assert abs(tt.dot(resid, t)) < 1e-10 * scale * float(t.norm())
    t = random_tangent(X, rng)
    assert abs(tt.dot(resid, t)) < 1e-10 * scale * float(t.norm())


def test_project_of_a_list_is_the_projection_of_the_sum():
    X = tt.rand([4, 4, 4], r=[1, 4, 4, 1]).round(0.0)
    zs = [tt.rand([4, 4, 4], r=2) for _ in range(5)]
    got = flat(project(X, zs))
    total = zs[0]
    for z in zs[1:]:
        total = total + z
    want = dense_tangent_projector(X) @ flat(total)
    assert rel(got, want) < 1e-10


def test_project_complex():
    rng = np.random.default_rng(5)
    X = low_rank_tt([3, 4, 3], 3, rng, dtype=np.complex128)
    Z = low_rank_tt([3, 4, 3], 2, rng, dtype=np.complex128)
    got = flat(project(X, Z))
    want = dense_tangent_projector(X) @ flat(Z)
    assert rel(got, want) < 1e-10


def test_project_rank_is_twice_the_rank_of_x():
    X = tt.rand([4, 4, 4, 4], r=[1, 3, 3, 3, 1]).round(0.0)
    p = project(X, tt.rand([4, 4, 4, 4], r=5))
    assert list(p.r) == [1, 6, 6, 6, 1]


@pytest.mark.parametrize("n,ranks", [([5, 2, 3], [1, 2, 3, 1]),
                                     ([4, 4, 4], [1, 3, 3, 1]),
                                     ([3, 4, 5, 3], [1, 2, 4, 2, 1])])
def test_projector_splitting_add_is_exact_on_the_manifold(n, ranks):
    """LOV Thm 4.1: if Y + delta has the rank of Y, the splitting returns it."""
    rng = np.random.default_rng(0)
    cores_y, cores_w = [], []
    for k in range(len(n)):
        cores_y.append(rng.standard_normal((ranks[k], n[k], ranks[k + 1])))
        cores_w.append(rng.standard_normal((ranks[k], n[k], ranks[k + 1])))
    Y = tt.vector.from_list(cores_y)
    W = tt.vector.from_list(cores_w)
    got = projector_splitting_add(Y, W - Y)
    assert rel(got.full(), W.full()) < 1e-9
    assert list(got.r) == ranks


def test_projector_splitting_add_doubling():
    Y = tt.rand([5, 2, 3], r=[1, 2, 3, 1]).round(0.0)
    got = projector_splitting_add(Y, Y)
    assert rel(got.full(), 2 * np.asarray(Y.full())) < 1e-10


def test_projector_splitting_add_is_first_order():
    """``psa(Y, t Z) = Y + t P_Y Z + O(t^2)``: the error must drop by ~4 when t halves."""
    n = [4, 4, 4]
    Y = tt.rand(n, r=[1, 3, 3, 1]).round(0.0)
    Y = (1.0 / Y.norm()) * Y
    Z = tt.rand(n, r=2)
    Z = (1.0 / Z.norm()) * Z
    pz = flat(project(Y, Z))
    errs = []
    for t in (1e-2, 5e-3):
        got = flat(projector_splitting_add(Y, t * Z))
        errs.append(np.linalg.norm(got - (flat(Y) + t * pz)) / t ** 2)
    assert errs[0] > 0.0
    assert errs[1] / errs[0] < 2.0, f"not second order: {errs}"


@pytest.mark.parametrize("left_to_right", [True, False])
def test_tt_qr(left_to_right):
    n = [2, 3, 4, 5]
    X = tt.rand(n, r=3)
    q, r = tt_qr(X, left_to_right=left_to_right)
    assert r.shape == (1, 1)
    assert rel(float(r[0, 0]) * np.asarray(q.full()), X.full()) < 1e-10
    assert abs(float(q.norm()) - 1.0) < 1e-12
    assert abs(abs(float(r[0, 0])) - float(X.norm())) < 1e-10 * float(X.norm())

    cores = [np.asarray(c) for c in q.cores]
    if left_to_right:
        for c in cores:
            m = c.reshape((-1, c.shape[2]), order="C")
            assert rel(m.T.conj() @ m, np.eye(c.shape[2])) < 1e-10
    else:
        for c in cores:
            m = c.reshape((c.shape[0], -1), order="C")
            assert rel(m @ m.T.conj(), np.eye(c.shape[0])) < 1e-10


def test_tt_qr_reproduces_a_block_tensor():
    """``X = Q x_last R`` also when the last rank is not 1."""
    rng = np.random.default_rng(3)
    cores = [rng.standard_normal(s) for s in [(1, 3, 4), (4, 4, 3), (3, 2, 2)]]
    X = tt.vector.from_list(cores)
    q, r = tt_qr(X, left_to_right=True)
    qc = [np.asarray(c) for c in q.cores]
    qc[-1] = np.einsum("anb,bc->anc", qc[-1], np.asarray(r))
    assert rel(tt.vector.from_list(qc).full(), X.full()) < 1e-10


# =============================================================================
# 4. tt.algs.solvers -- GMRES
# =============================================================================

def laplace_problem(d, rank=3, seed=0):
    """``A x = b`` with ``A`` the QTT Laplacian on ``2**d`` points."""
    rng = np.random.default_rng(seed)
    A = tt.qlaplace_dd([d])
    x_exact = low_rank_tt([2] * d, rank, rng)
    b = tt.matvec(A, x_exact).round(1e-14)
    return A, x_exact, b


def matvec_of(A):
    return lambda x, eps: tt.matvec(A, x).round(eps)


def test_gmres_solves_qlaplace_and_matches_the_dense_solve():
    d, eps = 4, 1e-8
    A, x_exact, b = laplace_problem(d)
    u0 = tt.zeros([2] * d)
    x, res, hist = GMRES(matvec_of(A), u0, b, eps=eps, maxit=200, m=20,
                         verbose=0, return_history=True)
    assert res <= eps
    assert hist.converged

    dense_x = np.linalg.solve(np.asarray(A.full()), flat(b))
    assert rel(flat(x), dense_x) < 1e-6
    assert rel(flat(x), flat(x_exact)) < 1e-6

    # the reported residual is the measured one, not an estimate
    true_res = float((b - tt.matvec(A, x)).norm()) / float(b.norm())
    assert res == pytest.approx(true_res, rel=1e-8)


def test_gmres_restarts_are_iterative():
    """A tiny Krylov dimension with many restarts must still converge (and not
    grow the python stack, which the recursive legacy version did)."""
    d, eps = 4, 1e-6
    A, x_exact, b = laplace_problem(d, rank=2, seed=1)
    x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps,
                         maxit=600, m=3, verbose=0, return_history=True)
    assert hist.converged, hist.message
    assert res <= eps
    assert len(hist.cycles) > 5, f"expected many restarts, got {len(hist.cycles)}"
    assert rel(flat(x), flat(x_exact)) < 1e-4


def test_gmres_does_not_mutate_the_initial_guess():
    d = 4
    A, x_exact, b = laplace_problem(d, seed=2)
    u0 = low_rank_tt([2] * d, 2, np.random.default_rng(0))
    before = [np.asarray(c).copy() for c in u0.cores]
    x, res = GMRES(matvec_of(A), u0, b, eps=1e-8, maxit=200, m=20, verbose=0)
    for old, new in zip(before, u0.cores):
        assert np.array_equal(old, np.asarray(new))
    assert res <= 1e-8


def test_gmres_starts_from_a_nonzero_guess():
    d = 4
    A, x_exact, b = laplace_problem(d, seed=3)
    u0 = low_rank_tt([2] * d, 2, np.random.default_rng(1))
    x, res = GMRES(matvec_of(A), u0, b, eps=1e-8, maxit=300, m=20, verbose=0)
    assert res <= 1e-8
    assert rel(flat(x), flat(x_exact)) < 1e-5


def test_gmres_reports_non_convergence():
    d = 6
    A, x_exact, b = laplace_problem(d, rank=2, seed=4)
    with pytest.warns(RuntimeWarning, match="stopped after"):
        x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=1e-12,
                             maxit=3, m=3, verbose=0, return_history=True)
    assert not hist.converged
    assert hist.iterations == 3
    assert res > 1e-12
    true_res = float((b - tt.matvec(A, x)).norm()) / float(b.norm())
    assert res == pytest.approx(true_res, rel=1e-8)


def test_gmres_callback_and_history():
    d = 4
    A, x_exact, b = laplace_problem(d, seed=5)
    seen = []
    x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=1e-8,
                         maxit=200, m=5, callback=lambda y: seen.append(max(y.r)),
                         verbose=0, return_history=True)
    assert len(seen) == len(hist.cycles)
    assert hist.iterations == sum(c["iterations"] for c in hist.cycles)
    assert len(hist.residuals) == len(hist.cycles) + 1
    assert hist.residuals[-1] == pytest.approx(hist.true_res)
    assert all(np.isfinite(c["res_est"]) for c in hist.cycles)


def test_gmres_zero_rhs_is_an_error():
    d = 3
    A = tt.qlaplace_dd([d])
    with pytest.raises(ValueError, match="right-hand side is zero"):
        GMRES(matvec_of(A), tt.zeros([2] * d), 0.0 * tt.ones(2, d), eps=1e-8,
              verbose=0)


def test_gmres_estimate_tracks_the_true_residual():
    """The Krylov estimate and the measured residual must agree at the end."""
    d = 4
    A, x_exact, b = laplace_problem(d, seed=7)
    x, res, hist = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=1e-6,
                         maxit=300, m=4, verbose=0, return_history=True)
    last = hist.cycles[-1]
    assert last["res_end"] == pytest.approx(res)
    assert last["res_est"] == pytest.approx(res, rel=0.5), (last, res)


def test_gmres_on_a_diagonally_dominant_random_operator():
    """A non-symmetric operator, checked against the dense solve."""
    rng = np.random.default_rng(6)
    d, n = 3, 4
    cores = [rng.standard_normal((1 if k == 0 else 2, n, n, 1 if k == d - 1 else 2))
             for k in range(d)]
    A = tt.matrix.from_list(cores)
    A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A + 3.0 * tt.eye([n] * d)
    x_exact = low_rank_tt([n] * d, 2, rng)
    b = tt.matvec(A, x_exact).round(1e-14)
    x, res = GMRES(matvec_of(A), tt.zeros([n] * d), b, eps=1e-9, maxit=300,
                   m=20, verbose=0)
    assert res <= 1e-9
    dense_x = np.linalg.solve(np.asarray(A.full()), flat(b))
    assert rel(flat(x), dense_x) < 1e-7


# =============================================================================
# 5. the torch backend, and a core defect found on the way
# =============================================================================

@pytest.fixture
def torch_default():
    """Run the body with ``torch`` as the default backend, then restore."""
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = tt.get_backend()
    tt.set_backend("torch", device, "float64")
    try:
        yield device
    finally:
        tt.set_backend(old)


def test_riemannian_on_torch(torch_default):
    """``project`` / ``projector_splitting_add`` / ``tt_qr`` are backend-agnostic."""
    X = tt.rand([4, 4, 4], r=[1, 4, 4, 1]).round(0.0)
    Z = tt.rand([4, 4, 4], r=3)
    assert X.backend.name == "torch"
    got = flat(project(X, Z).to("numpy"))
    want = dense_tangent_projector(X.to("numpy")) @ flat(Z.to("numpy"))
    assert rel(got, want) < 1e-10

    Y = tt.rand([5, 2, 3], r=[1, 2, 3, 1])
    W = tt.rand([5, 2, 3], r=[1, 2, 3, 1])
    add = projector_splitting_add(Y, W - Y).to("numpy")
    assert rel(add.full(), W.to("numpy").full()) < 1e-9

    q, r = tt_qr(X)
    assert rel(float(r[0, 0]) * np.asarray(q.to("numpy").full()),
               X.to("numpy").full()) < 1e-10


def test_gmres_on_torch(torch_default):
    d, eps = 4, 1e-8
    A = tt.qlaplace_dd([d]).to("torch", torch_default)
    x_exact = tt.rand([2] * d, r=3)
    b = tt.matvec(A, x_exact).round(1e-14)
    x, res = GMRES(matvec_of(A), tt.zeros([2] * d), b, eps=eps, maxit=200,
                   m=20, verbose=0)
    assert x.backend.name == "torch"
    assert res <= eps
    assert rel(flat(x.to("numpy")), flat(x_exact.to("numpy"))) < 1e-6


def test_min_tens_on_torch(torch_default):
    """The sweep is numpy, but it must accept a torch tensor and be right."""
    x = tt.rand([3, 4, 5, 4, 3], r=3)
    val, point = min_tens(x, rmax=10, nswp=20, verb=False, seed=0)
    dense = np.asarray(x.to("numpy").full())
    assert val == pytest.approx(dense.min(), rel=1e-10)


@pytest.mark.xfail(strict=True, reason=(
    "CORE DEFECT (tt/core/tools.py, frozen): the constructors qlaplace_dd, "
    "sin, cos, delta, stepfun, qshift, unit and xfun build their cores with "
    "np.zeros/np.eye instead of tt.backend, so they ignore tt.set_backend and "
    "return numpy tensors -- while ones/zeros/rand/eye in the same module do "
    "honour it, and tt/backend.py documents the default backend as 'used by "
    "constructors'.  Consequence: tt.matvec(qlaplace_dd(...), torch_vector) "
    "dies inside einops with 'can't convert cuda:0 device type tensor to "
    "numpy' instead of either working or raising the loud "
    "'cores live on different backends' error of bk.same_backend."))
def test_core_constructors_honour_the_default_backend(torch_default):
    from tt import backend as bkend
    for name, build in [("qlaplace_dd", lambda: tt.matrix.to_list(tt.qlaplace_dd([3]))[0]),
                        ("sin", lambda: tt.sin(4).cores[0]),
                        ("delta", lambda: tt.delta(2, 4).cores[0]),
                        ("xfun", lambda: tt.xfun(2, 4).cores[0])]:
        assert bkend.backend_of(build()).name == "torch", name


@pytest.mark.xfail(strict=True, reason=(
    "CORE DEFECT (tt/core/tools.py, frozen): tt.linspace mixes a numpy array "
    "into a core allocated through the backend, so with the torch default it "
    "raises TypeError: can't assign a numpy.ndarray to a torch.cuda."
    "DoubleTensor.  A constructor must either honour the default backend or "
    "ignore it; raising is neither."))
def test_core_linspace_works_under_the_torch_backend(torch_default):
    tt.linspace(2, 4)
