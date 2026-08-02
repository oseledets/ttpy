"""Adversarial verification of ``tt.algs.amen_mv``.

Everything here is checked against a dense numpy computation, against a
*different* TT algorithm (``tt.matvec(...).round``), or against a mathematical
invariant (scaling equivariance, orthogonality, monotonicity in ``tol``, a
residual being a lower bound).  Nothing is checked against ``amen_mv`` itself,
except the two places where self-consistency *is* the property under test
(determinism from a seed, and numpy vs torch agreement).

The cases here are the ones the module's own suite does not reach: exactly zero
input, ``d=1`` and ``d=2``, mode sizes that differ per mode, float32, a regime
where the truncation is actually active (the module's own ``tol`` sweep runs on
a product that is exactly representable, so its error is 3e-15 at every ``tol``
and its monotonicity assertion cannot fail), and the argument validation that
used to fall through into einops errors or an infinite loop.
"""

import warnings

import numpy as np
import pytest

import tt
from tt import backend as bk
from tt.algs.amen_mv import amen_mv, _gram_svd


# --- helpers ------------------------------------------------------------------

def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def _modes(v, d, k):
    return v[k] if isinstance(v, (list, tuple)) else v


def rand_matrix(n, m, d, r, seed=0, complex_=False):
    """Random TT-matrix; ``n``/``m`` may be ints or per-mode lists."""
    rng = np.random.default_rng(seed)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((rr[k], _modes(n, d, k), _modes(m, d, k),
                                 rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c)
    return tt.matrix.from_list(cores)


def rand_vector(m, d, r, seed=1, complex_=False):
    rng = np.random.default_rng(seed)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        c = rng.standard_normal((rr[k], _modes(m, d, k), rr[k + 1]))
        if complex_:
            c = c + 1j * rng.standard_normal(c.shape)
        cores.append(c)
    return tt.vector.from_list(cores)


def graded_vector(m, d, nterms, ratio, seed=2):
    """Sum of ``nterms`` rank-one terms with weights ``ratio**s``.

    Unlike a random TT of rank ``r``, this has a genuinely decaying singular
    spectrum, so a truncation at ``tol`` actually removes something.
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


def left_orthogonalize(cores):
    out = [np.asarray(c).copy() for c in cores]
    for k in range(len(out) - 1):
        r0, n, r1 = out[k].shape
        q, r = np.linalg.qr(out[k].reshape((r0 * n, r1)))
        out[k] = q.reshape((r0, n, q.shape[1]))
        out[k + 1] = np.einsum("ab,bnc->anc", r, out[k + 1])
    return out


# --- the truncation is actually exercised -------------------------------------

def test_error_tracks_tol_where_truncation_is_active():
    """The module's own tol sweep runs on an exactly representable product.

    Regime: float64, ``d=6``, ``n=m=5``, ``A`` of TT rank 1, ``x`` a sum of 10
    rank-one terms with weights ``0.45**s`` (exact TT ranks 1-5-10-10-10-5-1).
    Here a truncation at ``tol`` removes real singular values, so the error has
    to *follow* ``tol`` instead of sitting at 1e-15 -- which is what the
    module's own ``test_error_decreases_with_tol`` measures (3e-15 at every tol
    from 1e-2 to 1e-10, an assertion that cannot fail).
    """
    A = tt.matrix.from_list(
        [np.random.default_rng(50 + k).standard_normal((1, 5, 5, 1))
         for k in range(6)])
    x = graded_vector(5, 6, 10, 0.45, seed=51)
    exact = tt.matvec(A, x)
    ref = exact.full()

    errs, ranks = [], []
    for tol in [1e-1, 1e-2, 1e-3]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y, _ = amen_mv(A, x, tol, verb=0, nswp=40)
        e = rel(y.full(), ref)
        best = rel(exact.round(tol).full(), ref)
        errs.append(e)
        ranks.append(max(y.r))
        # (a) the request is met, (b) the truncation is genuinely active -- the
        # error is within an order of magnitude of tol, not 1e-15, and (c) the
        # adaptive ranks are no worse than the optimal SVD truncation.
        assert e <= tol, (tol, e)
        assert e >= 0.01 * tol, (tol, e)
        assert e <= 1.5 * best, (tol, e, best)
    assert errs[0] > errs[1] > errs[2], errs
    assert ranks[0] < ranks[1] < ranks[2], ranks


def test_error_can_exceed_tol_and_depends_on_the_random_guess():
    """Pins the honest caveat: ``tol`` is a block threshold, not a bound.

    Regime float64, ``d=6``, ``n=m=4``, ``r_A=r_x=4``, ``tol=1e-1``, one fixed
    problem, 15 different random initial guesses (``seed=0..14``).  The outcome
    is bimodal: 10 runs land at 7.2e-2 (below the request, near the optimal
    6.0e-2 of an SVD truncation of the same product) and 5 land at 1.8e-1 to
    2.0e-1, i.e. twice the request.  Which of the two the caller gets is decided
    by the random ``y0``, and the default ``seed=None`` makes that a run-to-run
    lottery -- so no accuracy assertion at a loose ``tol`` is meaningful unless
    the seed is pinned.
    """
    A = rand_matrix(4, 4, 6, 4, seed=22)
    x = rand_vector(4, 6, 4, seed=23)
    exact = tt.matvec(A, x)
    ref = exact.full()
    best = rel(exact.round(1e-1).full(), ref)
    errs = []
    for s in range(15):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y, _ = amen_mv(A, x, 1e-1, verb=0, nswp=30, seed=s)
        errs.append(rel(y.full(), ref))
    assert max(errs) > 1e-1, errs        # tol is not a bound
    assert min(errs) < 1e-1, errs        # and usually it is met
    assert min(errs) > best              # never better than the optimal
    assert max(errs) < 1.0


# --- degenerate and edge-case inputs ------------------------------------------

@pytest.mark.parametrize("renorm", ["direct", "gram"])
def test_zero_input_is_zero_not_nan(renorm):
    """``renorm='gram'`` used to divide by ``s[0] = 0`` and return NaN.

    The only sign was a numpy RuntimeWarning ("invalid value encountered in
    divide"), and on the torch backend there is not even that -- a plausible
    looking ``tt.vector`` full of NaN came back from a perfectly well posed
    problem.  Both paths must produce exactly zero.
    """
    A = rand_matrix(3, 3, 4, 2, seed=1)
    x = tt.zeros(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        y, z = amen_mv(A, x, 1e-10, verb=0, renorm=renorm)
    f = np.asarray(y.full(asvector=True))
    assert np.all(np.isfinite(f))
    assert np.linalg.norm(f) == 0.0


def test_zero_matrix_gives_zero():
    A = tt.matrix.from_list([np.zeros((1, 3, 3, 1)) for _ in range(4)])
    x = rand_vector(3, 4, 2, seed=2)
    y, _ = amen_mv(A, x, 1e-10, verb=0)
    f = np.asarray(y.full(asvector=True))
    assert np.all(np.isfinite(f))
    assert np.linalg.norm(f) == 0.0


def test_gram_svd_of_a_zero_matrix_is_an_exact_factorization():
    a = np.zeros((64, 8))
    u, s, vh = _gram_svd(a)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(s))
    assert np.linalg.norm(u.T.conj() @ u - np.eye(u.shape[1])) < 1e-14
    assert np.linalg.norm((u * s) @ vh - a) == 0.0


def test_d_equals_two():
    A = rand_matrix(4, 4, 2, 3, seed=3)
    x = rand_vector(4, 2, 3, seed=4)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12


def test_d_equals_one_with_enrichment():
    A = rand_matrix(5, 5, 1, 1, seed=60)
    x = rand_vector(5, 1, 1, seed=61)
    y, z, h = amen_mv(A, x, 1e-10, verb=0, return_history=True)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-13
    assert isinstance(z, tt.vector) and z.d == 1
    assert h.res_est < 1e-13


def test_mode_sizes_differ_per_mode_and_are_rectangular():
    """``n_k != m_k`` and both varying with ``k`` -- the index bookkeeping."""
    n, m = [2, 5, 3, 4], [3, 2, 4, 2]
    A = rand_matrix(n, m, 4, 2, seed=5)
    x = rand_vector(m, 4, 3, seed=6)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert list(y.n) == n
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12


def test_complex_rectangular_varying_modes():
    """Complex + rectangular + per-mode sizes: conjugation and reshapes at once.

    d=5, n=[3,2,4,3,2], m=[2,3,2,4,3], r_A=r_x=2, tol=1e-10, complex128.
    """
    n, m = [3, 2, 4, 3, 2], [2, 3, 2, 4, 3]
    A = rand_matrix(n, m, 5, 2, seed=30, complex_=True)
    x = rand_vector(m, 5, 2, seed=31, complex_=True)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=40)
    assert np.iscomplexobj(np.asarray(y.full()))
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12


@pytest.mark.parametrize("kw", [{"fkick": True}, {"kickrank2": 2},
                                {"renorm": "gram"}, {}])
def test_complex_through_every_option_path(kw):
    """fkick / kickrank2 / gram all touch conjugation; complex is the detector.

    ``nswp=6`` on purpose: with 30 sweeps plain ALS eventually finds the answer
    even when the residual enrichment is built from a wrongly conjugated
    ``Z^H Y`` interface, so a generous sweep budget hides that class of bug.
    Six sweeps is enough only if the enrichment really points at the residual.
    """
    A = rand_matrix(3, 3, 5, 2, seed=32, complex_=True)
    x = rand_vector(3, 5, 2, seed=33, complex_=True)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=6, seed=11, **kw)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12


def test_rank_one_input():
    A = rand_matrix(4, 4, 5, 3, seed=7)
    x = rand_vector(4, 5, 1, seed=8)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-12
    # A rank-1 x makes A x of rank at most r(A) = 3
    assert max(y.r) <= 3


def test_singular_diagonal_matrix():
    """A that annihilates part of the space, with a 1e-12 direction in it."""
    d = 5
    cores = []
    for _ in range(d):
        c = np.zeros((1, 4, 4, 1))
        c[0, :, :, 0] = np.diag([1.0, 1e-12, 0.0, 3.0])
        cores.append(c)
    A = tt.matrix.from_list(cores)
    x = rand_vector(4, d, 3, seed=40)
    y, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30)
    assert rel(y.full(asvector=True), A.full() @ x.full(asvector=True)) < 1e-9


# --- invariants ---------------------------------------------------------------

@pytest.mark.parametrize("c", [1e-150, 1e150])
def test_scaling_equivariance_over_300_orders(c):
    """``amen_mv(A, c x) == c amen_mv(A, x)``: the ``nrms`` bookkeeping.

    This is what the "scaling invariant" section of the module docstring
    promises; without it the interfaces overflow long before ``d`` is large.
    """
    A = rand_matrix(4, 4, 6, 3, seed=20)
    x = rand_vector(4, 6, 3, seed=21)
    y0, _ = amen_mv(A, x, 1e-10, verb=0, nswp=30, seed=1)
    cores = list(tt.vector.to_list(x))
    cores[0] = c * cores[0]
    ys, _ = amen_mv(A, tt.vector.from_list(cores), 1e-10, verb=0, nswp=30,
                    seed=1)
    f = np.asarray(ys.full())
    assert np.all(np.isfinite(f))
    assert rel(f / c, y0.full()) < 1e-12


def test_large_d_qtt_laplacian():
    """d=60 QTT: dense truth is impossible, so the oracle is TT arithmetic.

    ``tt.matvec`` builds the exact product (a different algorithm), and the
    error is measured with ``(y - exact).norm()``, which never forms anything
    dense.  Also the regime where an un-normalized implementation overflows.
    """
    d = 60
    A = tt.qlaplace_dd([d])
    x = tt.rand([2] * d, d, r=3)
    exact = tt.matvec(A, x)
    y, _, h = amen_mv(A, x, 1e-8, verb=0, nswp=30, return_history=True)
    assert h.converged is True
    assert (y - exact).norm() / exact.norm() < 1e-7
    assert max(y.r) <= max(exact.round(1e-8).r)


def test_residual_estimate_is_a_lower_bound_on_the_true_residual():
    """``res_est`` is advertised as a lower estimate; check it really is one.

    Regime float64, ``d=8``, ``n=m=4``, ``r_A=r_x=6``, ``tol=1e-8``,
    ``kickrank=2``, sweeps starved to 1..4 so the residual is O(1).  Measured
    ratios ``res_est / true`` are 0.06..0.09 -- an order of magnitude *under*
    the truth, which is why it must never be sold as a stopping certificate.
    """
    A = rand_matrix(4, 4, 8, 6, seed=7)
    x = rand_vector(4, 8, 6, seed=8)
    exact = tt.matvec(A, x)          # TT arithmetic: A.full() here is 34 GB
    ratios = []
    for nswp in (1, 2, 3, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y, z, h = amen_mv(A, x, 1e-8, verb=0, nswp=nswp, kickrank=2,
                              seed=nswp, return_history=True)
        true = (y - exact).norm() / y.norm()
        assert h.converged is False
        assert h.res_est <= true, (nswp, h.res_est, true)
        ratios.append(h.res_est / true)
    assert max(ratios) < 0.5, ratios       # it underestimates by ~10x


def test_ranks_never_exceed_the_optimal_truncation_over_many_seeds():
    """The ``y.r <= exact.r`` claim, over 12 independent problems, not one."""
    for s in range(12):
        A = rand_matrix(4, 4, 6, 3, seed=1000 + s)
        x = rand_vector(4, 6, 4, seed=2000 + s)
        exact = tt.matvec(A, x).round(1e-6)
        y, _ = amen_mv(A, x, 1e-6, verb=0)
        assert np.all(np.asarray(y.r) <= np.asarray(exact.r)), \
            (s, list(y.r), list(exact.r))
        assert rel(y.full(), exact.full()) < 1e-5


# --- dtype and backend --------------------------------------------------------

def test_float32_input_gives_float32_output():
    """A np.float64 scale factor used to promote the whole answer to float64.

    NEP 50: ``float32_array * np.float64(scalar) -> float64``.  The numpy
    backend therefore returned float64 for a float32 problem while the torch
    backend returned float32 -- the same code, two answers.
    """
    A = tt.matrix.from_list([c.astype(np.float32)
                             for c in tt.matrix.to_list(
                                 rand_matrix(4, 4, 5, 2, seed=3))])
    x = tt.vector.from_list([c.astype(np.float32)
                             for c in tt.vector.to_list(
                                 rand_vector(4, 5, 2, seed=4))])
    y, z = amen_mv(A, x, 1e-5, verb=0, nswp=20)
    assert bk.dtype_of(y.cores[0]) == "float32"
    assert bk.dtype_of(z.cores[0]) == "float32"
    exact = tt.matvec(A, x)
    assert rel(y.full(), exact.full()) < 1e-5


def test_float32_orthogonal_guess_is_accepted_with_init_qr_false():
    """The check used a fixed 1e-8; float32 QR lands at 5e-8 and was rejected."""
    A = tt.matrix.from_list([c.astype(np.float32)
                             for c in tt.matrix.to_list(
                                 rand_matrix(4, 4, 5, 2, seed=3))])
    x = tt.vector.from_list([c.astype(np.float32)
                             for c in tt.vector.to_list(
                                 rand_vector(4, 5, 2, seed=4))])
    exact = tt.matvec(A, x).round(1e-6)
    y0 = tt.rand(list(exact.n), exact.d, r=list(exact.r))
    y0 = tt.vector.from_list([c.astype(np.float32) for c in
                              left_orthogonalize(tt.vector.to_list(y0))])
    y, _ = amen_mv(A, x, 1e-5, y=y0, init_qr=False, verb=0, nswp=20)
    assert rel(y.full(), exact.full()) < 1e-4


def test_float32_non_orthogonal_guess_is_still_rejected():
    """Loosening the threshold must not turn the check into a rubber stamp."""
    A = tt.matrix.from_list([c.astype(np.float32)
                             for c in tt.matrix.to_list(
                                 rand_matrix(4, 4, 5, 2, seed=3))])
    x = tt.vector.from_list([c.astype(np.float32)
                             for c in tt.vector.to_list(
                                 rand_vector(4, 5, 2, seed=4))])
    y0 = tt.vector.from_list([c.astype(np.float32) for c in
                              tt.vector.to_list(rand_vector(4, 5, 3, seed=74))])
    with pytest.raises(ValueError, match="left-orthogonal"):
        amen_mv(A, x, 1e-5, y=y0, init_qr=False, verb=0)


# --- loud failure -------------------------------------------------------------

def test_nswp_below_one_raises_instead_of_looping_forever():
    """``swp == nswp`` never fired for ``nswp <= 0``: with ``tol=0`` (which can
    never satisfy the stopping test either) the call never returned."""
    A = rand_matrix(3, 3, 4, 2, seed=40)
    x = rand_vector(3, 4, 2, seed=41)
    for bad in (0, -5):
        with pytest.raises(ValueError, match="nswp"):
            amen_mv(A, x, 1e-8, verb=0, nswp=bad)


def test_negative_tol_raises():
    """It used to run and then warn ``max_dx=3.4e-16 > tol=-1.0e-08``."""
    A = rand_matrix(3, 3, 4, 2, seed=42)
    x = rand_vector(3, 4, 2, seed=43)
    with pytest.raises(ValueError, match="tol"):
        amen_mv(A, x, -1e-8, verb=0)


def test_negative_kickrank_raises():
    A = rand_matrix(3, 3, 4, 2, seed=44)
    x = rand_vector(3, 4, 2, seed=45)
    with pytest.raises(ValueError, match="kickrank"):
        amen_mv(A, x, 1e-8, verb=0, kickrank=-2)


def test_bad_z0_modes_raise():
    """z0 was taken on trust: a wrong mode reached einops, a wrong length
    raised IndexError from inside the sweep."""
    A = rand_matrix(3, 3, 4, 2, seed=16)
    x = rand_vector(3, 4, 2, seed=17)
    with pytest.raises(ValueError, match="z has modes"):
        amen_mv(A, x, 1e-8, z=rand_vector(4, 4, 3, seed=18), verb=0)
    with pytest.raises(ValueError, match="z has 3 cores"):
        amen_mv(A, x, 1e-8, z=rand_vector(3, 3, 3, seed=19), verb=0)


def test_starved_kickrank_warns_and_the_history_agrees_with_the_answer():
    """Non-convergence must be reported *and* consistent with what came back."""
    A = rand_matrix(4, 4, 8, 6, seed=122)
    x = rand_vector(4, 8, 6, seed=123)
    exact = tt.matvec(A, x).round(1e-8)
    with pytest.warns(UserWarning, match="did not converge"):
        y, z, h = amen_mv(A, x, 1e-8, verb=0, nswp=3, kickrank=2,
                          return_history=True)
    assert h.converged is False
    assert h.nswp_done == 3 and len(h.sweeps) >= 3
    assert list(h.ranks) == list(y.r)
    assert max(y.r) < max(exact.r)
    # and the answer really is as bad as the warning implies
    assert rel(y.full(), exact.full()) > 1e-3


def test_converged_run_does_not_warn():
    A = rand_matrix(3, 3, 5, 2, seed=46)
    x = rand_vector(3, 5, 2, seed=47)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, z, h = amen_mv(A, x, 1e-8, verb=0, nswp=30, return_history=True)
    assert h.converged is True


def test_verb_one_prints_and_verb_zero_does_not(capsys):
    A = rand_matrix(3, 3, 4, 2, seed=48)
    x = rand_vector(3, 4, 2, seed=49)
    amen_mv(A, x, 1e-8, verb=1)
    out1 = capsys.readouterr().out
    assert "amen_mv" in out1 and "max_dx" in out1
    _, _, h = amen_mv(A, x, 1e-8, verb=0, return_history=True)
    assert capsys.readouterr().out == ""
    assert len(h.sweeps) >= 2 and h.time > 0


def test_kickrank_zero_reports_the_unknown_residual_as_nan():
    """Unknown is allowed; a plausible fake number would not be."""
    A = rand_matrix(4, 4, 5, 2, seed=80)
    x = rand_vector(4, 5, 2, seed=81)
    exact = tt.matvec(A, x).round(1e-12)
    y, z, h = amen_mv(A, x, 1e-10, y=tt.rand(list(exact.n), exact.d,
                                             r=list(exact.r)),
                      kickrank=0, verb=0, nswp=30, return_history=True)
    assert z is None
    assert np.isnan(h.res_est)
    assert rel(y.full(), exact.full()) < 1e-8


def test_matrix_sum_with_mismatched_modes_raises():
    A1 = rand_matrix(3, 3, 4, 2, seed=50)
    A2 = rand_matrix(4, 4, 4, 2, seed=51)
    x = rand_vector(3, 4, 2, seed=52)
    with pytest.raises(ValueError):
        amen_mv([A1, A2], x, 1e-8, verb=0)


# --- torch backend ------------------------------------------------------------

def test_zero_input_on_torch_is_zero_not_nan():
    """On torch the NaN had no RuntimeWarning at all to give it away."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    gpu = bk.TorchBackend("cuda", "float64")
    A = rand_matrix(3, 3, 4, 2, seed=1)
    At = tt.matrix.from_list([bk.asarray(c, backend=gpu)
                              for c in tt.matrix.to_list(A)])
    xt = tt.vector.from_list([bk.asarray(c, backend=gpu) for c in
                              tt.vector.to_list(tt.zeros(3, 4))])
    for renorm in ("direct", "gram"):
        y, _ = amen_mv(At, xt, 1e-10, verb=0, renorm=renorm)
        f = np.asarray(bk.to_numpy(y.full(asvector=True)))
        assert np.all(np.isfinite(f)), renorm
        assert np.linalg.norm(f) == 0.0


def test_torch_float32_keeps_float32():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    gpu = bk.TorchBackend("cuda", "float32")
    A = rand_matrix(4, 4, 5, 2, seed=3)
    x = rand_vector(4, 5, 2, seed=4)
    At = tt.matrix.from_list([bk.asarray(c, "float32", backend=gpu)
                              for c in tt.matrix.to_list(A)])
    xt = tt.vector.from_list([bk.asarray(c, "float32", backend=gpu)
                              for c in tt.vector.to_list(x)])
    y, _ = amen_mv(At, xt, 1e-5, verb=0, nswp=20)
    assert bk.dtype_of(y.cores[0]) == "float32"
    ref = tt.matvec(A, x).full()
    assert rel(np.asarray(bk.to_numpy(y.full())), ref) < 1e-4


# --- what z actually is -------------------------------------------------------

@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("d,nswp", [(7, 4), (8, 5)])
def test_z_is_the_projection_of_the_residual(complex_, d, nswp):
    """``z`` must be the *residual*, not merely some enrichment subspace.

    AMEn only ever uses the span of ``z``, so a ``z`` computed from a corrupted
    projection still grows the ranks and still converges -- the delivered
    accuracy is blind to it.  (Measured: dropping the conjugation in the
    ``Z^H Y`` interface changes the final error of a d=7 complex problem by
    nothing at all, 4.4e-1 vs 4.5e-1 at 3 sweeps and 2.3e-15 vs 2.4e-15 at 8.)
    What does pin it is that ``z`` is an orthogonal projection of the residual
    ``r = (A x - y) / ||y||``: for any orthogonal projector ``P``,
    ``<P r, r> = ||P r||^2``.  Measured on the clean code this holds to four
    digits (1.0000) in every case below; with the conjugation dropped it reads
    0.026.

    Regime: float64/complex128, ``n=m=4``, ``r_A=r_x=4``, ``tol=1e-10``,
    ``kickrank=2``, sweeps starved so that ``||z||/||r||`` is 0.06 to 0.15 --
    a converged run has ``||z|| -> 0`` and the ratio becomes noise.
    """
    A = rand_matrix(4, 4, d, 4, seed=32, complex_=complex_)
    x = rand_vector(4, d, 4, seed=33, complex_=complex_)
    for seed in (1, 2, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y, z = amen_mv(A, x, 1e-10, verb=0, nswp=nswp, kickrank=2,
                           seed=seed)
        r = (tt.matvec(A, x) - y) * (1.0 / y.norm())
        ratio = abs(tt.dot(z, r)) / z.norm() ** 2
        assert abs(ratio - 1.0) < 1e-3, (complex_, d, seed, ratio)
        assert 0.01 < z.norm() / r.norm() < 0.5
