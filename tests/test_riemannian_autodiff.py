"""The tangent-space SSOT and the Riemannian autodiff, against dense truth.

The contracts here are the ones ``docs/plans/riemannian-autodiff.md`` measured
with prototypes before anything was built: the delta cores rebuild ``project``
to roundoff, the gauge holds, the cheap inner product equals the TT
contraction, and the AD gradient reproduces the projected dense Euclidean
gradient -- computed on the numpy backend through ``tt_svd``, i.e. through a
route that shares no code with the thing under test.
"""

import warnings

import numpy as np
import pytest

import tt
from tt.algs import riemannian as rm
from tt.core import _ops
from tt.core.vector import vector


def rel(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def gauge_defect(deltas, fr):
    worst = 0.0
    for k in range(len(deltas) - 1):
        g = np.asarray(deltas[k]).reshape(-1, deltas[k].shape[2])
        u = np.asarray(fr.U[k]).reshape(-1, fr.U[k].shape[2])
        worst = max(worst, float(np.abs(g.conj().T @ u).max()))
    return worst


# --- the tangent representation ----------------------------------------------

@pytest.mark.parametrize("complex_", [False, True])
def test_project_delta_rebuilds_project(complex_):
    """``tangent_to_tt(X, project_delta(X, Z)) == project(X, Z)`` to roundoff."""
    rng = np.random.default_rng(0)
    n = [5, 4, 6, 5]
    X = tt.rand(n, r=3)
    Z = tt.rand(n, r=4)
    if complex_:
        X = X + 1j * tt.rand(n, r=3)
        Z = Z + 1j * tt.rand(n, r=4)
    deltas, fr = rm.project_delta(X, Z)
    asm = rm.tangent_to_tt(X, deltas, frames_=fr)
    ref = rm.project(X, Z)
    assert float((asm - ref).norm() / ref.norm()) < 1e-13
    assert gauge_defect(deltas, fr) < 1e-12 * float(Z.norm())


def test_tangent_inner_matches_the_tt_contraction():
    n = [5] * 4
    X, Z1, Z2 = tt.rand(n, r=3), tt.rand(n, r=3), tt.rand(n, r=2)
    d1, fr = rm.project_delta(X, Z1)
    d2, _ = rm.project_delta(X, Z2, frames_=fr)
    cheap = complex(rm.tangent_inner(d1, d2))
    full = complex(tt.dot(rm.project(X, Z1), rm.project(X, Z2)))
    assert abs(cheap - full) < 1e-10 * abs(full)


def test_weighted_list_is_the_weighted_sum_of_projections():
    n = [4] * 4
    X = tt.rand(n, r=2)
    Z1, Z2 = tt.rand(n, r=3), tt.rand(n, r=2)
    deltas, fr = rm.project_delta(X, [Z1, Z2], weights=[2.0, -0.5])
    asm = rm.tangent_to_tt(X, deltas, frames_=fr)
    ref = 2.0 * rm.project(X, Z1) - 0.5 * rm.project(X, Z2)
    assert float((asm - ref).norm() / ref.norm()) < 1e-12


def test_tangent_gram():
    n = [4] * 4
    X = tt.rand(n, r=2)
    lists = [rm.project_delta(X, tt.rand(n, r=2))[0] for _ in range(3)]
    g = rm.tangent_gram(lists)
    assert g.shape == (3, 3)
    assert np.allclose(g, g.T.conj())
    assert np.all(np.linalg.eigvalsh(g) > -1e-10 * np.abs(g).max())


def test_frames_mu_and_rank_refusal():
    n = [4] * 4
    X = tt.rand(n, r=2)
    fr1 = rm.frames(X, mu=1)
    frd = rm.frames(X, mu=4)
    assert fr1.S is fr1.V[0]
    assert frd.S is frd.U[-1]
    with pytest.raises(ValueError, match="mu"):
        rm.frames(X, mu=5)

    # a rank-2 representation of a rank-1 tensor has no tangent space there
    ones = tt.ones(4, 4)
    bad = ones + ones
    with pytest.raises(ValueError):
        rm.frames(bad)


# --- retraction and transport ------------------------------------------------

def test_retract_keeps_ranks_and_reports_discarded():
    n = [5] * 4
    X = tt.rand(n, r=3)
    xi = rm.project(X, tt.rand(n, r=3))
    y, disc = rm.retract(X, 0.1 * xi, return_discarded=True)
    assert [int(v) for v in y.r] == [int(v) for v in X.r]
    assert disc >= 0.0
    # a zero step retracts to X itself
    y0 = rm.retract(X, 0.0 * xi)
    assert float((y0 - X).norm() / X.norm()) < 1e-13


def test_retract_psa_agrees_at_first_order():
    n = [5] * 4
    X = tt.rand(n, r=2)
    xi = rm.project(X, tt.rand(n, r=2))
    h = 1e-6
    a = rm.retract(X, h * xi, method="svd")
    b = rm.retract(X, h * xi, method="psa")
    assert float((a - b).norm() / X.norm()) < 1e-9


def test_transport_lands_in_the_new_tangent_space():
    n = [5] * 4
    X = tt.rand(n, r=2)
    Y = tt.rand(n, r=2)
    deltas, _ = rm.project_delta(X, tt.rand(n, r=2))
    moved, fr_new = rm.transport(deltas, X, Y)
    assert gauge_defect(moved, fr_new) < 1e-11
    ref = rm.project(Y, rm.tangent_to_tt(X, deltas))
    asm = rm.tangent_to_tt(Y, moved, frames_=fr_new)
    assert float((asm - ref).norm() / ref.norm()) < 1e-12


# --- riemannian autodiff -----------------------------------------------------

def _torch_point(n, r, seed):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    Xnp = tt.rand(n, r=r)
    return torch, Xnp, Xnp.to("torch", "cpu", "float64")


def test_riemannian_grad_matches_the_projected_dense_gradient():
    """AD route vs dense-gradient route, sharing no code.

    ``f = <A x, x>`` with symmetric ``A``: the Euclidean gradient is
    ``(A + A^T) x``, computed DENSELY on numpy, compressed by ``tt_svd`` and
    pushed through ``project`` -- the plan measured 1.5e-15 for exactly this
    pairing.
    """
    from tt.algs.autodiff import riemannian_grad
    torch, Xnp, X = _torch_point([4] * 5, 3, 0)
    rng = np.random.default_rng(1)
    d, n, R = 5, 4, 2
    cores = []
    for k in range(d):
        c = rng.standard_normal((1 if k == 0 else R, n, n,
                                 1 if k == d - 1 else R))
        cores.append(c + np.transpose(c, (0, 2, 1, 3)))
    Anp = tt.matrix.from_list(cores)
    A = Anp.to("torch", "cpu", "float64")

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        return tt.dot(x, tt.matvec(A, x))

    val, deltas, fr = riemannian_grad(f, X)
    g_ad = rm.tangent_to_tt(X, deltas, frames_=fr).to("numpy", None, "float64")

    Ad = np.asarray(Anp.full())
    xd = np.asarray(Xnp.full(asvector=True))
    grad_dense = (Ad + Ad.T) @ xd
    import tt.backend as bk
    gtt = vector.from_list(_ops.tt_svd(
        bk.get_backend().asarray(grad_dense.reshape([n] * d, order="F")),
        1e-13))
    ref = rm.project(Xnp, gtt)
    assert float((g_ad - ref).norm() / ref.norm()) < 1e-12
    assert abs(val - float(xd @ Ad @ xd)) < 1e-8 * abs(val)
    # the gauge sign: with the paper's Alg-5.2 plus this is O(1), not 1e-12
    assert gauge_defect(deltas, fr) < 1e-10 * float(ref.norm())


def test_runtime_check_catches_a_representation_dependent_f():
    from tt.algs.autodiff import riemannian_grad
    torch, _, X = _torch_point([4] * 4, 2, 2)

    def f(cores_list):
        return (cores_list[0] ** 2).sum()      # depends on the gauge

    with pytest.raises(ValueError, match="not a function of the tensor"):
        riemannian_grad(f, X)


def test_numpy_backend_raises_actionably():
    from tt.algs.autodiff import riemannian_grad
    pytest.importorskip("torch")
    X = tt.rand([4] * 4, r=2)
    with pytest.raises(ValueError, match="numpy.*grad="):
        riemannian_grad(lambda cores: 0.0, X)


# --- rgd ---------------------------------------------------------------------

def test_rgd_logcosh_completion_recovers_the_target():
    """The headline: a non-quadratic loss, where ALS has no local problem.

    ``log cosh`` completion of an on-manifold rank-2 target; the minimum is
    zero at the target by construction, and the held-out error against the
    dense truth is the oracle.
    """
    from tt.algs.cross import element
    from tt.algs.autodiff import rgd
    torch, _, _ = _torch_point([2] * 2, 1, 0)
    import tt.backend as bk
    rng = np.random.default_rng(3)
    d, n, r = 4, 6, 2
    target = tt.rand([n] * d, r=r)
    dense = np.asarray(target.full())
    idx = np.stack([rng.integers(0, n, 4000) for _ in range(d)], 1)
    X0 = tt.rand([n] * d, r=r).to("torch", "cpu", "float64")
    t = bk.backend_of(X0.cores[0]).torch
    tvals = t.as_tensor(dense[tuple(idx.T)])

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        z = element(x, idx) - tvals
        return t.log(t.cosh(z)).sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, h = rgd(f, X0, maxit=300, tol=1e-7)
    got = np.asarray(bk.to_numpy(x.full()))
    assert rel(got, dense) < 1e-6
    fs = [it["f"] for it in h.iterations]
    assert all(a >= b for a, b in zip(fs, fs[1:])), "Armijo must be monotone"
    assert h.ranks == [int(v) for v in x.r]
    assert h.fun_calls > h.grad_calls >= len(h.iterations)


def test_rgd_numpy_route_with_an_explicit_gradient():
    """``grad=`` escape hatch, no torch anywhere: minimise ``|x - A|^2``.

    Fully observed on purpose: a sampled functional at a few samples per dof
    sits below the recovery threshold where *both* Riemannian GD and ALS stall
    on a plateau (measured in the plan, section 5.2) -- not test material.
    """
    from tt.algs.autodiff import rgd
    d, n, r = 4, 5, 2
    target = tt.rand([n] * d, r=r)

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        diff = x - target
        return float(tt.dot(diff, diff))

    def grad(x):
        return (2.0 * (x - target)).round(1e-13)

    x0 = tt.rand([n] * d, r=r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x, h = rgd(f, x0, grad=grad, maxit=300, tol=1e-9)
    assert h.f < 1e-10
    assert rel(np.asarray(x.full()), np.asarray(target.full())) < 1e-5


def test_rgd_reports_a_budget_stop_honestly():
    from tt.algs.autodiff import rgd
    torch, _, X0 = _torch_point([5] * 4, 2, 5)
    import tt.backend as bk
    t = bk.backend_of(X0.cores[0]).torch
    target = tt.rand([5] * 4, r=2).to("torch", "cpu", "float64")

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        d = x - target
        return tt.dot(d, d)

    with pytest.warns(RuntimeWarning, match="iterations without"):
        _, h = rgd(f, X0, maxit=2, tol=1e-14)
    assert h.stop_reason == "maxit"
    assert not h.converged
