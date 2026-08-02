"""The compiled kernels must be the numpy path, only faster.

They are a second implementation of the same mathematics, which is exactly the
situation that rots: these tests are what keeps the two honest. Skipped when
numba is absent -- the package is fully functional without it.
"""

import numpy as np
import pytest

import tt
from tt.algs import _fast
from tt.algs.amen import (_blas_layout, _gmres, _jacobi, _local_matvec,
                          _local_operator, amen_solve)

pytestmark = pytest.mark.skipif(not _fast.HAVE_NUMBA, reason="numba not installed")


def problem(r1=12, r2=10, n=2, R=3, seed=0, shift=3.0):
    rng = np.random.default_rng(seed)
    phiL = rng.standard_normal((r1, r1, R))
    phiR = rng.standard_normal((r2, r2, R))
    A = rng.standard_normal((R, n, n, R)) * 0.1
    for k in range(R):
        A[k, :, :, k] += np.eye(n) * shift
    rhs = rng.standard_normal((r1, n, r2))
    return phiL, A, phiR, rhs


def call_kernel(phiL, A, phiR, rhs, tol, restart, iters, prec=False):
    a, i, p = phiL.shape
    b, j, c = phiR.shape
    _, n, m, _ = A.shape
    phi1, amat, phi2 = _blas_layout(phiL, A, phiR)
    if prec:
        invT = _jacobi("c", phiL, A, phiR).invT
    else:
        invT = np.zeros((n, n, 1, 1))
    return _fast.gmres_local(phi1, amat, phi2, invT, prec,
                             np.ascontiguousarray(rhs.reshape(-1)), tol,
                             restart, iters, i, m, j, p, n, c, b, a)


def test_kernel_matvec_matches_the_numpy_operator():
    phiL, A, phiR, rhs = problem()
    op, _ = _local_operator(phiL, A, phiR)
    ref = np.asarray(_local_matvec(phiL, A, phiR, rhs))
    assert np.linalg.norm(np.asarray(op(rhs)) - ref) < 1e-13 * np.linalg.norm(ref)


def test_jacobi_kernel_matches_the_numpy_apply():
    phiL, A, phiR, rhs = problem()
    prec = _jacobi("c", phiL, A, phiR)
    out = np.zeros_like(rhs)
    _fast.jacobi_c_apply(prec.invT, rhs, out)
    assert np.linalg.norm(out - prec(rhs)) < 1e-13 * np.linalg.norm(out)


@pytest.mark.parametrize("prec", [False, True])
@pytest.mark.parametrize("tol", [1e-4, 1e-10])
def test_compiled_gmres_matches_the_numpy_gmres(prec, tol):
    """Same iterate, same iteration count, same verdict."""
    phiL, A, phiR, rhs = problem(seed=3)
    op, _ = _local_operator(phiL, A, phiR)
    p_np = _jacobi("c", phiL, A, phiR) if prec else None
    sol_np, res_np, nmv_np, ok_np = _gmres(op, rhs, tol, 20, 2, p_np)
    sol_c, res_c, nmv_c, ok_c = call_kernel(phiL, A, phiR, rhs, tol, 20, 2, prec)

    assert nmv_c == nmv_np
    assert ok_c == ok_np
    assert abs(res_c - res_np) <= 1e-8 * max(res_np, 1e-30)
    scale = np.linalg.norm(sol_np)
    assert np.linalg.norm(sol_c.reshape(rhs.shape) - sol_np) <= 1e-9 * scale
    # and it really solved the system
    resid = np.asarray(op(sol_c.reshape(rhs.shape))) - rhs
    assert np.linalg.norm(resid) / np.linalg.norm(rhs) == pytest.approx(res_c, rel=1e-6)


def test_solver_agrees_with_and_without_the_compiled_path(monkeypatch):
    """Turning numba off must change the timing, not the answer."""
    A = tt.qlaplace_dd([6, 6])
    f = tt.ones(2, 12)
    fast_x = amen_solve(A, f, f, 1e-8, verb=0, seed=0)
    monkeypatch.setattr(_fast, "HAVE_NUMBA", False)
    slow_x = amen_solve(A, f, f, 1e-8, verb=0, seed=0)
    err = (fast_x - slow_x).norm() / slow_x.norm()
    assert err < 1e-6, f"compiled and numpy paths disagree by {err:.2e}"
    for x in (fast_x, slow_x):
        assert (tt.matvec(A, x) - f).norm() / f.norm() < 1e-8
