"""Tests for the AMEn linear solver (``tt.algs.amen``).

Every accuracy claim here is checked against dense truth (a full ``numpy``
solve / a full matrix-vector product) or against a mathematical invariant
(residual bound, error bound through the condition number, block-diagonal
identity of the preconditioner).  Nothing is compared to the legacy output.
"""

import numpy as np
import pytest

import tt
from tt.algs.amen import (_gmres, _jacobi, _local_matrix, _local_matvec,
                          _local_rhs, _phi_next, _phi_yy_next, _project,
                          amen_solve)
from tt.core import _ops

rel = lambda a, b: np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b))


# --- helpers -----------------------------------------------------------------

def dense_matvec(A, x):
    """``A x`` computed densely, in the F-order flattening of the package."""
    return np.asarray(A.full()) @ np.asarray(x.full(asvector=True))


def residual(A, x, f):
    """``||A x - f|| / ||f||`` computed densely (the oracle)."""
    fv = np.asarray(f.full(asvector=True))
    return np.linalg.norm(dense_matvec(A, x) - fv) / np.linalg.norm(fv)


def tt_residual(A, x, f):
    """``||A x - f|| / ||f||`` computed in the TT format (for large d)."""
    r = _ops.sub(_ops.matvec_cores(tt.matrix.to_list(A), x.cores), f.cores)
    return float(_ops.norm(r) / _ops.norm(f.cores))


def random_matrix(d, n, ra, rng, dtype=np.float64, diag_shift=0.0):
    """A random TT-matrix; with ``diag_shift`` normalized and shifted.

    The random part is scaled to unit spectral norm *before* the shift is
    added, so ``diag_shift`` is a real distance of the spectrum from the
    origin.  Without that, a "diagonally dominant" random TT-matrix is
    nothing of the kind: its own norm grows with ``d`` and ``ra``, the
    spectrum ends up surrounding the origin, and restarted GMRES stagnates on
    the local systems -- a property of the test problem, not of the solver.
    """
    cores = []
    for k in range(d):
        left = 1 if k == 0 else ra
        right = 1 if k == d - 1 else ra
        c = rng.standard_normal((left, n, n, right))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal((left, n, n, right))
        cores.append(c.astype(dtype))
    A = tt.matrix.from_list(cores)
    if diag_shift:
        A = (1.0 / np.linalg.norm(np.asarray(A.full()), 2)) * A
        A = A + diag_shift * tt.eye([n] * d)
    return A


def random_vector(d, n, r, rng, dtype=np.float64):
    cores = []
    for k in range(d):
        left = 1 if k == 0 else r
        right = 1 if k == d - 1 else r
        c = rng.standard_normal((left, n, right))
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal((left, n, right))
        cores.append(c.astype(dtype))
    return tt.vector.from_list(cores)


def build_interfaces(acores, xcores, fcores, k):
    """The four interfaces around block ``k``, built the way the solver does."""
    one = np.ones((1, 1, 1))
    one2 = np.ones((1, 1))
    phiL, phifL = one, one2
    for j in range(k):
        phiL = _phi_next(_project(phiL, acores[j], xcores[j], "lr"),
                         xcores[j], "lr")
        phifL = _phi_yy_next(phifL, xcores[j], fcores[j], "lr")
    phiR, phifR = one, one2
    for j in range(len(acores) - 1, k, -1):
        phiR = _phi_next(_project(phiR, acores[j], xcores[j], "rl"),
                         xcores[j], "rl")
        phifR = _phi_yy_next(phifR, xcores[j], fcores[j], "rl")
    return phiL, phiR, phifL, phifR


# --- the local system is the exact projection of the global one --------------

@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_local_system_is_the_galerkin_projection(dtype):
    """``<u, B_k v> == <X_u, A X_v>`` and ``<u, rhs_k> == <X_u, f>``.

    This is the definition of the local system.  Checking it against the dense
    contraction pins down every index order and every conjugation in the
    interfaces at once -- with a complex, non-symmetric ``A`` so that no
    accidental symmetry can hide a transposition.
    """
    rng = np.random.default_rng(0)
    d, n = 4, 3
    A = random_matrix(d, n, 2, rng, dtype)
    x = random_vector(d, n, 3, rng, dtype)
    f = random_vector(d, n, 2, rng, dtype)
    acores, xcores, fcores = tt.matrix.to_list(A), x.cores, f.cores
    Afull = np.asarray(A.full())
    ffull = np.asarray(f.full(asvector=True))

    for k in range(d):
        phiL, phiR, phifL, phifR = build_interfaces(acores, xcores, fcores, k)
        shape = xcores[k].shape
        u = rng.standard_normal(shape) + (1j * rng.standard_normal(shape)
                                          if np.dtype(dtype).kind == "c" else 0)
        v = rng.standard_normal(shape) + (1j * rng.standard_normal(shape)
                                          if np.dtype(dtype).kind == "c" else 0)
        u, v = u.astype(dtype), v.astype(dtype)

        def substituted(block):
            cores = list(xcores)
            cores[k] = block
            return np.asarray(tt.vector.from_list(cores).full(asvector=True))

        lhs = np.vdot(u, _local_matvec(phiL, acores[k], phiR, v))
        ref = np.vdot(substituted(u), Afull @ substituted(v))
        assert abs(lhs - ref) <= 1e-10 * abs(ref)

        lhs = np.vdot(u, _local_rhs(phifL, fcores[k], phifR))
        ref = np.vdot(substituted(u), ffull)
        assert abs(lhs - ref) <= 1e-10 * abs(ref)

        # the dense local matrix must agree with the matrix-free application
        mat = _local_matrix(phiL, acores[k], phiR)
        assert rel(mat @ v.reshape(-1),
                   _local_matvec(phiL, acores[k], phiR, v).reshape(-1)) < 1e-12


@pytest.mark.parametrize("kind", ["c", "l", "r"])
def test_jacobi_inverts_the_block_diagonal(kind):
    """Each Jacobi variant is the inverse of a block-diagonal part of ``B_k``.

    Truth: the dense local matrix, sliced by hand with the grouping the
    variant claims to use.
    """
    rng = np.random.default_rng(1)
    d, n, k = 4, 3, 2
    A = random_matrix(d, n, 2, rng, diag_shift=6.0)
    x = random_vector(d, n, 3, rng)
    f = random_vector(d, n, 2, rng)
    acores = tt.matrix.to_list(A)
    phiL, phiR, _, _ = build_interfaces(acores, x.cores, f.cores, k)
    r1, r2 = x.cores[k].shape[0], x.cores[k].shape[2]
    mat = np.asarray(_local_matrix(phiL, acores[k], phiR)).reshape(
        (r1, n, r2, r1, n, r2))

    block_diag = np.zeros_like(mat)
    for b1 in range(r1):
        for b2 in range(r2):
            if kind == "c":
                block_diag[b1, :, b2, b1, :, b2] = mat[b1, :, b2, b1, :, b2]
            elif kind == "l":
                block_diag[:, :, b2, :, :, b2] = mat[:, :, b2, :, :, b2]
            else:
                block_diag[b1, :, :, b1, :, :] = mat[b1, :, :, b1, :, :]
    bd = block_diag.reshape((r1 * n * r2, r1 * n * r2))

    apply_prec = _jacobi(kind, phiL, acores[k], phiR)
    w = rng.standard_normal((r1, n, r2))
    got = np.asarray(apply_prec(w)).reshape(-1)
    want = np.linalg.solve(bd, w.reshape(-1))
    assert rel(got, want) < 1e-10


def test_gmres_matches_the_dense_solve():
    """The local GMRES on a small system must reproduce ``numpy.linalg.solve``."""
    rng = np.random.default_rng(2)
    d, n, k = 4, 3, 2
    A = random_matrix(d, n, 2, rng, diag_shift=8.0)
    x = random_vector(d, n, 3, rng)
    f = random_vector(d, n, 2, rng)
    acores = tt.matrix.to_list(A)
    phiL, phiR, phifL, phifR = build_interfaces(acores, x.cores, f.cores, k)
    rhs = _local_rhs(phifL, f.cores[k], phifR)
    mat = np.asarray(_local_matrix(phiL, acores[k], phiR))
    want = np.linalg.solve(mat, np.asarray(rhs).reshape(-1))

    sol, relres, nmv, ok = _gmres(
        lambda w: _local_matvec(phiL, acores[k], phiR, w), rhs, 1e-12, 60, 4)
    assert ok and relres <= 1e-12
    assert rel(np.asarray(sol).reshape(-1), want) < 1e-9
    assert nmv > 0

    # ... and with each preconditioner, which must not change the answer
    for kind in ("c", "l", "r"):
        prec = _jacobi(kind, phiL, acores[k], phiR)
        sol, relres, _, ok = _gmres(
            lambda w: _local_matvec(phiL, acores[k], phiR, w), rhs, 1e-12,
            60, 4, prec=prec)
        assert ok, f"preconditioned GMRES ({kind}) stopped at {relres:.2e}"
        assert rel(np.asarray(sol).reshape(-1), want) < 1e-9


# --- end to end: the QTT Laplacian -------------------------------------------

@pytest.mark.parametrize("d, eps", [(6, 1e-6), (12, 1e-6),
                                    (6, 1e-10), (10, 1e-10)])
def test_qlaplace_residual(d, eps):
    """``||A x - f|| / ||f|| <= eps`` for the QTT Laplacian, rhs of ones.

    ``d = 12`` is 4096 unknowns; the residual is measured independently of the
    solver (TT arithmetic on ``A x - f``), and the residual the solver
    *reports* must agree with it.

    ``(d, eps) = (12, 1e-10)`` is deliberately absent: for that system a
    relative residual of 1e-10 is *below the float64 floor* and no algorithm
    can reach it -- see
    :func:`test_illconditioned_qlaplace_reports_failure`, which measures the
    floor with LAPACK and checks that the solver reports the failure.
    """
    A = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d)
    x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0, return_info=True)
    measured = tt_residual(A, x, rhs)
    assert measured <= eps, f"residual {measured:.3E} > eps {eps:.3E}"
    assert info.converged
    assert abs(info.true_res - measured) <= 1e-10 * max(measured, 1e-30)
    assert info.nswp_done == len(info.sweeps)
    for entry in info.sweeps:
        assert set(entry) >= {"sweep", "max_dx", "max_res", "max_rank",
                              "true_res", "time"}


def test_qlaplace_matches_dense_solve():
    """Against ``numpy.linalg.solve`` on the full 256x256 system.

    The residual bound only implies the error bound through the condition
    number, so that is what is asserted; the raw error is asserted as well, at
    a level measured to hold with room to spare.
    """
    d, eps = 8, 1e-10
    A = tt.qlaplace_dd([d])
    rhs = tt.ones(2, d)
    x = amen_solve(A, rhs, None, eps, verb=0, seed=0)

    Afull = np.asarray(A.full())
    fv = np.asarray(rhs.full(asvector=True))
    xdense = np.linalg.solve(Afull, fv)
    err = rel(np.asarray(x.full(asvector=True)), xdense)
    cond = np.linalg.cond(Afull)
    assert err <= 1.01 * cond * eps          # the mathematical bound
    assert err < 1e-7                        # measured: ~1e-11
    assert residual(A, x, rhs) <= eps


def test_direct_and_gmres_local_solvers_agree():
    """The two local solvers are two ways to solve the same local system."""
    d, eps = 8, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x_direct = amen_solve(A, rhs, None, eps, verb=0, seed=0,
                          max_full_size=10 ** 6)
    x_gmres = amen_solve(A, rhs, None, eps, verb=0, seed=0, max_full_size=0,
                         local_iters=6, local_restart=60)
    assert residual(A, x_direct, rhs) <= eps
    assert residual(A, x_gmres, rhs) <= eps
    assert rel(np.asarray(x_gmres.full(asvector=True)),
               np.asarray(x_direct.full(asvector=True))) < 1e-6


@pytest.mark.parametrize("prec", ["c", "l", "r"])
def test_local_preconditioners_end_to_end(prec):
    """Every Jacobi variant must still deliver the requested accuracy."""
    d, eps = 8, 1e-8
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x = amen_solve(A, rhs, None, eps, verb=0, seed=0, max_full_size=0,
                   local_prec=prec, local_iters=4, local_restart=40)
    assert residual(A, x, rhs) <= eps


def test_trunc_norm_frobenius():
    """``trunc_norm='fro'`` is a different truncation, not a different answer."""
    d, eps = 8, 1e-9
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x = amen_solve(A, rhs, None, eps, verb=0, seed=0, trunc_norm='fro')
    assert residual(A, x, rhs) <= eps


# --- non-symmetric problems ---------------------------------------------------

@pytest.mark.parametrize("a", [-0.5, -0.9])
def test_nonsymmetric_ipas(a):
    """``I + a S_{-1}``: bidiagonal, non-symmetric, exact dense comparison."""
    d, eps = 8, 1e-10
    A = tt.IpaS(d, a)
    rhs = tt.ones(2, d)
    x = amen_solve(A, rhs, None, eps, verb=0, seed=0)
    Afull = np.asarray(A.full())
    xdense = np.linalg.solve(Afull, np.asarray(rhs.full(asvector=True)))
    assert residual(A, x, rhs) <= eps
    assert rel(np.asarray(x.full(asvector=True)), xdense) \
        <= 1.01 * np.linalg.cond(Afull) * eps


def test_nonsymmetric_random_matrix():
    """A random (non-symmetric, non-normal) TT-matrix against the dense solve."""
    rng = np.random.default_rng(3)
    d, n, eps = 4, 3, 1e-10
    A = random_matrix(d, n, 2, rng, diag_shift=8.0)
    f = random_vector(d, n, 2, rng)
    x = amen_solve(A, f, None, eps, verb=0, seed=0)
    Afull = np.asarray(A.full())
    xdense = np.linalg.solve(Afull, np.asarray(f.full(asvector=True)))
    assert residual(A, x, f) <= eps
    assert rel(np.asarray(x.full(asvector=True)), xdense) \
        <= 1.01 * np.linalg.cond(Afull) * eps


@pytest.mark.parametrize("max_full_size", [50, 10 ** 6])
def test_complex_nonhermitian(max_full_size):
    """A complex non-Hermitian system -- the conjugation convention, end to end.

    Run twice, with the local systems solved by GMRES (the default 50) and
    densely, because the two paths share nothing but the interfaces.
    """
    rng = np.random.default_rng(4)
    d, n, eps = 4, 3, 1e-10
    A = random_matrix(d, n, 2, rng, dtype=np.complex128, diag_shift=2.0)
    f = random_vector(d, n, 2, rng, dtype=np.complex128)
    x = amen_solve(A, f, None, eps, verb=0, seed=0, max_full_size=max_full_size)
    assert x.is_complex
    Afull = np.asarray(A.full())
    xdense = np.linalg.solve(Afull, np.asarray(f.full(asvector=True)))
    assert residual(A, x, f) <= eps
    assert rel(np.asarray(x.full(asvector=True)), xdense) \
        <= 1.01 * np.linalg.cond(Afull) * eps


# --- honesty about failure ----------------------------------------------------

def test_als_without_enrichment_reports_its_failure():
    """``kickrank=0`` is plain ALS: it cannot grow the rank-1 initial guess.

    The solution of the QTT Laplacian with a constant rhs has TT rank > 1, so
    the requested accuracy is unreachable *by construction*.  The solver must
    say so instead of returning a plausible-looking vector.
    """
    d, eps = 10, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x0 = tt.ones(2, d)                                  # rank 1
    with pytest.warns(UserWarning, match="did NOT reach"):
        x, info = amen_solve(A, rhs, x0, eps, kickrank=0, nswp=8, verb=0,
                             return_info=True)
    assert not info.converged
    assert max(info.ranks) == 1
    measured = tt_residual(A, x, rhs)
    assert measured > eps                                # it really did fail
    assert abs(info.true_res - measured) <= 1e-10 * measured
    assert "did NOT reach" in info.message


def test_illconditioned_qlaplace_reports_failure():
    """``d = 12``, ``eps = 1e-10``: unreachable in float64, must be reported.

    The 4096x4096 QTT Laplacian with a constant right-hand side has
    ``||A|| ||x|| / ||f|| ~ 2e6``, so *any* backward-stable solver leaves a
    relative residual of order ``eps_machine * 2e6 ~ 1e-9``.  LAPACK's own
    dense solve is used here as the oracle for that floor.  The solver must
    (a) get within a small factor of the floor, and (b) say that it did not
    reach the requested accuracy instead of returning a plausible answer.
    """
    d, eps = 12, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)

    Afull = np.asarray(A.full())
    fv = np.asarray(rhs.full(asvector=True))
    floor = np.linalg.norm(Afull @ np.linalg.solve(Afull, fv) - fv) \
        / np.linalg.norm(fv)
    assert floor > eps, "the premise of this test (eps below the floor) broke"

    with pytest.warns(UserWarning, match="did NOT reach"):
        x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0,
                             return_info=True)
    assert not info.converged
    measured = tt_residual(A, x, rhs)
    assert measured > eps                                # it really did fail
    assert measured <= 10 * floor                        # ... but only barely
    assert abs(info.true_res - measured) <= 1e-8 * measured
    assert "did NOT reach" in info.message
    assert f"{measured:.3E}" in info.message


def test_shifted_qlaplace_d12():
    """Same 4096 QTT system, shifted to be well conditioned: 1e-10 is reached.

    This is the control for the test above: nothing about ``d = 12`` stops the
    solver, only the conditioning of that particular operator does.
    """
    d, eps = 12, 1e-10
    A = (tt.qlaplace_dd([d]) + tt.eye([2] * d)).round(1e-14)
    rhs = tt.ones(2, d)
    x, info = amen_solve(A, rhs, None, eps, verb=0, seed=0, return_info=True)
    assert info.converged
    assert tt_residual(A, x, rhs) <= eps


def test_history_is_recorded_even_when_silent(capsys):
    """``verb=0`` prints nothing but the history is complete."""
    A, rhs = tt.qlaplace_dd([8]), tt.ones(2, 8)
    x, info = amen_solve(A, rhs, None, 1e-8, verb=0, seed=0, return_info=True)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert info.nswp_done >= 1
    assert all(np.isfinite([s["max_dx"], s["max_res"], s["true_res"],
                            s["time"]]).all() for s in info.sweeps)
    assert x.amen_info is info
    assert "converged" in repr(info)


def test_verbose_prints(capsys):
    A, rhs = tt.qlaplace_dd([8]), tt.ones(2, 8)
    amen_solve(A, rhs, None, 1e-8, verb=1, seed=0)
    out = capsys.readouterr().out
    assert "amen_solve: swp=1" in out and "max_rank" in out
    amen_solve(A, rhs, None, 1e-8, verb=2, seed=0)
    out = capsys.readouterr().out
    assert "block=" in out and ("direct" in out or "gmres" in out)


def test_check_true_res_false_leaves_true_res_unknown():
    """Without the residual check the solver must not invent a residual."""
    A, rhs = tt.qlaplace_dd([8]), tt.ones(2, 8)
    x, info = amen_solve(A, rhs, None, 1e-8, verb=0, seed=0,
                         check_true_res=False, return_info=True)
    assert np.isnan(info.true_res)
    assert all(np.isnan(s["true_res"]) for s in info.sweeps)
    # the legacy criterion is optimistic, but it must not be *wrong*
    assert residual(A, x, rhs) < 1e-6


# --- argument handling --------------------------------------------------------

def test_unknown_preconditioner_raises():
    A, rhs = tt.qlaplace_dd([4]), tt.ones(2, 4)
    with pytest.raises(NotImplementedError, match="local_prec"):
        amen_solve(A, rhs, None, 1e-6, local_prec='gauss-seidel', verb=0)
    with pytest.raises(ValueError, match="trunc_norm"):
        amen_solve(A, rhs, None, 1e-6, trunc_norm='both', verb=0)


def test_legacy_aliases_are_accepted():
    A, rhs = tt.qlaplace_dd([6]), tt.ones(2, 6)
    x = amen_solve(A, rhs, rhs, 1e-8, local_prec='cjacobi',
                   trunc_norm='residual', verb=0, max_full_size=0,
                   local_iters=4)
    assert residual(A, x, rhs) <= 1e-8


def test_rectangular_operator_is_refused():
    cores = [np.zeros((1, 3, 2, 1))]
    A = tt.matrix.from_list(cores * 1)
    f = tt.vector.from_list([np.ones((1, 3, 1))])
    with pytest.raises(ValueError, match="square"):
        amen_solve(A, f, None, 1e-6, verb=0)


def test_mode_mismatch_is_refused():
    A = tt.qlaplace_dd([4])
    with pytest.raises(ValueError, match="modes"):
        amen_solve(A, tt.ones(3, 4), None, 1e-6, verb=0)      # wrong mode size
    with pytest.raises(ValueError, match="cores"):
        amen_solve(A, tt.ones(2, 3), None, 1e-6, verb=0)      # wrong d
    with pytest.raises(ValueError, match="modes"):
        amen_solve(A, tt.ones(2, 4), tt.ones(3, 4), 1e-6, verb=0)   # x0


def test_one_dimensional_problem():
    """d = 1: no sweep structure, a single dense local solve."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((5, 5)) + 5 * np.eye(5)
    A = tt.matrix.from_list([a.reshape((1, 5, 5, 1))])
    b = rng.standard_normal(5)
    f = tt.vector.from_list([b.reshape((1, 5, 1))])
    x = amen_solve(A, f, None, 1e-12, verb=0, seed=0)
    assert rel(np.asarray(x.full(asvector=True)), np.linalg.solve(a, b)) < 1e-10


def test_rmax_caps_the_rank():
    """``rmax`` bounds the truncated rank; the enrichment may add ``kickrank``."""
    d, rmax, kick = 10, 6, 3
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    with pytest.warns(UserWarning):        # 1e-12 is not reachable at rank 6+3
        x, info = amen_solve(A, rhs, None, 1e-12, verb=0, seed=0, rmax=rmax,
                             kickrank=kick, nswp=4, return_info=True)
    assert max(info.ranks) <= rmax + kick
    assert tt_residual(A, x, rhs) < 1e-3   # it still solves it roughly


def test_plain_als_with_a_good_start():
    """``kickrank=0`` (no enrichment) converges if the frames are good enough."""
    d, eps = 8, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x0 = amen_solve(A, rhs, None, 1e-4, verb=0, seed=0)
    x, info = amen_solve(A, rhs, x0, eps, kickrank=0, verb=0, return_info=True)
    assert info.converged
    assert residual(A, x, rhs) <= eps


def test_torch_backend_gives_the_same_answer():
    """The same solve on the torch backend, checked against numpy and dense."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    d, eps = 8, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x_np = amen_solve(A, rhs, None, eps, verb=0, seed=0)
    x_gpu, info = amen_solve(A.to("torch", "cuda", "float64"),
                             rhs.to("torch", "cuda", "float64"), None, eps,
                             verb=0, seed=0, return_info=True)
    assert info.converged
    got = np.asarray(tt.core._ops.bk.to_numpy(x_gpu.full(asvector=True)))
    xdense = np.linalg.solve(np.asarray(A.full()),
                             np.asarray(rhs.full(asvector=True)))
    assert rel(got, xdense) < 1e-7
    assert rel(got, np.asarray(x_np.full(asvector=True))) < 1e-7


def test_core_copy_on_torch_backend():
    """Was a core bug (copy() called ndarray.copy() on torch tensors).

    Fixed with a bk.copy() dispatch; kept as a regression test. Note the
    to_numpy(): a CUDA tensor cannot be handed to np.asarray directly.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tt import backend as bk

    x = tt.rand([2] * 4, r=2).to("torch", "cuda", "float64")
    assert rel(bk.to_numpy(x.copy().full()), bk.to_numpy(x.full())) == 0.0


def test_legacy_namespace():
    """``from tt.amen import amen_solve`` (ttpy 1.x) and ``tt.amen_solve``."""
    from tt.amen import amen_solve as legacy
    assert legacy is amen_solve
    assert tt.amen_solve is amen_solve


def test_raw_core_lists_in_and_out():
    """Given raw core lists, the solver returns a raw core list."""
    d, eps = 6, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    cores = amen_solve(tt.matrix.to_list(A), rhs.cores, None, eps, verb=0,
                       seed=0)
    assert isinstance(cores, list)
    x = tt.vector.from_list(cores)
    assert residual(A, x, rhs) <= eps


def test_x0_is_used():
    """A good initial guess must not be thrown away."""
    d, eps = 8, 1e-10
    A, rhs = tt.qlaplace_dd([d]), tt.ones(2, d)
    x_ref = amen_solve(A, rhs, None, 1e-11, verb=0, seed=0)
    x, info = amen_solve(A, rhs, x_ref, eps, verb=0, seed=0, return_info=True)
    assert info.nswp_done == 1
    assert residual(A, x, rhs) <= eps
