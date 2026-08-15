"""Tests for the fixed-rank transported-direction linear solver."""

import warnings

import numpy as np
import pytest

import tt
import tt.algs.lobpcg as lobpcg_module
from tt.algs.amen_mv import _matrix_cores, _vector_cores
from tt.algs.lobpcg import (
    _generic_augmented_pcg,
    _projected_gradient_norm,
    _transport_direction,
    lobpcg_solve,
)
from tt.core import _ops


def feasible_profile(depth, rank):
    return [min(rank, 2 ** min(k, depth - k)) for k in range(depth + 1)]


def random_profile_tt(profile, seed):
    rng = np.random.default_rng(seed)
    value = tt.vector.from_list([
        rng.standard_normal((profile[k], 2, profile[k + 1]))
        for k in range(len(profile) - 1)
    ])
    return value / value.norm()


def random_complex_profile_tt(profile, seed):
    rng = np.random.default_rng(seed)
    cores = []
    for k in range(len(profile) - 1):
        shape = (profile[k], 2, profile[k + 1])
        cores.append(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
    value = tt.vector.from_list(cores)
    return value / value.norm()


def dense_horizontal_gradient(operator, rhs, value):
    """Independent dense construction of an orthonormal horizontal basis."""
    left = [np.array(core, copy=True) for core in value.cores]
    right = [np.array(core, copy=True) for core in value.cores]
    depth = len(left)
    for site in range(depth - 1):
        r0, mode, r1 = left[site].shape
        qmat, factor = np.linalg.qr(left[site].reshape(r0 * mode, r1))
        left[site] = qmat.reshape(r0, mode, r1)
        following = left[site + 1]
        left[site + 1] = (
            factor @ following.reshape(r1, -1)
        ).reshape(following.shape)
    for site in range(depth - 1, 0, -1):
        r0, mode, r1 = right[site].shape
        qmat, factor = np.linalg.qr(
            right[site].reshape(r0, mode * r1).conj().T
        )
        right[site] = qmat.conj().T.reshape(r0, mode, r1)
        previous = right[site - 1]
        right[site - 1] = (
            previous.reshape(-1, r0) @ factor.conj().T
        ).reshape(previous.shape)

    dense_residual = (
        np.asarray(operator.full()) @ np.asarray(value.full(asvector=True))
        - np.asarray(rhs.full(asvector=True))
    )
    coordinates = []
    for site, core in enumerate(left):
        r0, mode, r1 = core.shape
        if site < depth - 1:
            complete, _ = np.linalg.qr(
                core.reshape(r0 * mode, r1), mode="complete"
            )
            local = complete[:, r1:]
            variations = []
            for row in range(local.shape[1]):
                for column in range(r1):
                    delta = np.zeros((r0 * mode, r1), dtype=core.dtype)
                    delta[:, column] = local[:, row]
                    variations.append(delta.reshape(r0, mode, r1))
        else:
            variations = np.eye(r0 * mode * r1).reshape(
                r0 * mode * r1, r0, mode, r1
            )
        for delta in variations:
            tangent = tt.vector.from_list(
                left[:site] + [delta] + right[site + 1:]
            )
            coordinates.append(
                np.vdot(
                    np.asarray(tangent.full(asvector=True)), dense_residual
                )
            )
    return np.linalg.norm(coordinates) / rhs.norm()


def test_transport_is_the_factored_frame_projection():
    rng = np.random.default_rng(0)
    new_left, old_left, mode = 3, 4, 2
    old_right, new_right = 5, 2
    left = rng.standard_normal((new_left, old_left))
    right = rng.standard_normal((old_right, new_right))
    direction = rng.standard_normal((old_left, mode, old_right))

    got = _transport_direction(direction, left, right)
    expected = np.einsum(
        "Aa,aib,bB->AiB", left, direction, right, optimize=True
    )
    assert np.linalg.norm(got - expected) < 1e-13 * np.linalg.norm(expected)


def test_augmented_pcg_stays_in_the_operator_orthogonal_complement():
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((12, 12))
    matrix = matrix.T @ matrix + np.eye(12)
    diagonal = np.diag(matrix)
    rhs = rng.standard_normal(12)
    coarse = rng.standard_normal(12)
    coarse /= np.linalg.norm(coarse)

    correction, memory, info = _generic_augmented_pcg(
        lambda value: matrix @ value,
        lambda value: value / diagonal,
        rhs,
        coarse,
        1e-14,
        6,
    )
    assert info["coarse_used"] == 1
    assert info["fresh_steps"] > 0
    assert memory is not None
    defect = abs(np.vdot(coarse, matrix @ memory))
    scale = np.linalg.norm(matrix @ coarse) * np.linalg.norm(memory)
    assert defect < 1e-12 * scale
    initial_energy = 0.0
    final_energy = 0.5 * correction @ matrix @ correction - rhs @ correction
    assert final_energy < initial_energy


def test_projected_gradient_is_zero_at_a_representable_solution():
    bits, rank = 3, 4
    depth = 2 * bits
    profile = feasible_profile(depth, rank)
    exact = random_profile_tt(profile, 1)
    operator = tt.eye(2, depth) + 0.1 * tt.qlaplace_dd([bits, bits])
    rhs = tt.matvec(operator, exact)
    acores = _matrix_cores(operator, depth)
    fcores, _ = _vector_cores(rhs)
    xcores, _ = _vector_cores(exact)
    dtype = np.result_type(acores[0], fcores[0], xcores[0])
    gradient = _projected_gradient_norm(
        [np.asarray(core, dtype=dtype) for core in acores],
        [np.asarray(core, dtype=dtype) for core in fcores],
        [np.asarray(core, dtype=dtype) for core in xcores],
        float(_ops.norm(fcores)),
        np.ones((1, 1, 1), dtype=dtype),
        np.ones((1, 1), dtype=dtype),
    )
    assert gradient < 2e-12


def test_projected_gradient_matches_an_explicit_dense_tangent_basis():
    depth, rank = 5, 3
    profile = feasible_profile(depth, rank)
    value = random_profile_tt(profile, 7)
    operator = tt.eye(2, depth) + 0.2 * tt.qlaplace_dd([depth])
    rhs = random_profile_tt(feasible_profile(depth, 2), 8)
    acores = _matrix_cores(operator, depth)
    fcores, _ = _vector_cores(rhs)
    xcores, _ = _vector_cores(value)
    dtype = np.result_type(acores[0], fcores[0], xcores[0])
    got = _projected_gradient_norm(
        [np.asarray(core, dtype=dtype) for core in acores],
        [np.asarray(core, dtype=dtype) for core in fcores],
        [np.asarray(core, dtype=dtype) for core in xcores],
        float(_ops.norm(fcores)),
        np.ones((1, 1, 1), dtype=dtype),
        np.ones((1, 1), dtype=dtype),
    )
    expected = dense_horizontal_gradient(operator, rhs, value)
    assert got == pytest.approx(expected, rel=2e-12, abs=2e-13)


def test_manufactured_system_converges_without_changing_ranks(monkeypatch):
    bits, rank = 4, 4
    depth = 2 * bits
    profile = feasible_profile(depth, rank)
    operator = tt.eye(2, depth) + 0.1 * tt.qlaplace_dd([bits, bits])
    exact = random_profile_tt(profile, 2)
    rhs = tt.matvec(operator, exact)
    initial = random_profile_tt(profile, 3)

    # A fixed-rank QR sweep and its LOBPCG memory need no SVD.
    monkeypatch.setattr(
        np.linalg,
        "svd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("lobpcg_solve must not call SVD")
        ),
    )
    result, info = lobpcg_solve(
        operator,
        rhs,
        initial,
        1e-8,
        nswp=60,
        local_steps=8,
        verb=0,
        check_true_res=True,
        return_info=True,
    )
    assert info.converged
    assert list(map(int, result.r)) == profile
    assert info.projected_gradient <= 1e-8
    assert info.true_res < 2e-8
    assert (result - exact).norm() / exact.norm() < 2e-8
    assert info.recycled_directions > 0
    assert tt.lobpcg_solve is lobpcg_solve


def test_numpy_fallback_without_numba(monkeypatch):
    depth = 6
    profile = feasible_profile(depth, 2)
    operator = tt.eye(2, depth) + 0.1 * tt.qlaplace_dd([depth])
    exact = random_profile_tt(profile, 21)
    rhs = tt.matvec(operator, exact)
    monkeypatch.setattr(lobpcg_module._fast, "HAVE_NUMBA", False)
    result, info = lobpcg_solve(
        operator,
        rhs,
        random_profile_tt(profile, 22),
        1e-7,
        nswp=30,
        local_steps=5,
        verb=0,
        check_true_res=True,
        return_info=True,
    )
    assert info.converged
    assert info.true_res < 1e-7
    assert (result - exact).norm() / exact.norm() < 1e-7


def test_complex_hermitian_system_uses_the_generic_jacobi_path():
    depth = 6
    profile = feasible_profile(depth, 2)
    operator = tt.eye(2, depth) + 0.1 * tt.qlaplace_dd([depth])
    exact = random_complex_profile_tt(profile, 31)
    rhs = tt.matvec(operator, exact)
    result, info = lobpcg_solve(
        operator,
        rhs,
        random_complex_profile_tt(profile, 32),
        1e-8,
        nswp=40,
        local_steps=8,
        verb=0,
        check_true_res=True,
        return_info=True,
    )
    assert info.converged
    assert info.true_res < 1e-8
    assert (result - exact).norm() / exact.norm() < 1e-8


def test_infeasible_fixed_profile_is_rejected():
    depth = 4
    operator = tt.eye(2, depth)
    rhs = tt.ones(2, depth)
    bad = tt.vector.from_list([
        np.ones((1, 2, 4)),
        np.ones((4, 2, 4)),
        np.ones((4, 2, 2)),
        np.ones((2, 2, 1)),
    ])
    with pytest.raises(ValueError, match="not left-feasible"):
        lobpcg_solve(operator, rhs, bad, 1e-6, verb=0)


def test_nonconvergence_is_reported_not_hidden():
    depth = 5
    profile = feasible_profile(depth, 2)
    operator = tt.qlaplace_dd([depth])
    rhs = tt.ones(2, depth)
    initial = random_profile_tt(profile, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = lobpcg_solve(
            operator,
            rhs,
            initial,
            1e-14,
            nswp=1,
            local_steps=1,
            verb=0,
            return_info=True,
        )
    assert not info.converged
    assert caught and "did NOT reach" in str(caught[0].message)
