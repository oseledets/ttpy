"""Ordinary conservative finite differences assembled directly in QTT."""

from __future__ import annotations

import numpy as np
import pytest

import tt


def _dense_reference(bits, coefficient):
    """Independent row-by-row central-flux assembly in Fortran ordering."""
    shape = tuple(2 ** value for value in bits)
    spacing = np.array([1.0 / (value + 1) for value in shape])
    size = int(np.prod(shape))
    result = np.zeros((size, size))
    for index in np.ndindex(shape):
        row = np.ravel_multi_index(index, shape, order="F")
        node = (np.array(index) + 1.0) * spacing
        for axis in range(len(bits)):
            left = node.copy()
            right = node.copy()
            left[axis] -= 0.5 * spacing[axis]
            right[axis] += 0.5 * spacing[axis]
            kleft = float(coefficient(left[None, :])[0])
            kright = float(coefficient(right[None, :])[0])
            scale = spacing[axis] ** -2
            result[row, row] += scale * (kleft + kright)
            if index[axis] > 0:
                neighbour = list(index)
                neighbour[axis] -= 1
                column = np.ravel_multi_index(tuple(neighbour), shape,
                                               order="F")
                result[row, column] -= scale * kleft
            if index[axis] + 1 < shape[axis]:
                neighbour = list(index)
                neighbour[axis] += 1
                column = np.ravel_multi_index(tuple(neighbour), shape,
                                               order="F")
                result[row, column] -= scale * kright
    return result


def _profile(depth, rank):
    return [min(rank, 2 ** min(k, depth - k)) for k in range(depth + 1)]


def _random_profile(profile, seed):
    rng = np.random.default_rng(seed)
    value = tt.vector.from_list([
        rng.standard_normal((profile[k], 2, profile[k + 1]))
        for k in range(len(profile) - 1)
    ])
    return value / value.norm()


@pytest.mark.parametrize("bits", [3, [3, 3]])
def test_constant_coefficient_is_the_standard_dirichlet_laplacian(bits):
    levels = [bits] if isinstance(bits, int) else bits
    operator = tt.qtt_divgrad(bits)
    h = 1.0 / (2 ** levels[0] + 1)
    reference = h ** -2 * tt.qlaplace_dd(levels)
    assert (operator - reference).norm() / operator.norm() < 2e-14


def test_variable_coefficient_matches_independent_dense_flux_assembly():
    bits = [2, 2]

    def coefficient(points):
        return 1.0 + points[:, 0] + 2.0 * points[:, 1]

    operator, info = tt.qtt_divgrad(
        bits,
        coefficient,
        coefficient_eps=1e-13,
        round_eps=1e-14,
        n_check=100,
        return_info=True,
    )
    reference = _dense_reference(bits, coefficient)
    error = np.linalg.norm(operator.full() - reference) / np.linalg.norm(reference)
    assert error < 2e-13
    symmetry_error = np.linalg.norm(operator.full() - operator.full().T) \
        / np.linalg.norm(operator.full())
    assert symmetry_error < 2e-14
    assert np.linalg.eigvalsh(operator.full())[0] > 0.0
    assert max(info.operator_ranks) <= 4
    assert len(info.cross_histories) == 4


def test_nonpositive_sampled_coefficient_is_rejected():
    with pytest.raises(ValueError, match="positive coefficient"):
        tt.qtt_divgrad([2, 2], lambda points: -np.ones(points.shape[0]))


def test_amen_and_fixed_rank_lobpcg_solve_the_variable_operator():
    bits = 3
    depth = 2 * bits

    def coefficient(points):
        return 1.0 + 0.25 * np.sin(2.0 * np.pi * points[:, 0]) \
            * np.sin(2.0 * np.pi * points[:, 1])

    operator = tt.qtt_divgrad([bits, bits], coefficient, n_check=0)
    profile = _profile(depth, 4)
    exact = _random_profile(profile, 1)
    rhs = tt.matvec(operator, exact)

    adaptive, amen_info = tt.amen_solve(
        operator,
        rhs,
        None,
        1e-9,
        kickrank=4,
        rmax=16,
        nswp=30,
        verb=0,
        seed=0,
        check_true_res=True,
        return_info=True,
    )
    fixed, lobpcg_info = tt.lobpcg_solve(
        operator,
        rhs,
        _random_profile(profile, 2),
        1e-9,
        nswp=60,
        local_steps=12,
        verb=0,
        check_true_res=True,
        return_info=True,
    )

    assert amen_info.converged and amen_info.true_res < 1e-9
    assert lobpcg_info.converged and lobpcg_info.true_res < 2e-9
    assert (adaptive - exact).norm() / exact.norm() < 2e-8
    assert (fixed - exact).norm() / exact.norm() < 5e-9
    assert list(map(int, fixed.r)) == profile
