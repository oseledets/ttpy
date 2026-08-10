from __future__ import annotations

import numpy as np
import pytest

from tt.transport import TruncatedGaussianMixture


def test_single_component_transport_matches_truncated_normal_formula():
    from scipy.special import ndtr, ndtri

    mean = np.array([[0.2, -0.4]])
    scale = np.array([[0.3, 0.7]])
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.5, 1.0])
    mixture = TruncatedGaussianMixture(
        mean, scale, lower=lower, upper=upper
    )
    uniform = np.random.default_rng(2).random((100, 2))
    low_z = (lower - mean[0]) / scale[0]
    high_z = (upper - mean[0]) / scale[0]
    physical = mean[0] + scale[0] * ndtri(
        ndtr(low_z) + uniform * (ndtr(high_z) - ndtr(low_z))
    )
    expected = (physical - lower) / (upper - lower)

    actual = mixture.inverse_rosenblatt(uniform)
    assert np.max(np.abs(actual - expected)) < 8e-15
    assert np.max(np.abs(mixture.rosenblatt(actual) - uniform)) < 8e-15


def test_gaussian_mixture_analytic_rosenblatt_roundtrip():
    means = np.array([
        [-0.7, -0.5, 0.4],
        [0.6, -0.3, -0.2],
        [0.2, 0.7, 0.6],
    ])
    mixture = TruncatedGaussianMixture(
        means,
        scales=np.array([0.2, 0.3, 0.25]),
        weights=np.array([0.2, 0.5, 0.3]),
        lower=-np.ones(3),
        upper=np.ones(3),
    )
    uniform = np.random.default_rng(3).random((300, 3))
    samples = mixture.inverse_rosenblatt(uniform)
    recovered = mixture.rosenblatt(samples)

    assert np.max(np.abs(recovered - uniform)) < 3e-14
    assert mixture.roundtrip_error(samples) < 3e-14
    assert np.all(np.isfinite(mixture.log_density(samples)))


def test_direct_and_transport_sampling_have_matching_moments():
    means = np.array([
        [-0.8, -0.8],
        [-0.8, 0.8],
        [0.8, -0.8],
    ])
    mixture = TruncatedGaussianMixture(
        means, 0.12, lower=[-1.5, -1.5], upper=[1.5, 1.5]
    )
    direct = mixture.sample(30_000, seed=4)
    uniform = np.random.default_rng(5).random((30_000, 2))
    transported = mixture.inverse_rosenblatt(uniform)

    assert np.max(np.abs(direct.mean(0) - transported.mean(0))) < 8e-3
    assert np.max(np.abs(np.cov(direct, rowvar=False) -
                         np.cov(transported, rowvar=False))) < 8e-3


def test_density_is_normalized_by_quadrature():
    mixture = TruncatedGaussianMixture(
        [[-0.5], [0.4]],
        [[0.2], [0.35]],
        weights=[0.3, 0.7],
        lower=[-1.5],
        upper=[1.5],
    )
    nodes, weights = np.polynomial.legendre.leggauss(300)
    unit = (0.5 * (nodes + 1.0)).reshape(-1, 1)
    integral = 0.5 * weights @ mixture.density(unit)

    assert integral == pytest.approx(1.0, abs=2e-13)


def test_piecewise_constant_oracle_density_and_transport_are_exact():
    mixture = TruncatedGaussianMixture(
        [[-0.8, -0.8], [0.8, 0.8]],
        0.16,
        lower=[-1.5, -1.5],
        upper=[1.5, 1.5],
    )
    modes = np.array([5, 6])
    grid = np.stack(
        np.meshgrid(
            (np.arange(modes[0]) + 0.5) / modes[0],
            (np.arange(modes[1]) + 0.5) / modes[1],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    assert np.mean(mixture.cell_density(grid, modes)) == pytest.approx(
        1.0, abs=3e-14
    )
    component_masses = mixture.component_cell_masses(modes)
    dense_probability = np.einsum(
        "c,ci,cj->ij", mixture.weights, component_masses[0], component_masses[1]
    )
    evaluated_probability = (
        mixture.cell_density(grid, modes).reshape(tuple(modes)) / np.prod(modes)
    )
    assert np.max(np.abs(evaluated_probability - dense_probability)) < 3e-15

    uniform = np.random.default_rng(8).random((400, 2))
    samples = mixture.inverse_cell_rosenblatt(uniform, modes)
    recovered = mixture.cell_rosenblatt(samples, modes)
    assert np.max(np.abs(recovered - uniform)) < 3e-14
