from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

import tt
from tt.transport import (
    AdaptiveLinearSquaredTTDensity,
    DirectTTDensity,
    LinearSquaredTTDensity,
    NonnegativeLinearTTDensity,
    ProbitOrthogonalTTDensity,
    PurifiedLinearTTDensity,
    QuadraticSquaredTTDensity,
    PermutedTTDensity,
    SampleDIRT,
    SquaredTTDensity,
    damp_tt_density,
    enable_linear_sample_dirt_probit_rotations,
    enrich_linear_sample_dirt_ranks,
    fit_centered_tt_density,
    fit_centered_tt_ratio,
    fit_adaptive_linear_squared_tt_density,
    fit_conditional_twists_to_probit_density,
    fine_tune_linear_sample_dirt,
    fit_linear_squared_tt_density,
    fit_nonnegative_linear_tt_density,
    fit_probit_rotated_linear_squared_tt_density,
    fit_radial_twists_to_probit_density,
    fit_purified_linear_tt_density,
    fit_quadratic_squared_tt_density,
    refine_linear_sample_dirt_modes,
    score_linear_sample_dirt_mode_refinement,
    truncate_linear_sample_dirt_ranks,
)
from tt.transport import sample_dirt as sd


def _nontrivial_root() -> tt.vector:
    # A rank-two three-dimensional root with genuinely conditional cell masses.
    rng = np.random.default_rng(4)
    cores = [
        rng.normal(size=(1, 5, 2)),
        rng.normal(size=(2, 4, 2)),
        rng.normal(size=(2, 6, 1)),
    ]
    return tt.vector.from_list(cores)


def test_linear_hat_mass_qr_is_function_preserving_and_exactly_canonical():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    cores = [
        torch.tensor(rng.normal(size=(1, 5, 3)), dtype=torch.float64),
        torch.tensor(rng.normal(size=(3, 7, 4)), dtype=torch.float64),
        torch.tensor(rng.normal(size=(4, 6, 1)), dtype=torch.float64),
    ]
    old = [core.clone() for core in cores]
    points = torch.tensor(rng.random((100, 3)), dtype=torch.float64)
    before = sd._torch_sample_linear_tt(old, points)

    sd._torch_linear_right_orthogonalize(cores)

    after = sd._torch_sample_linear_tt(cores, points)
    torch.testing.assert_close(after, before, rtol=2e-14, atol=2e-14)
    right = sd._torch_linear_right_environments(cores)
    for environment in right[1:]:
        torch.testing.assert_close(
            environment,
            torch.eye(
                environment.shape[0],
                dtype=environment.dtype,
                device=environment.device,
            ),
            rtol=2e-14,
            atol=2e-14,
        )


def test_linear_root_sobolev_ratio_is_exact_and_scale_invariant():
    torch = pytest.importorskip("torch")
    slope = 0.7
    dimension = 3
    first = torch.tensor(
        [[[1.0], [1.0 + slope]]], dtype=torch.float64
    )
    constant = torch.ones((1, 2, 1), dtype=torch.float64)
    cores = [first, constant.clone(), constant.clone()]
    expected_mass = 1.0 + slope + slope * slope / 3.0
    expected = slope * slope / (dimension * expected_mass)

    value = sd._torch_linear_root_sobolev_ratio(cores)
    scaled = sd._torch_linear_root_sobolev_ratio([
        4.2 * cores[0], *cores[1:]
    ])

    assert float(value) == pytest.approx(expected, rel=2e-13)
    assert float(scaled) == pytest.approx(expected, rel=2e-13)


def test_squared_density_is_normalized_and_l2_is_exact():
    density = SquaredTTDensity(_nontrivial_root(), gamma=1e-3)
    root = np.asarray(density.root.full())
    dense = (density.gamma + root * root) / density.normalization

    assert np.mean(dense) == pytest.approx(1.0, abs=1e-13)
    assert density.model_l2_norm_sq() == pytest.approx(
        float(np.mean(dense * dense)), rel=2e-13
    )
    assert density.chi2_to_reference == pytest.approx(
        float(np.mean((dense - 1.0) ** 2)), rel=2e-13
    )


@pytest.mark.parametrize(
    "density_type",
    (SquaredTTDensity, LinearSquaredTTDensity, QuadraticSquaredTTDensity),
)
def test_damped_squared_density_is_exact_uniform_mixture(density_type):
    rng = np.random.default_rng(451)
    root = tt.vector.from_list([
        rng.normal(size=(1, 7, 3)),
        rng.normal(size=(3, 6, 1)),
    ])
    density = density_type(root, gamma=2e-3)
    weight = 0.37
    damped = damp_tt_density(density, weight)
    probes = rng.random((300, 2))

    assert damped.reference_floor_mass == pytest.approx(
        1.0 - weight * (1.0 - density.reference_floor_mass), abs=2e-14
    )
    assert np.max(np.abs(
        damped.density(probes)
        - ((1.0 - weight) + weight * density.density(probes))
    )) < 2e-12
    assert np.max(np.abs(
        damped.inverse_rosenblatt(damped.rosenblatt(probes)) - probes
    )) < 3e-10


def test_damped_direct_and_permuted_densities_preserve_rank_and_mixture():
    correction = tt.vector.from_list([
        np.array([[[-0.3], [0.1], [0.2], [0.0]]], dtype=np.float64)
    ])
    direct = DirectTTDensity(correction, conditional_floor=1e-12)
    damped = damp_tt_density(direct, 0.4)
    probes = np.linspace(0.01, 0.99, 101)[:, None]

    assert np.array_equal(damped.ranks, direct.ranks)
    assert np.max(np.abs(
        damped.density(probes) - (0.6 + 0.4 * direct.density(probes))
    )) < 2e-14

    rng = np.random.default_rng(452)
    root = tt.vector.from_list([
        rng.normal(size=(1, 5, 2)), rng.normal(size=(2, 6, 1))
    ])
    permuted = PermutedTTDensity(SquaredTTDensity(root), [1, 0])
    permuted_damped = damp_tt_density(permuted, 0.25)
    points = rng.random((200, 2))
    assert np.array_equal(permuted_damped.permutation, permuted.permutation)
    assert np.max(np.abs(
        permuted_damped.density(points)
        - (0.75 + 0.25 * permuted.density(points))
    )) < 2e-12


@pytest.mark.parametrize("weight", (0.0, -0.1, 1.01, np.nan))
def test_damped_density_rejects_invalid_weight(weight):
    with pytest.raises(ValueError):
        damp_tt_density(SquaredTTDensity(_nontrivial_root()), weight)


def test_linear_squared_density_has_exact_gram_normalization_and_roundtrip(
    tmp_path,
):
    rng = np.random.default_rng(230)
    root = tt.vector.from_list([
        rng.normal(size=(1, 9, 3)),
        rng.normal(size=(3, 8, 1)),
    ])
    density = LinearSquaredTTDensity(root, gamma=1e-3)
    torch = pytest.importorskip("torch")
    torch_cores = [torch.tensor(core) for core in density._cores]
    grid = np.linspace(0.0, 1.0, 501)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    values = density.density(points).reshape(len(grid), len(grid))
    integral = np.trapezoid(np.trapezoid(values, grid, axis=1), grid)
    generated = density.sample(500, seed=231)

    assert integral == pytest.approx(1.0, rel=3e-5)
    assert float(sd._torch_linear_root_second_moment(torch_cores)) == pytest.approx(
        density._root_second_moment, rel=2e-13
    )
    assert np.max(np.abs(
        sd._torch_sample_linear_tt(
            torch_cores, torch.tensor(generated)
        ).numpy() - density.root_values(generated)
    )) < 2e-12
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 2e-12
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=73
        ) - density.log_density(generated)
    )) < 2e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=61
        ) - density.rosenblatt(generated)
    )) < 2e-12
    model = SampleDIRT(2, [PermutedTTDensity(density, [1, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 2e-12


def test_adaptive_linear_density_exact_gram_transport_and_serialization(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(701)
    root = tt.vector.from_list([
        rng.normal(size=(1, 6, 3)),
        rng.normal(size=(3, 5, 1)),
    ])
    knots = [
        np.array([0.0, 0.04, 0.17, 0.46, 0.81, 1.0]),
        np.array([0.0, 0.13, 0.31, 0.76, 1.0]),
    ]
    density = AdaptiveLinearSquaredTTDensity(root, knots, gamma=2e-3)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(2)
    grids, weights = [], []
    for nodes in knots:
        widths = np.diff(nodes)
        grids.append(np.concatenate([
            left + 0.5 * width * (gauss_nodes + 1.0)
            for left, width in zip(nodes[:-1], widths)
        ]))
        weights.append(np.concatenate([
            0.5 * width * gauss_weights for width in widths
        ]))
    xx, yy = np.meshgrid(*grids, indexing="ij")
    probes = np.column_stack([xx.ravel(), yy.ravel()])
    values = density.density(probes).reshape(len(grids[0]), len(grids[1]))
    integral = np.einsum("ij,i,j->", values, *weights)
    generated = density.sample(400, seed=702)
    torch_cores = [torch.tensor(core) for core in density._cores]
    torch_knots = [torch.tensor(nodes) for nodes in knots]

    assert integral == pytest.approx(1.0, abs=4e-13)
    assert float(sd._torch_adaptive_linear_root_second_moment(
        torch_cores, torch_knots
    )) == pytest.approx(density._root_second_moment, rel=3e-13)
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 3e-12
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=67
        ) - density.log_density(generated)
    )) < 3e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=59
        ) - density.rosenblatt(generated)
    )) < 3e-12

    damped = damp_tt_density(density, 0.37)
    assert isinstance(damped, AdaptiveLinearSquaredTTDensity)
    assert np.max(np.abs(
        damped.density(generated) - (0.63 + 0.37 * density.density(generated))
    )) < 3e-12
    model = SampleDIRT(2, [PermutedTTDensity(density, [1, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 3e-12


def test_adaptive_uniform_knots_match_uniform_linear_density_exactly():
    rng = np.random.default_rng(703)
    root = tt.vector.from_list([
        rng.normal(size=(1, 7, 3)),
        rng.normal(size=(3, 8, 1)),
    ])
    uniform = LinearSquaredTTDensity(root, gamma=1e-3)
    adaptive = AdaptiveLinearSquaredTTDensity(
        root,
        [np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 8)],
        gamma=1e-3,
    )
    probes = rng.random((500, 2))

    assert adaptive.size == uniform.size + 11
    assert adaptive.normalization == pytest.approx(
        uniform.normalization, rel=3e-14
    )
    assert np.max(np.abs(
        adaptive.log_density(probes) - uniform.log_density(probes)
    )) < 3e-13
    assert np.max(np.abs(
        adaptive.rosenblatt(probes) - uniform.rosenblatt(probes)
    )) < 3e-13


def test_adaptive_knot_fit_has_uniform_validation_fallback():
    pytest.importorskip("torch")
    rng = np.random.default_rng(704)
    training = np.column_stack([
        rng.beta(0.8, 3.0, size=900),
        rng.beta(3.0, 0.8, size=900),
    ])
    validation = np.column_stack([
        rng.beta(0.8, 3.0, size=300),
        rng.beta(3.0, 0.8, size=300),
    ])
    density, history = fit_adaptive_linear_squared_tt_density(
        training,
        validation_samples=validation,
        modes=7,
        rank=2,
        floor_mass=1e-3,
        warm_start_epochs=25,
        epochs=12,
        batch_size=256,
        learning_rate=2e-2,
        knot_learning_rate=2e-3,
        knot_initialization="quantile",
        minimum_knot_width=2e-3,
        patience=20,
        seed=705,
    )
    final_validation = -float(np.mean(density.log_density(validation)))

    assert history.optimizer.startswith("linear-adaptive-knots-adam")
    assert final_validation == pytest.approx(
        min(history.validation_loss), rel=2e-11
    )
    assert final_validation <= history.validation_loss[0] + 2e-11
    assert all(np.all(np.diff(nodes) >= 2e-3 - 2e-12)
               for nodes in density.knots)


def test_probit_orthogonal_density_is_exact_transport_and_serialized(tmp_path):
    rng = np.random.default_rng(706)
    root = tt.vector.from_list([
        rng.normal(size=(1, 8, 3)),
        rng.normal(size=(3, 7, 1)),
    ])
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    base = LinearSquaredTTDensity(root, gamma=2e-3)
    density = ProbitOrthogonalTTDensity(base, rotation)
    generated = density.sample(500, seed=707)
    rotated = density._rotate(generated)

    assert np.max(np.abs(
        density.log_density(generated) - base.log_density(rotated)
    )) < 2e-13
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 2e-10
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=73
        ) - density.log_density(generated)
    )) < 2e-12
    damped = damp_tt_density(density, 0.41)
    assert np.max(np.abs(
        damped.density(generated) - (0.59 + 0.41 * density.density(generated))
    )) < 3e-12

    model = SampleDIRT(2, [PermutedTTDensity(density, [1, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 3e-12


def test_probit_radial_twist_is_exact_volume_one_and_differentiable(tmp_path):
    torch = pytest.importorskip("torch")
    from scipy.special import ndtri

    rng = np.random.default_rng(1706)
    root = tt.vector.from_list([
        rng.normal(size=(1, 7, 3)),
        rng.normal(size=(3, 7, 1)),
    ])
    angle = 0.23
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    pairs = [np.array([[0, 1]])]
    coefficients = [np.array([[0.19, -0.08, 0.035]])]
    base = LinearSquaredTTDensity(root, gamma=1e-3)
    density = ProbitOrthogonalTTDensity(
        base,
        rotation,
        radial_twist_pairs=pairs,
        radial_twist_coefficients=coefficients,
    )
    points = rng.uniform(0.08, 0.92, size=(80, 2))
    mapped = density._rotate(points)
    recovered = density._rotate(mapped, inverse=True)

    assert np.max(np.abs(recovered - points)) < 2e-12
    gaussian = ndtri(points)
    mapped_gaussian = ndtri(mapped)
    np.testing.assert_allclose(
        np.sum(np.square(mapped_gaussian), axis=1),
        np.sum(np.square(gaussian), axis=1),
        atol=3e-12,
    )
    step = 2e-6
    for point in points[:8]:
        jacobian = np.empty((2, 2))
        for coordinate in range(2):
            offset = np.zeros(2)
            offset[coordinate] = step
            jacobian[:, coordinate] = (
                density._rotate((point + offset)[None])[0]
                - density._rotate((point - offset)[None])[0]
            ) / (2.0 * step)
        assert np.linalg.det(jacobian) == pytest.approx(1.0, abs=2e-8)

    torch_points = torch.tensor(points, dtype=torch.float64)
    torch_rotation = torch.tensor(rotation, dtype=torch.float64)
    torch_coefficients = [torch.tensor(
        coefficients[0], dtype=torch.float64, requires_grad=True
    )]
    torch_mapped = sd._torch_probit_orthogonal_map(
        torch_points,
        torch_rotation,
        radial_twist_pairs=pairs,
        radial_twist_coefficients=torch_coefficients,
    )
    np.testing.assert_allclose(
        torch_mapped.detach().numpy(), mapped, atol=3e-15
    )
    torch_mapped.square().sum().backward()
    assert torch.all(torch.isfinite(torch_coefficients[0].grad))
    assert float(torch.linalg.norm(torch_coefficients[0].grad)) > 0.0

    damped = damp_tt_density(density, 0.37)
    np.testing.assert_allclose(
        damped.density(points), 0.63 + 0.37 * density.density(points),
        atol=3e-12,
    )
    model = SampleDIRT(2, [PermutedTTDensity(density, [1, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    np.testing.assert_allclose(
        loaded.log_density(points), model.log_density(points), atol=3e-12
    )
    assert loaded.layers[0].base.size == base.size + 4 + 3


def test_probit_rotation_fit_has_identity_validation_fallback():
    pytest.importorskip("torch")
    rng = np.random.default_rng(708)
    training = np.column_stack([
        rng.beta(2.0, 6.0, size=1000),
        rng.beta(6.0, 2.0, size=1000),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 6.0, size=300),
        rng.beta(6.0, 2.0, size=300),
    ])
    density, history = fit_probit_rotated_linear_squared_tt_density(
        training,
        validation_samples=validation,
        modes=7,
        rank=2,
        floor_mass=1e-3,
        warm_start_epochs=20,
        epochs=10,
        batch_size=256,
        learning_rate=2e-2,
        rotation_learning_rate=2e-3,
        rotation_block_size=2,
        patience=20,
        seed=709,
    )
    final_validation = -float(np.mean(density.log_density(validation)))

    assert history.optimizer.startswith("linear-probit-block-orthogonal")
    assert final_validation == pytest.approx(
        min(history.validation_loss), rel=2e-10
    )
    assert final_validation <= history.validation_loss[0] + 2e-10
    assert np.linalg.norm(
        density.rotation.T @ density.rotation - np.eye(2), ord=2
    ) < 2e-12


def test_radial_twist_fit_recovers_curved_coordinate_signal():
    pytest.importorskip("torch")
    grid = np.linspace(0.0, 1.0, 9)
    first = np.sqrt(0.3 + 1.4 * grid)[None, :, None]
    second = np.sqrt(1.7 - 1.1 * grid)[None, :, None]
    base = LinearSquaredTTDensity(
        tt.vector.from_list([first, second]), gamma=1e-3
    )
    identity = np.eye(2)
    untwisted = ProbitOrthogonalTTDensity(base, identity)
    target = ProbitOrthogonalTTDensity(
        base,
        identity,
        radial_twist_pairs=[np.array([[0, 1]])],
        radial_twist_coefficients=[np.array([[0.32, -0.14, 0.06]])],
    )
    samples = target.sample(2200, seed=1710)
    training, validation = samples[:1700], samples[1700:]
    initial = float(np.mean(untwisted.log_density(validation)))

    fitted, history = fit_radial_twists_to_probit_density(
        untwisted,
        training,
        validation_samples=validation,
        stages=1,
        basis_count=3,
        epochs=40,
        learning_rate=2e-2,
        batch_size=512,
        validation_interval=2,
        patience=20,
        dtype="float64",
        seed=1711,
    )
    final = float(np.mean(fitted.log_density(validation)))

    assert history.optimizer.startswith("probit-radial-twist-adam")
    assert final >= initial - 2e-12
    assert final > initial + 2e-3
    assert -final == pytest.approx(min(history.validation_loss), abs=2e-11)


def test_probit_conditional_twist_is_exact_and_differentiable(tmp_path):
    torch = pytest.importorskip("torch")
    from scipy.special import ndtri

    rng = np.random.default_rng(2710)
    root = tt.vector.from_list([
        rng.normal(size=(1, 7, 3)),
        rng.normal(size=(3, 6, 3)),
        rng.normal(size=(3, 7, 1)),
    ])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    pairs = [np.array([[0, 1]]), np.array([[1, 2]])]
    conditioners = [np.array([2]), np.array([0])]
    coefficients = [
        np.array([[0.31, -0.09, 0.04]]),
        np.array([[-0.18, 0.07, 0.03]]),
    ]
    base = LinearSquaredTTDensity(root, gamma=1e-3)
    density = ProbitOrthogonalTTDensity(
        base,
        rotation,
        conditional_twist_pairs=pairs,
        conditional_twist_conditioners=conditioners,
        conditional_twist_coefficients=coefficients,
    )
    points = rng.uniform(0.12, 0.88, size=(70, 3))
    mapped = density._rotate(points)
    recovered = density._rotate(mapped, inverse=True)

    assert np.max(np.abs(recovered - points)) < 3e-12
    np.testing.assert_allclose(
        np.sum(np.square(ndtri(mapped)), axis=1),
        np.sum(np.square(ndtri(points)), axis=1),
        atol=5e-12,
    )
    step = 2e-6
    for point in points[:5]:
        jacobian = np.empty((3, 3))
        for coordinate in range(3):
            offset = np.zeros(3)
            offset[coordinate] = step
            jacobian[:, coordinate] = (
                density._rotate((point + offset)[None])[0]
                - density._rotate((point - offset)[None])[0]
            ) / (2.0 * step)
        assert np.linalg.det(jacobian) == pytest.approx(1.0, abs=4e-8)

    torch_coefficients = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in coefficients
    ]
    torch_mapped = sd._torch_probit_orthogonal_map(
        torch.tensor(points, dtype=torch.float64),
        torch.tensor(rotation, dtype=torch.float64),
        conditional_twist_pairs=pairs,
        conditional_twist_conditioners=conditioners,
        conditional_twist_coefficients=torch_coefficients,
    )
    np.testing.assert_allclose(
        torch_mapped.detach().numpy(), mapped, atol=4e-15
    )
    torch_mapped.square().sum().backward()
    assert all(
        torch.all(torch.isfinite(value.grad))
        and float(torch.linalg.norm(value.grad)) > 0.0
        for value in torch_coefficients
    )

    damped = damp_tt_density(density, 0.37)
    np.testing.assert_allclose(
        damped.density(points), 0.63 + 0.37 * density.density(points),
        atol=5e-12,
    )
    model = SampleDIRT(3, [PermutedTTDensity(density, [2, 0, 1])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    np.testing.assert_allclose(
        loaded.log_density(points), model.log_density(points), atol=5e-12
    )
    assert loaded.layers[0].base.size == base.size + 9 + 6

    fine_tuned, _ = fine_tune_linear_sample_dirt(
        SampleDIRT(3, [density]),
        points[:50],
        validation_samples=points[50:],
        epochs=1,
        learning_rate=1e-20,
        batch_size=25,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=2713,
    )
    np.testing.assert_allclose(
        fine_tuned.log_density(points), density.log_density(points),
        atol=3e-11,
    )
    assert len(fine_tuned.layers[0].conditional_twist_pairs) == 2


def test_conditional_twist_fit_recovers_nonlinear_coordinate_signal():
    pytest.importorskip("torch")
    grid = np.linspace(0.0, 1.0, 9)
    base = LinearSquaredTTDensity(
        tt.vector.from_list([
            np.sqrt(0.25 + 1.5 * grid)[None, :, None],
            np.sqrt(1.6 - 1.05 * grid)[None, :, None],
            np.sqrt(0.45 + 1.0 * grid)[None, :, None],
        ]),
        gamma=1e-3,
    )
    identity = np.eye(3)
    untwisted = ProbitOrthogonalTTDensity(base, identity)
    target = ProbitOrthogonalTTDensity(
        base,
        identity,
        conditional_twist_pairs=[np.array([[0, 1]])],
        conditional_twist_conditioners=[np.array([2])],
        conditional_twist_coefficients=[
            np.array([[0.55, -0.22, 0.08]])
        ],
    )
    samples = target.sample(3200, seed=2711)
    training, validation = samples[:2500], samples[2500:]
    initial = float(np.mean(untwisted.log_density(validation)))

    fitted, history = fit_conditional_twists_to_probit_density(
        untwisted,
        training,
        validation_samples=validation,
        stages=1,
        basis_count=3,
        epochs=60,
        learning_rate=2e-2,
        batch_size=512,
        validation_interval=2,
        patience=25,
        dtype="float64",
        seed=2712,
    )
    final = float(np.mean(fitted.log_density(validation)))

    assert history.optimizer.startswith("probit-conditional-twist-adam")
    assert final >= initial - 2e-12
    assert final > initial + 3e-3
    assert -final == pytest.approx(min(history.validation_loss), abs=3e-11)

    chain_fitted, chain_history = fine_tune_linear_sample_dirt(
        SampleDIRT(3, [untwisted]),
        training,
        validation_samples=validation,
        epochs=60,
        learning_rate=2e-2,
        batch_size=512,
        validation_interval=2,
        patience=25,
        dtype="float64",
        seed=2712,
        optimize_cores=False,
        optimize_conditional_twists=True,
        conditional_twist_stages=1,
        conditional_twist_basis_count=3,
    )
    chain_final = float(np.mean(
        chain_fitted.log_density(validation)
    ))
    assert chain_history.optimizer == (
        "joint-linear-probit-conditional-adam"
    )
    assert chain_final > initial + 3e-3
    assert -chain_final <= min(chain_history.validation_loss) + 5e-5

    selected_layers, _ = fine_tune_linear_sample_dirt(
        SampleDIRT(3, [untwisted, untwisted]),
        training[:128],
        validation_samples=validation[:64],
        epochs=1,
        learning_rate=1e-20,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=2714,
        optimize_cores=False,
        optimize_conditional_twists=True,
        conditional_twist_stages=1,
        conditional_twist_basis_count=3,
        conditional_twist_layer_indices=[0],
    )
    assert len(selected_layers.layers[0].conditional_twist_pairs) == 0
    assert len(selected_layers.layers[1].conditional_twist_pairs) == 0


def test_nonnegative_linear_density_is_exact_normalized_transport_and_serialized(
    tmp_path,
):
    rng = np.random.default_rng(710)
    tensor = tt.vector.from_list([
        rng.lognormal(size=(1, 8, 3)),
        rng.lognormal(size=(3, 7, 2)),
        rng.lognormal(size=(2, 6, 1)),
    ])
    density = NonnegativeLinearTTDensity(tensor, gamma=2e-3)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(2)
    grids = []
    weights = []
    for mode in (8, 7, 6):
        h = 1.0 / (mode - 1)
        grids.append(np.concatenate([
            h * (interval + 0.5 * (gauss_nodes + 1.0))
            for interval in range(mode - 1)
        ]))
        weights.append(np.tile(0.5 * h * gauss_weights, mode - 1))
    xx, yy, zz = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = density.density(points).reshape(*(len(grid) for grid in grids))
    integral = np.einsum(
        "ijk,i,j,k->",
        values,
        *weights,
    )
    generated = density.sample(500, seed=711)

    assert integral == pytest.approx(1.0, abs=3e-13)
    assert np.min(density.density(generated)) > 0.0
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 3e-12
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=73
        ) - density.log_density(generated)
    )) < 3e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=67
        ) - density.rosenblatt(generated)
    )) < 3e-12

    damped = damp_tt_density(density, 0.31)
    assert np.max(np.abs(
        damped.density(generated) - (0.69 + 0.31 * density.density(generated))
    )) < 3e-12
    model = SampleDIRT(3, [PermutedTTDensity(density, [2, 0, 1])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 3e-12


def test_nonnegative_linear_log_fit_learns_dependence_with_one_tt():
    pytest.importorskip("torch")
    rng = np.random.default_rng(712)

    def correlated_beta(count):
        component = rng.integers(0, 2, size=count)
        first = np.where(
            component == 0, rng.beta(2.0, 8.0, count),
            rng.beta(8.0, 2.0, count),
        )
        second = np.where(
            component == 0, rng.beta(2.0, 8.0, count),
            rng.beta(8.0, 2.0, count),
        )
        return np.column_stack([first, second])

    training = correlated_beta(1600)
    validation = correlated_beta(500)
    density, history = fit_nonnegative_linear_tt_density(
        training,
        validation_samples=validation,
        modes=14,
        rank=3,
        floor_mass=1e-3,
        epochs=100,
        batch_size=256,
        learning_rate=2e-2,
        patience=25,
        initialization="product",
        seed=713,
    )

    assert history.optimizer == "linear-nonnegative-log-adam"
    assert min(history.validation_loss) < history.validation_loss[0] - 0.15
    assert -np.mean(density.log_density(validation)) == pytest.approx(
        min(history.validation_loss), rel=3e-12
    )
    assert density.reference_floor_mass == pytest.approx(1e-3, abs=3e-14)
    assert all(np.all(core > 0.0) for core in density._cores)
    # Exact suffix-stochastic gauge: every suffix integral is one.
    assert np.max(np.abs(density._right[0] - 1.0)) < 3e-12


def test_nonnegative_log_stochastic_gauge_is_exact_and_gauge_invariant():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(714)
    log_cores = [
        torch.randn((1, 7, 3), generator=generator, dtype=torch.float64),
        torch.randn((3, 6, 4), generator=generator, dtype=torch.float64),
        torch.randn((4, 8, 1), generator=generator, dtype=torch.float64),
    ]
    probes = torch.rand((300, 3), generator=generator, dtype=torch.float64)
    normalized = sd._torch_normalize_nonnegative_linear_log_cores(log_cores)
    log_values = sd._torch_sample_nonnegative_linear_log_tt(
        normalized, probes
    )
    density = NonnegativeLinearTTDensity(
        tt.vector.from_list([torch.exp(core).numpy() for core in normalized]),
        gamma=1e-4,
    )

    assert abs(density._component_integral - 1.0) < 3e-13
    probabilities = density.bond_state_probabilities()
    effective = density.effective_state_ranks()
    assert all(abs(value.sum() - 1.0) < 3e-13 for value in probabilities)
    assert all(
        1.0 <= value <= rank + 1e-12
        for value, rank in zip(effective["entropy"], density.ranks[1:-1])
    )
    assert torch.max(torch.abs(
        log_values
        - torch.log(torch.as_tensor(density.component_values(probes.numpy())))
    )) < 3e-12

    # An arbitrary positive diagonal bond gauge changes raw cores but not the
    # function; the suffix-stochastic canonical representation is identical.
    first_gauge = torch.randn(3, generator=generator, dtype=torch.float64)
    second_gauge = torch.randn(4, generator=generator, dtype=torch.float64)
    gauged = [
        log_cores[0] + first_gauge[None, None, :],
        log_cores[1]
        - first_gauge[:, None, None]
        + second_gauge[None, None, :],
        log_cores[2] - second_gauge[:, None, None],
    ]
    gauged_normalized = sd._torch_normalize_nonnegative_linear_log_cores(gauged)
    gauged_values = sd._torch_sample_nonnegative_linear_log_tt(
        gauged_normalized, probes
    )
    assert torch.max(torch.abs(gauged_values - log_values)) < 3e-12


def test_purified_linear_density_exact_gram_transport_rotation_and_save(tmp_path):
    rng = np.random.default_rng(720)
    cores = [
        rng.normal(size=(1, 8, 3)),
        rng.normal(size=(3, 7, 2)),
        rng.normal(size=(2, 6, 4)),
    ]
    density = PurifiedLinearTTDensity(cores, gamma=2e-3)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(2)
    grids = []
    weights = []
    for mode in (8, 7, 6):
        h = 1.0 / (mode - 1)
        grids.append(np.concatenate([
            h * (interval + 0.5 * (gauss_nodes + 1.0))
            for interval in range(mode - 1)
        ]))
        weights.append(np.tile(0.5 * h * gauss_weights, mode - 1))
    xx, yy, zz = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = density.density(points).reshape(*(len(grid) for grid in grids))
    integral = np.einsum(
        "ijk,i,j,k->",
        values,
        *weights,
    )
    generated = density.sample(400, seed=721)

    assert integral == pytest.approx(1.0, abs=4e-13)
    assert density.terminal_channels == 4
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 4e-12
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=61
        ) - density.log_density(generated)
    )) < 4e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=57
        ) - density.rosenblatt(generated)
    )) < 4e-12

    orthogonal, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated_cores = [*cores[:-1], np.einsum(
        "aip,pq->aiq", cores[-1], orthogonal
    )]
    rotated = PurifiedLinearTTDensity(rotated_cores, gamma=density.gamma)
    assert np.max(np.abs(
        rotated.density(generated) - density.density(generated)
    )) < 4e-12
    np.testing.assert_allclose(
        rotated.channel_spectrum(), density.channel_spectrum(), atol=2e-12
    )
    np.testing.assert_allclose(density.channel_spectrum().sum(), 1.0)
    assert 1.0 <= density.effective_channels()["entropy"] <= 4.0
    assert np.max(np.abs(
        rotated.rosenblatt(generated) - density.rosenblatt(generated)
    )) < 4e-12

    damped = damp_tt_density(density, 0.29)
    assert np.max(np.abs(
        damped.density(generated) - (0.71 + 0.29 * density.density(generated))
    )) < 4e-12
    model = SampleDIRT(3, [PermutedTTDensity(density, [1, 2, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 4e-12


def test_locally_purified_linear_density_exact_gram_and_transport(tmp_path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(726)
    cores = [
        rng.normal(size=(1, 7, 2, 3)),
        rng.normal(size=(3, 6, 2, 2)),
        rng.normal(size=(2, 5, 2, 1)),
    ]
    density = sd.LocallyPurifiedLinearTTDensity(cores, gamma=2e-3)
    nodes, weights = np.polynomial.legendre.leggauss(2)
    grids, quadrature = [], []
    for mode in density.modes:
        h = 1.0 / (mode - 1)
        grids.append(np.concatenate([
            h * (interval + 0.5 * (nodes + 1.0))
            for interval in range(mode - 1)
        ]))
        quadrature.append(np.tile(0.5 * h * weights, mode - 1))
    xx, yy, zz = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = density.density(points).reshape(*(len(grid) for grid in grids))
    integral = np.einsum("ijk,i,j,k->", values, *quadrature)
    generated = density.sample(300, seed=727)

    assert integral == pytest.approx(1.0, abs=5e-13)
    spectra = density.local_channel_spectra()
    assert len(spectra) == density.d
    np.testing.assert_allclose(
        [spectrum.sum() for spectrum in spectra], 1.0, atol=2e-13
    )
    assert all(
        1.0 <= value <= 2.0
        for value in density.effective_local_channels()["entropy"]
    )
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 5e-12
    assert np.max(np.abs(
        density.log_density_device(
            generated, device="cpu", dtype="float64", batch_size=47
        ) - density.log_density(generated)
    )) < 5e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=43
        ) - density.rosenblatt(generated)
    )) < 5e-12
    torch_cores = [torch.tensor(core, dtype=torch.float64) for core in cores]
    joint_log, joint_map = sd._torch_local_purified_log_density_rosenblatt(
        torch_cores,
        torch.tensor(generated, dtype=torch.float64),
        density.gamma,
        transform=True,
    )
    assert np.max(np.abs(
        joint_log.detach().numpy() - density.log_density(generated)
    )) < 5e-12
    assert np.max(np.abs(
        joint_map.detach().numpy() - density.rosenblatt(generated)
    )) < 5e-12
    path = tmp_path / "local-purified.npz"
    density.save(path)
    loaded = sd.LocallyPurifiedLinearTTDensity.load(path)
    assert np.max(np.abs(
        loaded.log_density(generated) - density.log_density(generated)
    )) < 4e-12
    damped = damp_tt_density(density, 0.31)
    assert np.max(np.abs(
        damped.density(generated) - (0.69 + 0.31 * density.density(generated))
    )) < 5e-12
    model = SampleDIRT(3, [PermutedTTDensity(density, [2, 0, 1])])
    model.save(tmp_path / "local-chain")
    restored = SampleDIRT.load(tmp_path / "local-chain")
    assert np.max(np.abs(
        restored.log_density(generated) - model.log_density(generated)
    )) < 5e-12


def test_purified_linear_fit_uses_shared_channels_for_multimodality():
    pytest.importorskip("torch")
    rng = np.random.default_rng(722)

    def draw(count):
        state = rng.integers(0, 2, size=count)
        first = np.where(state == 0, 2.0, 8.0)
        second = np.where(state == 0, 8.0, 2.0)
        return np.column_stack([
            rng.beta(first, second), rng.beta(first, second)
        ])

    training = draw(1600)
    validation = draw(500)
    density, history = fit_purified_linear_tt_density(
        training,
        validation_samples=validation,
        modes=14,
        rank=3,
        purification_channels=2,
        floor_mass=1e-3,
        epochs=100,
        batch_size=256,
        learning_rate=2e-2,
        patience=25,
        initialization="cluster",
        seed=723,
    )

    assert history.optimizer == "linear-purified-adam"
    assert min(history.validation_loss) < history.validation_loss[0] - 0.005
    assert -np.mean(density.log_density(validation)) == pytest.approx(
        min(history.validation_loss), rel=3e-12
    )
    assert density.terminal_channels == 2
    assert density.reference_floor_mass == pytest.approx(1e-3, abs=3e-14)


def test_purified_scalar_warm_start_cannot_lose_heldout_likelihood():
    pytest.importorskip("torch")
    rng = np.random.default_rng(724)
    training = np.column_stack([
        rng.beta(2.5, 4.0, 500), rng.beta(5.0, 2.0, 500)
    ])
    validation = np.column_stack([
        rng.beta(2.5, 4.0, 180), rng.beta(5.0, 2.0, 180)
    ])
    common = dict(
        modes=9, rank=3, floor_mass=1e-3, learning_rate=1e-2,
        batch_size=128, validation_samples=validation, patience=8,
        validation_interval=2, initialization_noise=1e-2, seed=725,
    )
    scalar, _ = fit_linear_squared_tt_density(
        training, epochs=30, initialization="mixture", **common
    )
    purified, history = fit_purified_linear_tt_density(
        training,
        epochs=10,
        scalar_warm_start_epochs=30,
        purification_channels=3,
        initialization="scalar",
        **common,
    )

    scalar_nll = -float(np.mean(scalar.log_density(validation)))
    purified_nll = -float(np.mean(purified.log_density(validation)))
    assert purified_nll <= scalar_nll + 2e-12
    assert "scalar-warm-start" in history.optimizer
    assert history.epochs >= 30


def test_local_purification_warm_start_cannot_lose_heldout_likelihood():
    pytest.importorskip("torch")
    rng = np.random.default_rng(728)
    training = np.column_stack([
        rng.beta(3.0, 5.0, 420), rng.beta(6.0, 2.5, 420)
    ])
    validation = np.column_stack([
        rng.beta(3.0, 5.0, 150), rng.beta(6.0, 2.5, 150)
    ])
    common = dict(
        modes=8, rank=3, floor_mass=1e-3, learning_rate=1e-2,
        batch_size=128, validation_samples=validation, patience=6,
        validation_interval=2, initialization_noise=1e-3, seed=729,
    )
    scalar, _ = fit_linear_squared_tt_density(
        training, epochs=24, initialization="mixture", **common
    )
    local, history = sd.fit_locally_purified_linear_tt_density(
        training, epochs=8, scalar_warm_start_epochs=24,
        local_channels=[1, 2], scalar_initialization_noise=1e-3, **common
    )

    assert -np.mean(local.log_density(validation)) <= (
        -np.mean(scalar.log_density(validation)) + 3e-12
    )
    assert "local-purified" in history.optimizer
    np.testing.assert_array_equal(local.local_channels, [1, 2])
    chain = SampleDIRT(2, [local])
    refined, joint_history = fine_tune_linear_sample_dirt(
        chain,
        training,
        validation_samples=validation,
        epochs=4,
        batch_size=128,
        validation_interval=1,
        patience=4,
        learning_rate=1e-3,
        canonicalize=True,
        seed=730,
    )
    assert np.mean(refined.log_density(validation)) >= (
        np.mean(chain.log_density(validation)) - 1e-6
    )
    assert joint_history.optimizer == "joint-linear-adam"


def test_linear_mass_coordinates_make_exact_integral_euclidean():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(239)
    nodal = [
        torch.randn((1, 7, 3), generator=generator, dtype=torch.float64),
        torch.randn((3, 8, 2), generator=generator, dtype=torch.float64),
        torch.randn((2, 6, 1), generator=generator, dtype=torch.float64),
    ]
    probes = torch.rand(
        (300, 3), generator=generator, dtype=torch.float64
    )
    factors, inverses = sd._torch_linear_mass_factors(nodal)
    mass = sd._torch_linear_to_mass_cores(nodal, factors)

    assert float(sd._torch_tt_frobenius_sq(mass)) == pytest.approx(
        float(sd._torch_linear_root_second_moment(nodal)), rel=3e-13
    )
    recovered = sd._torch_linear_from_mass_cores(mass, inverses)
    assert torch.max(torch.abs(
        sd._torch_sample_linear_tt(recovered, probes)
        - sd._torch_sample_linear_tt(nodal, probes)
    )) < 3e-12

    before = sd._torch_sample_linear_tt(recovered, probes)
    sd._torch_right_orthogonalize(mass)
    canonical = sd._torch_linear_from_mass_cores(mass, inverses)
    assert torch.max(torch.abs(
        sd._torch_sample_linear_tt(canonical, probes) - before
    )) < 3e-12


@pytest.mark.parametrize(
    "optimizer", ("orthogonal-adam", "riemannian-adam")
)
def test_linear_mass_optimizers_improve_validation_and_remain_normalized(
    optimizer,
):
    pytest.importorskip("torch")
    rng = np.random.default_rng(240)
    training = np.column_stack([
        rng.beta(2.0, 5.0, 700), rng.beta(5.0, 2.0, 700)
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, 250), rng.beta(5.0, 2.0, 250)
    ])
    density, history = fit_linear_squared_tt_density(
        training,
        validation_samples=validation,
        modes=10,
        rank=2,
        epochs=12,
        batch_size=256,
        learning_rate=3e-3,
        patience=20,
        floor_mass=0.1,
        initialization="mixture",
        optimizer=optimizer,
        orthogonalization_interval=3,
        riemannian_retraction="psa",
        seed=241,
    )

    assert history.optimizer == f"linear-{optimizer}"
    assert np.all(np.isfinite(history.loss))
    assert min(history.validation_loss) < history.validation_loss[0]
    assert density.normalization > 0.0
    assert density.reference_floor_mass == pytest.approx(0.1, abs=2e-14)


def test_linear_squared_nll_fit_improves_heldout_likelihood():
    pytest.importorskip("torch")
    rng = np.random.default_rng(232)
    training = np.column_stack([
        rng.beta(2.0, 5.0, 1200), rng.beta(5.0, 2.0, 1200)
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, 400), rng.beta(5.0, 2.0, 400)
    ])
    density, history = fit_linear_squared_tt_density(
        training,
        validation_samples=validation,
        modes=16,
        rank=3,
        epochs=100,
        batch_size=256,
        learning_rate=2e-2,
        patience=20,
        initialization="uniform",
        seed=233,
    )

    assert history.optimizer == "linear-adam"
    assert min(history.validation_loss) < history.validation_loss[0] - 0.1
    assert np.mean(density.log_density(validation)) > 0.1
    assert -np.mean(density.log_density(validation)) == pytest.approx(
        min(history.validation_loss), rel=2e-12
    )


def test_linear_squared_fixed_background_fit_improves_mixture_likelihood():
    pytest.importorskip("torch")
    rng = np.random.default_rng(234)
    training = rng.beta(2.0, 6.0, size=(1600, 1))
    validation = rng.beta(2.0, 6.0, size=(500, 1))
    # The background and the uncorrected source component have equal weighted
    # density, so the identity layer starts at log(2) up to a constant.
    training_background = np.zeros(len(training))
    validation_background = np.zeros(len(validation))
    density, history = fit_linear_squared_tt_density(
        training,
        validation_samples=validation,
        mixture_background_log_ratio=training_background,
        validation_mixture_background_log_ratio=validation_background,
        modes=20,
        rank=2,
        epochs=120,
        batch_size=256,
        learning_rate=2e-2,
        patience=20,
        initialization="uniform",
        seed=235,
    )
    gain = (
        np.logaddexp(0.0, density.log_density(validation))
        - np.log(2.0)
    )

    assert min(history.validation_loss) < history.validation_loss[0] - 0.02
    assert gain.mean() > 0.02


def test_linear_squared_fixed_background_requires_matching_contexts():
    pytest.importorskip("torch")
    samples = np.linspace(0.01, 0.99, 20)[:, None]
    with pytest.raises(ValueError, match="provided together"):
        fit_linear_squared_tt_density(
            samples,
            validation_samples=samples,
            mixture_background_log_ratio=np.zeros(len(samples)),
            epochs=1,
        )


@pytest.mark.parametrize(
    ("fit", "density_type", "modes"),
    [
        (fit_linear_squared_tt_density, LinearSquaredTTDensity, 10),
        (fit_quadratic_squared_tt_density, QuadraticSquaredTTDensity, 11),
    ],
)
def test_continuous_squared_fit_has_scale_invariant_reference_floor(
    fit, density_type, modes,
):
    pytest.importorskip("torch")
    rng = np.random.default_rng(237)
    samples = np.column_stack([
        rng.beta(2.0, 5.0, 500), rng.beta(5.0, 2.0, 500)
    ])
    floor = 0.07
    density, _ = fit(
        samples,
        modes=modes,
        rank=2,
        floor_mass=floor,
        epochs=5,
        batch_size=128,
        learning_rate=1e-2,
        seed=238,
    )
    probes = rng.random((1000, 2))

    assert density.reference_floor_mass == pytest.approx(floor, abs=2e-14)
    assert np.min(density.density(probes)) >= floor - 2e-14

    scale = 17.0
    scaled_root = tt.vector.from_list([
        scale * density._cores[0], *density._cores[1:]
    ])
    scaled_second = scale * scale * density._root_second_moment
    scaled_gamma = floor * scaled_second / (1.0 - floor)
    scaled = density_type(scaled_root, gamma=scaled_gamma)
    assert scaled.reference_floor_mass == pytest.approx(floor, abs=2e-14)
    assert np.max(np.abs(
        scaled.density(probes) - density.density(probes)
    )) < 2e-12


def test_joint_linear_chain_nll_differentiates_exact_rosenblatt():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(239)

    def layer(seed):
        local = np.random.default_rng(seed)
        first = 0.03 * local.normal(size=(1, 7, 3))
        second = 0.03 * local.normal(size=(3, 7, 1))
        first[0, :, 0] += 1.0
        second[0, :, 0] += 1.0
        return LinearSquaredTTDensity(
            tt.vector.from_list([first, second]), gamma=1e-4
        )

    first = layer(240)
    second = PermutedTTDensity(layer(241), [1, 0])
    model = SampleDIRT(2, [first, second])
    points = rng.random((300, 2))
    validation = rng.random((120, 2))
    cores = [torch.tensor(core) for core in first._cores]
    canonical = [core.clone() for core in cores]
    sd._torch_right_orthogonalize(canonical, uniform_measure=True)
    assert torch.max(torch.abs(
        sd._torch_sample_linear_tt(canonical, torch.tensor(points))
        - sd._torch_sample_linear_tt(cores, torch.tensor(points))
    )).item() < 2e-12
    log_density, transformed = sd._torch_linear_log_density_rosenblatt(
        cores, torch.tensor(points), first.gamma, transform=True
    )

    assert np.max(np.abs(
        log_density.numpy() - first.log_density(points)
    )) < 2e-12
    assert np.max(np.abs(
        transformed.numpy() - first.rosenblatt(points)
    )) < 2e-12
    initial = np.mean(model.log_density(validation))
    fitted, history = fine_tune_linear_sample_dirt(
        model,
        points,
        validation_samples=validation,
        epochs=8,
        learning_rate=1e-3,
        batch_size=64,
        validation_interval=1,
        patience=8,
        dtype="float64",
        seed=242,
        orthogonalization_interval=2,
    )
    assert history.optimizer == "joint-linear-orthogonal-adam"
    assert len(history.regularization) == history.epochs
    assert np.mean(fitted.log_density(validation)) >= initial - 2e-12

    coefficient_fitted, _ = fine_tune_linear_sample_dirt(
        model,
        points,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-20,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=242,
        orthogonalization_interval=1,
        orthogonalization_metric="coefficient-uniform",
    )
    assert (
        np.mean(coefficient_fitted.log_density(validation))
        >= initial - 2e-12
    )

    _, uncanonicalized_history = fine_tune_linear_sample_dirt(
        model,
        points,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-3,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=242,
        canonicalize=False,
    )
    assert uncanonicalized_history.optimizer == "joint-linear-adam"

    _, regularized_history = fine_tune_linear_sample_dirt(
        model,
        points,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-20,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=242,
        sobolev_penalty=1e-2,
    )
    assert regularized_history.regularization[0] > 0.0
    plus = points.copy()
    minus = points.copy()
    plus[:, 0] = np.minimum(plus[:, 0] + 0.01, 0.999)
    minus[:, 0] = np.maximum(minus[:, 0] - 0.01, 0.001)
    _, curvature_history = fine_tune_linear_sample_dirt(
        model,
        points,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-4,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=242,
        curvature_plus_samples=plus,
        curvature_minus_samples=minus,
        curvature_targets=np.full(len(points), -0.25),
        curvature_weight=1e-2,
    )
    assert np.isfinite(curvature_history.regularization[0])
    assert curvature_history.regularization[0] > 0.0
    with pytest.raises(ValueError, match="requires plus/minus"):
        fine_tune_linear_sample_dirt(
            model,
            points,
            validation_samples=validation,
            epochs=1,
            curvature_weight=1e-2,
        )
    with pytest.raises(ValueError, match="positive and min_delta"):
        fine_tune_linear_sample_dirt(
            model,
            points,
            validation_samples=validation,
            epochs=1,
            sobolev_penalty=-1.0,
        )
    with pytest.raises(ValueError, match="orthogonalization_metric"):
        fine_tune_linear_sample_dirt(
            model,
            points,
            validation_samples=validation,
            epochs=1,
            orthogonalization_metric="unknown",
        )


def test_linear_chain_rank_enrichment_is_exact_and_trainable():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1240)
    cores = [
        rng.normal(size=(1, 6, 2)),
        rng.normal(size=(2, 7, 2)),
        rng.normal(size=(2, 5, 1)),
    ]
    base = LinearSquaredTTDensity(
        tt.vector.from_list(cores), gamma=1e-3
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    wrapped = PermutedTTDensity(
        ProbitOrthogonalTTDensity(base, rotation), [2, 0, 1]
    )
    model = SampleDIRT(3, [wrapped])
    points = rng.uniform(0.05, 0.95, size=(180, 3))
    enriched = enrich_linear_sample_dirt_ranks(
        model, 5, initialization_noise=2e-2, seed=1241
    )

    np.testing.assert_allclose(
        enriched.log_density(points), model.log_density(points), atol=2e-12
    )
    np.testing.assert_allclose(
        enriched.inverse(points), model.inverse(points), atol=3e-12
    )
    enriched_base = enriched.layers[0].base.base
    np.testing.assert_array_equal(enriched_base.ranks, [1, 5, 5, 1])
    # New output columns are live seeds, while matching next-core input rows
    # are exact zeros and hence kill every enriched path at checkpoint zero.
    assert np.linalg.norm(enriched_base._cores[0][:, :, 2:]) > 0.0
    assert np.linalg.norm(enriched_base._cores[1][2:, :, :]) == 0.0

    import torch

    parameters = [
        torch.tensor(core, dtype=torch.float64, requires_grad=True)
        for core in enriched_base._cores
    ]
    internal = enriched.layers[0].base._rotate(
        points[:, enriched.layers[0].permutation]
    )
    log_density, _ = sd._torch_linear_log_density_rosenblatt(
        parameters,
        torch.tensor(internal, dtype=torch.float64),
        enriched_base.gamma,
        transform=False,
    )
    (-log_density.mean()).backward()
    assert torch.linalg.norm(parameters[1].grad[2:, :, :2]).item() > 0.0

    fitted, _ = fine_tune_linear_sample_dirt(
        enriched,
        points[:128],
        validation_samples=points[128:],
        epochs=1,
        learning_rate=1e-20,
        batch_size=64,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=1242,
    )
    np.testing.assert_array_equal(
        fitted.layers[0].base.base.ranks, [1, 5, 5, 1]
    )
    np.testing.assert_allclose(
        fitted.log_density(points), model.log_density(points), atol=2e-11
    )


def test_linear_chain_rank_truncation_uses_exact_hat_mass_metric():
    rng = np.random.default_rng(1243)
    cores = [
        rng.normal(size=(1, 5, 4)),
        rng.normal(size=(4, 6, 4)),
        rng.normal(size=(4, 4, 1)),
    ]
    base = LinearSquaredTTDensity(
        tt.vector.from_list(cores), gamma=2e-3
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    model = SampleDIRT(3, [PermutedTTDensity(
        ProbitOrthogonalTTDensity(base, rotation), [2, 0, 1]
    )])
    points = rng.uniform(0.05, 0.95, size=(300, 3))

    unchanged = truncate_linear_sample_dirt_ranks(
        model, 4, tolerance=1e-14
    )
    np.testing.assert_allclose(
        unchanged.log_density(points), model.log_density(points), atol=2e-11
    )
    np.testing.assert_allclose(
        unchanged.inverse(points), model.inverse(points), atol=2e-11
    )

    compressed = truncate_linear_sample_dirt_ranks(
        model, 2, tolerance=1e-14
    )
    compressed_base = compressed.layers[0].base.base
    assert compressed_base.ranks.tolist() == [1, 2, 2, 1]
    assert compressed.stored_parameters < model.stored_parameters
    assert np.all(np.isfinite(compressed.log_density(points)))
    assert compressed.roundtrip_error(points) < 2e-11


def test_identity_probit_enablement_is_exact_selective_and_serializable(
    tmp_path,
):
    rng = np.random.default_rng(2239)
    cores = [
        rng.normal(size=(1, 4, 2)),
        rng.normal(size=(2, 4, 2)),
        rng.normal(size=(2, 4, 1)),
    ]
    base = LinearSquaredTTDensity(
        tt.vector.from_list(cores), gamma=2e-3
    )
    model = SampleDIRT(3, [
        PermutedTTDensity(base, [2, 0, 1]),
        PermutedTTDensity(base, [1, 2, 0]),
    ])
    points = rng.uniform(0.02, 0.98, size=(160, 3))
    enabled = enable_linear_sample_dirt_probit_rotations(
        model, layer_indices=[1], block_size=2
    )

    assert type(enabled.layers[0].base) is LinearSquaredTTDensity
    assert isinstance(enabled.layers[1].base, ProbitOrthogonalTTDensity)
    assert tuple(enabled.layers[1].base.rotation_block_sizes.tolist()) == (2, 1)
    np.testing.assert_allclose(
        enabled.layers[1].base.rotation, np.eye(3), atol=0.0
    )
    np.testing.assert_allclose(
        enabled.log_density(points), model.log_density(points), atol=3e-12
    )
    np.testing.assert_allclose(
        enabled.inverse(points), model.inverse(points), atol=4e-12
    )
    assert enabled.stored_parameters == model.stored_parameters + 5
    enabled.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    np.testing.assert_allclose(
        loaded.log_density(points), model.log_density(points), atol=3e-12
    )

    with pytest.raises(ValueError, match="no trainable rotation"):
        enable_linear_sample_dirt_probit_rotations(model, block_size=1)
    with pytest.raises(ValueError, match="unique valid"):
        enable_linear_sample_dirt_probit_rotations(
            model, layer_indices=[0, 0]
        )


def test_linear_chain_nested_mode_refinement_is_exact_and_trainable(tmp_path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2240)
    cores = [
        rng.normal(size=(1, 3, 2)),
        rng.normal(size=(2, 3, 2)),
        rng.normal(size=(2, 3, 1)),
    ]
    base = LinearSquaredTTDensity(
        tt.vector.from_list(cores), gamma=1e-3
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    first = PermutedTTDensity(
        ProbitOrthogonalTTDensity(base, rotation), [2, 0, 1]
    )
    second = PermutedTTDensity(
        ProbitOrthogonalTTDensity(base, rotation.T), [1, 2, 0]
    )
    model = SampleDIRT(3, [first, second])
    points = rng.uniform(0.04, 0.96, size=(180, 3))
    refined = refine_linear_sample_dirt_modes(
        model, 5, layer_indices=[1]
    )

    np.testing.assert_allclose(
        refined.log_density(points), model.log_density(points), atol=3e-12
    )
    np.testing.assert_allclose(
        refined.inverse(points), model.inverse(points), atol=4e-12
    )
    np.testing.assert_array_equal(
        refined.layers[1].base.base.modes, [5, 5, 5]
    )
    assert refined.stored_parameters > model.stored_parameters
    refined.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    np.testing.assert_allclose(
        loaded.log_density(points), model.log_density(points), atol=3e-12
    )

    parameters = [
        torch.tensor(core, dtype=torch.float64, requires_grad=True)
        for core in refined.layers[1].base.base._cores
    ]
    internal = refined.layers[1].base._rotate(
        points[:, refined.layers[1].permutation]
    )
    log_density, _ = sd._torch_linear_log_density_rosenblatt(
        parameters,
        torch.tensor(internal, dtype=torch.float64),
        refined.layers[1].base.base.gamma,
        transform=False,
    )
    (-log_density.mean()).backward()
    assert torch.linalg.norm(parameters[1].grad[:, 1::2, :]).item() > 0.0

    with pytest.raises(ValueError, match="contain all old uniform nodes"):
        refine_linear_sample_dirt_modes(model, 4)

    local = refine_linear_sample_dirt_modes(
        model, 5, layer_indices=[1], coordinate_indices=[0]
    )
    np.testing.assert_allclose(
        local.log_density(points), model.log_density(points), atol=3e-12
    )
    np.testing.assert_allclose(
        local.inverse(points), model.inverse(points), atol=4e-12
    )
    # Layer 1 has permutation [1, 2, 0], so physical coordinate zero is the
    # last internal TT core.
    np.testing.assert_array_equal(
        local.layers[1].base.base.modes, [3, 3, 5]
    )
    np.testing.assert_array_equal(local.layers[1].modes, [5, 3, 3])
    with pytest.raises(ValueError, match="physical coordinates"):
        refine_linear_sample_dirt_modes(
            model, 5, coordinate_indices=[0, 0]
        )


def test_cross_fitted_mode_refinement_score_finds_signal_and_is_gauge_invariant():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2250)
    uniform = LinearSquaredTTDensity(
        tt.vector.from_list([
            np.ones((1, 3, 1)),
            np.ones((1, 3, 1)),
        ]),
        gamma=1e-3,
    )
    model = SampleDIRT(2, [uniform])

    # Only coordinate zero contains a component outside the modes-3 linear
    # FE space.  Rejection sampling avoids relying on a fitted teacher.
    accepted = []
    while len(accepted) < 2048:
        candidates = rng.random(4096)
        keep = rng.random(4096) < (
            1.0 + 0.8 * np.cos(4.0 * np.pi * candidates)
        ) / 1.8
        accepted.extend(candidates[keep].tolist())
    samples = np.column_stack([
        np.asarray(accepted[:2048]), rng.random(2048)
    ])
    scores = score_linear_sample_dirt_mode_refinement(
        model,
        samples,
        5,
        layer_index=0,
        replicates=3,
        batch_size=512,
        confidence_z=1.0,
        dtype="float64",
        seed=2251,
    )
    by_coordinate = {score.physical_coordinate: score for score in scores}
    assert by_coordinate[0].lower_confidence_bound > 0.5
    assert by_coordinate[0].mean_score > 10.0 * by_coordinate[1].mean_score
    assert by_coordinate[0].added_parameters == 2

    # An arbitrary invertible TT gauge changes raw core gradients but not the
    # canonicalized environment-metric score.
    first = rng.normal(size=(1, 3, 2))
    second = rng.normal(size=(2, 3, 1))
    gauge = np.array([[1.4, 0.2], [-0.3, 0.8]])
    gauged_first = np.einsum("aib,bc->aic", first, gauge)
    gauged_second = np.einsum(
        "ab,bic->aic", np.linalg.inv(gauge), second
    )

    def make(candidate):
        return SampleDIRT(2, [LinearSquaredTTDensity(
            tt.vector.from_list(candidate), gamma=1e-3
        )])

    common = dict(
        modes=5,
        layer_index=0,
        replicates=2,
        batch_size=256,
        dtype="float64",
        seed=2252,
    )
    original_scores = score_linear_sample_dirt_mode_refinement(
        make([first, second]), samples[:1024], **common
    )
    gauged_scores = score_linear_sample_dirt_mode_refinement(
        make([gauged_first, gauged_second]), samples[:1024], **common
    )
    original_by_coordinate = {
        score.physical_coordinate: score for score in original_scores
    }
    gauged_by_coordinate = {
        score.physical_coordinate: score for score in gauged_scores
    }
    for coordinate in range(2):
        np.testing.assert_allclose(
            original_by_coordinate[coordinate].cross_scores,
            gauged_by_coordinate[coordinate].cross_scores,
            rtol=2e-11,
            atol=2e-11,
        )


def test_joint_linear_chain_supports_fixed_probit_rotations():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1242)

    def layer(seed):
        local = np.random.default_rng(seed)
        cores = [
            local.normal(size=(1, 6, 3)),
            local.normal(size=(3, 6, 1)),
        ]
        return LinearSquaredTTDensity(
            tt.vector.from_list(cores), gamma=1e-3
        )

    angle = 0.31
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    first = ProbitOrthogonalTTDensity(layer(1243), rotation)
    second = PermutedTTDensity(
        ProbitOrthogonalTTDensity(layer(1244), rotation.T), [1, 0]
    )
    model = SampleDIRT(2, [first, second])
    training = rng.uniform(0.02, 0.98, size=(96, 2))
    validation = rng.uniform(0.02, 0.98, size=(48, 2))

    fitted, history = fine_tune_linear_sample_dirt(
        model,
        training,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-20,
        batch_size=48,
        validation_interval=1,
        patience=1,
        dtype="float64",
        seed=1245,
    )

    assert history.optimizer == "joint-linear-orthogonal-adam"
    assert isinstance(fitted.layers[0], ProbitOrthogonalTTDensity)
    assert isinstance(fitted.layers[1], PermutedTTDensity)
    assert isinstance(fitted.layers[1].base, ProbitOrthogonalTTDensity)
    np.testing.assert_allclose(
        fitted.layers[0].rotation, rotation, atol=2e-15
    )
    np.testing.assert_allclose(
        fitted.log_density(validation), model.log_density(validation),
        atol=2e-11,
    )

    block_fitted, block_history = fine_tune_linear_sample_dirt(
        model,
        training,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-20,
        batch_size=48,
        validation_interval=1,
        patience=1,
        dtype="float64",
        core_layer_indices=[1],
        seed=1245,
    )
    assert block_history.optimizer == (
        "joint-linear-orthogonal-adam[fixed-prefix=1]"
    )
    for old, new in zip(
        model.layers[0].base._cores, block_fitted.layers[0].base._cores
    ):
        np.testing.assert_array_equal(new, old)
    np.testing.assert_allclose(
        block_fitted.log_density(validation), model.log_density(validation),
        atol=2e-11,
    )
    with pytest.raises(ValueError, match="unique and valid"):
        fine_tune_linear_sample_dirt(
            model,
            training,
            validation_samples=validation,
            epochs=1,
            core_layer_indices=[1, 1],
        )

    fitted32, _ = fine_tune_linear_sample_dirt(
        model,
        training,
        validation_samples=validation,
        epochs=1,
        learning_rate=1e-20,
        batch_size=48,
        validation_interval=1,
        patience=1,
        dtype="float32",
        seed=1245,
    )
    for candidate in (fitted32.layers[0], fitted32.layers[1].base):
        np.testing.assert_allclose(
            candidate.rotation.T @ candidate.rotation,
            np.eye(2),
            atol=2e-14,
        )

    rotated, rotation_history = fine_tune_linear_sample_dirt(
        model,
        training,
        validation_samples=validation,
        epochs=3,
        learning_rate=1e-3,
        batch_size=48,
        validation_interval=1,
        patience=3,
        dtype="float64",
        optimize_cores=False,
        optimize_rotations=True,
        seed=1246,
    )
    assert rotation_history.optimizer == "joint-linear-probit-rotation-adam"
    assert np.mean(rotated.log_density(validation)) >= (
        np.mean(model.log_density(validation)) - 2e-11
    )
    for candidate in (rotated.layers[0], rotated.layers[1].base):
        np.testing.assert_allclose(
            candidate.rotation.T @ candidate.rotation,
            np.eye(2),
            atol=2e-14,
        )


def test_quadratic_squared_density_exact_gram_roundtrip_and_fit(tmp_path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(234)
    root = tt.vector.from_list([
        rng.normal(size=(1, 8, 3)),
        rng.normal(size=(3, 9, 1)),
    ])
    density = QuadraticSquaredTTDensity(root, gamma=1e-3)
    torch_cores = [torch.tensor(core) for core in density._cores]
    grid = np.linspace(0.0, 1.0, 501)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    values = density.density(points).reshape(len(grid), len(grid))
    integral = np.trapezoid(np.trapezoid(values, grid, axis=1), grid)
    generated = density.sample(300, seed=235)

    assert integral == pytest.approx(1.0, rel=3e-5)
    assert float(sd._torch_quadratic_root_second_moment(
        torch_cores
    )) == pytest.approx(density._root_second_moment, rel=2e-13)
    assert np.max(np.abs(
        sd._torch_sample_quadratic_tt(
            torch_cores, torch.tensor(generated)
        ).numpy() - density.root_values(generated)
    )) < 2e-12
    assert np.max(np.abs(
        density.inverse_rosenblatt(density.rosenblatt(generated)) - generated
    )) < 3e-12
    assert np.max(np.abs(
        density.rosenblatt_device(
            generated, device="cpu", dtype="float64", batch_size=61
        ) - density.rosenblatt(generated)
    )) < 2e-12
    model = SampleDIRT(2, [PermutedTTDensity(density, [1, 0])])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert np.max(np.abs(
        loaded.log_density(generated) - model.log_density(generated)
    )) < 2e-12

    training = np.column_stack([
        rng.beta(2.0, 5.0, 900), rng.beta(5.0, 2.0, 900)
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, 300), rng.beta(5.0, 2.0, 300)
    ])
    fitted, history = fit_quadratic_squared_tt_density(
        training,
        validation_samples=validation,
        modes=14,
        rank=3,
        epochs=80,
        batch_size=256,
        learning_rate=2e-2,
        patience=20,
        initialization="uniform",
        seed=236,
    )
    assert history.optimizer == "quadratic-adam"
    assert min(history.validation_loss) < history.validation_loss[0] - 0.1
    assert np.mean(fitted.log_density(validation)) > 0.1


def test_incremental_rosenblatt_roundtrip():
    density = SquaredTTDensity(_nontrivial_root(), gamma=1e-3)
    rng = np.random.default_rng(1)
    uniform = rng.random((200, 3))
    samples = density.inverse_rosenblatt(uniform)
    recovered = density.rosenblatt(samples)

    assert np.max(np.abs(recovered - uniform)) < 2e-14
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_permuted_layer_is_exact_and_serializable(tmp_path):
    base = SquaredTTDensity(_nontrivial_root(), gamma=1e-3)
    permutation = np.array([2, 0, 1])
    density = PermutedTTDensity(base, permutation)
    rng = np.random.default_rng(71)
    points = rng.random((200, 3))

    transported = density.inverse_rosenblatt(points)
    assert np.max(np.abs(density.rosenblatt(transported) - points)) < 3e-14
    accelerated = base.rosenblatt_device(
        points[:, permutation], device="cpu", dtype="float64", batch_size=37
    )[:, np.argsort(permutation)]
    assert np.max(np.abs(accelerated - density.rosenblatt(points))) < 3e-14
    assert np.max(
        np.abs(density.log_density(points) - base.log_density(points[:, permutation]))
    ) < 1e-14
    assert np.max(np.abs(
        base.log_density_device(
            points[:, permutation], device="cpu", dtype="float64", batch_size=41
        ) - base.log_density(points[:, permutation])
    )) < 1e-14
    model = SampleDIRT(3, [density])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert isinstance(loaded.layers[0], PermutedTTDensity)
    assert np.array_equal(loaded.layers[0].permutation, permutation)
    assert np.max(np.abs(loaded.forward(points) - model.forward(points))) < 1e-14
    assert np.max(np.abs(
        loaded.log_density(points, device="cpu") - model.log_density(points)
    )) < 1e-14


def test_direct_density_uses_linear_marginals_and_exact_second_moment(tmp_path):
    rng = np.random.default_rng(101)
    dense = 0.2 + rng.random((4, 5, 3))
    density = DirectTTDensity(tt.vector(dense - 1.0, eps=1e-14))
    grid = np.stack(
        np.meshgrid(
            (np.arange(4) + 0.5) / 4,
            (np.arange(5) + 0.5) / 5,
            (np.arange(3) + 0.5) / 3,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    normalized = dense / dense.mean()
    uniform = rng.random((300, 3))

    assert np.max(
        np.abs(density.density(grid).reshape(dense.shape) - normalized)
    ) < 3e-14
    assert density.model_l2_norm_sq() == pytest.approx(
        float(np.mean(normalized * normalized)), rel=3e-13
    )
    assert density.conditional_clipping_fraction(grid) == 0.0
    assert np.max(
        np.abs(density.rosenblatt(density.inverse_rosenblatt(uniform)) - uniform)
    ) < 3e-14
    model = SampleDIRT(3, [density])
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert isinstance(loaded.layers[0], DirectTTDensity)
    assert np.max(np.abs(loaded.log_density(grid) - model.log_density(grid))) < 1e-14


def test_deep_composition_roundtrip_and_serialization(tmp_path):
    first = SquaredTTDensity(_nontrivial_root(), gamma=2e-3)
    second = SquaredTTDensity(
        tt.vector.from_list([
            np.array([[[1.0], [2.0], [0.5], [1.5], [0.8]]]),
            np.array([[[0.4], [1.0], [2.0], [0.7]]]),
            np.array([[[1.0], [0.5], [1.5], [0.9], [2.0], [0.6]]]),
        ]),
        gamma=1e-3,
    )
    model = SampleDIRT(3, [first, second])
    rng = np.random.default_rng(2)
    points = rng.random((100, 3))

    assert model.roundtrip_error(points) < 3e-14
    model.save(tmp_path)
    loaded = SampleDIRT.load(tmp_path)
    assert loaded.stored_parameters == model.stored_parameters
    assert np.max(np.abs(loaded.forward(points) - model.forward(points))) < 1e-14
    assert np.max(np.abs(loaded.log_density(points) - model.log_density(points))) < 1e-14
    manual = first.log_density(points)
    manual += second.log_density(first.rosenblatt(points))
    assert np.max(np.abs(model.log_density(points) - manual)) < 1e-14
    transformed, fused_log_density = model.inverse_and_log_density(points)
    assert np.max(np.abs(transformed - model.inverse(points))) < 1e-14
    assert np.max(np.abs(fused_log_density - manual)) < 1e-14


def test_sample_only_fit_moves_product_moments_toward_target():
    pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    target = np.column_stack([
        rng.beta(2.0, 5.0, size=3000),
        rng.beta(5.0, 2.0, size=3000),
    ])
    model = SampleDIRT(2)
    history = model.fit_layer(
        target,
        estimator="squared",
        modes=12,
        rank=2,
        gamma=1e-3,
        epochs=250,
        learning_rate=4e-2,
        seed=3,
    )
    generated = model.sample(5000, seed=12)
    target_mean = target.mean(axis=0)
    uniform_error = np.linalg.norm(target_mean - 0.5)
    model_error = np.linalg.norm(target_mean - generated.mean(axis=0))

    assert history.loss[-1] < history.loss[0] - 0.05
    assert model_error < 0.35 * uniform_error
    assert model.roundtrip_error(generated[:100]) < 3e-14


def test_squared_product_initialization_uses_validation_checkpoint():
    pytest.importorskip("torch")
    rng = np.random.default_rng(711)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=900),
        rng.beta(5.0, 2.0, size=900),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, size=300),
        rng.beta(5.0, 2.0, size=300),
    ])
    model = SampleDIRT(2)

    history = model.fit_layer(
        training,
        estimator="squared",
        validation_samples=validation,
        initialization="product",
        initialization_noise=1e-3,
        modes=8,
        rank=2,
        gamma=1e-5,
        epochs=8,
        learning_rate=1e-2,
        batch_size=256,
        patience=4,
        seed=712,
    )

    assert len(history.validation_loss) == history.epochs + 1
    assert 0 <= history.best_epoch <= history.epochs
    assert np.all(np.isfinite(model.log_density(validation)))
    assert np.min(model.density(validation)) > 0.0


def test_optimized_torch_contractions_equal_dense_cell_objective():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(21)
    cores_np = [rng.normal(size=(1, 5, 3)), rng.normal(size=(3, 6, 1))]
    cores = [torch.tensor(core, dtype=torch.float64) for core in cores_np]
    indices_np = np.stack(
        [rng.integers(0, 5, 200), rng.integers(0, 6, 200)], axis=1
    )
    unique, weights_np = sd._compress_empirical_cells(indices_np)
    indices = torch.tensor(unique, dtype=torch.long)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    gamma = 2e-3

    loss, h2, z = sd._torch_density_objective(
        cores, indices, weights, gamma
    )
    root = tt.vector.from_list(cores_np)
    density = SquaredTTDensity(root, gamma=gamma)
    values = density.density((unique + 0.5) / np.array([5, 6]))
    expected_loss = 0.5 * density.model_l2_norm_sq() - weights_np @ values

    assert float(z) == pytest.approx(density.normalization, rel=2e-13)
    assert float(h2) == pytest.approx(density.model_l2_norm_sq(), rel=2e-13)
    assert float(loss) == pytest.approx(expected_loss, rel=2e-13)


def test_squared_nll_uses_the_exact_tt_partition_function():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(221)
    cores_np = [rng.normal(size=(1, 4, 3)), rng.normal(size=(3, 5, 1))]
    cores = [torch.tensor(core, dtype=torch.float64) for core in cores_np]
    indices_np = np.stack(
        [rng.integers(0, 4, 180), rng.integers(0, 5, 180)], axis=1
    )
    unique, weights_np = sd._compress_empirical_cells(indices_np)
    indices = torch.tensor(unique, dtype=torch.long)
    weights = torch.tensor(weights_np, dtype=torch.float64)
    gamma = 1e-3

    loss, normalization = sd._torch_nll_objective(
        cores, indices, weights, gamma
    )
    density = SquaredTTDensity(tt.vector.from_list(cores_np), gamma=gamma)
    values = density.density((unique + 0.5) / np.array([4, 5]))

    assert float(normalization) == pytest.approx(
        density.normalization, rel=2e-13
    )
    assert float(loss) == pytest.approx(
        -float(weights_np @ np.log(values)), rel=2e-13
    )


def test_uniform_measure_orthogonalization_avoids_high_dimensional_overflow():
    torch = pytest.importorskip("torch")
    cores = [torch.ones((1, 64, 1), dtype=torch.float32) for _ in range(43)]

    sd._torch_right_orthogonalize(cores, uniform_measure=True)
    second_moment = sd._torch_root_second_moment(cores)

    assert all(torch.isfinite(core).all() for core in cores)
    assert float(second_moment) == pytest.approx(1.0, rel=2e-6)


def test_squared_fit_tapers_unattainable_boundary_ranks():
    pytest.importorskip("torch")
    rng = np.random.default_rng(224)
    points = rng.random((200, 3))

    density, _ = sd.fit_squared_tt_density(
        points,
        modes=4,
        rank=8,
        epochs=1,
        objective="nll",
        initialization="mixture",
        seed=225,
    )

    assert density.ranks.tolist() == [1, 4, 4, 1]


def test_squared_nll_fit_improves_heldout_likelihood():
    pytest.importorskip("torch")
    rng = np.random.default_rng(222)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=1200),
        rng.beta(5.0, 2.0, size=1200),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, size=400),
        rng.beta(5.0, 2.0, size=400),
    ])
    density, history = sd.fit_squared_tt_density(
        training,
        validation_samples=validation,
        validation_metric="log-likelihood",
        objective="nll",
        optimizer="adam",
        modes=12,
        rank=2,
        gamma=1e-5,
        epochs=80,
        batch_size=256,
        learning_rate=2e-2,
        patience=15,
        initialization="uniform",
        initialization_noise=2e-2,
        seed=223,
    )

    assert min(history.validation_loss) < history.validation_loss[0] - 0.1
    assert np.mean(density.log_density(validation)) > 0.1
    assert np.all(np.isfinite(history.loss))


def test_squared_adam_patience_counts_validation_checks_not_minibatches():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2231)
    training = rng.random((120, 2))
    validation = rng.random((40, 2))

    _, history = sd.fit_squared_tt_density(
        training,
        validation_samples=validation,
        validation_metric="log-likelihood",
        objective="nll",
        optimizer="adam",
        modes=4,
        rank=2,
        epochs=7,
        batch_size=32,
        learning_rate=0.0,
        patience=2,
        validation_interval=3,
        initialization="uniform",
        seed=2232,
    )

    # Initial validation plus checks after minibatches 3 and 6.  With the old
    # per-minibatch check this stopped after only two optimizer steps.
    assert history.epochs == 6
    assert len(history.validation_loss) == 3
    assert history.converged


def test_centered_rank_enrichment_reaches_requested_fixed_rank():
    modes = np.full(4, 4, dtype=np.int64)
    zero = tt.ones(4, 4) * 0.0

    enriched = sd._enrich_correction_ranks(
        zero, modes, rank=4, noise=1e-2, seed=2233
    )

    assert enriched.r.tolist() == [1, 4, 4, 4, 1]
    assert sd._mean_numpy(sd._as_numpy_cores(enriched)) == pytest.approx(
        0.0, abs=1e-12
    )
    assert sd._second_moment_numpy(
        sd._as_numpy_cores(enriched)
    ) == pytest.approx(1e-4, rel=2e-8)


def test_centered_adam_patience_counts_validation_checks():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2234)
    training = rng.random((120, 2))
    validation = rng.random((40, 2))

    _, history = sd.fit_centered_tt_density(
        training,
        validation_samples=validation,
        modes=4,
        rank=2,
        epochs=7,
        batch_size=32,
        learning_rate=0.0,
        patience=2,
        validation_interval=3,
        rank_enrichment_noise=1e-3,
        seed=2235,
    )

    assert history.epochs == 6
    assert len(history.validation_loss) == 3
    assert history.converged


def test_cellwise_fixed_floor_nll_is_scale_invariant():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(231)
    cores = [
        torch.tensor(rng.normal(size=(1, 5, 2)), dtype=torch.float64),
        torch.tensor(rng.normal(size=(2, 6, 1)), dtype=torch.float64),
    ]
    indices = torch.tensor(
        np.column_stack([rng.integers(5, size=40), rng.integers(6, size=40)]),
        dtype=torch.long,
    )
    weights = torch.full((40,), 1.0 / 40, dtype=torch.float64)

    first = sd._torch_nll_objective(
        cores, indices, weights, gamma=1e-4, floor_mass=0.03
    )[0]
    scaled = [cores[0] * 19.0, cores[1]]
    second = sd._torch_nll_objective(
        scaled, indices, weights, gamma=1e-4, floor_mass=0.03
    )[0]

    assert float(first) == pytest.approx(float(second), abs=2e-13)


@pytest.mark.parametrize(
    ("optimizer", "options"),
    [
        ("adam", {"epochs": 15, "batch_size": 128, "learning_rate": 3e-2}),
        (
            "riemannian-sgd",
            {"epochs": 12, "batch_size": 128, "learning_rate": 3e-2},
        ),
        ("als", {"epochs": 1, "als_inner_steps": 2, "learning_rate": 1.0}),
        (
            "kaczmarz",
            {
                "epochs": 1,
                "batch_size": 128,
                "als_inner_steps": 2,
                "learning_rate": 0.2,
            },
        ),
    ],
)
def test_cellwise_fixed_floor_supported_by_stochastic_and_sweep_optimizers(
    optimizer, options,
):
    pytest.importorskip("torch")
    rng = np.random.default_rng(232)
    samples = np.column_stack([
        rng.beta(2.0, 5.0, size=400),
        rng.beta(5.0, 2.0, size=400),
    ])

    density, history = sd.fit_squared_tt_density(
        samples,
        modes=6,
        rank=2,
        floor_mass=0.025,
        objective="nll",
        optimizer=optimizer,
        initialization="mixture",
        seed=233,
        **options,
    )

    assert density.reference_floor_mass == pytest.approx(0.025, abs=2e-12)
    assert np.all(np.isfinite(history.loss))
    assert np.min(density.density(rng.random((200, 2)))) >= 0.025 - 1e-12


def test_block_kaczmarz_nll_is_finite_and_improves_validation():
    pytest.importorskip("torch")
    rng = np.random.default_rng(226)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=1000),
        rng.beta(5.0, 2.0, size=1000),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, size=300),
        rng.beta(5.0, 2.0, size=300),
    ])
    density, history = sd.fit_squared_tt_density(
        training,
        validation_samples=validation,
        validation_metric="log-likelihood",
        objective="nll",
        optimizer="kaczmarz",
        modes=8,
        rank=2,
        gamma=1e-4,
        epochs=6,
        batch_size=256,
        learning_rate=0.2,
        als_inner_steps=3,
        patience=3,
        initialization="mixture",
        seed=227,
    )

    assert history.optimizer == "kaczmarz"
    assert np.all(np.isfinite(history.loss))
    assert np.mean(density.log_density(validation)) > 0.1


def test_mass_orthogonal_als_supports_exact_nll():
    pytest.importorskip("torch")
    rng = np.random.default_rng(228)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=800),
        rng.beta(5.0, 2.0, size=800),
    ])
    density, history = sd.fit_squared_tt_density(
        training,
        objective="nll",
        optimizer="als",
        modes=8,
        rank=2,
        gamma=1e-4,
        epochs=3,
        learning_rate=1.0,
        als_inner_steps=4,
        initialization="mixture",
        seed=229,
    )

    assert history.optimizer == "als"
    assert np.all(np.isfinite(history.loss))
    assert history.loss[-1] < history.loss[0] - 0.02
    assert np.mean(density.log_density(training)) > 0.1


def test_centered_correction_objective_equals_dense_quadratic_loss():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(121)
    cores_np = [rng.normal(size=(1, 4, 3)), rng.normal(size=(3, 5, 1))]
    cores = [torch.tensor(core, dtype=torch.float64) for core in cores_np]
    indices_np = np.stack(
        [rng.integers(0, 4, 300), rng.integers(0, 5, 300)], axis=1
    )
    unique, weights_np = sd._compress_empirical_cells(indices_np)
    indices = torch.tensor(unique, dtype=torch.long)
    weights = torch.tensor(weights_np, dtype=torch.float64)

    loss, second, mean = sd._torch_correction_objective(
        cores, indices, weights
    )
    correction = tt.vector.from_list(cores_np)
    dense = np.asarray(correction.full())
    sampled = np.array([dense[tuple(index)] for index in unique])

    assert float(mean) == pytest.approx(float(dense.mean()), rel=2e-13)
    assert float(second) == pytest.approx(float(np.mean(dense * dense)), rel=2e-13)
    assert float(loss) == pytest.approx(
        0.5 * float(np.mean(dense * dense))
        - float(weights_np @ sampled)
        + float(dense.mean()),
        rel=2e-13,
    )


def test_centered_fit_is_positive_compressed_and_moves_product_moments():
    pytest.importorskip("torch")
    rng = np.random.default_rng(122)
    target = np.column_stack([
        rng.beta(2.0, 5.0, size=1800),
        rng.beta(5.0, 2.0, size=1800),
    ])
    density, history = fit_centered_tt_density(
        target,
        modes=8,
        rank=4,
        epochs=8,
        learning_rate=3e-3,
        seed=5,
        initialization_coarse_bins=2,
        initialization_tolerance=5e-2,
        projection_tolerance=2e-2,
        projection_rank=5,
        projection_sweeps=5,
    )
    generated = density.sample(4000, seed=6)

    assert history.optimizer == "centered-adam"
    assert history.validation_loss
    assert history.correction_ranks[0] == history.correction_ranks[-1] == 1
    assert history.projected_ranks[0] == history.projected_ranks[-1] == 1
    assert history.negative_ratio_fraction == 0.0
    assert np.min(density.density(rng.random((1000, 2)))) > 0.0
    assert np.linalg.norm(generated.mean(0) - target.mean(0)) < 0.03


def test_centered_kernel_objective_matches_dense_exact_contractions():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17001)
    cores_np = [
        rng.normal(size=(1, 3, 2)),
        rng.normal(size=(2, 4, 1)),
    ]
    cores = [torch.tensor(core, dtype=torch.float64) for core in cores_np]
    target = rng.random((7, 2))
    weights_np = rng.random(7)
    weights_np /= weights_np.sum()
    weights = torch.tensor(weights_np, dtype=torch.float64)
    bandwidths = (0.17, 0.53)
    matrix_sets = []
    feature_sets = []
    expected = expected_quadratic = expected_uniform = 0.0
    dense = np.asarray(tt.vector.from_list(cores_np).full()).reshape(-1)
    for bandwidth in bandwidths:
        matrices_np = [
            sd._rbf_cell_kernel_matrix(3, bandwidth),
            sd._rbf_cell_kernel_matrix(4, bandwidth),
        ]
        features_np = [
            sd._rbf_cell_target_features(target[:, 0], 3, bandwidth),
            sd._rbf_cell_target_features(target[:, 1], 4, bandwidth),
        ]
        matrix_sets.append([
            torch.tensor(value, dtype=torch.float64)
            for value in matrices_np
        ])
        feature_sets.append([
            torch.tensor(value, dtype=torch.float64)
            for value in features_np
        ])
        product_matrix = np.kron(matrices_np[0], matrices_np[1])
        quadratic = float(dense @ product_matrix @ dense)
        target_features = np.einsum(
            "bi,bj->bij", features_np[0], features_np[1]
        ).reshape(len(target), -1)
        target_cross = float(weights_np @ (target_features @ dense))
        uniform_vector = np.kron(
            matrices_np[0].sum(axis=1),
            matrices_np[1].sum(axis=1),
        )
        uniform_cross = float(uniform_vector @ dense)
        expected += 0.5 * quadratic - target_cross + uniform_cross
        expected_quadratic += quadratic
        expected_uniform += uniform_cross
    expected /= len(bandwidths)
    expected_quadratic /= len(bandwidths)
    expected_uniform /= len(bandwidths)

    value, quadratic, uniform = sd._torch_kernel_correction_objective(
        cores, feature_sets, weights, matrix_sets
    )

    assert float(value) == pytest.approx(expected, rel=2e-12, abs=2e-12)
    assert float(quadratic) == pytest.approx(expected_quadratic, rel=2e-12)
    assert float(uniform) == pytest.approx(expected_uniform, rel=2e-12)


def test_centered_kernel_fit_reduces_held_out_mmd_objective():
    pytest.importorskip("torch")
    rng = np.random.default_rng(17002)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=600),
        rng.beta(5.0, 2.0, size=600),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, size=200),
        rng.beta(5.0, 2.0, size=200),
    ])

    density, history = fit_centered_tt_density(
        training,
        validation_samples=validation,
        modes=6,
        rank=3,
        epochs=12,
        learning_rate=1e-2,
        batch_size=300,
        objective="kernel",
        kernel_bandwidths=(0.2, 0.5),
        initialization="uniform",
        rank_enrichment_noise=1e-2,
        patience=12,
        min_delta=0.0,
        positivity_margin=1e-3,
        seed=17003,
    )

    assert history.optimizer == "centered-kernel-adam"
    assert history.validation_loss[-1] < history.validation_loss[0]
    assert np.all(np.isfinite(density.log_density(validation[:32])))


def test_centered_kernel_als_solves_exact_block_quadratics():
    pytest.importorskip("torch")
    rng = np.random.default_rng(17004)
    training = np.column_stack([
        rng.beta(2.0, 5.0, size=400),
        rng.beta(5.0, 2.0, size=400),
    ])
    validation = np.column_stack([
        rng.beta(2.0, 5.0, size=150),
        rng.beta(5.0, 2.0, size=150),
    ])

    density, history = fit_centered_tt_density(
        training,
        validation_samples=validation,
        modes=5,
        rank=2,
        epochs=1,
        optimizer="als",
        objective="kernel",
        kernel_bandwidths=(0.2, 0.5),
        initialization="uniform",
        rank_enrichment_noise=1e-2,
        als_regularization=1e-6,
        positivity_margin=1e-3,
        min_delta=0.0,
        seed=17005,
    )

    assert history.optimizer == "centered-kernel-als"
    assert history.function_calls == 4
    assert history.validation_loss[-1] < history.validation_loss[0] - 0.05
    assert np.all(np.isfinite(density.log_density(validation[:32])))


def test_fit_layer_defaults_to_direct_centered_correction():
    pytest.importorskip("torch")
    samples = np.random.default_rng(703).beta(2.0, 3.0, size=(500, 2))
    model = SampleDIRT(2)

    history = model.fit_layer(
        samples,
        modes=8,
        rank=2,
        epochs=0,
        seed=704,
    )

    assert isinstance(model.layers[0], DirectTTDensity)
    assert history.optimizer == "centered-adam"


def test_centered_auto_initialization_scales_to_tabular_dimension():
    pytest.importorskip("torch")
    rng = np.random.default_rng(705)
    samples = rng.beta(2.0, 3.0, size=(500, 21))

    density, history = fit_centered_tt_density(
        samples,
        modes=4,
        rank=2,
        epochs=0,
        initialization="auto",
        initialization_coarse_bins=2,
        initialization_tolerance=5e-2,
        seed=706,
    )

    assert density.d == 21
    assert max(density.ranks) <= 2
    assert history.optimizer == "centered-adam"
    assert np.all(np.isfinite(density.log_density(rng.random((32, 21)))))


def test_centered_uniform_initialization_starts_at_controlled_l2_scale():
    pytest.importorskip("torch")
    rng = np.random.default_rng(1705)
    samples = rng.beta(8.0, 12.0, size=(600, 40))

    density, history = fit_centered_tt_density(
        samples,
        modes=4,
        rank=3,
        epochs=0,
        initialization="uniform",
        rank_enrichment_noise=2e-3,
        seed=1706,
    )

    assert density.d == 40
    assert max(history.initial_ranks) == 3
    assert history.l2_norm_sq[0] == pytest.approx(4e-6, rel=2e-8)
    assert abs(history.normalization[0] - 1.0) < 1e-12
    assert np.all(np.isfinite(density.log_density(rng.random((32, 40)))))


def test_direct_positivity_scaling_is_uniform_mixture_and_preserves_rank():
    correction = tt.vector(np.array([[-3.0, 2.0], [1.0, 0.0]]), eps=1e-14)
    grid = np.stack(
        np.meshgrid([0.25, 0.75], [0.25, 0.75], indexing="ij"), axis=-1
    ).reshape(-1, 2)

    scaled, scale = sd._scale_correction_for_positive_conditionals(
        correction, grid, margin=1e-4
    )
    density = DirectTTDensity(scaled)

    assert 0.0 < scale < 1.0
    assert np.array_equal(scaled.r, correction.r)
    assert density.raw_density(grid).min() >= 1e-4 - 1e-12
    assert density.model_l2_norm_sq() == pytest.approx(
        1.0 + scale * scale * np.mean(np.asarray(correction.full()) ** 2)
    )


def test_centered_als_uses_closed_form_quadratic_core_updates():
    pytest.importorskip("torch")
    rng = np.random.default_rng(124)
    target = np.column_stack([
        rng.beta(2.0, 4.0, size=1600),
        rng.beta(4.0, 2.0, size=1600),
        rng.beta(3.0, 3.0, size=1600),
    ])
    _, history = fit_centered_tt_density(
        target,
        modes=8,
        rank=2,
        epochs=1,
        optimizer="als",
        als_relaxation=1.0,
        als_regularization=0.0,
        seed=8,
        initialization_coarse_bins=2,
        initialization_tolerance=5e-2,
    )

    assert history.optimizer == "centered-als"
    assert history.function_calls == 5  # one forward/backward three-core sweep
    assert history.loss[-1] < history.loss[0]


def test_high_dimensional_als_cell_normalization_does_not_overflow_int64():
    pytest.importorskip("torch")
    rng = np.random.default_rng(713)
    training = rng.beta(2.0, 3.0, size=(96, 21))
    validation = rng.beta(2.0, 3.0, size=(32, 21))

    squared, squared_history = sd.fit_squared_tt_density(
        training,
        validation_samples=validation,
        validation_metric="log-likelihood",
        modes=8,
        rank=1,
        gamma=1e-4,
        epochs=1,
        optimizer="als",
        als_inner_steps=1,
        learning_rate=0.1,
        initialization="product",
        seed=714,
    )
    centered, centered_history = fit_centered_tt_density(
        training,
        validation_samples=validation,
        modes=8,
        rank=1,
        epochs=1,
        optimizer="als",
        initialization="product",
        seed=715,
    )

    assert np.all(np.isfinite(squared_history.loss))
    assert np.all(np.isfinite(centered_history.loss))
    assert np.all(np.isfinite(squared.log_density(validation)))
    assert np.all(np.isfinite(centered.log_density(validation)))


def test_two_sample_centered_ratio_moves_denominator_toward_numerator():
    pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    denominator = rng.random((2200, 2))
    numerator = np.column_stack([
        rng.beta(3.0, 4.0, size=2200),
        rng.beta(4.0, 3.0, size=2200),
    ])
    density, history = fit_centered_tt_ratio(
        denominator,
        numerator,
        modes=8,
        rank=4,
        epochs=8,
        learning_rate=2e-3,
        seed=7,
        initialization_coarse_bins=2,
        initialization_tolerance=5e-2,
        projection_tolerance=2e-2,
        projection_rank=5,
        projection_sweeps=5,
    )
    residual = density.sample(5000, seed=8)

    assert history.optimizer == "centered-ratio-adam"
    assert history.validation_loss
    assert np.linalg.norm(residual.mean(0) - numerator.mean(0)) < 0.04
    assert np.min(density.density(rng.random((1000, 2)))) > 0.0


def test_coarse_initialization_preserves_disconnected_joint_modes():
    rng = np.random.default_rng(31)
    count = 8000
    component = rng.integers(2, size=count)
    means = np.where(component[:, None] == 0, 0.18, 0.82)
    points = np.clip(means + 0.025 * rng.standard_normal((count, 4)), 0.0, 1.0)
    root = sd._coarse_to_fine_initial_root(
        points,
        np.full(4, 8, dtype=np.int64),
        rank=2,
        coarse_bins=2,
        pseudocount=0.5,
    )
    density = SquaredTTDensity(root, gamma=1e-8)
    generated = density.inverse_rosenblatt(rng.random((10_000, 4)))
    bits = generated >= 0.5
    correct = np.all(bits == bits[:, :1], axis=1)

    assert root.r.tolist() == [1, 2, 2, 2, 1]
    assert np.mean(correct) > 0.995


@pytest.mark.parametrize(
    ("optimizer", "options"),
    [
        ("adam", {"epochs": 40, "learning_rate": 4e-2}),
        ("riemannian", {"epochs": 15, "learning_rate": 1e-1}),
        (
            "riemannian-sgd",
            {
                "epochs": 60,
                "batch_size": 128,
                "learning_rate": 3e-2,
                "riemannian_retraction": "psa",
            },
        ),
        (
            "als",
            {
                "epochs": 2,
                "learning_rate": 1.0,
                "als_inner_steps": 5,
            },
        ),
    ],
)
def test_all_sample_density_optimizers_decrease_exact_loss(optimizer, options):
    pytest.importorskip("torch")
    rng = np.random.default_rng(22)
    target = np.column_stack(
        [rng.beta(2.0, 5.0, size=1000), rng.beta(5.0, 2.0, size=1000)]
    )
    with pytest.warns(RuntimeWarning) if optimizer == "riemannian" else nullcontext():
        density, history = sd.fit_squared_tt_density(
            target,
            modes=7,
            rank=2,
            gamma=1e-3,
            optimizer=optimizer,
            tolerance=1e-8,
            seed=4,
            **options,
        )

    assert history.optimizer == optimizer
    assert history.loss[-1] < history.loss[0] - 0.25
    assert history.function_calls >= history.epochs
    assert density.ranks.tolist() == [1, 2, 1]
