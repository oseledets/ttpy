from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

import tt
from tt.transport import SampleDIRT, SquaredTTDensity
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


def test_incremental_rosenblatt_roundtrip():
    density = SquaredTTDensity(_nontrivial_root(), gamma=1e-3)
    rng = np.random.default_rng(1)
    uniform = rng.random((200, 3))
    samples = density.inverse_rosenblatt(uniform)
    recovered = density.rosenblatt(samples)

    assert np.max(np.abs(recovered - uniform)) < 2e-14
    assert np.all((samples >= 0.0) & (samples <= 1.0))


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
