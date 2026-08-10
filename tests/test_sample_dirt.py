from __future__ import annotations

import numpy as np
import pytest

import tt
from tt.transport import SampleDIRT, SquaredTTDensity


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


