"""Fitters for radial and conditional coordinate twists."""

from __future__ import annotations

import time

import numpy as np

from ._basis import _validate_points
from ._torch_density import _torch_probit_orthogonal_map, _torch_sample_linear_tt
from ._types import FitHistory
from .coordinates import ProbitOrthogonalTTDensity
from .scalar import LinearSquaredTTDensity

def _cyclic_radial_twist_pairs(dimension: int, stages: int):
    """Deterministic alternating matchings for pairwise radial twists."""
    if dimension < 2 or stages < 1:
        raise ValueError("radial twists need dimension >= 2 and stages >= 1")
    pairs = []
    coordinates = np.arange(dimension, dtype=np.int64)
    paired_count = 2 * (dimension // 2)
    for stage in range(int(stages)):
        order = np.roll(coordinates, -stage)
        pairs.append(order[:paired_count].reshape(-1, 2).copy())
    return tuple(pairs)


def fit_radial_twists_to_probit_density(
    density: ProbitOrthogonalTTDensity,
    samples,
    *,
    validation_samples=None,
    stages: int = 2,
    basis_count: int = 4,
    epochs: int = 400,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_interval: int = 10,
    patience: int = 25,
    min_delta: float = 1e-4,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[ProbitOrthogonalTTDensity, FitHistory]:
    """Fit nonlinear exact radial coordinate twists with the TT fixed.

    Each stage contains disjoint coordinate pairs.  In Gaussian coordinates a
    pair is represented in polar form and updated by

    ``(r, phi) -> (r, phi + sum_m a_m sin(pi m q))``,
    ``q = 1 - exp(-r^2/2)``.

    Radius and polar area are unchanged, so the map preserves standard
    Gaussian measure exactly.  Probit conjugation therefore gives a
    volume-one cube automorphism with analytic inverse.  Zero coefficients
    are retained as validation checkpoint zero.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_radial_twists_to_probit_density requires torch"
        ) from exc
    if (
        not isinstance(density, ProbitOrthogonalTTDensity)
        or type(density.base) is not LinearSquaredTTDensity
    ):
        raise TypeError(
            "radial twist fitting requires a probit-orthogonal scalar "
            "linear squared TT density"
        )
    training = _validate_points(samples, density.d, name="samples")
    validation = (
        None if validation_samples is None else _validate_points(
            validation_samples, density.d, name="validation_samples"
        )
    )
    if (
        stages < 1 or basis_count < 1 or epochs < 1
        or learning_rate <= 0.0 or batch_size < 1
        or validation_interval < 1 or patience < 1 or min_delta < 0.0
    ):
        raise ValueError("invalid radial twist optimizer options")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("invalid radial twist tail objective")

    started = time.perf_counter()
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    cores = [
        torch.as_tensor(core, dtype=torch_dtype, device=device)
        for core in density.base._cores
    ]
    rotation = torch.as_tensor(
        density.rotation, dtype=torch_dtype, device=device
    )
    new_pairs = _cyclic_radial_twist_pairs(density.d, int(stages))
    new_coefficients = [
        torch.nn.Parameter(torch.zeros(
            (len(stage_pairs), int(basis_count)),
            dtype=torch_dtype, device=device,
        ))
        for stage_pairs in new_pairs
    ]
    existing_pairs = density.radial_twist_pairs
    existing_coefficients = [
        torch.as_tensor(value, dtype=torch_dtype, device=device)
        for value in density.radial_twist_coefficients
    ]
    all_pairs = (*existing_pairs, *new_pairs)
    training_tensor = torch.as_tensor(
        training, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )
    normalization = torch.as_tensor(
        density.base.normalization, dtype=torch_dtype, device=device
    )

    def point_nll(batch, coefficients):
        transformed = _torch_probit_orthogonal_map(
            batch,
            rotation,
            radial_twist_pairs=all_pairs,
            radial_twist_coefficients=(
                *existing_coefficients, *coefficients
            ),
            conditional_twist_pairs=density.conditional_twist_pairs,
            conditional_twist_conditioners=(
                density.conditional_twist_conditioners
            ),
            conditional_twist_coefficients=(
                density.conditional_twist_coefficients
            ),
        )
        values = _torch_sample_linear_tt(cores, transformed)
        return (
            torch.log(normalization)
            - torch.log(density.base.gamma + values.square())
        )

    def heldout(coefficients):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            values = []
            for start in range(0, len(validation_tensor), batch_size):
                values.append(point_nll(
                    validation_tensor[start:start + batch_size], coefficients
                ))
            return float(torch.cat(values).mean())

    history = FitHistory(
        optimizer="probit-radial-twist-adam[fixed-tt-rotation]"
    )
    history.initial_ranks = [sum(
        int(value.numel()) for value in new_coefficients
    )]
    best_validation = heldout(new_coefficients)
    patience_validation = best_validation
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    best_coefficients = [
        value.detach().clone() for value in new_coefficients
    ]
    optimizer_options = {}
    if torch.device(device).type == "cuda":
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.Adam(
            new_coefficients, lr=learning_rate, **optimizer_options
        )
    except TypeError:
        optimizer = torch.optim.Adam(
            new_coefficients, lr=learning_rate
        )
    rng = np.random.default_rng(seed)
    count = len(training)
    effective_batch = min(int(batch_size), count)
    stale = 0
    for epoch in range(int(epochs)):
        if effective_batch == count:
            batch = training_tensor
        else:
            chosen = torch.as_tensor(
                rng.choice(count, size=effective_batch, replace=False),
                dtype=torch.long, device=device,
            )
            batch = training_tensor[chosen]
        optimizer.zero_grad(set_to_none=True)
        losses = point_nll(batch, new_coefficients)
        loss = losses.mean()
        if tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(losses.numel())))
            )
            tail = torch.topk(
                losses, tail_count, sorted=False
            ).values.mean()
            loss = (1.0 - tail_weight) * loss + tail_weight * tail
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            new_coefficients, 100.0
        )
        optimizer.step()
        history.loss.append(float(loss.detach()))
        history.gradient_norm.append(float(gradient_norm))
        history.function_calls += 1
        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            value = heldout(new_coefficients)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_coefficients = [
                    coefficient.detach().clone()
                    for coefficient in new_coefficients
                ]
                history.best_epoch = epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - value > threshold:
                patience_validation = value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                break
    if validation_tensor is None:
        best_coefficients = [
            coefficient.detach().clone()
            for coefficient in new_coefficients
        ]
    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started
    fitted_coefficients = [
        value.detach().cpu().numpy().astype(np.float64)
        for value in best_coefficients
    ]
    selected_new = [
        (pairs, coefficients)
        for pairs, coefficients in zip(new_pairs, fitted_coefficients)
        if np.any(coefficients != 0.0)
    ]
    return ProbitOrthogonalTTDensity(
        density.base,
        density.rotation,
        rotation_block_sizes=density.rotation_block_sizes,
        radial_twist_pairs=(
            *existing_pairs, *[pairs for pairs, _ in selected_new]
        ),
        radial_twist_coefficients=(
            *density.radial_twist_coefficients,
            *[coefficients for _, coefficients in selected_new],
        ),
        conditional_twist_pairs=density.conditional_twist_pairs,
        conditional_twist_conditioners=(
            density.conditional_twist_conditioners
        ),
        conditional_twist_coefficients=(
            density.conditional_twist_coefficients
        ),
    ), history


def _alternating_conditional_twist_stages(
    dimension: int, stages: int,
):
    """Build balanced target/conditioner partitions for exact twist stages."""
    if dimension < 3 or stages < 1:
        raise ValueError(
            "conditional twists need dimension >= 3 and stages >= 1"
        )
    coordinates = np.arange(dimension, dtype=np.int64)
    conditioner_count = max(1, int(np.ceil(dimension / 3.0)))
    target_count = 2 * ((dimension - conditioner_count) // 2)
    if target_count < 2:
        raise ValueError("conditional twist stage has no target pair")
    pair_stages, conditioner_stages = [], []
    for stage in range(int(stages)):
        # A large cyclic shift makes successive target and conditioner sets
        # complementary instead of changing only one boundary coordinate.
        order = np.roll(coordinates, -stage * target_count)
        targets = order[:target_count]
        conditioner_pool = order[target_count:]
        pairs = targets.reshape(-1, 2).copy()
        conditioners = np.resize(
            conditioner_pool, len(pairs)
        ).astype(np.int64, copy=False)
        pair_stages.append(pairs)
        conditioner_stages.append(conditioners.copy())
    return tuple(pair_stages), tuple(conditioner_stages)


def fit_conditional_twists_to_probit_density(
    density: ProbitOrthogonalTTDensity,
    samples,
    *,
    validation_samples=None,
    stages: int = 2,
    basis_count: int = 4,
    epochs: int = 400,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_interval: int = 10,
    patience: int = 25,
    min_delta: float = 1e-4,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[ProbitOrthogonalTTDensity, FitHistory]:
    """Fit exact conditional coordinate rotations with the TT fixed.

    In one stage, disjoint Gaussian coordinate pairs are rotated by

    ``theta(z_c) = sum_m a_m sin(pi m Phi(z_c))``,

    where every conditioner ``c`` is outside the target set and remains
    unchanged in that stage.  The Jacobian is block triangular with planar
    rotations on its diagonal, while Gaussian norm is preserved pointwise.
    Probit conjugation is therefore an exact volume-one cube automorphism.
    Stages use alternating target/conditioner partitions, giving a nonlinear
    replacement of separation coordinates without a learned log determinant.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_conditional_twists_to_probit_density requires torch"
        ) from exc
    if (
        not isinstance(density, ProbitOrthogonalTTDensity)
        or type(density.base) is not LinearSquaredTTDensity
    ):
        raise TypeError(
            "conditional twist fitting requires a probit-orthogonal scalar "
            "linear squared TT density"
        )
    training = _validate_points(samples, density.d, name="samples")
    validation = (
        None if validation_samples is None else _validate_points(
            validation_samples, density.d, name="validation_samples"
        )
    )
    if (
        stages < 1 or basis_count < 1 or epochs < 1
        or learning_rate <= 0.0 or batch_size < 1
        or validation_interval < 1 or patience < 1 or min_delta < 0.0
    ):
        raise ValueError("invalid conditional twist optimizer options")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("invalid conditional twist tail objective")

    started = time.perf_counter()
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    cores = [
        torch.as_tensor(core, dtype=torch_dtype, device=device)
        for core in density.base._cores
    ]
    rotation = torch.as_tensor(
        density.rotation, dtype=torch_dtype, device=device
    )
    new_pairs, new_conditioners = _alternating_conditional_twist_stages(
        density.d, int(stages)
    )
    new_coefficients = [
        torch.nn.Parameter(torch.zeros(
            (len(stage_pairs), int(basis_count)),
            dtype=torch_dtype, device=device,
        ))
        for stage_pairs in new_pairs
    ]
    existing_pairs = density.conditional_twist_pairs
    existing_conditioners = density.conditional_twist_conditioners
    existing_coefficients = [
        torch.as_tensor(value, dtype=torch_dtype, device=device)
        for value in density.conditional_twist_coefficients
    ]
    all_pairs = (*existing_pairs, *new_pairs)
    all_conditioners = (*existing_conditioners, *new_conditioners)
    training_tensor = torch.as_tensor(
        training, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )
    normalization = torch.as_tensor(
        density.base.normalization, dtype=torch_dtype, device=device
    )

    def point_nll(batch, coefficients):
        transformed = _torch_probit_orthogonal_map(
            batch,
            rotation,
            radial_twist_pairs=density.radial_twist_pairs,
            radial_twist_coefficients=density.radial_twist_coefficients,
            conditional_twist_pairs=all_pairs,
            conditional_twist_conditioners=all_conditioners,
            conditional_twist_coefficients=(
                *existing_coefficients, *coefficients
            ),
        )
        values = _torch_sample_linear_tt(cores, transformed)
        return (
            torch.log(normalization)
            - torch.log(density.base.gamma + values.square())
        )

    def heldout(coefficients):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            values = []
            for start in range(0, len(validation_tensor), batch_size):
                values.append(point_nll(
                    validation_tensor[start:start + batch_size], coefficients
                ))
            return float(torch.cat(values).mean())

    history = FitHistory(
        optimizer="probit-conditional-twist-adam[fixed-tt-rotation]"
    )
    history.initial_ranks = [sum(
        int(value.numel()) for value in new_coefficients
    )]
    best_validation = heldout(new_coefficients)
    patience_validation = best_validation
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    best_coefficients = [
        value.detach().clone() for value in new_coefficients
    ]
    optimizer_options = {}
    if torch.device(device).type == "cuda":
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.Adam(
            new_coefficients, lr=learning_rate, **optimizer_options
        )
    except TypeError:
        optimizer = torch.optim.Adam(
            new_coefficients, lr=learning_rate
        )
    rng = np.random.default_rng(seed)
    count = len(training)
    effective_batch = min(int(batch_size), count)
    stale = 0
    for epoch in range(int(epochs)):
        if effective_batch == count:
            batch = training_tensor
        else:
            chosen = torch.as_tensor(
                rng.choice(count, size=effective_batch, replace=False),
                dtype=torch.long, device=device,
            )
            batch = training_tensor[chosen]
        optimizer.zero_grad(set_to_none=True)
        losses = point_nll(batch, new_coefficients)
        loss = losses.mean()
        if tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(losses.numel())))
            )
            tail = torch.topk(
                losses, tail_count, sorted=False
            ).values.mean()
            loss = (1.0 - tail_weight) * loss + tail_weight * tail
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            new_coefficients, 100.0
        )
        optimizer.step()
        history.loss.append(float(loss.detach()))
        history.gradient_norm.append(float(gradient_norm))
        history.function_calls += 1
        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            value = heldout(new_coefficients)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_coefficients = [
                    coefficient.detach().clone()
                    for coefficient in new_coefficients
                ]
                history.best_epoch = epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - value > threshold:
                patience_validation = value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                break
    if validation_tensor is None:
        best_coefficients = [
            coefficient.detach().clone()
            for coefficient in new_coefficients
        ]
    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started
    fitted_coefficients = [
        value.detach().cpu().numpy().astype(np.float64)
        for value in best_coefficients
    ]
    selected_new = [
        (pairs, conditioners, coefficients)
        for pairs, conditioners, coefficients in zip(
            new_pairs, new_conditioners, fitted_coefficients
        )
        if np.any(coefficients != 0.0)
    ]
    return ProbitOrthogonalTTDensity(
        density.base,
        density.rotation,
        rotation_block_sizes=density.rotation_block_sizes,
        radial_twist_pairs=density.radial_twist_pairs,
        radial_twist_coefficients=density.radial_twist_coefficients,
        conditional_twist_pairs=(
            *existing_pairs,
            *[pairs for pairs, _, _ in selected_new],
        ),
        conditional_twist_conditioners=(
            *existing_conditioners,
            *[conditioners for _, conditioners, _ in selected_new],
        ),
        conditional_twist_coefficients=(
            *density.conditional_twist_coefficients,
            *[coefficients for _, _, coefficients in selected_new],
        ),
    ), history

__all__ = [
    "_cyclic_radial_twist_pairs",
    "fit_radial_twists_to_probit_density",
    "_alternating_conditional_twist_stages",
    "fit_conditional_twists_to_probit_density",
]
