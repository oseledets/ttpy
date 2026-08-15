"""Fitters for purified, nonnegative, and quadratic TT densities."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _as_numpy_cores,
    _linear_hat_integral_weights,
    _linear_right_gram_environments,
    _local_purified_right_environments,
    _quadratic_gram_bands,
    _quadratic_right_gram_environments,
    _validate_points,
)
from ._fit_spline import fit_linear_squared_tt_density
from ._torch_als import (
    _torch_linear_right_orthogonalize,
    _torch_right_orthogonalize,
)
from ._torch_density import (
    _torch_linear_root_second_moment,
    _torch_local_purified_integral,
    _torch_nonnegative_linear_log_integral,
    _torch_normalize_nonnegative_linear_log_cores,
    _torch_sample_linear_tt_vector,
    _torch_sample_local_purified,
    _torch_sample_nonnegative_linear_log_tt,
    _torch_sample_quadratic_tt,
)
from ._types import FitHistory
from .polynomial import QuadraticSquaredTTDensity
from .positive import (
    LocallyPurifiedLinearTTDensity,
    NonnegativeLinearTTDensity,
    PurifiedLinearTTDensity,
)

def fit_purified_linear_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 4,
    purification_channels: int = 4,
    gamma: float = 1e-5,
    floor_mass: float | None = 1e-4,
    epochs: int = 500,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "cluster",
    scalar_warm_start_epochs: int | None = None,
    initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    verbose: bool = False,
) -> tuple[PurifiedLinearTTDensity, FitHistory]:
    """Fit one shared vector-valued TT root by exact-normalization NLL.

    ``purification_channels`` is the final auxiliary leg of a single tensor
    network, not a collection of independently normalized models.  All cores
    are optimized jointly and normalization is one exact Gram contraction.
    A train-only cluster construction only breaks amplitude-channel symmetry at
    initialization; dense shared cores and terminal channels remain free.
    ``initialization="scalar"`` first fits the scalar squared-TT submodel,
    embeds it exactly into the first terminal channel, and then activates the
    remaining channels with small perturbations.  Held-out checkpointing keeps
    the exact scalar embedding unless purification genuinely improves it.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_purified_linear_tt_density requires the 'torch' extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    if (
        rank < 1 or purification_channels < 1 or gamma <= 0.0
        or epochs < 1 or batch_size < 1 or validation_interval < 1
        or patience < 1 or initialization_noise < 0.0
        or initialization_pseudocount <= 0.0
    ):
        raise ValueError("invalid rank, purification or optimizer option")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0,1] and tail_weight in [0,1]")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in (
        "uniform", "product", "cluster", "random", "scalar"
    ):
        raise ValueError(
            "initialization must be uniform, product, cluster, random or scalar"
        )
    if scalar_warm_start_epochs is not None and scalar_warm_start_epochs < 1:
        raise ValueError("scalar_warm_start_epochs must be positive")
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )

    ranks = [1] * (d + 1)
    ranks[-1] = int(purification_channels)
    capacity = 1
    for k in range(1, d):
        capacity = min(int(rank), capacity * int(modes_array[k - 1]))
        ranks[k] = capacity
    capacity = int(purification_channels)
    for k in range(d - 1, 0, -1):
        capacity = min(int(rank), capacity * int(modes_array[k]))
        ranks[k] = min(ranks[k], capacity)

    scalar_history = None
    exact_scalar_cores = None
    if initialization == "scalar":
        scalar_model, scalar_history = fit_linear_squared_tt_density(
            points,
            modes=modes_array,
            rank=rank,
            gamma=gamma,
            floor_mass=floor_mass,
            epochs=(
                epochs if scalar_warm_start_epochs is None
                else int(scalar_warm_start_epochs)
            ),
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_samples=validation,
            patience=patience,
            validation_interval=validation_interval,
            min_delta=min_delta,
            seed=seed,
            device=device,
            dtype=dtype,
            initialization="mixture",
            initialization_noise=initialization_noise,
            initialization_pseudocount=initialization_pseudocount,
            tail_fraction=tail_fraction,
            tail_weight=tail_weight,
            optimizer="adam",
            verbose=verbose,
        )
        source_cores = scalar_model._cores
        exact_scalar_cores = [
            np.zeros((ranks[k], int(modes_array[k]), ranks[k + 1]))
            for k in range(d)
        ]
        for target, source in zip(exact_scalar_cores, source_cores):
            target[:source.shape[0], :, :source.shape[2]] = source

    def nodal_marginal(selected: np.ndarray, coordinate: int) -> np.ndarray:
        n = int(modes_array[coordinate])
        counts = np.histogram(
            selected[:, coordinate], bins=n - 1, range=(0.0, 1.0)
        )[0].astype(np.float64)
        cells = (
            counts + initialization_pseudocount
        ) / (
            len(selected) + initialization_pseudocount * (n - 1)
        ) * (n - 1)
        nodal = np.empty(n, dtype=np.float64)
        nodal[[0, -1]] = cells[[0, -1]]
        if n > 2:
            nodal[1:-1] = 0.5 * (cells[:-1] + cells[1:])
        nodal /= np.dot(_linear_hat_integral_weights(n), nodal)
        return nodal

    product_marginals = [nodal_marginal(points, k) for k in range(d)]
    cluster_count = min(
        int(purification_channels),
        int(min(ranks[1:-1])) if d > 1 else int(purification_channels),
    )
    cluster_weights = np.ones(1)
    cluster_marginals = [product_marginals]
    if initialization == "cluster" and cluster_count > 1:
        labels = _kmeans_labels_for_tt_initialization(
            points, cluster_count, seed=seed + 104729
        )
        sizes = np.bincount(labels, minlength=cluster_count)
        cluster_weights = (
            sizes + initialization_pseudocount
        ) / (
            len(points) + initialization_pseudocount * cluster_count
        )
        cluster_marginals = [
            [nodal_marginal(points[labels == component], k) for k in range(d)]
            for component in range(cluster_count)
        ]
    elif initialization in ("uniform", "random"):
        cluster_marginals = [[
            np.ones(int(mode), dtype=np.float64) for mode in modes_array
        ]]

    rng = np.random.default_rng(seed)
    effective_noise = (
        max(initialization_noise, 0.1)
        if initialization == "random" else initialization_noise
    )
    if exact_scalar_cores is None:
        numpy_cores = [
            effective_noise * rng.normal(size=(
                ranks[k], int(modes_array[k]), ranks[k + 1]
            ))
            for k in range(d)
        ]
    else:
        numpy_cores = [core.copy() for core in exact_scalar_cores]
        for target, source in zip(numpy_cores, source_cores):
            perturbation = initialization_noise * rng.normal(size=target.shape)
            perturbation[:source.shape[0], :, :source.shape[2]] = 0.0
            target += perturbation
    active_components = cluster_count if initialization == "cluster" else 1
    for component in range(active_components if exact_scalar_cores is None else 0):
        weight = (
            np.sqrt(cluster_weights[component])
            if initialization == "cluster" else 1.0
        )
        if d == 1:
            numpy_cores[0][0, :, component] += (
                weight * np.sqrt(cluster_marginals[component][0])
            )
            continue
        numpy_cores[0][0, :, component] += (
            weight * np.sqrt(cluster_marginals[component][0])
        )
        for k in range(1, d - 1):
            numpy_cores[k][component, :, component] += np.sqrt(
                cluster_marginals[component][k]
            )
        numpy_cores[-1][component, :, component] += np.sqrt(
            cluster_marginals[component][-1]
        )

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    canonical = [
        torch.as_tensor(core, dtype=torch_dtype, device=device)
        for core in numpy_cores
    ]
    _torch_linear_right_orthogonalize(canonical)
    params = [torch.nn.Parameter(core) for core in canonical]
    exact_scalar_torch = None
    if exact_scalar_cores is not None:
        exact_scalar_torch = [
            torch.as_tensor(core, dtype=torch_dtype, device=device)
            for core in exact_scalar_cores
        ]
        _torch_linear_right_orthogonalize(exact_scalar_torch)
    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )

    def nll(cores, batch, *, tail_aware: bool):
        second = _torch_linear_root_second_moment(cores)
        values = _torch_sample_linear_tt_vector(cores, batch)
        square = values.square().sum(dim=1)
        if floor_mass is None:
            normalization = gamma + second
            losses = torch.log(normalization) - torch.log(gamma + square)
        else:
            normalized_square = square / torch.clamp_min(
                second, torch.finfo(second.dtype).tiny
            )
            density = floor_mass + (1.0 - floor_mass) * normalized_square
            normalization = second
            losses = -torch.log(density)
        empirical = losses.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            count = max(1, int(np.ceil(tail_fraction * int(losses.numel()))))
            tail = torch.topk(losses, count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization

    def heldout(cores):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(
                cores, validation_tensor, tail_aware=False
            )[0])

    history = FitHistory(optimizer=(
        "linear-purified-adam[scalar-warm-start]"
        if scalar_history is not None else "linear-purified-adam"
    ))
    best_validation = heldout(
        exact_scalar_torch if exact_scalar_torch is not None else params
    )
    patience_validation = best_validation
    best_cores = [
        parameter.detach().clone()
        for parameter in (
            exact_scalar_torch if exact_scalar_torch is not None else params
        )
    ]
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    stale = 0
    count = len(points)
    batch_size = min(int(batch_size), count)
    batch_rng = np.random.default_rng(seed + 1)
    adam_options = {"fused": True} if torch.device(device).type == "cuda" else {}
    try:
        adam = torch.optim.Adam(params, lr=learning_rate, **adam_options)
    except TypeError:  # pragma: no cover - old torch
        adam = torch.optim.Adam(params, lr=learning_rate)

    started = time.perf_counter()
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            chosen = batch_rng.choice(count, size=batch_size, replace=False)
            batch = training_tensor[torch.as_tensor(
                chosen, dtype=torch.long, device=device
            )]
        adam.zero_grad(set_to_none=True)
        loss, normalization = nll(params, batch, tail_aware=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 100.0)
        adam.step()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(normalization.detach()))
        history.gradient_norm.append(float(gradient_norm))

        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            value = heldout(params)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_cores = [parameter.detach().clone() for parameter in params]
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
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(
                f"epoch {epoch + 1:5d}: purified_nll={float(loss):.7e}"
            )

    purified_epochs = len(history.loss)
    purified_best_epoch = history.best_epoch
    history.epochs = purified_epochs
    history.wall_time = time.perf_counter() - started
    if scalar_history is not None:
        history.epochs += scalar_history.epochs
        history.function_calls += scalar_history.function_calls
        history.wall_time += scalar_history.wall_time
        history.best_epoch = (
            scalar_history.epochs + purified_best_epoch
            if purified_best_epoch > 0 else scalar_history.best_epoch
        )
    fitted_cores = (
        best_cores if validation_tensor is not None
        else [parameter.detach().clone() for parameter in params]
    )
    fitted_second = float(_linear_right_gram_environments([
        core.cpu().numpy() for core in fitted_cores
    ])[0][0, 0])
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    return PurifiedLinearTTDensity(
        [core.cpu().numpy() for core in fitted_cores], gamma=fitted_gamma
    ), history


def fit_locally_purified_linear_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 4,
    local_channels: int | Sequence[int] = 2,
    gamma: float = 1e-5,
    floor_mass: float | None = 1e-4,
    epochs: int = 500,
    scalar_warm_start_epochs: int | None = None,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization_noise: float = 1e-3,
    scalar_initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    verbose: bool = False,
) -> tuple[LocallyPurifiedLinearTTDensity, FitHistory]:
    """Scalar-warm-started exact-NLL fit of one locally purified MPS.

    The scalar linear squared TT is embedded exactly in Kraus channel zero of
    every core.  Other local channels are perturbed for optimization, while
    held-out checkpointing starts at the exact scalar density.  Consequently
    the returned validation likelihood cannot be worse than the nested scalar
    warm start.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_locally_purified_linear_tt_density requires the torch extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    if isinstance(local_channels, (int, np.integer)):
        local_channels_array = np.full(d, int(local_channels), dtype=np.int64)
    else:
        local_channels_array = np.asarray(list(local_channels), dtype=np.int64)
    if (
        local_channels_array.shape != (d,) or np.any(local_channels_array < 1)
    ):
        raise ValueError("local_channels must provide one positive value per core")
    if (
        rank < 1 or epochs < 1 or batch_size < 1
        or patience < 1 or validation_interval < 1
        or initialization_noise < 0.0 or scalar_initialization_noise < 0.0
    ):
        raise ValueError("invalid local-purification optimizer option")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    validation = None if validation_samples is None else _validate_points(
        validation_samples, d, name="validation_samples"
    )

    scalar, scalar_history = fit_linear_squared_tt_density(
        points,
        modes=modes_array,
        rank=rank,
        gamma=gamma,
        floor_mass=floor_mass,
        epochs=(
            epochs if scalar_warm_start_epochs is None
            else int(scalar_warm_start_epochs)
        ),
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_samples=validation,
        patience=patience,
        validation_interval=validation_interval,
        min_delta=min_delta,
        seed=seed,
        device=device,
        dtype=dtype,
        initialization="mixture",
        initialization_noise=scalar_initialization_noise,
        initialization_pseudocount=initialization_pseudocount,
        tail_fraction=tail_fraction,
        tail_weight=tail_weight,
        optimizer="adam",
        verbose=verbose,
    )
    exact_cores = []
    rng = np.random.default_rng(seed + 161803)
    perturbed_cores = []
    for coordinate, core in enumerate(scalar._cores):
        channels = int(local_channels_array[coordinate])
        exact = np.zeros((
            core.shape[0], core.shape[1], channels, core.shape[2]
        ), dtype=np.float64)
        exact[:, :, 0, :] = core
        perturbed = exact.copy()
        if channels > 1:
            perturbed[:, :, 1:, :] = initialization_noise * rng.normal(
                size=perturbed[:, :, 1:, :].shape
            )
        exact_cores.append(exact)
        perturbed_cores.append(perturbed)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    exact_torch = [
        torch.as_tensor(core, dtype=torch_dtype, device=device)
        for core in exact_cores
    ]
    params = [torch.nn.Parameter(torch.as_tensor(
        core, dtype=torch_dtype, device=device
    )) for core in perturbed_cores]
    training_tensor = torch.as_tensor(points, dtype=torch_dtype, device=device)
    validation_tensor = None if validation is None else torch.as_tensor(
        validation, dtype=torch_dtype, device=device
    )

    def nll(cores, batch, *, tail_aware):
        integral = _torch_local_purified_integral(cores)
        signal = _torch_sample_local_purified(cores, batch)
        if floor_mass is None:
            losses = torch.log(gamma + integral) - torch.log(gamma + signal)
        else:
            normalized = signal / torch.clamp_min(
                integral, torch.finfo(integral.dtype).tiny
            )
            losses = -torch.log(floor_mass + (1.0 - floor_mass) * normalized)
        value = losses.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            count = max(1, int(np.ceil(tail_fraction * len(losses))))
            tail = torch.topk(losses, count, sorted=False).values.mean()
            value = (1.0 - tail_weight) * value + tail_weight * tail
        return value, integral

    def heldout(cores):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(cores, validation_tensor, tail_aware=False)[0])

    history = FitHistory(optimizer="linear-local-purified-adam[scalar-warm-start]")
    best_validation = heldout(exact_torch)
    patience_validation = best_validation
    best_cores = [core.detach().clone() for core in exact_torch]
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    stale = 0
    count = len(points)
    batch_size = min(int(batch_size), count)
    batch_rng = np.random.default_rng(seed + 1)
    adam_options = {"fused": True} if torch.device(device).type == "cuda" else {}
    try:
        adam = torch.optim.Adam(params, lr=learning_rate, **adam_options)
    except TypeError:  # pragma: no cover
        adam = torch.optim.Adam(params, lr=learning_rate)
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            selected = batch_rng.choice(count, size=batch_size, replace=False)
            batch = training_tensor[torch.as_tensor(
                selected, dtype=torch.long, device=device
            )]
        adam.zero_grad(set_to_none=True)
        loss, integral = nll(params, batch, tail_aware=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 100.0)
        adam.step()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(integral.detach()))
        history.gradient_norm.append(float(gradient_norm))
        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            value = heldout(params)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_cores = [parameter.detach().clone() for parameter in params]
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
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(f"epoch {epoch + 1:5d}: local_purified_nll={float(loss):.7e}")

    local_best_epoch = history.best_epoch
    history.epochs = scalar_history.epochs + len(history.loss)
    history.function_calls += scalar_history.function_calls
    history.wall_time = scalar_history.wall_time + time.perf_counter() - started
    history.best_epoch = (
        scalar_history.epochs + local_best_epoch
        if local_best_epoch > 0 else scalar_history.best_epoch
    )
    fitted_numpy = [core.cpu().numpy() for core in best_cores]
    fitted_integral = float(
        _local_purified_right_environments(fitted_numpy)[0][0, 0]
    )
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_integral / (1.0 - floor_mass)
    )
    return LocallyPurifiedLinearTTDensity(
        fitted_numpy, gamma=fitted_gamma
    ), history


def _kmeans_labels_for_tt_initialization(
    points: np.ndarray,
    clusters: int,
    *,
    seed: int,
    maximum_fit_points: int = 20_000,
    iterations: int = 25,
) -> np.ndarray:
    """Small dependency-free k-means used only to break TT-state symmetry."""
    if clusters <= 1:
        return np.zeros(len(points), dtype=np.int64)
    rng = np.random.default_rng(seed)
    scale = np.std(points, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (points - np.mean(points, axis=0)) / scale
    if len(points) > maximum_fit_points:
        subset_indices = rng.choice(
            len(points), size=maximum_fit_points, replace=False
        )
        fitting = standardized[subset_indices]
    else:
        fitting = standardized

    centers = [fitting[rng.integers(len(fitting))]]
    squared_distance = np.sum((fitting - centers[0]) ** 2, axis=1)
    for _ in range(1, clusters):
        total = float(squared_distance.sum())
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            centers.append(fitting[rng.integers(len(fitting))])
        else:
            centers.append(fitting[rng.choice(
                len(fitting), p=squared_distance / total
            )])
        candidate_distance = np.sum(
            (fitting - centers[-1]) ** 2, axis=1
        )
        squared_distance = np.minimum(squared_distance, candidate_distance)
    centers = np.asarray(centers)

    for _ in range(int(iterations)):
        distance = np.sum(
            (fitting[:, None, :] - centers[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distance, axis=1)
        updated = centers.copy()
        for component in range(clusters):
            selected = fitting[labels == component]
            if len(selected):
                updated[component] = selected.mean(axis=0)
            else:
                updated[component] = fitting[rng.integers(len(fitting))]
        if np.max(np.abs(updated - centers)) < 1e-6:
            centers = updated
            break
        centers = updated

    labels = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 50_000):
        stop = min(start + 50_000, len(points))
        distance = np.sum(
            (standardized[start:stop, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        labels[start:stop] = np.argmin(distance, axis=1)
    return labels


def fit_nonnegative_linear_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 4,
    gamma: float = 1e-6,
    floor_mass: float | None = 1e-4,
    epochs: int = 500,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "product",
    initialization_noise: float = 5e-2,
    initialization_pseudocount: float = 0.5,
    initialization_transition_leak: float = 1e-4,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    verbose: bool = False,
) -> tuple[NonnegativeLinearTTDensity, FitHistory]:
    """Globally fit all cores of one directly nonnegative linear TT.

    Core entries are parameterized in log space.  On every objective call a
    differentiable positive diagonal gauge makes all exact suffix integrals
    stochastic, so the TT component is normalized without unstable products
    across dimensions.  For the default explicit ``floor_mass`` the model is

    ``floor_mass + (1-floor_mass) * f_theta``,  ``integral f_theta = 1``.

    The minibatch term is stochastic maximum likelihood, while the
    normalization and every TT contraction are exact.  All dense core entries
    are optimized simultaneously.  The product histogram is only an initial
    point; after the first gradient step the internal state may change at every
    coordinate, and the returned artifact contains one TT density.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_nonnegative_linear_tt_density requires the 'torch' extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    if (
        rank < 1 or gamma <= 0.0 or epochs < 1 or batch_size < 1
        or validation_interval < 1 or patience < 1
        or initialization_noise < 0.0 or initialization_pseudocount <= 0.0
        or not 0.0 < initialization_transition_leak < 1.0
    ):
        raise ValueError("invalid rank, floor, epoch, batch or initialization option")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1] and tail_weight in [0, 1]")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("uniform", "product", "cluster", "random"):
        raise ValueError(
            "initialization must be uniform, product, cluster or random"
        )
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )

    ranks = [1] * (d + 1)
    capacity = 1
    for k in range(1, d):
        capacity = min(int(rank), capacity * int(modes_array[k - 1]))
        ranks[k] = capacity
    capacity = 1
    for k in range(d - 1, 0, -1):
        capacity = min(int(rank), capacity * int(modes_array[k]))
        ranks[k] = min(ranks[k], capacity)

    rng = np.random.default_rng(seed)
    marginals: list[np.ndarray] = []
    for k, n_value in enumerate(modes_array):
        n = int(n_value)
        if initialization in ("uniform", "random"):
            nodal = np.ones(n, dtype=np.float64)
        else:
            counts = np.histogram(
                points[:, k], bins=n - 1, range=(0.0, 1.0)
            )[0].astype(np.float64)
            cells = (
                counts + initialization_pseudocount
            ) / (
                len(points) + initialization_pseudocount * (n - 1)
            ) * (n - 1)
            nodal = np.empty(n, dtype=np.float64)
            nodal[[0, -1]] = cells[[0, -1]]
            if n > 2:
                nodal[1:-1] = 0.5 * (cells[:-1] + cells[1:])
            nodal /= np.dot(_linear_hat_integral_weights(n), nodal)
        marginals.append(nodal)

    cluster_marginals = None
    cluster_weights = None
    cluster_count = 1 if d == 1 else min(ranks[1:-1])
    if initialization == "cluster" and cluster_count > 1:
        labels = _kmeans_labels_for_tt_initialization(
            points, cluster_count, seed=seed + 7919
        )
        cluster_sizes = np.bincount(labels, minlength=cluster_count)
        cluster_weights = (
            cluster_sizes + initialization_pseudocount
        ) / (
            len(points) + initialization_pseudocount * cluster_count
        )
        cluster_marginals = []
        for component in range(cluster_count):
            selected = points[labels == component]
            component_marginals = []
            for k, n_value in enumerate(modes_array):
                n = int(n_value)
                counts = np.histogram(
                    selected[:, k], bins=n - 1, range=(0.0, 1.0)
                )[0].astype(np.float64)
                cells = (
                    counts + initialization_pseudocount
                ) / (
                    len(selected) + initialization_pseudocount * (n - 1)
                ) * (n - 1)
                nodal = np.empty(n, dtype=np.float64)
                nodal[[0, -1]] = cells[[0, -1]]
                if n > 2:
                    nodal[1:-1] = 0.5 * (cells[:-1] + cells[1:])
                nodal /= np.dot(_linear_hat_integral_weights(n), nodal)
                component_marginals.append(nodal)
            cluster_marginals.append(component_marginals)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    params = []
    effective_noise = (
        max(initialization_noise, 0.25)
        if initialization == "random" else initialization_noise
    )
    for k, n_value in enumerate(modes_array):
        left_rank, right_rank = ranks[k], ranks[k + 1]
        transition = np.exp(
            effective_noise * rng.normal(size=(left_rank, right_rank))
        )
        transition /= transition.sum(axis=1, keepdims=True)
        if cluster_marginals is None:
            core = transition[:, None, :] * marginals[k][None, :, None]
        else:
            core = (
                initialization_transition_leak
                * transition[:, None, :]
                * marginals[k][None, :, None]
            )
            if k == 0:
                for state in range(cluster_count):
                    core[0, :, state] += (
                        cluster_weights[state]
                        * cluster_marginals[state][k]
                    )
            elif k == d - 1:
                for state in range(cluster_count):
                    core[state, :, 0] += cluster_marginals[state][k]
            else:
                for state in range(cluster_count):
                    core[state, :, state] += cluster_marginals[state][k]
        core *= np.exp(effective_noise * rng.normal(
            size=(left_rank, int(n_value), right_rank)
        ))
        params.append(torch.nn.Parameter(torch.as_tensor(
            np.log(core), dtype=torch_dtype, device=device
        )))

    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )
    log_floor = (
        None if floor_mass is None else torch.tensor(
            np.log(floor_mass), dtype=torch_dtype, device=device
        )
    )
    log_nonfloor = (
        None if floor_mass is None else torch.tensor(
            np.log1p(-floor_mass), dtype=torch_dtype, device=device
        )
    )
    log_gamma = torch.tensor(
        np.log(gamma), dtype=torch_dtype, device=device
    )

    def point_nll(log_cores, batch, *, tail_aware: bool):
        if floor_mass is None:
            log_component = _torch_sample_nonnegative_linear_log_tt(
                log_cores, batch
            )
            log_integral = _torch_nonnegative_linear_log_integral(log_cores)
            log_density = (
                torch.logaddexp(log_gamma, log_component)
                - torch.logaddexp(log_gamma, log_integral)
            )
            normalization_record = torch.exp(
                torch.logaddexp(log_gamma, log_integral)
            )
        else:
            normalized = _torch_normalize_nonnegative_linear_log_cores(
                log_cores
            )
            log_component = _torch_sample_nonnegative_linear_log_tt(
                normalized, batch
            )
            log_density = torch.logaddexp(
                log_floor, log_nonfloor + log_component
            )
            normalization_record = torch.ones(
                (), dtype=torch_dtype, device=device
            )
        losses = -log_density
        empirical = losses.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(losses.numel())))
            )
            tail = torch.topk(losses, tail_count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization_record

    def heldout(log_cores):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(point_nll(
                log_cores, validation_tensor, tail_aware=False
            )[0])

    history = FitHistory(optimizer="linear-nonnegative-log-adam")
    best_validation = heldout(params)
    patience_validation = best_validation
    best_cores = [parameter.detach().clone() for parameter in params]
    stale = 0
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    count = len(points)
    batch_size = min(int(batch_size), count)
    batch_rng = np.random.default_rng(seed + 1)
    adam_options = {}
    if torch.device(device).type == "cuda":
        adam_options["fused"] = True
    try:
        optimizer = torch.optim.Adam(
            params, lr=learning_rate, **adam_options
        )
    except TypeError:  # older torch without fused Adam
        optimizer = torch.optim.Adam(params, lr=learning_rate)

    started = time.perf_counter()
    terminal_cores = params
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            chosen = batch_rng.choice(count, size=batch_size, replace=False)
            batch = training_tensor[torch.as_tensor(
                chosen, dtype=torch.long, device=device
            )]
        optimizer.zero_grad(set_to_none=True)
        loss, normalization_record = point_nll(
            params, batch, tail_aware=True
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 100.0)
        optimizer.step()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(normalization_record.detach()))
        history.gradient_norm.append(float(gradient_norm))

        check_validation = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check_validation:
            validation_value = heldout(params)
            history.validation_loss.append(validation_value)
            if validation_value < best_validation:
                best_validation = validation_value
                best_cores = [parameter.detach().clone() for parameter in params]
                history.best_epoch = epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - validation_value > threshold:
                patience_validation = validation_value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                terminal_cores = params
                break
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            validation_message = (
                "" if validation_tensor is None
                else f", val={history.validation_loss[-1]:.7e}"
            )
            print(
                f"epoch {epoch + 1:5d}: nonnegative_nll={float(loss):.7e}"
                f"{validation_message}"
            )

    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started
    selected = (
        best_cores if validation_tensor is not None
        else [parameter.detach().clone() for parameter in terminal_cores]
    )
    with torch.no_grad():
        log_integral = _torch_nonnegative_linear_log_integral(selected)
        normalized_logs = _torch_normalize_nonnegative_linear_log_cores(
            selected
        )
    fitted = vector.from_list([
        torch.exp(core).cpu().numpy().astype(np.float64, copy=True)
        for core in normalized_logs
    ])
    if floor_mass is None:
        fitted_gamma = float(torch.exp(log_gamma - log_integral).cpu())
    else:
        fitted_gamma = floor_mass / (1.0 - floor_mass)
    density = NonnegativeLinearTTDensity(fitted, gamma=fitted_gamma)
    history.chi2_to_reference.append(density.chi2_to_reference)
    return density, history


def fit_quadratic_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 64,
    rank: int = 4,
    gamma: float = 1e-5,
    floor_mass: float | None = None,
    epochs: int = 500,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "mixture",
    initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    verbose: bool = False,
) -> tuple[QuadraticSquaredTTDensity, FitHistory]:
    """Fit a quadratic B-spline squared TT by exact-normalization NLL.

    ``floor_mass`` has the same scale-invariant reference-mixture meaning as
    in :func:`fit_linear_squared_tt_density`.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_quadratic_squared_tt_density requires the 'torch' extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 3):
        raise ValueError("modes must provide at least three quadratic functions")
    if (
        rank < 1 or gamma <= 0.0 or epochs < 1 or batch_size < 1
        or validation_interval < 1
    ):
        raise ValueError("invalid rank, gamma, epochs, batch or validation interval")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1] and tail_weight in [0, 1]")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("uniform", "product", "mixture"):
        raise ValueError(
            "quadratic basis supports uniform, product or mixture initialization"
        )
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ranks = [1] * (d + 1)
    capacity = 1
    for k in range(1, d):
        capacity = min(int(rank), capacity * int(modes_array[k - 1]))
        ranks[k] = capacity
    capacity = 1
    for k in range(d - 1, 0, -1):
        capacity = min(int(rank), capacity * int(modes_array[k]))
        ranks[k] = min(ranks[k], capacity)

    marginal_roots = []
    if initialization in ("product", "mixture"):
        for k, n_value in enumerate(modes_array):
            n = int(n_value)
            intervals = n - 2
            counts = np.histogram(
                points[:, k], bins=intervals, range=(0.0, 1.0)
            )[0].astype(np.float64)
            density = (
                counts + initialization_pseudocount
            ) / (
                len(points) + initialization_pseudocount * intervals
            ) * intervals
            centers = (np.arange(intervals) + 0.5) / intervals
            greville = np.clip(
                (np.arange(n) - 0.5) / intervals, 0.0, 1.0
            )
            nodal = np.interp(
                greville, centers, density,
                left=density[0], right=density[-1],
            )
            marginal_roots.append(np.sqrt(np.maximum(nodal, 1e-12)))

    params = []
    for k, n_value in enumerate(modes_array):
        n = int(n_value)
        core = initialization_noise * torch.randn(
            (ranks[k], n, ranks[k + 1]),
            generator=generator, dtype=torch_dtype,
        )
        if initialization == "uniform":
            core[0, :, 0] += 1.0
        elif initialization == "product":
            core[0, :, 0] += torch.as_tensor(
                marginal_roots[k], dtype=torch_dtype
            )
        else:
            base = torch.as_tensor(marginal_roots[k], dtype=torch_dtype)
            component_noise = max(2e-2, 5.0 * initialization_noise)
            if k == 0:
                for right_index in range(ranks[k + 1]):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[0, :, right_index] += (
                        base * jitter / ranks[k + 1]
                    )
            elif k == d - 1:
                for left_index in range(ranks[k]):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[left_index, :, 0] += base * jitter
            else:
                for channel in range(min(ranks[k], ranks[k + 1])):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[channel, :, channel] += base * jitter
        params.append(torch.nn.Parameter(core.to(device)))

    canonical = [parameter.detach().clone() for parameter in params]
    _torch_right_orthogonalize(canonical, uniform_measure=True)
    params = [torch.nn.Parameter(core) for core in canonical]
    initial_cores = [parameter.detach().clone() for parameter in params]
    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )
    gram_bands = [
        tuple(
            torch.as_tensor(band, dtype=torch_dtype, device=device)
            for band in _quadratic_gram_bands(int(n))
        )
        for n in modes_array
    ]

    def second_moment(cores):
        environment = torch.ones(
            (1, 1), dtype=torch_dtype, device=device
        )
        for core, bands in zip(cores, gram_bands):
            projected = torch.einsum("ac,aib->cib", environment, core)
            weighted = projected * bands[0][None, :, None]
            environment = torch.einsum("cib,cid->bd", weighted, core)
            for offset, band in ((1, bands[1]), (2, bands[2])):
                weighted_left = projected[:, :-offset] * band[None, :, None]
                environment = environment + torch.einsum(
                    "cib,cid->bd", weighted_left, core[:, offset:]
                )
                weighted_right = projected[:, offset:] * band[None, :, None]
                environment = environment + torch.einsum(
                    "cib,cid->bd", weighted_right, core[:, :-offset]
                )
        return environment.reshape(())

    def nll(cores, batch, *, tail_aware: bool):
        second = second_moment(cores)
        values = _torch_sample_quadratic_tt(cores, batch)
        if floor_mass is None:
            normalization = gamma + second
            point_nll = (
                torch.log(normalization)
                - torch.log(gamma + values.square())
            )
        else:
            normalized_square = values.square() / torch.clamp_min(
                second, torch.finfo(second.dtype).tiny
            )
            density = (
                floor_mass + (1.0 - floor_mass) * normalized_square
            )
            normalization = second
            point_nll = -torch.log(density)
        empirical = point_nll.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(point_nll.numel())))
            )
            tail = torch.topk(point_nll, tail_count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization

    def heldout(cores):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(cores, validation_tensor, tail_aware=False)[0])

    history = FitHistory(optimizer="quadratic-adam")
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    rng = np.random.default_rng(seed + 1)
    best_validation = heldout(params)
    best_cores = initial_cores
    stale = 0
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    started = time.perf_counter()
    count = len(points)
    batch_size = min(int(batch_size), count)
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            chosen = rng.choice(count, size=batch_size, replace=False)
            batch = training_tensor[torch.as_tensor(
                chosen, dtype=torch.long, device=device
            )]
        optimizer.zero_grad(set_to_none=True)
        loss, normalization = nll(params, batch, tail_aware=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(params, 100.0)
        optimizer.step()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(normalization.detach()))
        history.gradient_norm.append(float(gradient_norm))
        check_validation = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0 or epoch + 1 == epochs)
        )
        validation_value = heldout(params) if check_validation else None
        if check_validation:
            history.validation_loss.append(validation_value)
            threshold = min_delta * max(1.0, abs(best_validation))
            if best_validation - validation_value > threshold:
                best_validation = validation_value
                best_cores = [p.detach().clone() for p in params]
                history.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                break
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            print(
                f"epoch {epoch + 1:5d}: quadratic_nll={float(loss):.7e}, "
                f"Z={float(normalization):.4e}"
            )
    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started
    fitted_cores = (
        best_cores if validation_tensor is not None
        else [p.detach() for p in params]
    )
    fitted = vector.from_list([
        core.cpu().numpy().copy() for core in fitted_cores
    ])
    fitted_second = float(_quadratic_right_gram_environments(
        _as_numpy_cores(fitted)
    )[0][0, 0])
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    return QuadraticSquaredTTDensity(fitted, gamma=fitted_gamma), history

__all__ = [
    "fit_purified_linear_tt_density",
    "fit_locally_purified_linear_tt_density",
    "_kmeans_labels_for_tt_initialization",
    "fit_nonnegative_linear_tt_density",
    "fit_quadratic_squared_tt_density",
]
