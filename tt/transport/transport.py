"""Sample-DIRT composition, serialization, and exact continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import _as_numpy_cores, _linear_hat_mass_matrix, _validate_points
from ._fit_cell import (
    fit_centered_tt_density,
    fit_centered_tt_ratio,
    fit_squared_tt_density,
)
from ._fit_positive import (
    fit_locally_purified_linear_tt_density,
    fit_nonnegative_linear_tt_density,
    fit_purified_linear_tt_density,
    fit_quadratic_squared_tt_density,
)
from ._fit_spline import (
    fit_adaptive_linear_squared_tt_density,
    fit_linear_squared_tt_density,
    fit_probit_rotated_linear_squared_tt_density,
)
from ._types import FitHistory
from .coordinates import ProbitOrthogonalTTDensity
from .polynomial import DirectTTDensity, QuadraticSquaredTTDensity
from .positive import (
    LocallyPurifiedLinearTTDensity,
    NonnegativeLinearTTDensity,
    PurifiedLinearTTDensity,
)
from .scalar import (
    AdaptiveLinearSquaredTTDensity,
    LinearSquaredTTDensity,
    SquaredTTDensity,
)
from .wrappers import PermutedTTDensity

class SampleDIRT:
    """Composition of sample-fitted incremental TT Rosenblatt maps."""

    def __init__(
        self,
        dimension: int,
        layers: Iterable[
            DirectTTDensity | SquaredTTDensity | LinearSquaredTTDensity
            | PurifiedLinearTTDensity | LocallyPurifiedLinearTTDensity
            | NonnegativeLinearTTDensity
            | ProbitOrthogonalTTDensity
            | QuadraticSquaredTTDensity
            | PermutedTTDensity
        ] = (),
    ):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.layers: list[
            DirectTTDensity | SquaredTTDensity | LinearSquaredTTDensity
            | PurifiedLinearTTDensity | LocallyPurifiedLinearTTDensity
            | NonnegativeLinearTTDensity
            | ProbitOrthogonalTTDensity
            | QuadraticSquaredTTDensity
            | PermutedTTDensity
        ] = []
        for layer in layers:
            self.append(layer)

    def append(
        self,
        layer: DirectTTDensity | SquaredTTDensity | LinearSquaredTTDensity
        | PurifiedLinearTTDensity | LocallyPurifiedLinearTTDensity
        | NonnegativeLinearTTDensity
        | ProbitOrthogonalTTDensity
        | QuadraticSquaredTTDensity
        | PermutedTTDensity,
    ) -> None:
        if not isinstance(
            layer,
            (
                DirectTTDensity, SquaredTTDensity,
                LinearSquaredTTDensity, PurifiedLinearTTDensity,
                LocallyPurifiedLinearTTDensity,
                NonnegativeLinearTTDensity,
                ProbitOrthogonalTTDensity,
                QuadraticSquaredTTDensity,
                PermutedTTDensity,
            ),
        ):
            raise TypeError(
                "layer must be a supported TT density or PermutedTTDensity"
            )
        if layer.d != self.dimension:
            raise ValueError(
                f"layer dimension {layer.d} does not match model dimension {self.dimension}"
            )
        self.layers.append(layer)

    def forward(self, uniform) -> np.ndarray:
        """Generate physical samples; newest incremental layer is applied first."""
        value = _validate_points(uniform, self.dimension, name="uniform").copy()
        for layer in reversed(self.layers):
            value = layer.inverse_rosenblatt(value)
        return value

    def inverse(
        self,
        points,
        *,
        device: str | None = None,
        dtype: str = "float32",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Map physical points to the reference cube."""
        value = _validate_points(points, self.dimension).copy()
        for layer_index, layer in enumerate(self.layers):
            if (
                device is not None
                and str(device) != "cpu"
                and hasattr(layer, "rosenblatt_device")
            ):
                options = {"device": device, "dtype": dtype}
                if batch_size is not None:
                    options["batch_size"] = int(batch_size)
                value = layer.rosenblatt_device(value, **options)
            else:
                value = layer.rosenblatt(value)
        return value

    def inverse_and_log_density(
        self,
        points,
        *,
        device: str | None = None,
        dtype: str = "float32",
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map to the cube and accumulate the exact Jacobian in one pass."""
        value = _validate_points(points, self.dimension).copy()
        result = np.zeros(value.shape[0], dtype=np.float64)
        for layer in self.layers:
            accelerated = (
                device is not None
                and str(device) != "cpu"
                and hasattr(layer, "log_density_device")
                and hasattr(layer, "rosenblatt_device")
            )
            if accelerated:
                options = {"device": device, "dtype": dtype}
                if batch_size is not None:
                    options["batch_size"] = int(batch_size)
                result += layer.log_density_device(value, **options)
                value = layer.rosenblatt_device(value, **options)
            else:
                result += layer.log_density(value)
                value = layer.rosenblatt(value)
        return value, result

    def sample(self, count: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.forward(rng.random((count, self.dimension)))

    def log_density(
        self,
        points,
        *,
        device: str | None = None,
        dtype: str = "float32",
    ) -> np.ndarray:
        """Evaluate the density induced by the complete transport composition."""
        value = _validate_points(points, self.dimension).copy()
        result = np.zeros(value.shape[0], dtype=np.float64)
        for layer_index, layer in enumerate(self.layers):
            accelerated = (
                device is not None
                and str(device) != "cpu"
                and hasattr(layer, "log_density_device")
                and hasattr(layer, "rosenblatt_device")
            )
            if accelerated:
                result += layer.log_density_device(
                    value, device=device, dtype=dtype
                )
                if layer_index + 1 < len(self.layers):
                    value = layer.rosenblatt_device(
                        value, device=device, dtype=dtype
                    )
            else:
                result += layer.log_density(value)
                if layer_index + 1 < len(self.layers):
                    value = layer.rosenblatt(value)
        return result

    def density(self, points) -> np.ndarray:
        return np.exp(self.log_density(points))

    def fit_layer(
        self,
        samples_next,
        *,
        estimator: str = "centered",
        permutation: Sequence[int] | None = None,
        **fit_options,
    ) -> FitHistory:
        samples_next = _validate_points(samples_next, self.dimension,
                                        name="samples_next")
        transport_device = fit_options.get("device")
        # Evaluating an existing composition is numerically more demanding
        # than optimizing one local layer.  Keep the two dtypes independent:
        # a long squared-TT Rosenblatt chain may need float64 residual
        # coordinates even when the next layer is trained in float32.
        transport_dtype = fit_options.pop(
            "transport_dtype", fit_options.get("dtype", "float32")
        )
        transport_batch_size = fit_options.pop("transport_batch_size", None)
        residual_samples = self.inverse(
            samples_next,
            device=transport_device,
            dtype=transport_dtype,
            batch_size=transport_batch_size,
        )
        estimator = str(estimator).lower().replace("_", "-")
        validation_next = fit_options.pop("validation_samples", None)
        if validation_next is not None:
            validation_next = _validate_points(
                validation_next,
                self.dimension,
                name="validation_samples",
            )
            validation_next = self.inverse(
                validation_next,
                device=transport_device,
                dtype=transport_dtype,
                batch_size=transport_batch_size,
            )
        if permutation is not None:
            permutation = np.asarray(permutation, dtype=np.int64)
            if permutation.shape != (self.dimension,) or not np.array_equal(
                np.sort(permutation), np.arange(self.dimension)
            ):
                raise ValueError(
                    "permutation must contain every dimension exactly once"
                )
            residual_samples = residual_samples[:, permutation]
            if validation_next is not None:
                validation_next = validation_next[:, permutation]
        if estimator == "squared":
            layer, history = fit_squared_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator == "linear-squared":
            layer, history = fit_linear_squared_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator in ("linear-adaptive-squared", "adaptive-linear-squared"):
            layer, history = fit_adaptive_linear_squared_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator in (
            "linear-probit-orthogonal", "probit-orthogonal-linear"
        ):
            layer, history = fit_probit_rotated_linear_squared_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator in ("linear-purified", "purified-linear"):
            layer, history = fit_purified_linear_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator in (
            "linear-local-purified", "locally-purified-linear"
        ):
            layer, history = fit_locally_purified_linear_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator in ("linear-nonnegative", "nonnegative-linear"):
            layer, history = fit_nonnegative_linear_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator == "quadratic-squared":
            layer, history = fit_quadratic_squared_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        elif estimator == "centered":
            layer, history = fit_centered_tt_density(
                residual_samples,
                validation_samples=validation_next,
                **fit_options,
            )
        else:
            raise ValueError(
                "estimator must be 'squared', 'linear-squared', "
                "'linear-adaptive-squared', "
                "'linear-probit-orthogonal', "
                "'linear-purified', 'linear-local-purified', "
                "'linear-nonnegative', "
                "'quadratic-squared' or 'centered'"
            )
        if permutation is not None:
            layer = PermutedTTDensity(layer, permutation)
        self.append(layer)
        return history

    def fit_ratio_layer(
        self,
        samples_current,
        samples_next,
        *,
        validation_current=None,
        validation_next=None,
        **fit_options,
    ) -> FitHistory:
        """Append a nominal centered-ratio layer from two bridge sample sets.

        Both laws are pulled back through the same existing composition.  The
        density ratio is invariant under this common invertible change of
        coordinates, so the fitted correction approximates
        ``p_next / p_current - 1`` rather than absorbing all prior transport
        error.  This is the sample analogue of approximate-ratio DIRT.
        """
        current = self.inverse(_validate_points(
            samples_current, self.dimension, name="samples_current"
        ))
        next_ = self.inverse(_validate_points(
            samples_next, self.dimension, name="samples_next"
        ))
        if (validation_current is None) != (validation_next is None):
            raise ValueError(
                "provide both validation_current and validation_next or neither"
            )
        if validation_current is not None:
            validation_current = self.inverse(_validate_points(
                validation_current,
                self.dimension,
                name="validation_current",
            ))
            validation_next = self.inverse(_validate_points(
                validation_next,
                self.dimension,
                name="validation_next",
            ))
        layer, history = fit_centered_tt_ratio(
            current,
            next_,
            validation_denominator=validation_current,
            validation_numerator=validation_next,
            **fit_options,
        )
        self.append(layer)
        return history

    def roundtrip_error(self, points) -> float:
        points = _validate_points(points, self.dimension)
        if points.shape[0] == 0:
            return 0.0
        return float(np.max(np.abs(self.forward(self.inverse(points)) - points)))

    @property
    def stored_parameters(self) -> int:
        return int(sum(layer.size for layer in self.layers))

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "metadata.npz",
            dimension=np.array([self.dimension], dtype=np.int64),
            layers=np.array([len(self.layers)], dtype=np.int64),
        )
        for k, layer in enumerate(self.layers):
            layer.save(directory / f"layer_{k:03d}.npz")

    @classmethod
    def load(cls, directory) -> "SampleDIRT":
        directory = Path(directory)
        with np.load(directory / "metadata.npz") as data:
            dimension = int(data["dimension"][0])
            count = int(data["layers"][0])
        layers = []
        for k in range(count):
            path = directory / f"layer_{k:03d}.npz"
            with np.load(path) as data:
                kind = str(data["kind"][0]) if "kind" in data else "squared"
            if kind == "direct":
                layers.append(DirectTTDensity.load(path))
            elif kind == "squared":
                layers.append(SquaredTTDensity.load(path))
            elif kind == "linear-squared":
                layers.append(LinearSquaredTTDensity.load(path))
            elif kind == "linear-adaptive-squared":
                layers.append(AdaptiveLinearSquaredTTDensity.load(path))
            elif kind == "probit-orthogonal-linear-squared":
                layers.append(ProbitOrthogonalTTDensity.load(path))
            elif kind == "linear-purified":
                layers.append(PurifiedLinearTTDensity.load(path))
            elif kind == "linear-local-purified":
                layers.append(LocallyPurifiedLinearTTDensity.load(path))
            elif kind == "linear-nonnegative":
                layers.append(NonnegativeLinearTTDensity.load(path))
            elif kind == "quadratic-squared":
                layers.append(QuadraticSquaredTTDensity.load(path))
            elif kind in (
                "permuted-direct", "permuted-squared",
                "permuted-linear-squared",
                "permuted-linear-adaptive-squared",
                "permuted-probit-orthogonal-linear-squared",
                "permuted-linear-purified",
                "permuted-linear-local-purified",
                "permuted-linear-nonnegative",
                "permuted-quadratic-squared",
            ):
                layers.append(PermutedTTDensity.load(path))
            else:
                raise ValueError(f"unknown SampleDIRT layer kind {kind!r}")
        return cls(dimension, layers)


def enable_linear_sample_dirt_probit_rotations(
    model: SampleDIRT,
    *,
    layer_indices: Sequence[int] | None = None,
    block_size: int = 0,
) -> SampleDIRT:
    """Add identity probit-orthogonal coordinates to scalar TT layers.

    The probit conjugate of an identity Gaussian rotation is the identity map
    of the open unit cube.  Wrapping a fitted scalar linear-TT density this way
    therefore preserves its density and Rosenblatt map at checkpoint zero,
    while exposing tangent-skew rotation parameters to
    :func:`fine_tune_linear_sample_dirt`.  ``block_size <= 0`` enables one full
    rotation; a positive size partitions coordinates into consecutive blocks.
    Existing probit-orthogonal layers are retained unchanged.
    """
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    if block_size < 0:
        raise ValueError("block_size must be non-negative")
    if layer_indices is None:
        selected = set(range(len(model.layers)))
    else:
        selected = {int(index) for index in layer_indices}
        if (
            not selected
            or len(selected) != len(layer_indices)
            or any(index < 0 or index >= len(model.layers) for index in selected)
        ):
            raise ValueError("layer_indices must be unique valid layer indices")
    if block_size == 1:
        raise ValueError("block_size one has no trainable rotation directions")
    if block_size <= 0 or block_size >= model.dimension:
        rotation_block_sizes = (model.dimension,)
    else:
        quotient, remainder = divmod(model.dimension, int(block_size))
        rotation_block_sizes = (int(block_size),) * quotient
        if remainder:
            rotation_block_sizes += (remainder,)

    def wrap_density(density):
        permutation = None
        if isinstance(density, PermutedTTDensity):
            permutation = density.permutation.copy()
            density = density.base
        if isinstance(density, ProbitOrthogonalTTDensity):
            result = density
        else:
            if type(density) is not LinearSquaredTTDensity:
                raise TypeError(
                    "identity probit coordinates support scalar linear-"
                    "squared TT layers, optionally wrapped by permutations"
                )
            result = ProbitOrthogonalTTDensity(
                density,
                np.eye(model.dimension, dtype=np.float64),
                rotation_block_sizes=rotation_block_sizes,
            )
        return (
            result if permutation is None
            else PermutedTTDensity(result, permutation)
        )

    return SampleDIRT(model.dimension, [
        wrap_density(layer) if index in selected else layer
        for index, layer in enumerate(model.layers)
    ])


def refine_linear_sample_dirt_modes(
    model: SampleDIRT,
    modes: int | Sequence[int],
    *,
    layer_indices: Sequence[int] | None = None,
    coordinate_indices: Sequence[int] | None = None,
) -> SampleDIRT:
    """Embed uniform linear-TT layers in a nested finer nodal basis.

    A target grid is admissible when every old interval is split into an
    integer number of equal subintervals, i.e. ``(m_new - 1)`` is divisible
    by ``(m_old - 1)``.  Each new core slice is initialized by evaluating the
    old piecewise-linear matrix-valued core at the fine node.  The represented
    TT root, normalized density and Rosenblatt map are consequently unchanged
    at checkpoint zero, while subsequent optimization may move the new nodal
    values independently.  ``coordinate_indices`` restricts this operation
    to physical (pre-permutation) coordinates, leaving every other core at
    its stored resolution.  This is exact finite-element prolongation, not a
    higher-resolution reinitialization.
    """
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    if isinstance(modes, (int, np.integer)):
        target_modes = np.full(model.dimension, int(modes), dtype=np.int64)
    else:
        target_modes = np.asarray([int(value) for value in modes], dtype=np.int64)
        if target_modes.shape != (model.dimension,):
            raise ValueError("modes must be one integer or one value per coordinate")
    if np.any(target_modes < 2):
        raise ValueError("refined linear modes must be at least two")
    if layer_indices is None:
        selected = set(range(len(model.layers)))
    else:
        selected = {int(index) for index in layer_indices}
        if (
            not selected
            or len(selected) != len(layer_indices)
            or any(index < 0 or index >= len(model.layers) for index in selected)
        ):
            raise ValueError("layer_indices must be unique valid layer indices")
    if coordinate_indices is None:
        selected_coordinates = None
    else:
        selected_coordinates = {int(index) for index in coordinate_indices}
        if (
            not selected_coordinates
            or len(selected_coordinates) != len(coordinate_indices)
            or any(
                index < 0 or index >= model.dimension
                for index in selected_coordinates
            )
        ):
            raise ValueError(
                "coordinate_indices must be unique valid physical coordinates"
            )

    def refine_density(density):
        permutation = None
        if isinstance(density, PermutedTTDensity):
            permutation = density.permutation.copy()
            density = density.base
        wrapper = None
        if isinstance(density, ProbitOrthogonalTTDensity):
            wrapper = density
            density = density.base
        if type(density) is not LinearSquaredTTDensity:
            raise TypeError(
                "mode refinement supports scalar linear-squared TT layers, "
                "optionally wrapped by probit coordinates and permutations"
            )
        requested_internal_modes = (
            target_modes
            if permutation is None else target_modes[permutation]
        )
        old_internal_modes = np.asarray(
            [core.shape[1] for core in density._cores], dtype=np.int64
        )
        if selected_coordinates is None:
            internal_modes = requested_internal_modes
        else:
            physical_coordinates = (
                np.arange(model.dimension, dtype=np.int64)
                if permutation is None else permutation
            )
            active = np.asarray([
                int(index) in selected_coordinates
                for index in physical_coordinates
            ])
            internal_modes = np.where(
                active, requested_internal_modes, old_internal_modes
            )
        refined_cores = []
        for coordinate, (core, target) in enumerate(
            zip(density._cores, internal_modes)
        ):
            old = int(core.shape[1])
            target = int(target)
            if target < old or (target - 1) % (old - 1) != 0:
                raise ValueError(
                    "every refined grid must contain all old uniform nodes; "
                    f"coordinate {coordinate} cannot refine {old} to {target}"
                )
            factor = (target - 1) // (old - 1)
            position = np.arange(target, dtype=np.float64) / factor
            lower = np.minimum(np.floor(position).astype(np.int64), old - 2)
            fraction = position - lower
            refined_cores.append(
                (1.0 - fraction)[None, :, None] * core[:, lower, :]
                + fraction[None, :, None] * core[:, lower + 1, :]
            )
        result = LinearSquaredTTDensity(
            vector.from_list(refined_cores), gamma=density.gamma
        )
        if wrapper is not None:
            result = ProbitOrthogonalTTDensity(
                result,
                wrapper.rotation,
                rotation_block_sizes=wrapper.rotation_block_sizes,
                radial_twist_pairs=wrapper.radial_twist_pairs,
                radial_twist_coefficients=wrapper.radial_twist_coefficients,
                conditional_twist_pairs=wrapper.conditional_twist_pairs,
                conditional_twist_conditioners=(
                    wrapper.conditional_twist_conditioners
                ),
                conditional_twist_coefficients=(
                    wrapper.conditional_twist_coefficients
                ),
            )
        if permutation is not None:
            result = PermutedTTDensity(result, permutation)
        return result

    return SampleDIRT(model.dimension, [
        refine_density(layer) if index in selected else layer
        for index, layer in enumerate(model.layers)
    ])


def enrich_linear_sample_dirt_ranks(
    model: SampleDIRT,
    rank: int | Sequence[int],
    *,
    initialization_noise: float = 1e-2,
    seed: int = 0,
) -> SampleDIRT:
    """Embed scalar linear-TT layers at larger ranks without changing them.

    Old cores occupy the upper-left rank blocks.  At every internal bond the
    left core receives small old-input/new-output columns, while the matching
    new-input rows of the next core are initially zero.  Consequently every
    new path is killed and the represented root, density and Rosenblatt map
    are exactly the input model at checkpoint zero.  The zero rows nevertheless
    have nonzero first-order gradients through the random columns, so global
    NLL optimization can immediately activate new Schmidt directions.  This
    is a one-sided DMRG-style rank enrichment, not random reinitialization.
    """
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    if initialization_noise <= 0.0:
        raise ValueError("initialization_noise must be positive")
    if isinstance(rank, (int, np.integer)):
        layer_ranks = [int(rank)] * len(model.layers)
    else:
        layer_ranks = [int(value) for value in rank]
        if len(layer_ranks) != len(model.layers):
            raise ValueError("rank must provide one cap per layer")
    if any(value < 1 for value in layer_ranks):
        raise ValueError("enriched ranks must be positive")
    generator = np.random.default_rng(seed)

    def enrich_density(density, target_rank):
        permutation = None
        if isinstance(density, PermutedTTDensity):
            permutation = density.permutation.copy()
            density = density.base
        wrapper = None
        if isinstance(density, ProbitOrthogonalTTDensity):
            wrapper = density
            density = density.base
        if type(density) is not LinearSquaredTTDensity:
            raise TypeError(
                "rank enrichment supports scalar linear-squared TT layers, "
                "optionally wrapped by probit coordinates and permutations"
            )
        old_cores = density._cores
        old_ranks = [old_cores[0].shape[0]] + [
            core.shape[2] for core in old_cores
        ]
        # Boundary bonds cannot exceed the dimension of either matricization
        # side (e.g. a modes-6 first bond has maximum rank six).  Cap the
        # requested continuation by these exact algebraic limits instead of
        # asking QR to preserve an unattainable padded rank.
        left_capacities = []
        capacity = 1
        for core in old_cores[:-1]:
            capacity = min(
                int(target_rank), capacity * int(core.shape[1])
            )
            left_capacities.append(capacity)
        right_capacities = []
        capacity = 1
        for core in reversed(old_cores[1:]):
            capacity = min(
                int(target_rank), capacity * int(core.shape[1])
            )
            right_capacities.append(capacity)
        right_capacities.reverse()
        new_ranks = [1] + [
            max(
                int(old_rank),
                min(
                    int(target_rank), int(left_capacity), int(right_capacity)
                ),
            )
            for old_rank, left_capacity, right_capacity in zip(
                old_ranks[1:-1], left_capacities, right_capacities
            )
        ] + [1]
        enriched_cores = []
        for coordinate, core in enumerate(old_cores):
            old_left, modes, old_right = core.shape
            new_core = np.zeros(
                (new_ranks[coordinate], modes, new_ranks[coordinate + 1]),
                dtype=np.float64,
            )
            new_core[:old_left, :, :old_right] = core
            if new_ranks[coordinate + 1] > old_right:
                scale = float(np.linalg.norm(core)) / np.sqrt(core.size)
                scale = max(scale, np.finfo(np.float64).eps)
                new_core[:old_left, :, old_right:] = (
                    float(initialization_noise) * scale
                    * generator.normal(size=(
                        old_left,
                        modes,
                        new_ranks[coordinate + 1] - old_right,
                    ))
                )
            enriched_cores.append(new_core)
        result = LinearSquaredTTDensity(
            vector.from_list(enriched_cores), gamma=density.gamma
        )
        if wrapper is not None:
            result = ProbitOrthogonalTTDensity(
                result,
                wrapper.rotation,
                rotation_block_sizes=wrapper.rotation_block_sizes,
                radial_twist_pairs=wrapper.radial_twist_pairs,
                radial_twist_coefficients=wrapper.radial_twist_coefficients,
                conditional_twist_pairs=wrapper.conditional_twist_pairs,
                conditional_twist_conditioners=(
                    wrapper.conditional_twist_conditioners
                ),
                conditional_twist_coefficients=(
                    wrapper.conditional_twist_coefficients
                ),
            )
        if permutation is not None:
            result = PermutedTTDensity(result, permutation)
        return result

    return SampleDIRT(model.dimension, [
        enrich_density(layer, target_rank)
        for layer, target_rank in zip(model.layers, layer_ranks)
    ])


def truncate_linear_sample_dirt_ranks(
    model: SampleDIRT,
    rank: int | Sequence[int],
    *,
    tolerance: float = 1e-12,
    layer_indices: Sequence[int] | None = None,
) -> SampleDIRT:
    """Best functional-L2 TT rank truncation of scalar linear layers.

    Nodal coefficients are not Euclidean coordinates of
    ``L2([0,1]^d)``.  Each physical core is first multiplied by the Cholesky
    factor of its exact linear-hat mass matrix.  Standard TT-SVD is then the
    Hilbert-space truncation in this weighted tensor product.  Transforming
    the rounded cores back retains the original nodal basis, gamma floor,
    permutations and probit/orthogonal coordinate wrappers.
    """
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if isinstance(rank, (int, np.integer)):
        layer_ranks = [int(rank)] * len(model.layers)
    else:
        layer_ranks = [int(value) for value in rank]
        if len(layer_ranks) != len(model.layers):
            raise ValueError("rank must provide one cap per layer")
    if any(value < 1 for value in layer_ranks):
        raise ValueError("truncated ranks must be positive")
    if layer_indices is None:
        selected = set(range(len(model.layers)))
    else:
        selected = {int(index) for index in layer_indices}
        if (
            not selected
            or len(selected) != len(layer_indices)
            or any(index < 0 or index >= len(model.layers) for index in selected)
        ):
            raise ValueError("layer_indices must be unique valid layer indices")

    def truncate_density(density, target_rank):
        permutation = None
        if isinstance(density, PermutedTTDensity):
            permutation = density.permutation.copy()
            density = density.base
        wrapper = None
        if isinstance(density, ProbitOrthogonalTTDensity):
            wrapper = density
            density = density.base
        if type(density) is not LinearSquaredTTDensity:
            raise TypeError(
                "rank truncation supports scalar linear-squared TT layers, "
                "optionally wrapped by probit coordinates and permutations"
            )
        cholesky_factors = [
            np.linalg.cholesky(_linear_hat_mass_matrix(core.shape[1]))
            for core in density._cores
        ]
        weighted = [
            np.einsum("aic,ij->ajc", core, cholesky, optimize=True)
            for core, cholesky in zip(density._cores, cholesky_factors)
        ]
        rounded = vector.from_list(weighted).round(
            eps=float(tolerance), rmax=int(target_rank)
        )
        nodal = []
        for core, cholesky in zip(
            _as_numpy_cores(rounded), cholesky_factors
        ):
            inverse = np.linalg.solve(
                cholesky,
                np.eye(cholesky.shape[0], dtype=np.float64),
            )
            nodal.append(np.einsum(
                "ajc,ji->aic", core, inverse, optimize=True
            ))
        result = LinearSquaredTTDensity(
            vector.from_list(nodal), gamma=density.gamma
        )
        if wrapper is not None:
            result = ProbitOrthogonalTTDensity(
                result,
                wrapper.rotation,
                rotation_block_sizes=wrapper.rotation_block_sizes,
                radial_twist_pairs=wrapper.radial_twist_pairs,
                radial_twist_coefficients=wrapper.radial_twist_coefficients,
                conditional_twist_pairs=wrapper.conditional_twist_pairs,
                conditional_twist_conditioners=(
                    wrapper.conditional_twist_conditioners
                ),
                conditional_twist_coefficients=(
                    wrapper.conditional_twist_coefficients
                ),
            )
        if permutation is not None:
            result = PermutedTTDensity(result, permutation)
        return result

    return SampleDIRT(model.dimension, [
        truncate_density(layer, layer_ranks[index])
        if index in selected else layer
        for index, layer in enumerate(model.layers)
    ])

__all__ = [
    "SampleDIRT",
    "enable_linear_sample_dirt_probit_rotations",
    "refine_linear_sample_dirt_modes",
    "enrich_linear_sample_dirt_ranks",
    "truncate_linear_sample_dirt_ranks",
]
