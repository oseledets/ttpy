"""Measure-preserving coordinate transforms for TT densities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ._basis import _validate_points
from .scalar import AdaptiveLinearSquaredTTDensity, LinearSquaredTTDensity

def _validate_radial_twists(dimension, pairs, coefficients):
    """Validate disjoint-pair radial Gaussian twists stage by stage."""
    if pairs is None and coefficients is None:
        return (), ()
    if pairs is None or coefficients is None or len(pairs) != len(coefficients):
        raise ValueError("radial twist pairs and coefficients must match")
    validated_pairs = []
    validated_coefficients = []
    for stage_pairs, stage_coefficients in zip(pairs, coefficients):
        stage_pairs = np.asarray(stage_pairs, dtype=np.int64)
        stage_coefficients = np.asarray(stage_coefficients, dtype=np.float64)
        if (
            stage_pairs.ndim != 2 or stage_pairs.shape[1:] != (2,)
            or stage_coefficients.ndim != 2
            or stage_coefficients.shape[0] != len(stage_pairs)
            or stage_coefficients.shape[1] < 1
            or np.any(stage_pairs < 0) or np.any(stage_pairs >= dimension)
            or np.any(stage_pairs[:, 0] == stage_pairs[:, 1])
            or len(np.unique(stage_pairs)) != 2 * len(stage_pairs)
            or np.any(~np.isfinite(stage_coefficients))
        ):
            raise ValueError(
                "each radial twist stage needs disjoint valid pairs and "
                "one finite coefficient row per pair"
            )
        validated_pairs.append(stage_pairs.copy())
        validated_coefficients.append(stage_coefficients.copy())
    return tuple(validated_pairs), tuple(validated_coefficients)


def _validate_conditional_twists(
    dimension, pairs, conditioners, coefficients,
):
    """Validate conditional pair rotations with fixed stage conditioners."""
    if pairs is None and conditioners is None and coefficients is None:
        return (), (), ()
    if (
        pairs is None or conditioners is None or coefficients is None
        or len(pairs) != len(conditioners)
        or len(pairs) != len(coefficients)
    ):
        raise ValueError("conditional twist stage data must match")
    validated_pairs, validated_conditioners, validated_coefficients = [], [], []
    for stage_pairs, stage_conditioners, stage_coefficients in zip(
        pairs, conditioners, coefficients
    ):
        stage_pairs = np.asarray(stage_pairs, dtype=np.int64)
        stage_conditioners = np.asarray(stage_conditioners, dtype=np.int64)
        stage_coefficients = np.asarray(stage_coefficients, dtype=np.float64)
        targets = np.unique(stage_pairs)
        if (
            stage_pairs.ndim != 2 or stage_pairs.shape[1:] != (2,)
            or stage_conditioners.shape != (len(stage_pairs),)
            or stage_coefficients.ndim != 2
            or stage_coefficients.shape[0] != len(stage_pairs)
            or stage_coefficients.shape[1] < 1
            or np.any(stage_pairs < 0) or np.any(stage_pairs >= dimension)
            or np.any(stage_conditioners < 0)
            or np.any(stage_conditioners >= dimension)
            or np.any(stage_pairs[:, 0] == stage_pairs[:, 1])
            or len(targets) != 2 * len(stage_pairs)
            or np.intersect1d(targets, stage_conditioners).size
            or np.any(~np.isfinite(stage_coefficients))
        ):
            raise ValueError(
                "each conditional twist stage needs disjoint target pairs, "
                "unchanged conditioners and finite coefficient rows"
            )
        validated_pairs.append(stage_pairs.copy())
        validated_conditioners.append(stage_conditioners.copy())
        validated_coefficients.append(stage_coefficients.copy())
    return (
        tuple(validated_pairs), tuple(validated_conditioners),
        tuple(validated_coefficients),
    )


def _radial_twist_gaussian_numpy(
    gaussian: np.ndarray, pairs, coefficients, *, inverse: bool = False,
) -> np.ndarray:
    """Apply exact norm/volume-preserving radial twists in Gaussian space."""
    value = np.asarray(gaussian, dtype=np.float64).copy()
    stages = list(zip(pairs, coefficients))
    if inverse:
        stages.reverse()
    for stage_pairs, stage_coefficients in stages:
        first_index = stage_pairs[:, 0]
        second_index = stage_pairs[:, 1]
        first = value[:, first_index]
        second = value[:, second_index]
        radius_squared = first * first + second * second
        radial_quantile = -np.expm1(-0.5 * radius_squared)
        frequencies = np.arange(
            1, stage_coefficients.shape[1] + 1, dtype=np.float64
        )
        basis = np.sin(
            np.pi * radial_quantile[:, :, None]
            * frequencies[None, None, :]
        )
        angles = np.sum(
            basis * stage_coefficients[None, :, :], axis=2
        )
        if inverse:
            angles = -angles
        cosine, sine = np.cos(angles), np.sin(angles)
        value[:, first_index] = cosine * first - sine * second
        value[:, second_index] = sine * first + cosine * second
    return value


def _conditional_twist_gaussian_numpy(
    gaussian: np.ndarray, pairs, conditioners, coefficients,
    *, inverse: bool = False,
) -> np.ndarray:
    """Apply conditional pair rotations with exact triangular inverse."""
    from scipy.special import ndtr

    value = np.asarray(gaussian, dtype=np.float64).copy()
    stages = list(zip(pairs, conditioners, coefficients))
    if inverse:
        stages.reverse()
    for stage_pairs, stage_conditioners, stage_coefficients in stages:
        first_index = stage_pairs[:, 0]
        second_index = stage_pairs[:, 1]
        first = value[:, first_index]
        second = value[:, second_index]
        conditioning_uniform = ndtr(value[:, stage_conditioners])
        frequencies = np.arange(
            1, stage_coefficients.shape[1] + 1, dtype=np.float64
        )
        basis = np.sin(
            np.pi * conditioning_uniform[:, :, None]
            * frequencies[None, None, :]
        )
        angles = np.sum(
            basis * stage_coefficients[None, :, :], axis=2
        )
        if inverse:
            angles = -angles
        cosine, sine = np.cos(angles), np.sin(angles)
        value[:, first_index] = cosine * first - sine * second
        value[:, second_index] = sine * first + cosine * second
    return value


class ProbitOrthogonalTTDensity:
    """Exact Gaussian-measure-preserving coordinates followed by a TT density.

    For an orthogonal matrix ``R`` this wrapper evaluates the base density at

    ``C_R(u) = Phi(Phi^{-1}(u) R)``.

    Since ``||z R|| = ||z||``, the Jacobian determinant of ``C_R`` is exactly
    one.  Optional disjoint-pair radial twists add
    ``(r, phi) -> (r, phi + theta(1-exp(-r^2/2)))`` after the rotation.  They
    preserve both radius and polar area, hence the same exact invariant.
    Optional conditional twists rotate disjoint target pairs by angles that
    depend only on unchanged coordinates.  Their Jacobians are block
    triangular with unit-determinant rotation blocks, and they preserve the
    Gaussian norm pointwise.  The wrapper therefore changes TT separation
    coordinates without a density correction and has an analytic inverse.
    """

    def __init__(
        self, base, rotation: np.ndarray,
        rotation_block_sizes: Sequence[int] | None = None,
        radial_twist_pairs=None,
        radial_twist_coefficients=None,
        conditional_twist_pairs=None,
        conditional_twist_conditioners=None,
        conditional_twist_coefficients=None,
    ):
        if not all(hasattr(base, name) for name in (
            "d", "size", "density", "log_density", "rosenblatt",
            "inverse_rosenblatt", "sample", "save",
        )):
            raise TypeError("base must implement the TT density interface")
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (base.d, base.d):
            raise ValueError("rotation must have shape (dimension, dimension)")
        defect = np.linalg.norm(
            rotation.T @ rotation - np.eye(base.d), ord=2
        )
        if not np.isfinite(defect) or defect > 1e-8:
            raise ValueError("rotation must be orthogonal")
        self.base = base
        self.rotation = rotation.copy()
        self.d = int(base.d)
        if rotation_block_sizes is None:
            rotation_block_sizes = [self.d]
        self.rotation_block_sizes = np.asarray(
            rotation_block_sizes, dtype=np.int64
        )
        if (
            self.rotation_block_sizes.ndim != 1
            or np.any(self.rotation_block_sizes < 1)
            or int(self.rotation_block_sizes.sum()) != self.d
        ):
            raise ValueError("rotation block sizes must partition the dimension")
        start = 0
        for size in self.rotation_block_sizes:
            stop = start + int(size)
            outside = self.rotation[start:stop].copy()
            outside[:, start:stop] = 0.0
            if np.linalg.norm(outside, ord="fro") > 1e-8:
                raise ValueError("rotation must respect rotation_block_sizes")
            start = stop
        (
            self.radial_twist_pairs,
            self.radial_twist_coefficients,
        ) = _validate_radial_twists(
            self.d, radial_twist_pairs, radial_twist_coefficients
        )
        (
            self.conditional_twist_pairs,
            self.conditional_twist_conditioners,
            self.conditional_twist_coefficients,
        ) = _validate_conditional_twists(
            self.d,
            conditional_twist_pairs,
            conditional_twist_conditioners,
            conditional_twist_coefficients,
        )

    @property
    def size(self) -> int:
        stored_rotation = sum(
            int(size) ** 2 for size in self.rotation_block_sizes
        )
        stored_twists = sum(
            coefficients.size
            for coefficients in self.radial_twist_coefficients
        )
        stored_twists += sum(
            coefficients.size
            for coefficients in self.conditional_twist_coefficients
        )
        return int(self.base.size + stored_rotation + stored_twists)

    @property
    def ranks(self):
        return self.base.ranks

    @property
    def modes(self):
        return self.base.modes

    @property
    def reference_floor_mass(self) -> float:
        return self.base.reference_floor_mass

    def _rotate(self, points, *, inverse: bool = False) -> np.ndarray:
        from scipy.special import ndtr, ndtri

        points = _validate_points(points, self.d)
        epsilon = 1e-12
        gaussian = ndtri(np.clip(points, epsilon, 1.0 - epsilon))
        if inverse:
            gaussian = _conditional_twist_gaussian_numpy(
                gaussian,
                self.conditional_twist_pairs,
                self.conditional_twist_conditioners,
                self.conditional_twist_coefficients,
                inverse=True,
            )
            gaussian = _radial_twist_gaussian_numpy(
                gaussian,
                self.radial_twist_pairs,
                self.radial_twist_coefficients,
                inverse=True,
            )
            gaussian = gaussian @ self.rotation.T
        else:
            gaussian = gaussian @ self.rotation
            gaussian = _radial_twist_gaussian_numpy(
                gaussian,
                self.radial_twist_pairs,
                self.radial_twist_coefficients,
            )
            gaussian = _conditional_twist_gaussian_numpy(
                gaussian,
                self.conditional_twist_pairs,
                self.conditional_twist_conditioners,
                self.conditional_twist_coefficients,
            )
        return np.clip(ndtr(gaussian), 0.0, 1.0)

    def log_density(self, points) -> np.ndarray:
        return self.base.log_density(self._rotate(points))

    def density(self, points) -> np.ndarray:
        return np.exp(self.log_density(points))

    def log_density_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        rotated = self._rotate(points)
        if not hasattr(self.base, "log_density_device"):
            return self.base.log_density(rotated)
        return self.base.log_density_device(
            rotated, device=device, dtype=dtype, batch_size=batch_size
        )

    def rosenblatt(self, points) -> np.ndarray:
        return self.base.rosenblatt(self._rotate(points))

    def rosenblatt_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 8192,
    ) -> np.ndarray:
        rotated = self._rotate(points)
        if not hasattr(self.base, "rosenblatt_device"):
            return self.base.rosenblatt(rotated)
        return self.base.rosenblatt_device(
            rotated, device=device, dtype=dtype, batch_size=batch_size
        )

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        return self._rotate(
            self.base.inverse_rosenblatt(uniform), inverse=True
        )

    def sample(self, count: int, seed=None) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.inverse_rosenblatt(
            np.random.default_rng(seed).random((count, self.d))
        )

    def save(self, path) -> None:
        path = Path(path)
        self.base.save(path)
        with np.load(path) as stored:
            payload = {name: stored[name].copy() for name in stored.files}
        payload["kind"] = np.array([
            f"probit-orthogonal-{payload['kind'][0]}"
        ])
        payload["rotation_block_sizes"] = self.rotation_block_sizes
        start = 0
        for k, size in enumerate(self.rotation_block_sizes):
            stop = start + int(size)
            payload[f"rotation_block{k}"] = self.rotation[
                start:stop, start:stop
            ]
            start = stop
        payload["radial_twist_stages"] = np.array([
            len(self.radial_twist_pairs)
        ], dtype=np.int64)
        for k, (pairs, coefficients) in enumerate(zip(
            self.radial_twist_pairs, self.radial_twist_coefficients
        )):
            payload[f"radial_twist_pairs{k}"] = pairs
            payload[f"radial_twist_coefficients{k}"] = coefficients
        payload["conditional_twist_stages"] = np.array([
            len(self.conditional_twist_pairs)
        ], dtype=np.int64)
        for k, (pairs, conditioners, coefficients) in enumerate(zip(
            self.conditional_twist_pairs,
            self.conditional_twist_conditioners,
            self.conditional_twist_coefficients,
        )):
            payload[f"conditional_twist_pairs{k}"] = pairs
            payload[f"conditional_twist_conditioners{k}"] = conditioners
            payload[f"conditional_twist_coefficients{k}"] = coefficients
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "ProbitOrthogonalTTDensity":
        with np.load(path) as data:
            kind = str(data["kind"][0])
            if "rotation_block_sizes" in data:
                block_sizes = data["rotation_block_sizes"].copy()
                blocks = [
                    data[f"rotation_block{k}"].copy()
                    for k in range(len(block_sizes))
                ]
                from scipy.linalg import block_diag

                rotation = block_diag(*blocks)
            else:  # compatibility with initial experimental checkpoints
                rotation = data["rotation"].copy()
                block_sizes = np.array([len(rotation)], dtype=np.int64)
            stage_count = (
                int(data["radial_twist_stages"][0])
                if "radial_twist_stages" in data else 0
            )
            radial_pairs = [
                data[f"radial_twist_pairs{k}"].copy()
                for k in range(stage_count)
            ]
            radial_coefficients = [
                data[f"radial_twist_coefficients{k}"].copy()
                for k in range(stage_count)
            ]
            conditional_stage_count = (
                int(data["conditional_twist_stages"][0])
                if "conditional_twist_stages" in data else 0
            )
            conditional_pairs = [
                data[f"conditional_twist_pairs{k}"].copy()
                for k in range(conditional_stage_count)
            ]
            conditional_conditioners = [
                data[f"conditional_twist_conditioners{k}"].copy()
                for k in range(conditional_stage_count)
            ]
            conditional_coefficients = [
                data[f"conditional_twist_coefficients{k}"].copy()
                for k in range(conditional_stage_count)
            ]
        if kind.startswith("permuted-"):
            kind = kind[len("permuted-"):]
        prefix = "probit-orthogonal-"
        base_kind = kind[len(prefix):] if kind.startswith(prefix) else kind
        if base_kind == "linear-squared":
            base = LinearSquaredTTDensity.load(path)
        elif base_kind == "linear-adaptive-squared":
            base = AdaptiveLinearSquaredTTDensity.load(path)
        else:
            raise ValueError(f"unknown probit-orthogonal base kind {base_kind!r}")
        return cls(
            base, rotation, rotation_block_sizes=block_sizes,
            radial_twist_pairs=radial_pairs,
            radial_twist_coefficients=radial_coefficients,
            conditional_twist_pairs=conditional_pairs,
            conditional_twist_conditioners=conditional_conditioners,
            conditional_twist_coefficients=conditional_coefficients,
        )

__all__ = [
    "_validate_radial_twists",
    "_validate_conditional_twists",
    "_radial_twist_gaussian_numpy",
    "_conditional_twist_gaussian_numpy",
    "ProbitOrthogonalTTDensity",
]
