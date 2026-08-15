"""Permutation wrappers and rank-preserving density damping."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import _validate_points
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

class PermutedTTDensity:
    """A TT Rosenblatt layer conjugated by a coordinate permutation.

    Changing the triangular order between layers is the tensor-transport
    analogue of permutations between autoregressive flow blocks.  A
    permutation has unit Jacobian, so density evaluation and both directions
    of the transport remain exact while a deep composition is no longer
    triangular in one fixed ordering.
    """

    def __init__(
        self,
        density: DirectTTDensity | SquaredTTDensity | LinearSquaredTTDensity
        | PurifiedLinearTTDensity | LocallyPurifiedLinearTTDensity
        | NonnegativeLinearTTDensity
        | ProbitOrthogonalTTDensity
        | QuadraticSquaredTTDensity,
        permutation: Sequence[int],
    ):
        if not isinstance(
            density,
            (
                DirectTTDensity, SquaredTTDensity, LinearSquaredTTDensity,
                PurifiedLinearTTDensity, LocallyPurifiedLinearTTDensity,
                NonnegativeLinearTTDensity,
                ProbitOrthogonalTTDensity,
                QuadraticSquaredTTDensity,
            ),
        ):
            raise TypeError("density must be a supported TT density")
        permutation = np.asarray(permutation, dtype=np.int64)
        if permutation.shape != (density.d,) or not np.array_equal(
            np.sort(permutation), np.arange(density.d)
        ):
            raise ValueError("permutation must contain every dimension exactly once")
        self.base = density
        self.permutation = permutation.copy()
        self.inverse_permutation = np.argsort(permutation)

    @property
    def d(self) -> int:
        return self.base.d

    @property
    def modes(self) -> np.ndarray:
        result = np.empty(self.d, dtype=np.int64)
        result[self.permutation] = self.base.modes
        return result

    @property
    def ranks(self) -> np.ndarray:
        return self.base.ranks

    @property
    def size(self) -> int:
        return self.base.size

    @property
    def reference_floor_mass(self) -> float:
        """Uniform-reference mixture weight, unchanged by permutation."""
        return self.base.reference_floor_mass

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        internal = self.base.inverse_rosenblatt(uniform[:, self.permutation])
        return internal[:, self.inverse_permutation]

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        internal = self.base.rosenblatt(points[:, self.permutation])
        return internal[:, self.inverse_permutation]

    def rosenblatt_device(
        self,
        points,
        *,
        device: str,
        dtype: str = "float32",
        batch_size: int = 8192,
    ) -> np.ndarray:
        points = _validate_points(points, self.d)
        if not hasattr(self.base, "rosenblatt_device"):
            return self.rosenblatt(points)
        internal = self.base.rosenblatt_device(
            points[:, self.permutation],
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        return internal[:, self.inverse_permutation]

    def log_density(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return self.base.log_density(points[:, self.permutation])

    def log_density_device(
        self,
        points,
        *,
        device: str,
        dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        points = _validate_points(points, self.d)
        if not hasattr(self.base, "log_density_device"):
            return self.log_density(points)
        return self.base.log_density_device(
            points[:, self.permutation],
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    def density(self, points) -> np.ndarray:
        return np.exp(self.log_density(points))

    def sample(self, count: int, seed=None) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = np.random.default_rng(seed)
        return self.inverse_rosenblatt(rng.random((count, self.d)))

    def save(self, path) -> None:
        path = Path(path)
        self.base.save(path)
        with np.load(path) as stored:
            payload = {name: stored[name].copy() for name in stored.files}
        payload["kind"] = np.array([f"permuted-{payload['kind'][0]}"])
        payload["permutation"] = self.permutation
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "PermutedTTDensity":
        with np.load(path) as data:
            kind = str(data["kind"][0])
            permutation = data["permutation"].copy()
        if kind == "permuted-direct":
            base = DirectTTDensity.load(path)
        elif kind == "permuted-squared":
            base = SquaredTTDensity.load(path)
        elif kind == "permuted-linear-squared":
            base = LinearSquaredTTDensity.load(path)
        elif kind == "permuted-linear-adaptive-squared":
            base = AdaptiveLinearSquaredTTDensity.load(path)
        elif kind == "permuted-probit-orthogonal-linear-squared":
            base = ProbitOrthogonalTTDensity.load(path)
        elif kind == "permuted-linear-purified":
            base = PurifiedLinearTTDensity.load(path)
        elif kind == "permuted-linear-local-purified":
            base = LocallyPurifiedLinearTTDensity.load(path)
        elif kind == "permuted-linear-nonnegative":
            base = NonnegativeLinearTTDensity.load(path)
        elif kind == "permuted-quadratic-squared":
            base = QuadraticSquaredTTDensity.load(path)
        else:
            raise ValueError(f"unknown permuted density kind {kind!r}")
        return cls(base, permutation)


def damp_tt_density(
    density: DirectTTDensity | SquaredTTDensity | LinearSquaredTTDensity
    | PurifiedLinearTTDensity | LocallyPurifiedLinearTTDensity
    | NonnegativeLinearTTDensity
    | QuadraticSquaredTTDensity
    | PermutedTTDensity,
    weight: float,
):
    """Return ``(1-weight) * Uniform + weight * density`` exactly.

    Damping is a one-dimensional line search along the multiplicative DIRT
    update.  The returned object has the same TT ranks and basis as ``density``
    and remains exactly normalized.  For squared densities only the analytic
    uniform floor changes; for a direct centered density only one TT core is
    rescaled.  Consequently Rosenblatt evaluation and sampling need no new
    mixture wrapper or discrete latent variable.

    ``weight=0`` is deliberately excluded: the caller should reject the layer
    instead of storing a redundant identity transport.
    """
    weight = float(weight)
    if not np.isfinite(weight) or not 0.0 < weight <= 1.0:
        raise ValueError("weight must lie in (0, 1]")
    if isinstance(density, PermutedTTDensity):
        return PermutedTTDensity(
            damp_tt_density(density.base, weight), density.permutation
        )
    if isinstance(density, ProbitOrthogonalTTDensity):
        return ProbitOrthogonalTTDensity(
            damp_tt_density(density.base, weight), density.rotation,
            rotation_block_sizes=density.rotation_block_sizes,
            radial_twist_pairs=density.radial_twist_pairs,
            radial_twist_coefficients=density.radial_twist_coefficients,
            conditional_twist_pairs=density.conditional_twist_pairs,
            conditional_twist_conditioners=(
                density.conditional_twist_conditioners
            ),
            conditional_twist_coefficients=(
                density.conditional_twist_coefficients
            ),
        )
    if isinstance(density, DirectTTDensity):
        mean = density.normalization - 1.0
        scale_denominator = 1.0 + (1.0 - weight) * mean
        if not np.isfinite(scale_denominator) or scale_denominator <= 0.0:
            raise FloatingPointError("invalid direct-density damping scale")
        cores = [core.copy() for core in density._cores]
        cores[0] *= weight / scale_denominator
        return DirectTTDensity(
            vector.from_list(cores),
            conditional_floor=density.conditional_floor,
        )
    if isinstance(density, NonnegativeLinearTTDensity):
        cores = [core.copy() for core in density._cores]
        cores[0] *= weight
        gamma = (
            (1.0 - weight) * density.normalization
            + weight * density.gamma
        )
        return NonnegativeLinearTTDensity(
            vector.from_list(cores), gamma=gamma
        )
    if isinstance(density, AdaptiveLinearSquaredTTDensity):
        if weight == 1.0 or density._root_second_moment <= np.finfo(float).tiny:
            return AdaptiveLinearSquaredTTDensity(
                density.root, density.knots, gamma=density.gamma
            )
        old_floor = density.reference_floor_mass
        new_floor = 1.0 - weight * (1.0 - old_floor)
        if new_floor >= 1.0:
            raise FloatingPointError("damping weight is numerically too small")
        gamma = new_floor * density._root_second_moment / (1.0 - new_floor)
        return AdaptiveLinearSquaredTTDensity(
            density.root, density.knots, gamma=gamma
        )
    if isinstance(density, PurifiedLinearTTDensity):
        if weight == 1.0 or density._root_second_moment <= np.finfo(float).tiny:
            return PurifiedLinearTTDensity(
                density._cores, gamma=density.gamma
            )
        old_floor = density.reference_floor_mass
        new_floor = 1.0 - weight * (1.0 - old_floor)
        if new_floor >= 1.0:
            raise FloatingPointError("damping weight is numerically too small")
        gamma = new_floor * density._root_second_moment / (1.0 - new_floor)
        return PurifiedLinearTTDensity(density._cores, gamma=gamma)
    if isinstance(density, LocallyPurifiedLinearTTDensity):
        if weight == 1.0 or density._signal_integral <= np.finfo(float).tiny:
            return LocallyPurifiedLinearTTDensity(
                density._cores, gamma=density.gamma
            )
        old_floor = density.reference_floor_mass
        new_floor = 1.0 - weight * (1.0 - old_floor)
        if new_floor >= 1.0:
            raise FloatingPointError("damping weight is numerically too small")
        gamma = new_floor * density._signal_integral / (1.0 - new_floor)
        return LocallyPurifiedLinearTTDensity(density._cores, gamma=gamma)
    if isinstance(
        density,
        (SquaredTTDensity, LinearSquaredTTDensity, QuadraticSquaredTTDensity),
    ):
        if weight == 1.0 or density._root_second_moment <= np.finfo(float).tiny:
            return type(density)(density.root, gamma=density.gamma)
        old_floor = density.reference_floor_mass
        new_floor = 1.0 - weight * (1.0 - old_floor)
        if new_floor >= 1.0:
            raise FloatingPointError("damping weight is numerically too small")
        gamma = (
            new_floor * density._root_second_moment / (1.0 - new_floor)
        )
        return type(density)(density.root, gamma=gamma)
    raise TypeError("density must be a supported TT density")

__all__ = [
    "PermutedTTDensity",
    "damp_tt_density",
]
