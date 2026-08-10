"""Analytic Rosenblatt transport for diagonal Gaussian mixtures.

The mixture components are independently truncated to a rectangular physical
domain and that box is represented by the unit cube.  Marginal and conditional
densities are finite mixtures of one-dimensional truncated Gaussians.  Their
CDFs are analytic; inverse CDFs are evaluated to machine precision by a
vectorised scalar bisection.

This class is primarily an oracle for validating learned transports.  It also
provides exact target log densities and direct independent samples without
using either a grid or a tensor approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri


def _unit_points(points, dimension: int, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError(
            f"{name} must have shape (n, {dimension}), got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]^{dimension}")
    return points


class TruncatedGaussianMixture:
    """Mixture of axis-aligned truncated Gaussians on the unit cube.

    ``means`` and ``scales`` are specified in physical coordinates.  Every
    component is truncated and normalised on ``[lower, upper]`` before mixing,
    and the physical box is mapped affinely to ``[0, 1]^d``.
    """

    def __init__(
        self,
        means,
        scales,
        *,
        weights=None,
        lower=None,
        upper=None,
        bisection_steps: int = 60,
    ):
        means = np.asarray(means, dtype=np.float64)
        if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
            raise ValueError("means must have shape (components, dimension)")
        if not np.all(np.isfinite(means)):
            raise ValueError("means contain non-finite values")
        components, dimension = means.shape

        scales = np.asarray(scales, dtype=np.float64)
        if scales.ndim == 0:
            scales = np.full_like(means, float(scales))
        elif scales.shape == (dimension,):
            scales = np.broadcast_to(scales, means.shape).copy()
        elif scales.shape != means.shape:
            raise ValueError(
                "scales must be scalar, shape (dimension,), or match means"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("scales must be finite and strictly positive")

        if weights is None:
            weights = np.full(components, 1.0 / components)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (components,):
            raise ValueError(f"weights must have shape ({components},)")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("weights must be finite and strictly positive")
        weights = weights / weights.sum()

        if lower is None:
            lower = np.min(means - 8.0 * scales, axis=0)
        if upper is None:
            upper = np.max(means + 8.0 * scales, axis=0)
        lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (dimension,))
        upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (dimension,))
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("lower and upper must define a finite nonempty box")
        if bisection_steps < 1:
            raise ValueError("bisection_steps must be positive")

        lower_z = (lower[None, :] - means) / scales
        upper_z = (upper[None, :] - means) / scales
        normalizers = ndtr(upper_z) - ndtr(lower_z)
        if np.any(normalizers <= np.finfo(float).tiny):
            raise ValueError("a component has numerically zero mass in the box")

        self.means = means
        self.scales = scales
        self.weights = weights
        self.lower = lower.copy()
        self.upper = upper.copy()
        self.width = self.upper - self.lower
        self.bisection_steps = int(bisection_steps)
        self._lower_z = lower_z
        self._upper_z = upper_z
        self._normalizers = normalizers
        self._log_normalizers = np.log(normalizers)
        self._log_weights = np.log(weights)

    @property
    def components(self) -> int:
        return self.means.shape[0]

    @property
    def dimension(self) -> int:
        return self.means.shape[1]

    def to_physical(self, unit) -> np.ndarray:
        unit = _unit_points(unit, self.dimension, name="unit")
        return self.lower + unit * self.width

    def to_unit(self, physical) -> np.ndarray:
        physical = np.asarray(physical, dtype=np.float64)
        if physical.ndim == 1:
            physical = physical.reshape(1, -1)
        if physical.ndim != 2 or physical.shape[1] != self.dimension:
            raise ValueError(
                f"physical must have shape (n, {self.dimension})"
            )
        return _unit_points(
            (physical - self.lower) / self.width,
            self.dimension,
            name="physical",
        )

    def _component_log_pdf(self, coordinate, k: int) -> np.ndarray:
        coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1, 1)
        physical = self.lower[k] + self.width[k] * coordinate
        z = (physical - self.means[None, :, k]) / self.scales[None, :, k]
        return (
            np.log(self.width[k])
            - np.log(self.scales[None, :, k])
            - 0.5 * z * z
            - 0.5 * np.log(2.0 * np.pi)
            - self._log_normalizers[None, :, k]
        )

    def _component_cdf(self, coordinate, k: int) -> np.ndarray:
        coordinate = np.asarray(coordinate, dtype=np.float64).reshape(-1, 1)
        physical = self.lower[k] + self.width[k] * coordinate
        z = (physical - self.means[None, :, k]) / self.scales[None, :, k]
        cdf = (
            ndtr(z) - ndtr(self._lower_z[None, :, k])
        ) / self._normalizers[None, :, k]
        return np.clip(cdf, 0.0, 1.0)

    def _modes_array(self, modes) -> np.ndarray:
        if isinstance(modes, (int, np.integer)):
            result = np.full(self.dimension, int(modes), dtype=np.int64)
        else:
            result = np.asarray(modes, dtype=np.int64)
        if result.shape != (self.dimension,) or np.any(result < 2):
            raise ValueError(
                f"modes must contain {self.dimension} integers greater than one"
            )
        return result

    def component_cell_masses(self, modes) -> list[np.ndarray]:
        """Exact component probabilities of all one-dimensional grid cells."""
        modes = self._modes_array(modes)
        result = []
        for k, mode in enumerate(modes):
            edges = np.linspace(0.0, 1.0, int(mode) + 1)
            cdf = self._component_cdf(edges, k).T
            result.append(np.maximum(np.diff(cdf, axis=1), 0.0))
        return result

    def cell_log_density(self, unit, modes) -> np.ndarray:
        """Exact L2-optimal piecewise-constant projection on a product grid."""
        unit = _unit_points(unit, self.dimension, name="unit")
        modes = self._modes_array(modes)
        indices = np.floor(unit * modes).astype(np.int64)
        indices = np.minimum(indices, modes - 1)
        terms = np.broadcast_to(
            self._log_weights, (unit.shape[0], self.components)
        ).copy()
        masses = self.component_cell_masses(modes)
        for k in range(self.dimension):
            selected = masses[k][:, indices[:, k]].T
            terms += np.log(np.maximum(selected, np.finfo(float).tiny))
        return logsumexp(terms, axis=1) + np.log(modes).sum()

    def cell_density(self, unit, modes) -> np.ndarray:
        return np.exp(self.cell_log_density(unit, modes))

    def cell_rosenblatt(self, unit, modes) -> np.ndarray:
        """Exact Rosenblatt map of the piecewise-constant grid projection."""
        unit = _unit_points(unit, self.dimension, name="unit")
        modes = self._modes_array(modes)
        masses = self.component_cell_masses(modes)
        count = unit.shape[0]
        result = np.empty_like(unit)
        posterior = np.broadcast_to(
            self.weights, (count, self.components)
        ).copy()
        rows = np.arange(count)
        for k, mode in enumerate(modes):
            conditional = posterior @ masses[k]
            conditional /= conditional.sum(axis=1, keepdims=True)
            cell = np.minimum(
                np.floor(unit[:, k] * mode).astype(np.int64), mode - 1
            )
            fraction = unit[:, k] * mode - cell
            cdf = np.cumsum(conditional, axis=1)
            lower = np.where(
                cell == 0, 0.0, cdf[rows, np.maximum(cell - 1, 0)]
            )
            result[:, k] = lower + fraction * conditional[rows, cell]
            posterior *= masses[k][:, cell].T
            posterior /= posterior.sum(axis=1, keepdims=True)
        return np.clip(result, 0.0, 1.0)

    def inverse_cell_rosenblatt(self, uniform, modes) -> np.ndarray:
        """Exact inverse Rosenblatt map of the product-grid projection."""
        uniform = _unit_points(uniform, self.dimension, name="uniform")
        modes = self._modes_array(modes)
        masses = self.component_cell_masses(modes)
        count = uniform.shape[0]
        result = np.empty_like(uniform)
        posterior = np.broadcast_to(
            self.weights, (count, self.components)
        ).copy()
        rows = np.arange(count)
        for k, mode in enumerate(modes):
            conditional = posterior @ masses[k]
            conditional /= conditional.sum(axis=1, keepdims=True)
            cdf = np.cumsum(conditional, axis=1)
            uk = np.minimum(uniform[:, k], np.nextafter(1.0, 0.0))
            cell = np.minimum(np.sum(cdf <= uk[:, None], axis=1), mode - 1)
            lower = np.where(
                cell == 0, 0.0, cdf[rows, np.maximum(cell - 1, 0)]
            )
            mass = np.maximum(conditional[rows, cell], np.finfo(float).tiny)
            fraction = np.clip((uk - lower) / mass, 0.0, 1.0)
            result[:, k] = (cell + fraction) / mode
            posterior *= masses[k][:, cell].T
            posterior /= posterior.sum(axis=1, keepdims=True)
        return result

    def log_density(self, unit) -> np.ndarray:
        """Exact mixture log density relative to Lebesgue measure on the cube."""
        unit = _unit_points(unit, self.dimension, name="unit")
        terms = np.broadcast_to(
            self._log_weights, (unit.shape[0], self.components)
        ).copy()
        for k in range(self.dimension):
            terms += self._component_log_pdf(unit[:, k], k)
        return logsumexp(terms, axis=1)

    def density(self, unit) -> np.ndarray:
        return np.exp(self.log_density(unit))

    def rosenblatt(self, unit) -> np.ndarray:
        """Map mixture samples to the uniform cube using analytic CDFs."""
        unit = _unit_points(unit, self.dimension, name="unit")
        count = unit.shape[0]
        result = np.empty_like(unit)
        log_posterior = np.broadcast_to(
            self._log_weights, (count, self.components)
        ).copy()
        for k in range(self.dimension):
            posterior = np.exp(
                log_posterior - logsumexp(log_posterior, axis=1, keepdims=True)
            )
            component_cdf = self._component_cdf(unit[:, k], k)
            result[:, k] = np.sum(posterior * component_cdf, axis=1)
            log_posterior += self._component_log_pdf(unit[:, k], k)
        return np.clip(result, 0.0, 1.0)

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        """Map uniform points to the mixture; scalar inverses use bisection."""
        uniform = _unit_points(uniform, self.dimension, name="uniform")
        count = uniform.shape[0]
        result = np.empty_like(uniform)
        log_posterior = np.broadcast_to(
            self._log_weights, (count, self.components)
        ).copy()
        for k in range(self.dimension):
            posterior = np.exp(
                log_posterior - logsumexp(log_posterior, axis=1, keepdims=True)
            )
            low = np.zeros(count, dtype=np.float64)
            high = np.ones(count, dtype=np.float64)
            for _ in range(self.bisection_steps):
                middle = 0.5 * (low + high)
                value = np.sum(
                    posterior * self._component_cdf(middle, k), axis=1
                )
                below = value < uniform[:, k]
                low = np.where(below, middle, low)
                high = np.where(below, high, middle)
            coordinate = 0.5 * (low + high)
            coordinate = np.where(uniform[:, k] == 0.0, 0.0, coordinate)
            coordinate = np.where(uniform[:, k] == 1.0, 1.0, coordinate)
            result[:, k] = coordinate
            log_posterior += self._component_log_pdf(coordinate, k)
        return result

    def sample(self, count: int, seed=None) -> np.ndarray:
        """Draw independent samples directly from the component representation."""
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = np.random.default_rng(seed)
        component = rng.choice(self.components, size=count, p=self.weights)
        quantile = rng.random((count, self.dimension))
        low = ndtr(self._lower_z[component])
        probability = low + quantile * self._normalizers[component]
        eps = np.finfo(float).eps
        z = ndtri(np.clip(probability, eps, 1.0 - eps))
        physical = self.means[component] + self.scales[component] * z
        return (physical - self.lower) / self.width

    def roundtrip_error(self, unit) -> float:
        unit = _unit_points(unit, self.dimension, name="unit")
        recovered = self.inverse_rosenblatt(self.rosenblatt(unit))
        return float(np.max(np.abs(recovered - unit), initial=0.0))
