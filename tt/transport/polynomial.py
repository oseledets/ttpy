"""Quadratic and direct cellwise TT density models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _as_numpy_cores,
    _cell_indices,
    _quadratic_basis_weights,
    _quadratic_indices_fractions,
    _quadratic_right_gram_environments,
    _sample_quadratic_tt_numpy,
    _sample_tt_numpy,
    _second_moment_numpy,
    _validate_points,
)
from ._torch_density import _torch_sample_quadratic_tt

class QuadraticSquaredTTDensity:
    """Positive squared TT in a uniform cardinal-quadratic B-spline basis.

    There are ``n-2`` intervals for ``n`` basis functions. Exactly three
    basis functions are nonzero on each interval. Gram contractions and
    conditional CDFs are exact: the conditional density is quartic and its
    partial integral is quintic.
    """

    def __init__(self, root: vector, gamma: float = 1e-6):
        if not isinstance(root, vector) or root.d == 0:
            raise TypeError("root must be a non-empty tt.vector")
        if root.r[0] != 1 or root.r[-1] != 1:
            raise ValueError("root must be a scalar TT")
        if np.any(root.n < 3):
            raise ValueError("quadratic TT modes need at least three functions")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self.root = vector.from_list(_as_numpy_cores(root))
        self.gamma = float(gamma)
        self._cores = _as_numpy_cores(self.root)
        self._right = _quadratic_right_gram_environments(self._cores)
        self._root_second_moment = float(self._right[0][0, 0])
        self.normalization = self.gamma + self._root_second_moment

    @property
    def d(self) -> int:
        return self.root.d

    @property
    def modes(self) -> np.ndarray:
        return self.root.n.astype(np.int64)

    @property
    def ranks(self) -> np.ndarray:
        return self.root.r.astype(np.int64)

    @property
    def size(self) -> int:
        return self.root.size

    @property
    def reference_floor_mass(self) -> float:
        """Normalized weight of the uniform reference component."""
        return self.gamma / self.normalization

    def root_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_quadratic_tt_numpy(self._cores, points)

    def density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return (self.gamma + values * values) / self.normalization

    def log_density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return np.log(self.gamma + values * values) - np.log(
            self.normalization
        )

    def log_density_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        import torch

        points = _validate_points(points, self.d)
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        cores = [
            torch.as_tensor(core, dtype=torch_dtype, device=device)
            for core in self._cores
        ]
        output = np.empty(len(points), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(points), int(batch_size)):
                stop = min(start + int(batch_size), len(points))
                batch = torch.as_tensor(
                    points[start:stop], dtype=torch_dtype, device=device
                )
                values = _torch_sample_quadratic_tt(cores, batch)
                output[start:stop] = (
                    torch.log(self.gamma + values.square())
                    - np.log(self.normalization)
                ).cpu().numpy()
        return output

    def _conditional_coefficients_batch(
        self, left: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        core = self._cores[k]
        nodes = np.einsum("pa,aib->pib", left, core, optimize=True)
        first, middle, last = nodes[:, :-2], nodes[:, 1:-1], nodes[:, 2:]
        polynomial = (
            0.5 * (first + middle),
            -first + middle,
            0.5 * first - middle + 0.5 * last,
        )
        right = self._right[k + 1]
        projected = [
            np.einsum("pia,ab->pib", value, right, optimize=True)
            for value in polynomial
        ]
        a, b, c = polynomial
        pa, pb, pc = projected
        coefficients = np.stack([
            np.einsum("pib,pib->pi", pa, a, optimize=True),
            2.0 * np.einsum("pib,pib->pi", pb, a, optimize=True),
            np.einsum("pib,pib->pi", pb, b, optimize=True)
            + 2.0 * np.einsum("pib,pib->pi", pc, a, optimize=True),
            2.0 * np.einsum("pib,pib->pi", pc, b, optimize=True),
            np.einsum("pib,pib->pi", pc, c, optimize=True),
        ], axis=-1)
        inverse_powers = 1.0 / np.arange(1.0, 6.0)
        adjusted = coefficients.copy()
        adjusted[..., 0] += self.gamma
        masses = (
            adjusted * inverse_powers[None, None, :]
        ).sum(axis=-1) / (core.shape[1] - 2)
        masses = np.maximum(masses, np.finfo(float).tiny)
        totals = masses.sum(axis=1)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise FloatingPointError("invalid quadratic-TT conditional mass")
        return coefficients, masses

    @staticmethod
    def _partial_integral(coefficients, fraction, gamma):
        result = np.zeros_like(fraction)
        power = fraction
        for degree in range(5):
            coefficient = coefficients[..., degree]
            if degree == 0:
                coefficient = coefficient + gamma
            result += coefficient * power / (degree + 1)
            power = power * fraction
        return result

    @staticmethod
    def _selected_core(core, interval, fraction):
        weights = _quadratic_basis_weights(fraction)
        return sum(
            weight[:, None, None]
            * np.moveaxis(core[:, interval + offset, :], 1, 0)
            for offset, weight in enumerate(weights)
        )

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        interval, fraction = _quadratic_indices_fractions(points, self.modes)
        result = np.empty_like(points)
        left = np.ones((len(points), 1), dtype=np.float64)
        rows = np.arange(len(points))
        for k, n in enumerate(self.modes):
            coefficients, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            selected_interval = interval[:, k]
            before = np.where(
                selected_interval == 0, 0.0,
                cumulative[rows, np.maximum(selected_interval - 1, 0)],
            )
            local = self._partial_integral(
                coefficients[rows, selected_interval], fraction[:, k],
                self.gamma,
            ) / (int(n) - 2)
            result[:, k] = (before + local) / cumulative[:, -1]
            selected = self._selected_core(
                self._cores[k], selected_interval, fraction[:, k]
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return np.clip(result, 0.0, 1.0)

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        result = np.empty_like(uniform)
        left = np.ones((len(uniform), 1), dtype=np.float64)
        rows = np.arange(len(uniform))
        for k, n in enumerate(self.modes):
            coefficients, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            target = np.minimum(
                uniform[:, k], np.nextafter(1.0, 0.0)
            ) * cumulative[:, -1]
            interval = np.minimum(
                np.sum(cumulative <= target[:, None], axis=1), int(n) - 3
            )
            before = np.where(
                interval == 0, 0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            local_target = (target - before) * (int(n) - 2)
            selected_coefficients = coefficients[rows, interval]
            lo = np.zeros(len(uniform), dtype=np.float64)
            hi = np.ones(len(uniform), dtype=np.float64)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                value = self._partial_integral(
                    selected_coefficients, mid, self.gamma
                )
                lo = np.where(value < local_target, mid, lo)
                hi = np.where(value >= local_target, mid, hi)
            fraction = 0.5 * (lo + hi)
            result[:, k] = (interval + fraction) / (int(n) - 2)
            selected = self._selected_core(
                self._cores[k], interval, fraction
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return result

    def rosenblatt_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 4096,
    ) -> np.ndarray:
        import torch

        points = _validate_points(points, self.d)
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        cores = [
            torch.as_tensor(core, dtype=torch_dtype, device=device)
            for core in self._cores
        ]
        right = [
            torch.as_tensor(value, dtype=torch_dtype, device=device)
            for value in self._right
        ]
        output = np.empty_like(points)
        with torch.inference_mode():
            for start in range(0, len(points), int(batch_size)):
                stop = min(start + int(batch_size), len(points))
                value = torch.as_tensor(
                    points[start:stop], dtype=torch_dtype, device=device
                )
                result = torch.empty_like(value)
                left = torch.ones(
                    (len(value), 1), dtype=torch_dtype, device=device
                )
                rows = torch.arange(len(value), device=device)
                for k, core in enumerate(cores):
                    nodes = torch.einsum("pa,aib->pib", left, core)
                    first, middle, last = (
                        nodes[:, :-2], nodes[:, 1:-1], nodes[:, 2:]
                    )
                    polynomial = (
                        0.5 * (first + middle),
                        -first + middle,
                        0.5 * first - middle + 0.5 * last,
                    )
                    projected = [
                        torch.einsum("pia,ab->pib", item, right[k + 1])
                        for item in polynomial
                    ]
                    a, b, c = polynomial
                    pa, pb, pc = projected
                    coefficients = torch.stack((
                        torch.einsum("pib,pib->pi", pa, a),
                        2.0 * torch.einsum("pib,pib->pi", pb, a),
                        torch.einsum("pib,pib->pi", pb, b)
                        + 2.0 * torch.einsum("pib,pib->pi", pc, a),
                        2.0 * torch.einsum("pib,pib->pi", pc, b),
                        torch.einsum("pib,pib->pi", pc, c),
                    ), dim=-1)
                    adjusted = coefficients.clone()
                    adjusted[..., 0] += self.gamma
                    inverse_powers = torch.arange(
                        1, 6, dtype=torch_dtype, device=device
                    ).reciprocal()
                    masses = torch.sum(
                        adjusted * inverse_powers, dim=-1
                    ) / (core.shape[1] - 2)
                    masses = torch.clamp_min(
                        masses, torch.finfo(torch_dtype).tiny
                    )
                    cumulative = torch.cumsum(masses, dim=1)
                    position = value[:, k] * (core.shape[1] - 2)
                    interval = torch.clamp(
                        torch.floor(position).to(torch.long),
                        max=core.shape[1] - 3,
                    )
                    fraction = torch.clamp(position - interval, 0.0, 1.0)
                    before = torch.where(
                        interval == 0,
                        torch.zeros((), dtype=torch_dtype, device=device),
                        cumulative[rows, torch.clamp_min(interval - 1, 0)],
                    )
                    selected_coefficients = coefficients[rows, interval]
                    local = torch.zeros_like(fraction)
                    power = fraction
                    for degree in range(5):
                        coefficient = selected_coefficients[:, degree]
                        if degree == 0:
                            coefficient = coefficient + self.gamma
                        local = local + coefficient * power / (degree + 1)
                        power = power * fraction
                    local = local / (core.shape[1] - 2)
                    result[:, k] = (before + local) / cumulative[:, -1]
                    weights = (
                        0.5 * (1.0 - fraction).square(),
                        0.5 + fraction - fraction.square(),
                        0.5 * fraction.square(),
                    )
                    selected = sum(
                        weight[:, None, None]
                        * core[:, interval + offset, :].permute(1, 0, 2)
                        for offset, weight in enumerate(weights)
                    )
                    left = torch.einsum("pa,pab->pb", left, selected)
                output[start:stop] = torch.clamp(
                    result, 0.0, 1.0
                ).cpu().numpy()
        return output

    def sample(self, count: int, seed=None) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        return self.inverse_rosenblatt(
            np.random.default_rng(seed).random((count, self.d))
        )

    def save(self, path) -> None:
        payload = {
            "kind": np.array(["quadratic-squared"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "QuadraticSquaredTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            root = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(root, gamma=float(data["gamma"][0]))


class DirectTTDensity:
    """Cellwise density ``1 + correction`` represented by a centered TT.

    Only the correction is stored: the rank-one constant is added analytically
    to every marginal.  Thus a rank-``r`` correction costs rank-``r`` storage,
    not the rank-``r+1`` storage of an explicitly materialized ``1 + a``.
    All Rosenblatt marginals are linear suffix contractions and
    ``integral (1+a)**2`` is an exact second-order TT contraction.
    """

    def __init__(self, correction: vector, conditional_floor: float = 1e-12):
        if not isinstance(correction, vector) or correction.d == 0:
            raise TypeError("correction must be a non-empty tt.vector")
        if correction.r[0] != 1 or correction.r[-1] != 1:
            raise ValueError("correction must be a scalar TT")
        if conditional_floor <= 0.0 or not np.isfinite(conditional_floor):
            raise ValueError("conditional_floor must be finite and positive")
        self.correction = vector.from_list(_as_numpy_cores(correction))
        self.conditional_floor = float(conditional_floor)
        self._cores = _as_numpy_cores(self.correction)
        self._right = self._right_mean_environments(self._cores)
        self.normalization = 1.0 + float(self._right[0][0])
        if not np.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("direct TT density must have positive integral")

    @staticmethod
    def _right_mean_environments(cores):
        d = len(cores)
        right = [np.empty(0) for _ in range(d + 1)]
        right[d] = np.ones(1, dtype=np.float64)
        for k in range(d - 1, -1, -1):
            right[k] = cores[k].mean(axis=1) @ right[k + 1]
        return right

    @property
    def d(self) -> int:
        return self.correction.d

    @property
    def modes(self) -> np.ndarray:
        return self.correction.n.astype(np.int64)

    @property
    def ranks(self) -> np.ndarray:
        return self.correction.r.astype(np.int64)

    @property
    def size(self) -> int:
        return self.correction.size

    def raw_density(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        values = 1.0 + _sample_tt_numpy(
            self._cores, _cell_indices(points, self.modes)
        )
        return values / self.normalization

    def _conditional_probabilities_batch(self, left, k, *, diagnostics=False):
        core = self._cores[k]
        extended = np.einsum("pa,aib->pib", left, core, optimize=True)
        raw = 1.0 + np.einsum(
            "pib,b->pi", extended, self._right[k + 1], optimize=True
        )
        scale = np.maximum(
            np.mean(np.abs(raw), axis=1, keepdims=True),
            np.finfo(float).tiny,
        )
        floor = self.conditional_floor * scale
        clipped = np.maximum(raw, floor)
        totals = clipped.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise FloatingPointError("invalid conditional mass in direct TT density")
        probabilities = clipped / totals
        if diagnostics:
            return probabilities, raw <= floor
        return probabilities

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        out = np.empty_like(uniform)
        count = uniform.shape[0]
        left = np.ones((count, 1), dtype=np.float64)
        rows = np.arange(count)
        for k, n in enumerate(self.modes):
            probs = self._conditional_probabilities_batch(left, k)
            cdf = np.cumsum(probs, axis=1)
            uk = np.minimum(uniform[:, k], np.nextafter(1.0, 0.0))
            cell = np.minimum(np.sum(cdf <= uk[:, None], axis=1), int(n) - 1)
            lower = np.where(cell == 0, 0.0, cdf[rows, np.maximum(cell - 1, 0)])
            mass = np.maximum(probs[rows, cell], np.finfo(float).tiny)
            fraction = np.clip((uk - lower) / mass, 0.0, 1.0)
            out[:, k] = (cell + fraction) / n
            selected = np.moveaxis(self._cores[k][:, cell, :], 1, 0)
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return out

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        out = np.empty_like(points)
        indices = _cell_indices(points, self.modes)
        count = points.shape[0]
        left = np.ones((count, 1), dtype=np.float64)
        rows = np.arange(count)
        for k, n in enumerate(self.modes):
            probs = self._conditional_probabilities_batch(left, k)
            cdf = np.cumsum(probs, axis=1)
            cell = indices[:, k]
            lower = np.where(cell == 0, 0.0, cdf[rows, np.maximum(cell - 1, 0)])
            fraction = points[:, k] * n - cell
            out[:, k] = lower + probs[rows, cell] * fraction
            selected = np.moveaxis(self._cores[k][:, cell, :], 1, 0)
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return np.clip(out, 0.0, 1.0)

    def log_density(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        indices = _cell_indices(points, self.modes)
        left = np.ones((points.shape[0], 1), dtype=np.float64)
        rows = np.arange(points.shape[0])
        result = np.zeros(points.shape[0], dtype=np.float64)
        for k, n in enumerate(self.modes):
            probs = self._conditional_probabilities_batch(left, k)
            cell = indices[:, k]
            result += np.log(probs[rows, cell] * n)
            selected = np.moveaxis(self._cores[k][:, cell, :], 1, 0)
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return result

    def density(self, points) -> np.ndarray:
        return np.exp(self.log_density(points))

    def sample(self, count: int, seed=None) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = np.random.default_rng(seed)
        return self.inverse_rosenblatt(rng.random((count, self.d)))

    def model_l2_norm_sq(self) -> float:
        mean = float(self._right[0][0])
        second = _second_moment_numpy(self._cores)
        return (1.0 + 2.0 * mean + second) / self.normalization ** 2

    @property
    def chi2_to_reference(self) -> float:
        return max(0.0, self.model_l2_norm_sq() - 1.0)

    def conditional_clipping_fraction(self, points) -> float:
        points = _validate_points(points, self.d)
        indices = _cell_indices(points, self.modes)
        left = np.ones((points.shape[0], 1), dtype=np.float64)
        rows = np.arange(points.shape[0])
        clipped_count = 0
        total_count = 0
        for k in range(self.d):
            _, clipped = self._conditional_probabilities_batch(
                left, k, diagnostics=True
            )
            clipped_count += int(clipped.sum())
            total_count += int(clipped.size)
            cell = indices[:, k]
            selected = np.moveaxis(self._cores[k][:, cell, :], 1, 0)
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return clipped_count / max(total_count, 1)

    def save(self, path) -> None:
        payload = {
            "kind": np.array(["direct"]),
            "conditional_floor": np.array([self.conditional_floor]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "DirectTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            correction = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(
                correction,
                conditional_floor=float(data["conditional_floor"][0]),
            )

__all__ = [
    "QuadraticSquaredTTDensity",
    "DirectTTDensity",
]
