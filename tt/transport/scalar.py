"""Scalar squared TT density models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _adaptive_linear_indices_fractions,
    _adaptive_linear_right_gram_environments,
    _as_numpy_cores,
    _cell_indices,
    _linear_indices_fractions,
    _linear_right_gram_environments,
    _right_gram_environments,
    _sample_adaptive_linear_tt_numpy,
    _sample_linear_tt_numpy,
    _sample_tt_numpy,
    _second_moment_numpy,
    _validate_linear_knots,
    _validate_points,
)
from ._torch_density import (
    _torch_sample_adaptive_linear_tt,
    _torch_sample_linear_tt,
    _torch_sample_tt,
)

class SquaredTTDensity:
    """Positive cellwise density ``(gamma + root**2) / Z`` on ``[0, 1]^d``.

    The density is relative to the uniform probability measure.  ``root`` stores
    values on a tensor product of equal-width cells, not nodal values.  Within a
    cell the density is constant, so the Rosenblatt CDF is piecewise linear and
    its inverse is analytic.
    """

    def __init__(self, root: vector, gamma: float = 1e-6):
        if not isinstance(root, vector) or root.d == 0:
            raise TypeError("root must be a non-empty tt.vector")
        if root.r[0] != 1 or root.r[-1] != 1:
            raise ValueError("root must be a scalar TT (boundary ranks equal to one)")
        if gamma <= 0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self.root = vector.from_list(_as_numpy_cores(root))
        self.gamma = float(gamma)
        self._cores = _as_numpy_cores(self.root)
        self._right = _right_gram_environments(self._cores)
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
        """Normalized weight of the uniform reference component.

        A fixed pre-normalization ``gamma`` is not itself a fixed density
        floor because the actual mixture weight also depends on the root norm.
        """
        return self.gamma / self.normalization

    def root_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_tt_numpy(self._cores, _cell_indices(points, self.modes))

    def density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return (self.gamma + values * values) / self.normalization

    def log_density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return np.log(self.gamma + values * values) - np.log(self.normalization)

    def log_density_device(
        self,
        points,
        *,
        device: str,
        dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        """Evaluate log density in accelerator batches."""
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
                modes = torch.as_tensor(
                    self.modes, dtype=torch_dtype, device=device
                )
                indices = torch.minimum(
                    torch.floor(batch * modes).to(torch.long),
                    modes.to(torch.long) - 1,
                )
                values = _torch_sample_tt(cores, indices)
                result = torch.log(self.gamma + values.square()) - np.log(
                    self.normalization
                )
                output[start:stop] = result.cpu().numpy()
        return output

    def _conditional_probabilities(self, left: np.ndarray, k: int) -> np.ndarray:
        core = self._cores[k]
        extended = np.einsum("a,aib->ib", left, core, optimize=True)
        projected = np.einsum(
            "ia,ab->ib", extended, self._right[k + 1], optimize=True
        )
        square_mass = np.einsum("ib,ib->i", projected, extended, optimize=True)
        weights = self.gamma + np.maximum(square_mass, 0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("invalid conditional mass in squared TT density")
        return weights / total

    def _conditional_probabilities_batch(
        self, left: np.ndarray, k: int
    ) -> np.ndarray:
        """Conditional cell masses for every point in one TT contraction."""
        core = self._cores[k]
        extended = np.einsum("pa,aib->pib", left, core, optimize=True)
        projected = np.einsum(
            "pia,ab->pib", extended, self._right[k + 1], optimize=True
        )
        square_mass = np.einsum(
            "pib,pib->pi", projected, extended, optimize=True
        )
        weights = self.gamma + np.maximum(square_mass, 0.0)
        totals = weights.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise FloatingPointError("invalid conditional mass in squared TT density")
        return weights / totals

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        """Map uniform samples to this density (the incremental SIRT)."""
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
        """Map samples from this density to the uniform reference."""
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

    def rosenblatt_device(
        self,
        points,
        *,
        device: str,
        dtype: str = "float32",
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Evaluate the exact Rosenblatt map with torch contractions.

        This follows :meth:`rosenblatt` cell for cell; only the batched
        ``N x modes x rank`` contractions move to the requested accelerator.
        It is particularly important between deep Sample-DIRT layers, where
        all training samples have to be pulled back through the current map.
        """
        import torch

        points = _validate_points(points, self.d)
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
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
                modes = torch.as_tensor(
                    self.modes, dtype=torch_dtype, device=device
                )
                indices = torch.minimum(
                    torch.floor(value * modes).to(torch.long),
                    modes.to(torch.long) - 1,
                )
                result = torch.empty_like(value)
                left = torch.ones(
                    (len(value), 1), dtype=torch_dtype, device=device
                )
                rows = torch.arange(len(value), device=device)
                for k, n in enumerate(self.modes):
                    extended = torch.einsum("pa,aib->pib", left, cores[k])
                    projected = torch.einsum(
                        "pia,ab->pib", extended, right[k + 1]
                    )
                    square_mass = torch.einsum(
                        "pib,pib->pi", projected, extended
                    )
                    weights = self.gamma + torch.clamp_min(square_mass, 0.0)
                    probabilities = weights / weights.sum(dim=1, keepdim=True)
                    cdf = torch.cumsum(probabilities, dim=1)
                    cell = indices[:, k]
                    lower = torch.where(
                        cell == 0,
                        torch.zeros((), dtype=torch_dtype, device=device),
                        cdf[rows, torch.clamp_min(cell - 1, 0)],
                    )
                    fraction = value[:, k] * int(n) - cell
                    result[:, k] = (
                        lower + probabilities[rows, cell] * fraction
                    )
                    selected = cores[k][:, cell, :].permute(1, 0, 2)
                    left = torch.einsum("pa,pab->pb", left, selected)
                output[start:stop] = torch.clamp(
                    result, 0.0, 1.0
                ).cpu().numpy()
        return output

    def sample(self, count: int, seed=None) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = np.random.default_rng(seed)
        return self.inverse_rosenblatt(rng.random((count, self.d)))

    def model_l2_norm_sq(self) -> float:
        """Exact ``integral h**2 dmu`` by TT contractions."""
        squared_cores = [
            np.einsum("aic,bid->abicd", core, core, optimize=True).reshape(
                core.shape[0] ** 2, core.shape[1], core.shape[2] ** 2
            )
            for core in self._cores
        ]
        fourth = _second_moment_numpy(squared_cores)
        numerator = (
            self.gamma * self.gamma
            + 2.0 * self.gamma * self._root_second_moment
            + fourth
        )
        return numerator / (self.normalization * self.normalization)

    @property
    def chi2_to_reference(self) -> float:
        return max(0.0, self.model_l2_norm_sq() - 1.0)

    def save(self, path) -> None:
        path = Path(path)
        payload = {
            "kind": np.array(["squared"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "SquaredTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            root = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(root, gamma=float(data["gamma"][0]))


class LinearSquaredTTDensity:
    """Positive squared TT in a continuous piecewise-linear nodal basis.

    Each TT mode stores values at equally spaced nodes. The root is
    multilinear between neighbouring nodes and the density is
    ``(gamma + root**2) / Z``. Hat-function Gram contractions give ``Z``
    exactly, while every one-dimensional conditional is piecewise quadratic;
    its CDF is therefore an exactly integrated piecewise cubic.
    """

    def __init__(self, root: vector, gamma: float = 1e-6):
        if not isinstance(root, vector) or root.d == 0:
            raise TypeError("root must be a non-empty tt.vector")
        if root.r[0] != 1 or root.r[-1] != 1:
            raise ValueError("root must be a scalar TT")
        if np.any(root.n < 2):
            raise ValueError("linear TT modes need at least two nodes")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self.root = vector.from_list(_as_numpy_cores(root))
        self.gamma = float(gamma)
        self._cores = _as_numpy_cores(self.root)
        self._right = _linear_right_gram_environments(self._cores)
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
        return _sample_linear_tt_numpy(self._cores, points)

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
                values = _torch_sample_linear_tt(cores, batch)
                output[start:stop] = (
                    torch.log(self.gamma + values.square())
                    - np.log(self.normalization)
                ).cpu().numpy()
        return output

    def _conditional_coefficients_batch(
        self, left: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        core = self._cores[k]
        nodes = np.einsum("pa,aib->pib", left, core, optimize=True)
        first = nodes[:, :-1]
        delta = nodes[:, 1:] - first
        right = self._right[k + 1]
        projected_first = np.einsum(
            "pia,ab->pib", first, right, optimize=True
        )
        projected_delta = np.einsum(
            "pia,ab->pib", delta, right, optimize=True
        )
        a = np.einsum(
            "pib,pib->pi", projected_first, first, optimize=True
        )
        b = 2.0 * np.einsum(
            "pib,pib->pi", projected_delta, first, optimize=True
        )
        c = np.einsum(
            "pib,pib->pi", projected_delta, delta, optimize=True
        )
        h = 1.0 / (core.shape[1] - 1)
        masses = h * (self.gamma + a + 0.5 * b + c / 3.0)
        masses = np.maximum(masses, np.finfo(float).tiny)
        totals = masses.sum(axis=1)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise FloatingPointError("invalid linear-TT conditional mass")
        return a, b, c, masses

    @staticmethod
    def _partial_integral(a, b, c, fraction, gamma):
        return (
            (gamma + a) * fraction
            + 0.5 * b * fraction * fraction
            + (c / 3.0) * fraction * fraction * fraction
        )

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        lower, fraction = _linear_indices_fractions(points, self.modes)
        result = np.empty_like(points)
        left = np.ones((len(points), 1), dtype=np.float64)
        rows = np.arange(len(points))
        for k, n in enumerate(self.modes):
            a, b, c, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            interval = lower[:, k]
            before = np.where(
                interval == 0, 0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            local = self._partial_integral(
                a[rows, interval], b[rows, interval], c[rows, interval],
                fraction[:, k], self.gamma,
            ) / (int(n) - 1)
            result[:, k] = (before + local) / cumulative[:, -1]
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :], 1, 0
            )
            selected = (
                (1.0 - fraction[:, k])[:, None, None] * first
                + fraction[:, k, None, None] * second
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return np.clip(result, 0.0, 1.0)

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        result = np.empty_like(uniform)
        left = np.ones((len(uniform), 1), dtype=np.float64)
        rows = np.arange(len(uniform))
        for k, n in enumerate(self.modes):
            a, b, c, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            target = np.minimum(
                uniform[:, k], np.nextafter(1.0, 0.0)
            ) * cumulative[:, -1]
            interval = np.minimum(
                np.sum(cumulative <= target[:, None], axis=1), int(n) - 2
            )
            before = np.where(
                interval == 0, 0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            local_target = (target - before) * (int(n) - 1)
            aa = a[rows, interval]
            bb = b[rows, interval]
            cc = c[rows, interval]
            lo = np.zeros(len(uniform), dtype=np.float64)
            hi = np.ones(len(uniform), dtype=np.float64)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                value = self._partial_integral(
                    aa, bb, cc, mid, self.gamma
                )
                lo = np.where(value < local_target, mid, lo)
                hi = np.where(value >= local_target, mid, hi)
            fraction = 0.5 * (lo + hi)
            result[:, k] = (interval + fraction) / (int(n) - 1)
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :], 1, 0
            )
            selected = (
                (1.0 - fraction)[:, None, None] * first
                + fraction[:, None, None] * second
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return result

    def rosenblatt_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Accelerated exact piecewise-cubic Rosenblatt evaluation."""
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
                    first = nodes[:, :-1]
                    delta = nodes[:, 1:] - first
                    projected_first = torch.einsum(
                        "pia,ab->pib", first, right[k + 1]
                    )
                    projected_delta = torch.einsum(
                        "pia,ab->pib", delta, right[k + 1]
                    )
                    a = torch.einsum(
                        "pib,pib->pi", projected_first, first
                    )
                    b = 2.0 * torch.einsum(
                        "pib,pib->pi", projected_delta, first
                    )
                    c = torch.einsum(
                        "pib,pib->pi", projected_delta, delta
                    )
                    h = 1.0 / (core.shape[1] - 1)
                    masses = h * (
                        self.gamma + a + 0.5 * b + c / 3.0
                    )
                    masses = torch.clamp_min(
                        masses, torch.finfo(torch_dtype).tiny
                    )
                    cumulative = torch.cumsum(masses, dim=1)
                    position = value[:, k] * (core.shape[1] - 1)
                    interval = torch.clamp(
                        torch.floor(position).to(torch.long),
                        max=core.shape[1] - 2,
                    )
                    fraction = torch.clamp(position - interval, 0.0, 1.0)
                    before = torch.where(
                        interval == 0,
                        torch.zeros((), dtype=torch_dtype, device=device),
                        cumulative[rows, torch.clamp_min(interval - 1, 0)],
                    )
                    aa = a[rows, interval]
                    bb = b[rows, interval]
                    cc = c[rows, interval]
                    local = h * (
                        (self.gamma + aa) * fraction
                        + 0.5 * bb * fraction.square()
                        + (cc / 3.0) * fraction.pow(3)
                    )
                    result[:, k] = (before + local) / cumulative[:, -1]
                    node_first = core[:, interval, :].permute(1, 0, 2)
                    node_second = core[:, interval + 1, :].permute(1, 0, 2)
                    selected = (
                        (1.0 - fraction)[:, None, None] * node_first
                        + fraction[:, None, None] * node_second
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
            "kind": np.array(["linear-squared"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "LinearSquaredTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            root = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(root, gamma=float(data["gamma"][0]))


class AdaptiveLinearSquaredTTDensity(LinearSquaredTTDensity):
    """Squared linear TT on learned or prescribed nonuniform knot vectors.

    The root is linear on every interval ``[t_{k,i}, t_{k,i+1}]``.  Exact
    nonuniform hat Gram matrices give the normalizer, and each conditional is
    still piecewise quadratic with an exactly integrated cubic CDF.  Interior
    knot locations are counted as stored model parameters.
    """

    def __init__(
        self, root: vector, knots: Sequence[np.ndarray], gamma: float = 1e-6
    ):
        if not isinstance(root, vector) or root.d == 0:
            raise TypeError("root must be a non-empty tt.vector")
        if root.r[0] != 1 or root.r[-1] != 1:
            raise ValueError("root must be a scalar TT")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self.root = vector.from_list(_as_numpy_cores(root))
        self.gamma = float(gamma)
        self._cores = _as_numpy_cores(self.root)
        self.knots = _validate_linear_knots(knots, self.root.n)
        self._right = _adaptive_linear_right_gram_environments(
            self._cores, self.knots
        )
        self._root_second_moment = float(self._right[0][0, 0])
        self.normalization = self.gamma + self._root_second_moment
        if not np.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("adaptive linear TT has invalid normalization")

    @property
    def size(self) -> int:
        return int(self.root.size + sum(len(value) - 2 for value in self.knots))

    def root_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_adaptive_linear_tt_numpy(
            self._cores, self.knots, points
        )

    def _conditional_coefficients_batch(self, left, k):
        core = self._cores[k]
        nodes = np.einsum("pa,aib->pib", left, core, optimize=True)
        first = nodes[:, :-1]
        delta = nodes[:, 1:] - first
        right = self._right[k + 1]
        projected_first = np.einsum(
            "pia,ab->pib", first, right, optimize=True
        )
        projected_delta = np.einsum(
            "pia,ab->pib", delta, right, optimize=True
        )
        a = np.einsum("pib,pib->pi", projected_first, first, optimize=True)
        b = 2.0 * np.einsum(
            "pib,pib->pi", projected_delta, first, optimize=True
        )
        c = np.einsum(
            "pib,pib->pi", projected_delta, delta, optimize=True
        )
        widths = np.diff(self.knots[k])
        masses = widths[None, :] * (self.gamma + a + 0.5 * b + c / 3.0)
        masses = np.maximum(masses, np.finfo(float).tiny)
        totals = masses.sum(axis=1)
        if not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise FloatingPointError("invalid adaptive-TT conditional mass")
        return a, b, c, masses

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        lower, fraction = _adaptive_linear_indices_fractions(
            points, self.knots
        )
        result = np.empty_like(points)
        left = np.ones((len(points), 1), dtype=np.float64)
        rows = np.arange(len(points))
        for k in range(self.d):
            a, b, c, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            interval = lower[:, k]
            before = np.where(
                interval == 0, 0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            width = np.diff(self.knots[k])[interval]
            local = width * self._partial_integral(
                a[rows, interval], b[rows, interval], c[rows, interval],
                fraction[:, k], self.gamma,
            )
            result[:, k] = (before + local) / cumulative[:, -1]
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :], 1, 0
            )
            selected = (
                (1.0 - fraction[:, k])[:, None, None] * first
                + fraction[:, k, None, None] * second
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return np.clip(result, 0.0, 1.0)

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        result = np.empty_like(uniform)
        left = np.ones((len(uniform), 1), dtype=np.float64)
        rows = np.arange(len(uniform))
        for k in range(self.d):
            a, b, c, masses = self._conditional_coefficients_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            target = np.minimum(
                uniform[:, k], np.nextafter(1.0, 0.0)
            ) * cumulative[:, -1]
            interval = np.minimum(
                np.sum(cumulative <= target[:, None], axis=1),
                len(self.knots[k]) - 2,
            )
            before = np.where(
                interval == 0, 0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            width = np.diff(self.knots[k])[interval]
            local_target = (target - before) / width
            aa, bb, cc = a[rows, interval], b[rows, interval], c[rows, interval]
            lo = np.zeros(len(uniform), dtype=np.float64)
            hi = np.ones(len(uniform), dtype=np.float64)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                value = self._partial_integral(
                    aa, bb, cc, mid, self.gamma
                )
                lo = np.where(value < local_target, mid, lo)
                hi = np.where(value >= local_target, mid, hi)
            fraction = 0.5 * (lo + hi)
            result[:, k] = (
                self.knots[k][interval] + width * fraction
            )
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :], 1, 0
            )
            selected = (
                (1.0 - fraction)[:, None, None] * first
                + fraction[:, None, None] * second
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return result

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
        knots = [
            torch.as_tensor(value, dtype=torch_dtype, device=device)
            for value in self.knots
        ]
        output = np.empty(len(points), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(points), int(batch_size)):
                stop = min(start + int(batch_size), len(points))
                batch = torch.as_tensor(
                    points[start:stop], dtype=torch_dtype, device=device
                )
                values = _torch_sample_adaptive_linear_tt(
                    cores, knots, batch
                )
                output[start:stop] = (
                    torch.log(self.gamma + values.square())
                    - np.log(self.normalization)
                ).cpu().numpy()
        return output

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
        knots = [
            torch.as_tensor(value, dtype=torch_dtype, device=device)
            for value in self.knots
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
                    first, delta = nodes[:, :-1], nodes[:, 1:] - nodes[:, :-1]
                    projected_first = torch.einsum(
                        "pia,ab->pib", first, right[k + 1]
                    )
                    projected_delta = torch.einsum(
                        "pia,ab->pib", delta, right[k + 1]
                    )
                    a = torch.einsum("pib,pib->pi", projected_first, first)
                    b = 2.0 * torch.einsum(
                        "pib,pib->pi", projected_delta, first
                    )
                    c = torch.einsum(
                        "pib,pib->pi", projected_delta, delta
                    )
                    widths = knots[k][1:] - knots[k][:-1]
                    masses = torch.clamp_min(
                        widths[None, :] * (
                            self.gamma + a + 0.5 * b + c / 3.0
                        ),
                        torch.finfo(torch_dtype).tiny,
                    )
                    cumulative = torch.cumsum(masses, dim=1)
                    interval = torch.clamp(
                        torch.searchsorted(
                            knots[k], value[:, k].contiguous(), right=True
                        ) - 1,
                        min=0, max=core.shape[1] - 2,
                    )
                    width = widths[interval]
                    fraction = torch.clamp(
                        (value[:, k] - knots[k][interval]) / width, 0.0, 1.0
                    )
                    before = torch.where(
                        interval == 0,
                        torch.zeros((), dtype=torch_dtype, device=device),
                        cumulative[rows, torch.clamp_min(interval - 1, 0)],
                    )
                    aa, bb, cc = (
                        a[rows, interval], b[rows, interval], c[rows, interval]
                    )
                    local = width * (
                        (self.gamma + aa) * fraction
                        + 0.5 * bb * fraction.square()
                        + (cc / 3.0) * fraction.pow(3)
                    )
                    result[:, k] = (before + local) / cumulative[:, -1]
                    node_first = core[:, interval, :].permute(1, 0, 2)
                    node_second = core[:, interval + 1, :].permute(1, 0, 2)
                    selected = (
                        (1.0 - fraction)[:, None, None] * node_first
                        + fraction[:, None, None] * node_second
                    )
                    left = torch.einsum("pa,pab->pb", left, selected)
                output[start:stop] = torch.clamp(
                    result, 0.0, 1.0
                ).cpu().numpy()
        return output

    def save(self, path) -> None:
        payload = {
            "kind": np.array(["linear-adaptive-squared"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        payload.update({f"knots{k}": value for k, value in enumerate(self.knots)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "AdaptiveLinearSquaredTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            root = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(
                root,
                [data[f"knots{k}"] for k in range(d)],
                gamma=float(data["gamma"][0]),
            )

__all__ = [
    "SquaredTTDensity",
    "LinearSquaredTTDensity",
    "AdaptiveLinearSquaredTTDensity",
]
