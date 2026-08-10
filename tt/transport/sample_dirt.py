"""Sample-only deep inverse Rosenblatt transport in the TT format.

This module implements a discrete, piecewise-constant prototype of the
sample-driven DIRT construction.  A layer is a positive residual density

    h(i_1, ..., i_d) = (gamma + g(i_1, ..., i_d)**2) / Z,

where ``g`` is a :class:`tt.vector` and the reference measure is uniform on the
unit cube.  The cells are equiprobable under the reference.  Consequently all
normalising constants and L2 terms in the least-squares density-ratio loss are
exact TT contractions; only the linear target expectation is estimated from
samples.

If ``T`` is the transport accumulated so far and ``F = T^{-1}``, samples
``x ~ nu_next`` are mapped to ``v = F(x)``.  The population loss

    1/2 ||h||^2_{L2(mu)} - E[h(v)]

is minimised by ``d(F#nu_next)/dmu``.  The inverse Rosenblatt map of the fitted
``h`` is appended on the right: ``T_next = T o S``.  This is the sample analogue
of the exact-ratio DIRT update and corrects, rather than assumes away, the error
of the previous layer.

The piecewise-constant basis keeps the first implementation deliberately
small and auditable.  Replacing cell lookup by local polynomial evaluation and
the uniform cell mass by one-dimensional mass matrices gives the continuous
functional-TT version without changing the public ``SampleDIRT`` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core.vector import vector


def _as_numpy_cores(root: vector) -> list[np.ndarray]:
    from .. import backend as bk

    return [np.asarray(bk.to_numpy(core), dtype=np.float64) for core in root.cores]


def _validate_points(points, d: int, *, name: str = "points") -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != d:
        raise ValueError(f"{name} must have shape (n, {d}), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    # One is a valid right endpoint and belongs to the last cell.
    if np.any(points < 0.0) or np.any(points > 1.0):
        lo, hi = float(points.min()), float(points.max())
        raise ValueError(f"{name} must lie in [0, 1]^{d}; range is [{lo}, {hi}]")
    return points


def _cell_indices(points: np.ndarray, modes: np.ndarray) -> np.ndarray:
    idx = np.floor(points * modes.reshape(1, -1)).astype(np.int64)
    return np.minimum(idx, modes.reshape(1, -1) - 1)


def _sample_root_numpy(cores: Sequence[np.ndarray], indices: np.ndarray) -> np.ndarray:
    left = np.ones((indices.shape[0], 1), dtype=np.float64)
    for k, core in enumerate(cores):
        selected = np.moveaxis(core[:, indices[:, k], :], 1, 0)
        left = np.einsum("pa,pab->pb", left, selected, optimize=True)
    return left[:, 0]


def _second_moment_numpy(cores: Sequence[np.ndarray]) -> float:
    env = np.ones((1, 1), dtype=np.float64)
    for core in cores:
        projected = np.einsum("ab,aic->bic", env, core, optimize=True)
        env = np.einsum("bic,bid->cd", projected, core, optimize=True)
        env /= core.shape[1]
    return float(env[0, 0])


def _right_gram_environments(cores: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Right suffix averages of ``g**2`` under the product reference."""
    d = len(cores)
    right: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    right[d] = np.ones((1, 1), dtype=np.float64)
    for k in range(d - 1, -1, -1):
        core = cores[k]
        projected = np.einsum(
            "aic,cd->aid", core, right[k + 1], optimize=True
        )
        right[k] = (
            np.einsum("aid,bid->ab", projected, core, optimize=True)
            / core.shape[1]
        )
    return right


@dataclass
class FitHistory:
    """Diagnostics recorded while fitting one Sample-DIRT layer."""

    loss: list[float] = field(default_factory=list)
    l2_norm_sq: list[float] = field(default_factory=list)
    normalization: list[float] = field(default_factory=list)
    chi2_to_reference: list[float] = field(default_factory=list)
    epochs: int = 0
    converged: bool = False


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

    def root_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_root_numpy(self._cores, _cell_indices(points, self.modes))

    def density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return (self.gamma + values * values) / self.normalization

    def log_density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return np.log(self.gamma + values * values) - np.log(self.normalization)

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

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        """Map uniform samples to this density (the incremental SIRT)."""
        uniform = _validate_points(uniform, self.d, name="uniform")
        out = np.empty_like(uniform)
        for p in range(uniform.shape[0]):
            left = np.ones(1, dtype=np.float64)
            for k, n in enumerate(self.modes):
                probs = self._conditional_probabilities(left, k)
                cdf = np.cumsum(probs)
                uk = min(float(uniform[p, k]), np.nextafter(1.0, 0.0))
                cell = min(int(np.searchsorted(cdf, uk, side="right")), int(n) - 1)
                lower = 0.0 if cell == 0 else float(cdf[cell - 1])
                fraction = (uk - lower) / max(float(probs[cell]), np.finfo(float).tiny)
                fraction = min(max(fraction, 0.0), 1.0)
                out[p, k] = (cell + fraction) / n
                left = left @ self._cores[k][:, cell, :]
        return out

    def rosenblatt(self, points) -> np.ndarray:
        """Map samples from this density to the uniform reference."""
        points = _validate_points(points, self.d)
        out = np.empty_like(points)
        indices = _cell_indices(points, self.modes)
        for p in range(points.shape[0]):
            left = np.ones(1, dtype=np.float64)
            for k, n in enumerate(self.modes):
                probs = self._conditional_probabilities(left, k)
                cell = int(indices[p, k])
                lower = float(probs[:cell].sum())
                fraction = points[p, k] * n - cell
                out[p, k] = lower + float(probs[cell]) * fraction
                left = left @ self._cores[k][:, cell, :]
        return np.clip(out, 0.0, 1.0)

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


def _torch_sample_root(cores, indices):
    import torch

    left = torch.ones((indices.shape[0], 1), dtype=cores[0].dtype,
                      device=cores[0].device)
    for k, core in enumerate(cores):
        selected = core[:, indices[:, k], :].permute(1, 0, 2)
        left = torch.einsum("pa,pab->pb", left, selected)
    return left[:, 0]


def fit_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 3,
    gamma: float = 1e-4,
    epochs: int = 500,
    learning_rate: float = 3e-2,
    batch_size: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    tolerance: float = 1e-8,
    verbose: bool = False,
) -> tuple[SquaredTTDensity, FitHistory]:
    """Fit a positive TT density from samples by the exact-L2 ratio loss.

    Args:
        samples: Array of shape ``(N, d)`` in the unit cube.  These are samples
            from the residual law ``F_k#nu_{k+1}``.
        modes: Number of equal-width cells in each coordinate.
        rank: Fixed internal TT rank of the square root ``g``.
        gamma: Positive density floor before normalisation.
        epochs: Number of Adam steps.
        learning_rate: Adam learning rate.
        batch_size: Optional size of the empirical linear-term minibatch.  The
            quadratic L2 term remains an exact full TT contraction at every step.
        seed: Reproducible initialisation and minibatch seed.
        device, dtype: Torch training device and precision.
        tolerance: Relative loss-change stopping threshold, checked over 25 steps.

    Returns:
        ``(density, history)``.  The returned TT is converted to numpy and does
        not retain an autograd graph.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without torch
        raise ImportError("fit_squared_tt_density requires the 'torch' extra") from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"samples must have shape (N, d), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("at least one sample is required")
    d = points.shape[1]
    points = _validate_points(points, d, name="samples")
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
        if modes_array.shape != (d,):
            raise ValueError(f"modes must contain {d} entries")
    if np.any(modes_array < 2):
        raise ValueError("each mode must contain at least two cells")
    if rank < 1:
        raise ValueError("rank must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ranks = [1] + [int(rank)] * (d - 1) + [1]
    params = []
    for k, n in enumerate(modes_array):
        noise = 2e-2 * torch.randn(
            (ranks[k], int(n), ranks[k + 1]), generator=generator,
            dtype=torch_dtype,
        )
        # A constant rank-one path gives h=1 at initialisation; the small dense
        # perturbation lets all rank directions receive a gradient immediately.
        noise[0, :, 0] += 1.0
        params.append(torch.nn.Parameter(noise.to(device)))

    indices_np = _cell_indices(points, modes_array)
    indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    rng = np.random.default_rng(seed + 1)
    history = FitHistory()
    total_cells = int(np.prod(modes_array, dtype=np.int64))

    # Import here so merely importing tt.transport does not require torch.
    from ..core.tools import dot

    def objective(batch_indices):
        root = vector.from_list(params)
        square = root * root
        m2 = dot(root, root) / total_cells
        m4 = dot(square, square) / total_cells
        z = gamma + m2
        h2 = (gamma * gamma + 2.0 * gamma * m2 + m4) / (z * z)
        values = _torch_sample_root(params, batch_indices)
        target_mean = ((gamma + values * values) / z).mean()
        return 0.5 * h2 - target_mean, h2, z

    window = 25
    for epoch in range(epochs):
        if batch_size is None or batch_size >= indices.shape[0]:
            batch = indices
        else:
            chosen = rng.choice(indices.shape[0], size=batch_size, replace=False)
            batch = indices[torch.as_tensor(chosen, dtype=torch.long, device=device)]
        optimizer.zero_grad(set_to_none=True)
        loss, h2, z = objective(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        h2_value = float(h2.detach().cpu())
        z_value = float(z.detach().cpu())
        history.loss.append(loss_value)
        history.l2_norm_sq.append(h2_value)
        history.normalization.append(z_value)
        history.chi2_to_reference.append(max(0.0, h2_value - 1.0))
        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(
                f"epoch {epoch + 1:4d}: loss={loss_value:.7e}, "
                f"chi2={max(0.0, h2_value - 1.0):.3e}, Z={z_value:.3e}"
            )
        if len(history.loss) >= 2 * window:
            old = np.mean(history.loss[-2 * window:-window])
            new = np.mean(history.loss[-window:])
            scale = max(1.0, abs(old), abs(new))
            if abs(new - old) <= tolerance * scale:
                history.converged = True
                break

    history.epochs = len(history.loss)
    fitted = vector.from_list([
        core.detach().cpu().numpy().copy() for core in params
    ])
    return SquaredTTDensity(fitted, gamma=gamma), history


class SampleDIRT:
    """Composition of sample-fitted incremental squared Rosenblatt maps."""

    def __init__(self, dimension: int, layers: Iterable[SquaredTTDensity] = ()):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.layers: list[SquaredTTDensity] = []
        for layer in layers:
            self.append(layer)

    def append(self, layer: SquaredTTDensity) -> None:
        if not isinstance(layer, SquaredTTDensity):
            raise TypeError("layer must be a SquaredTTDensity")
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

    def inverse(self, points) -> np.ndarray:
        """Map physical points to the reference cube."""
        value = _validate_points(points, self.dimension).copy()
        for layer in self.layers:
            value = layer.rosenblatt(value)
        return value

    def sample(self, count: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.forward(rng.random((count, self.dimension)))

    def fit_layer(self, samples_next, **fit_options) -> FitHistory:
        samples_next = _validate_points(samples_next, self.dimension,
                                        name="samples_next")
        residual_samples = self.inverse(samples_next)
        layer, history = fit_squared_tt_density(residual_samples, **fit_options)
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
        layers = [
            SquaredTTDensity.load(directory / f"layer_{k:03d}.npz")
            for k in range(count)
        ]
        return cls(dimension, layers)

