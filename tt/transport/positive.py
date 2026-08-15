"""Purified and entrywise-nonnegative linear TT densities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _as_numpy_cores,
    _linear_hat_integral_weights,
    _linear_indices_fractions,
    _linear_right_gram_environments,
    _linear_right_mean_environments,
    _linear_terminal_gram,
    _local_purified_right_environments,
    _sample_linear_tt_numpy,
    _sample_linear_tt_vector_numpy,
    _sample_local_purified_numpy,
    _validate_points,
)
from ._torch_density import (
    _torch_sample_linear_tt,
    _torch_sample_linear_tt_vector,
    _torch_sample_local_purified,
)
from .scalar import LinearSquaredTTDensity

class PurifiedLinearTTDensity(LinearSquaredTTDensity):
    """One vector-valued TT root with an exact sum-of-squares density.

    The terminal TT leg has size ``P`` and

    ``density(u) = (gamma + sum_l root_l(u)**2) / Z``.

    This is a single purified/Born tensor network: all terminal amplitudes
    share every preceding core and may be orthogonally gauge-rotated without
    changing the density.  It is not an ensemble of separately normalized
    densities.  The identity metric on the terminal leg makes normalization
    and every Rosenblatt suffix contraction the same exact hat-Gram operations
    as for a scalar squared root.
    """

    def __init__(self, cores: Sequence[np.ndarray], gamma: float = 1e-6):
        cores = [np.asarray(core, dtype=np.float64) for core in cores]
        if not cores or any(core.ndim != 3 for core in cores):
            raise TypeError("cores must be a non-empty sequence of 3D arrays")
        if cores[0].shape[0] != 1:
            raise ValueError("the first purified TT boundary rank must be one")
        if any(core.shape[1] < 2 for core in cores):
            raise ValueError("linear TT modes need at least two nodes")
        if any(
            cores[k].shape[2] != cores[k + 1].shape[0]
            for k in range(len(cores) - 1)
        ):
            raise ValueError("adjacent purified TT ranks do not match")
        if any(np.any(~np.isfinite(core)) for core in cores):
            raise ValueError("purified TT cores must be finite")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self._cores = [core.copy() for core in cores]
        self.gamma = float(gamma)
        self._right = _linear_right_gram_environments(self._cores)
        self._root_second_moment = float(self._right[0][0, 0])
        self.normalization = self.gamma + self._root_second_moment
        if not np.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("purified TT must have positive finite normalization")

    @property
    def d(self) -> int:
        return len(self._cores)

    @property
    def modes(self) -> np.ndarray:
        return np.asarray([core.shape[1] for core in self._cores], dtype=np.int64)

    @property
    def ranks(self) -> np.ndarray:
        return np.asarray(
            [self._cores[0].shape[0]]
            + [core.shape[2] for core in self._cores],
            dtype=np.int64,
        )

    @property
    def terminal_channels(self) -> int:
        return int(self._cores[-1].shape[2])

    @property
    def size(self) -> int:
        return int(sum(core.size for core in self._cores))

    @property
    def reference_floor_mass(self) -> float:
        return self.gamma / self.normalization

    def channel_spectrum(self) -> np.ndarray:
        """Normalized exact energy spectrum of the terminal amplitude space.

        The spectrum is invariant under terminal orthogonal rotations.  Thus
        it measures the number of purification channels actually used without
        assigning mixture-component semantics to a particular channel basis.
        The uniform floor is intentionally excluded.
        """
        eigenvalues = np.linalg.eigvalsh(_linear_terminal_gram(self._cores))
        eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
        total = float(eigenvalues.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise FloatingPointError("purified TT has no finite channel energy")
        return eigenvalues / total

    def effective_channels(self) -> dict[str, float]:
        """Entropy and inverse-participation ranks of ``channel_spectrum``."""
        probabilities = self.channel_spectrum()
        positive = probabilities[probabilities > 0.0]
        return {
            "entropy": float(np.exp(-np.sum(positive * np.log(positive)))),
            "inverse_participation": float(
                1.0 / np.sum(probabilities * probabilities)
            ),
        }

    def root_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_linear_tt_vector_numpy(self._cores, points)

    def density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return (self.gamma + np.sum(values * values, axis=1)) / self.normalization

    def log_density(self, points) -> np.ndarray:
        values = self.root_values(points)
        return (
            np.log(self.gamma + np.sum(values * values, axis=1))
            - np.log(self.normalization)
        )

    def log_density_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        import torch

        points = _validate_points(points, self.d)
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
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
                values = _torch_sample_linear_tt_vector(cores, batch)
                output[start:stop] = (
                    torch.log(self.gamma + values.square().sum(dim=1))
                    - np.log(self.normalization)
                ).cpu().numpy()
        return output

    def save(self, path) -> None:
        payload = {
            "kind": np.array(["linear-purified"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "PurifiedLinearTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            return cls(
                [data[f"core{k}"] for k in range(d)],
                gamma=float(data["gamma"][0]),
            )


class LocallyPurifiedLinearTTDensity:
    """Linear-hat locally purified MPS density with exact transport.

    A core has shape ``(r_left, modes, kraus, r_right)``.  The density signal
    is the squared MPS amplitude summed over every sequence of local Kraus
    indices.  This represents exponentially many structured amplitudes with
    one tensor network; it is not a list or mixture of density models.
    """

    def __init__(self, cores: Sequence[np.ndarray], gamma: float = 1e-6):
        cores = [np.asarray(core, dtype=np.float64) for core in cores]
        if not cores or any(core.ndim != 4 for core in cores):
            raise TypeError("cores must be a non-empty sequence of 4D arrays")
        if cores[0].shape[0] != 1 or cores[-1].shape[3] != 1:
            raise ValueError("locally purified TT boundary ranks must be one")
        if any(core.shape[1] < 2 or core.shape[2] < 1 for core in cores):
            raise ValueError("every core needs two nodes and one Kraus channel")
        if any(
            cores[k].shape[3] != cores[k + 1].shape[0]
            for k in range(len(cores) - 1)
        ):
            raise ValueError("adjacent locally purified TT ranks do not match")
        if any(np.any(~np.isfinite(core)) for core in cores):
            raise ValueError("locally purified TT cores must be finite")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        self._cores = [core.copy() for core in cores]
        self.gamma = float(gamma)
        self._right = _local_purified_right_environments(self._cores)
        self._signal_integral = float(self._right[0][0, 0])
        self.normalization = self.gamma + self._signal_integral
        if not np.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("locally purified TT has invalid normalization")

    @property
    def d(self) -> int:
        return len(self._cores)

    @property
    def modes(self) -> np.ndarray:
        return np.asarray([core.shape[1] for core in self._cores], dtype=np.int64)

    @property
    def ranks(self) -> np.ndarray:
        return np.asarray(
            [self._cores[0].shape[0]]
            + [core.shape[3] for core in self._cores], dtype=np.int64
        )

    @property
    def local_channels(self) -> np.ndarray:
        return np.asarray([core.shape[2] for core in self._cores], dtype=np.int64)

    @property
    def size(self) -> int:
        return int(sum(core.size for core in self._cores))

    @property
    def reference_floor_mass(self) -> float:
        return self.gamma / self.normalization

    def local_channel_spectra(self) -> list[np.ndarray]:
        """Gauge-invariant exact energy spectra of all local Kraus spaces."""
        left = np.ones((1, 1), dtype=np.float64)
        spectra = []
        for k, core in enumerate(self._cores):
            n = core.shape[1]
            h = 1.0 / (n - 1)
            right = self._right[k + 1]
            projected = np.einsum(
                "aikr,rs->aiks", core, right, optimize=True
            )
            diagonal = np.full(n, 2.0 * h / 3.0)
            diagonal[[0, -1]] = h / 3.0
            weighted_left = np.einsum(
                "ab,aiks->biks", left, projected, optimize=True
            )
            gram = np.einsum(
                "biks,bils->kl",
                weighted_left * diagonal[None, :, None, None],
                core,
                optimize=True,
            )
            gram += (h / 6.0) * (
                np.einsum(
                    "biks,bils->kl",
                    np.einsum(
                        "ab,aiks->biks", left, projected[:, :-1],
                        optimize=True,
                    ),
                    core[:, 1:],
                    optimize=True,
                )
                + np.einsum(
                    "biks,bils->kl",
                    np.einsum(
                        "ab,aiks->biks", left, projected[:, 1:],
                        optimize=True,
                    ),
                    core[:, :-1],
                    optimize=True,
                )
            )
            eigenvalues = np.maximum(
                np.linalg.eigvalsh(0.5 * (gram + gram.T))[::-1], 0.0
            )
            total = float(eigenvalues.sum())
            if total <= 0.0 or not np.isfinite(total):
                raise FloatingPointError(
                    "locally purified TT has no finite channel energy"
                )
            spectra.append(eigenvalues / total)

            projected_left = np.einsum(
                "ab,aikr->bikr", left, core, optimize=True
            )
            left = np.einsum(
                "bikr,biks->rs",
                projected_left * diagonal[None, :, None, None],
                core,
                optimize=True,
            )
            left += (h / 6.0) * (
                np.einsum(
                    "bikr,biks->rs",
                    projected_left[:, :-1], core[:, 1:], optimize=True,
                )
                + np.einsum(
                    "bikr,biks->rs",
                    projected_left[:, 1:], core[:, :-1], optimize=True,
                )
            )
        return spectra

    def effective_local_channels(self) -> dict[str, list[float]]:
        """Entropy and participation ranks of local channel spectra."""
        entropy, participation = [], []
        for probabilities in self.local_channel_spectra():
            positive = probabilities[probabilities > 0.0]
            entropy.append(float(np.exp(-np.sum(positive * np.log(positive)))))
            participation.append(float(
                1.0 / np.sum(probabilities * probabilities)
            ))
        return {"entropy": entropy, "inverse_participation": participation}

    def signal(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_local_purified_numpy(self._cores, points)

    def density(self, points) -> np.ndarray:
        return (self.gamma + self.signal(points)) / self.normalization

    def log_density(self, points) -> np.ndarray:
        return np.log(self.gamma + self.signal(points)) - np.log(
            self.normalization
        )

    def log_density_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 16384,
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
                signal = _torch_sample_local_purified(cores, batch)
                output[start:stop] = (
                    torch.log(self.gamma + signal) - np.log(self.normalization)
                ).cpu().numpy()
        return output

    @staticmethod
    def _update_left(left, selected):
        contracted = np.einsum(
            "pab,pakr->pbkr", left, selected, optimize=True
        )
        return np.einsum(
            "pbkr,pbks->prs", contracted, selected, optimize=True
        )

    def _conditional_coefficients_batch(self, left, k):
        core = self._cores[k]
        first = core[:, :-1]
        delta = core[:, 1:] - first
        right = self._right[k + 1]
        projected_first = np.einsum(
            "aikr,rs->aiks", first, right, optimize=True
        )
        projected_delta = np.einsum(
            "aikr,rs->aiks", delta, right, optimize=True
        )
        first_gram = np.einsum(
            "aiks,biks->abi", projected_first, first, optimize=True
        )
        cross_gram = np.einsum(
            "aiks,biks->abi", projected_delta, first, optimize=True
        )
        delta_gram = np.einsum(
            "aiks,biks->abi", projected_delta, delta, optimize=True
        )
        a = np.einsum("pab,abi->pi", left, first_gram, optimize=True)
        b = 2.0 * np.einsum(
            "pab,abi->pi", left, cross_gram, optimize=True
        )
        c = np.einsum("pab,abi->pi", left, delta_gram, optimize=True)
        h = 1.0 / (core.shape[1] - 1)
        masses = h * (self.gamma + a + 0.5 * b + c / 3.0)
        masses = np.maximum(masses, np.finfo(float).tiny)
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
        left = np.ones((len(points), 1, 1), dtype=np.float64)
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
            first = np.moveaxis(self._cores[k][:, interval, :, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :, :], 1, 0
            )
            selected = (
                (1.0 - fraction[:, k])[:, None, None, None] * first
                + fraction[:, k, None, None, None] * second
            )
            left = self._update_left(left, selected)
        return np.clip(result, 0.0, 1.0)

    def inverse_rosenblatt(self, uniform) -> np.ndarray:
        uniform = _validate_points(uniform, self.d, name="uniform")
        result = np.empty_like(uniform)
        left = np.ones((len(uniform), 1, 1), dtype=np.float64)
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
            aa, bb, cc = a[rows, interval], b[rows, interval], c[rows, interval]
            lo = np.zeros(len(uniform), dtype=np.float64)
            hi = np.ones(len(uniform), dtype=np.float64)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                value = self._partial_integral(aa, bb, cc, mid, self.gamma)
                lo = np.where(value < local_target, mid, lo)
                hi = np.where(value >= local_target, mid, hi)
            fraction = 0.5 * (lo + hi)
            result[:, k] = (interval + fraction) / (int(n) - 1)
            first = np.moveaxis(self._cores[k][:, interval, :, :], 1, 0)
            second = np.moveaxis(
                self._cores[k][:, interval + 1, :, :], 1, 0
            )
            selected = (
                (1.0 - fraction)[:, None, None, None] * first
                + fraction[:, None, None, None] * second
            )
            left = self._update_left(left, selected)
        return result

    def rosenblatt_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 2048,
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
                    (len(value), 1, 1), dtype=torch_dtype, device=device
                )
                rows = torch.arange(len(value), device=device)
                for k, core in enumerate(cores):
                    first = core[:, :-1]
                    delta = core[:, 1:] - first
                    projected_first = torch.einsum(
                        "aikr,rs->aiks", first, right[k + 1]
                    )
                    projected_delta = torch.einsum(
                        "aikr,rs->aiks", delta, right[k + 1]
                    )
                    first_gram = torch.einsum(
                        "aiks,biks->abi", projected_first, first
                    )
                    cross_gram = torch.einsum(
                        "aiks,biks->abi", projected_delta, first
                    )
                    delta_gram = torch.einsum(
                        "aiks,biks->abi", projected_delta, delta
                    )
                    a = torch.einsum("pab,abi->pi", left, first_gram)
                    b = 2.0 * torch.einsum(
                        "pab,abi->pi", left, cross_gram
                    )
                    c = torch.einsum("pab,abi->pi", left, delta_gram)
                    h = 1.0 / (core.shape[1] - 1)
                    masses = torch.clamp_min(
                        h * (self.gamma + a + 0.5 * b + c / 3.0),
                        torch.finfo(torch_dtype).tiny,
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
                    aa, bb, cc = (
                        a[rows, interval], b[rows, interval], c[rows, interval]
                    )
                    local = h * (
                        (self.gamma + aa) * fraction
                        + 0.5 * bb * fraction.square()
                        + (cc / 3.0) * fraction.pow(3)
                    )
                    result[:, k] = (before + local) / cumulative[:, -1]
                    node_first = core[:, interval, :, :].permute(1, 0, 2, 3)
                    node_second = core[:, interval + 1, :, :].permute(1, 0, 2, 3)
                    selected = (
                        (1.0 - fraction)[:, None, None, None] * node_first
                        + fraction[:, None, None, None] * node_second
                    )
                    contracted = torch.einsum(
                        "pab,pakr->pbkr", left, selected
                    )
                    left = torch.einsum(
                        "pbkr,pbks->prs", contracted, selected
                    )
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
            "kind": np.array(["linear-local-purified"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "LocallyPurifiedLinearTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            return cls(
                [data[f"core{k}"] for k in range(d)],
                gamma=float(data["gamma"][0]),
            )


class NonnegativeLinearTTDensity:
    """A directly nonnegative multilinear TT density on ``[0, 1]^d``.

    Every TT-core entry and every nodal hat function is nonnegative, hence

    ``density(u) = (gamma + G_1(u_1) ... G_d(u_d)) / Z``

    is positive without squaring a TT root.  This distinction is important:
    the represented density has the stated TT ranks, rather than the squared
    ranks and cross terms induced by ``root**2``.  Hat-function integrals give
    ``Z`` and all suffix marginals exactly.  Conditional densities are
    piecewise linear, so their CDFs are piecewise quadratic and are inverted
    robustly by scalar bisection.

    The class stores one TT, not a wrapper around mixture components.  Dense
    positive internal cores allow latent TT states to change at every
    coordinate; a diagonal/product construction may be used to initialize a
    fit, but is not retained as a separate model.
    """

    def __init__(self, tensor: vector, gamma: float = 1e-6):
        if not isinstance(tensor, vector) or tensor.d == 0:
            raise TypeError("tensor must be a non-empty tt.vector")
        if tensor.r[0] != 1 or tensor.r[-1] != 1:
            raise ValueError("tensor must be a scalar TT")
        if np.any(tensor.n < 2):
            raise ValueError("linear TT modes need at least two nodes")
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and strictly positive")
        cores = _as_numpy_cores(tensor)
        if any(np.any(~np.isfinite(core)) or np.any(core < 0.0) for core in cores):
            raise ValueError("all nonnegative TT core entries must be finite and >= 0")
        self.tensor = vector.from_list(cores)
        self.gamma = float(gamma)
        self._cores = _as_numpy_cores(self.tensor)
        self._right = _linear_right_mean_environments(self._cores)
        self._component_integral = float(self._right[0][0])
        self.normalization = self.gamma + self._component_integral
        if not np.isfinite(self.normalization) or self.normalization <= 0.0:
            raise ValueError("nonnegative linear TT must have positive finite integral")

    @property
    def d(self) -> int:
        return self.tensor.d

    @property
    def modes(self) -> np.ndarray:
        return self.tensor.n.astype(np.int64)

    @property
    def ranks(self) -> np.ndarray:
        return self.tensor.r.astype(np.int64)

    @property
    def size(self) -> int:
        return self.tensor.size

    @property
    def reference_floor_mass(self) -> float:
        return self.gamma / self.normalization

    def bond_state_probabilities(self) -> list[np.ndarray]:
        """Exact integrated mass carried by every internal TT state."""
        left = np.ones(1, dtype=np.float64)
        result = []
        for k, core in enumerate(self._cores[:-1]):
            integrated = np.einsum(
                "i,aib->ab",
                _linear_hat_integral_weights(core.shape[1]),
                core,
                optimize=True,
            )
            left = left @ integrated
            mass = left * self._right[k + 1]
            total = float(mass.sum())
            if total <= 0.0 or not np.isfinite(total):
                raise FloatingPointError("invalid nonnegative TT bond mass")
            result.append(mass / total)
        return result

    def effective_state_ranks(self) -> dict[str, list[float]]:
        """Entropy and inverse-participation effective ranks at TT bonds."""
        entropy_ranks = []
        participation_ranks = []
        for probabilities in self.bond_state_probabilities():
            positive = probabilities[probabilities > 0.0]
            entropy_ranks.append(float(np.exp(
                -np.sum(positive * np.log(positive))
            )))
            participation_ranks.append(float(
                1.0 / np.sum(probabilities * probabilities)
            ))
        return {
            "entropy": entropy_ranks,
            "inverse_participation": participation_ranks,
        }

    def component_values(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        return _sample_linear_tt_numpy(self._cores, points)

    def density(self, points) -> np.ndarray:
        values = self.component_values(points)
        return (self.gamma + values) / self.normalization

    def log_density(self, points) -> np.ndarray:
        values = self.component_values(points)
        return np.log(self.gamma + values) - np.log(self.normalization)

    def log_density_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 32768,
    ) -> np.ndarray:
        import torch

        points = _validate_points(points, self.d)
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
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
                    torch.log(self.gamma + values)
                    - np.log(self.normalization)
                ).cpu().numpy()
        return output

    def _conditional_nodes_batch(
        self, left: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.einsum(
            "pa,aib->pib", left, self._cores[k], optimize=True
        )
        values = self.gamma + np.einsum(
            "pib,b->pi", nodes, self._right[k + 1], optimize=True
        )
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise FloatingPointError("invalid nonnegative-TT conditional values")
        h = 1.0 / (self._cores[k].shape[1] - 1)
        masses = 0.5 * h * (values[:, :-1] + values[:, 1:])
        totals = masses.sum(axis=1)
        if np.any(totals <= 0.0) or not np.all(np.isfinite(totals)):
            raise FloatingPointError("invalid nonnegative-TT conditional mass")
        return values, masses

    @staticmethod
    def _partial_linear_integral(first, second, fraction):
        return first * fraction + 0.5 * (second - first) * fraction * fraction

    def rosenblatt(self, points) -> np.ndarray:
        points = _validate_points(points, self.d)
        lower, fraction = _linear_indices_fractions(points, self.modes)
        result = np.empty_like(points)
        left = np.ones((len(points), 1), dtype=np.float64)
        rows = np.arange(len(points))
        for k, n_value in enumerate(self.modes):
            n = int(n_value)
            values, masses = self._conditional_nodes_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            interval = lower[:, k]
            before = np.where(
                interval == 0,
                0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            first_value = values[rows, interval]
            second_value = values[rows, interval + 1]
            local = self._partial_linear_integral(
                first_value, second_value, fraction[:, k]
            ) / (n - 1)
            result[:, k] = (before + local) / cumulative[:, -1]
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(self._cores[k][:, interval + 1, :], 1, 0)
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
        for k, n_value in enumerate(self.modes):
            n = int(n_value)
            values, masses = self._conditional_nodes_batch(left, k)
            cumulative = np.cumsum(masses, axis=1)
            target = np.minimum(
                uniform[:, k], np.nextafter(1.0, 0.0)
            ) * cumulative[:, -1]
            interval = np.minimum(
                np.sum(cumulative <= target[:, None], axis=1), n - 2
            )
            before = np.where(
                interval == 0,
                0.0,
                cumulative[rows, np.maximum(interval - 1, 0)],
            )
            local_target = (target - before) * (n - 1)
            first_value = values[rows, interval]
            second_value = values[rows, interval + 1]
            lo = np.zeros(len(uniform), dtype=np.float64)
            hi = np.ones(len(uniform), dtype=np.float64)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                partial = self._partial_linear_integral(
                    first_value, second_value, mid
                )
                lo = np.where(partial < local_target, mid, lo)
                hi = np.where(partial >= local_target, mid, hi)
            local_fraction = 0.5 * (lo + hi)
            result[:, k] = (interval + local_fraction) / (n - 1)
            first = np.moveaxis(self._cores[k][:, interval, :], 1, 0)
            second = np.moveaxis(self._cores[k][:, interval + 1, :], 1, 0)
            selected = (
                (1.0 - local_fraction)[:, None, None] * first
                + local_fraction[:, None, None] * second
            )
            left = np.einsum("pa,pab->pb", left, selected, optimize=True)
        return result

    def rosenblatt_device(
        self, points, *, device: str, dtype: str = "float32",
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Accelerated exact piecewise-quadratic Rosenblatt evaluation."""
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
                result = torch.empty_like(value)
                left = torch.ones(
                    (len(value), 1), dtype=torch_dtype, device=device
                )
                rows = torch.arange(len(value), device=device)
                for k, core in enumerate(cores):
                    nodes = torch.einsum("pa,aib->pib", left, core)
                    nodal_values = self.gamma + torch.einsum(
                        "pib,b->pi", nodes, right[k + 1]
                    )
                    h = 1.0 / (core.shape[1] - 1)
                    masses = 0.5 * h * (
                        nodal_values[:, :-1] + nodal_values[:, 1:]
                    )
                    cumulative = torch.cumsum(masses, dim=1)
                    position = value[:, k] * (core.shape[1] - 1)
                    interval = torch.clamp(
                        torch.floor(position).to(torch.long),
                        max=core.shape[1] - 2,
                    )
                    local_fraction = torch.clamp(
                        position - interval, 0.0, 1.0
                    )
                    before = torch.where(
                        interval == 0,
                        torch.zeros((), dtype=torch_dtype, device=device),
                        cumulative[rows, torch.clamp_min(interval - 1, 0)],
                    )
                    first_value = nodal_values[rows, interval]
                    second_value = nodal_values[rows, interval + 1]
                    local = h * self._partial_linear_integral(
                        first_value, second_value, local_fraction
                    )
                    result[:, k] = (before + local) / cumulative[:, -1]
                    first = core[:, interval, :].permute(1, 0, 2)
                    second = core[:, interval + 1, :].permute(1, 0, 2)
                    selected = (
                        (1.0 - local_fraction)[:, None, None] * first
                        + local_fraction[:, None, None] * second
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

    def model_l2_norm_sq(self) -> float:
        component_second = float(
            _linear_right_gram_environments(self._cores)[0][0, 0]
        )
        numerator = (
            self.gamma * self.gamma
            + 2.0 * self.gamma * self._component_integral
            + component_second
        )
        return numerator / (self.normalization * self.normalization)

    @property
    def chi2_to_reference(self) -> float:
        return max(0.0, self.model_l2_norm_sq() - 1.0)

    def save(self, path) -> None:
        payload = {
            "kind": np.array(["linear-nonnegative"]),
            "gamma": np.array([self.gamma]),
            "d": np.array([self.d], dtype=np.int64),
        }
        payload.update({f"core{k}": core for k, core in enumerate(self._cores)})
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path) -> "NonnegativeLinearTTDensity":
        with np.load(path) as data:
            d = int(data["d"][0])
            tensor = vector.from_list([data[f"core{k}"] for k in range(d)])
            return cls(tensor, gamma=float(data["gamma"][0]))

__all__ = [
    "PurifiedLinearTTDensity",
    "LocallyPurifiedLinearTTDensity",
    "NonnegativeLinearTTDensity",
]
