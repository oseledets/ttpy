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
import time
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
    gradient_norm: list[float] = field(default_factory=list)
    epochs: int = 0
    converged: bool = False
    optimizer: str = ""
    function_calls: int = 0
    wall_time: float = 0.0


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


def _torch_root_moments(cores):
    """Exact uniform second and fourth moments without Hadamard TT ranks.

    Contracting four copies of every core directly avoids materialising
    ``root * root`` (whose ranks are squared) and works unchanged under
    autograd, including on the rank-doubled tangent stack used by RGD.
    """
    import torch

    env2 = torch.ones((), dtype=cores[0].dtype, device=cores[0].device).reshape(1, 1)
    env4 = torch.ones((), dtype=cores[0].dtype, device=cores[0].device).reshape(
        1, 1, 1, 1
    )
    for core in cores:
        projected = torch.einsum("ac,aib->cib", env2, core)
        env2 = torch.einsum("cib,cid->bd", projected, core) / core.shape[1]
        env4 = _torch_fourth_left_step(env4, core) / core.shape[1]
    return env2.reshape(()), env4.reshape(())


def _torch_fourth_left_step(environment, core):
    """Contract one core into a fourth-order left environment, pairwise."""
    import torch

    work = torch.einsum("aceg,aib->cegib", environment, core)
    work = torch.einsum("cegib,cid->egibd", work, core)
    work = torch.einsum("egibd,eif->gibdf", work, core)
    return torch.einsum("gibdf,gih->bdfh", work, core)


def _torch_fourth_right_step(core, environment):
    """Contract one core into a fourth-order right environment, pairwise."""
    import torch

    work = torch.einsum("bdfh,gih->bdfgi", environment, core)
    work = torch.einsum("bdfgi,eif->bdgie", work, core)
    work = torch.einsum("bdgie,cid->bgiec", work, core)
    return torch.einsum("bgiec,aib->aceg", work, core)


def _torch_density_objective(cores, indices, weights, gamma: float):
    """Exact-contraction L2 ratio objective for a list of torch TT cores."""
    m2, m4 = _torch_root_moments(cores)
    z = gamma + m2
    h2 = (gamma * gamma + 2.0 * gamma * m2 + m4) / (z * z)
    values = _torch_sample_root(cores, indices)
    target_mean = (weights * (gamma + values * values) / z).sum()
    return 0.5 * h2 - target_mean, h2, z


def _compress_empirical_cells(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce repeated cells; exact because the basis is cellwise constant."""
    unique, counts = np.unique(indices, axis=0, return_counts=True)
    return unique, counts.astype(np.float64) / counts.sum()


def _coarse_to_fine_initial_root(
    points: np.ndarray,
    modes: np.ndarray,
    *,
    rank: int,
    coarse_bins: int,
    pseudocount: float,
) -> vector:
    """Build a sample-only multiscale root retaining coarse joint dependence."""
    d = points.shape[1]
    coarse_bins = int(coarse_bins)
    if coarse_bins < 2:
        raise ValueError("coarse_bins must be at least two")
    if np.any(modes % coarse_bins != 0):
        raise ValueError("every mode size must be divisible by coarse_bins")
    coarse_cells = coarse_bins ** d
    if coarse_cells > 1_048_576:
        raise ValueError(
            "coarse initializer would contain more than 1,048,576 cells"
        )
    if pseudocount <= 0.0 or not np.isfinite(pseudocount):
        raise ValueError("initialization_pseudocount must be positive")

    coarse = np.floor(points * coarse_bins).astype(np.int64)
    coarse = np.minimum(coarse, coarse_bins - 1)
    strides = coarse_bins ** np.arange(d - 1, -1, -1, dtype=np.int64)
    flat = coarse @ strides
    counts = np.bincount(flat, minlength=coarse_cells).astype(np.float64)
    counts += pseudocount
    probabilities = counts / counts.sum()
    coarse_density = (probabilities * coarse_cells).reshape(
        [coarse_bins] * d
    )
    coarse_root = vector(np.sqrt(coarse_density), eps=1e-13, rmax=rank)

    fine = _cell_indices(points, modes)
    lifted = []
    for k, (core, mode) in enumerate(zip(coarse_root.cores, modes)):
        width = int(mode // coarse_bins)
        basis = np.zeros((coarse_bins, int(mode)), dtype=np.float64)
        for coarse_cell in range(coarse_bins):
            start = coarse_cell * width
            selected = fine[coarse[:, k] == coarse_cell, k] - start
            local_counts = np.bincount(selected, minlength=width).astype(np.float64)
            local_counts += pseudocount
            local_probability = local_counts / local_counts.sum()
            # Density relative to the uniform measure inside the coarse cell.
            basis[coarse_cell, start:start + width] = np.sqrt(
                width * local_probability
            )
        lifted.append(np.einsum("asb,si->aib", core, basis, optimize=True))
    return vector.from_list(lifted)


def _torch_right_orthogonalize(cores) -> None:
    """Put a torch TT into mixed canonical form with centre zero, in place."""
    import torch

    for k in range(len(cores) - 1, 0, -1):
        r1, n, r2 = cores[k].shape
        q, r = torch.linalg.qr(cores[k].reshape(r1, n * r2).T, mode="reduced")
        if q.shape[1] != r1:
            raise ValueError(
                f"TT rank {r1} at bond {k} is not attainable for mode {n}; "
                "ALS orthogonalization would lower the requested fixed rank"
            )
        cores[k] = q.T.reshape(r1, n, r2).detach()
        cores[k - 1] = torch.einsum("aib,bc->aic", cores[k - 1], r.T).detach()


def _torch_move_center_right(cores, k: int) -> None:
    import torch

    r1, n, r2 = cores[k].shape
    q, r = torch.linalg.qr(cores[k].reshape(r1 * n, r2), mode="reduced")
    if q.shape[1] != r2:
        raise ValueError("left orthogonalization lowered the requested TT rank")
    cores[k] = q.reshape(r1, n, r2).detach()
    cores[k + 1] = torch.einsum("ab,bic->aic", r, cores[k + 1]).detach()


def _torch_move_center_left(cores, k: int) -> None:
    import torch

    r1, n, r2 = cores[k].shape
    q, r = torch.linalg.qr(cores[k].reshape(r1, n * r2).T, mode="reduced")
    if q.shape[1] != r1:
        raise ValueError("right orthogonalization lowered the requested TT rank")
    cores[k] = q.T.reshape(r1, n, r2).detach()
    cores[k - 1] = torch.einsum("aib,bc->aic", cores[k - 1], r.T).detach()


def _torch_sample_interfaces(cores, indices, centre: int):
    """Cached sample contractions on both sides of one ALS centre."""
    import torch

    count = indices.shape[0]
    left = torch.ones((count, 1), dtype=cores[0].dtype, device=cores[0].device)
    for k in range(centre):
        selected = cores[k][:, indices[:, k], :].permute(1, 0, 2)
        left = torch.einsum("pa,pab->pb", left, selected)
    right = torch.ones((count, 1), dtype=cores[0].dtype, device=cores[0].device)
    for k in range(len(cores) - 1, centre, -1):
        selected = cores[k][:, indices[:, k], :].permute(1, 0, 2)
        right = torch.einsum("pab,pb->pa", selected, right)
    return left, right


def _torch_fourth_interfaces(cores, centre: int):
    """Exact fourth-order uniform interfaces around an ALS centre."""
    import torch

    left = torch.ones((1, 1, 1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for k in range(centre):
        core = cores[k]
        left = _torch_fourth_left_step(left, core) / core.shape[1]
    right = torch.ones((1, 1, 1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for k in range(len(cores) - 1, centre, -1):
        core = cores[k]
        right = _torch_fourth_right_step(core, right) / core.shape[1]
    return left, right


def _torch_als_local_objective(
    core, *, centre, modes, total_cells, indices, weights, gamma,
    sample_left, sample_right, fourth_left, fourth_right,
):
    """Nonlinear ALS objective with every fixed-side contraction cached."""
    import torch

    # Mixed canonical form makes the global Frobenius norm a local norm.
    m2 = (core * core).sum() / total_cells
    local_fourth = _torch_fourth_left_step(fourth_left, core)
    m4 = torch.einsum("bdfh,bdfh->", local_fourth, fourth_right) / modes[centre]
    z = gamma + m2
    h2 = (gamma * gamma + 2.0 * gamma * m2 + m4) / (z * z)
    selected = core[:, indices[:, centre], :].permute(1, 0, 2)
    partial = torch.einsum("pa,pab->pb", sample_left, selected)
    values = (partial * sample_right).sum(dim=1)
    target_mean = (weights * (gamma + values * values) / z).sum()
    return 0.5 * h2 - target_mean


def _fit_tt_als(
    initial_cores,
    *,
    modes,
    indices,
    weights,
    gamma,
    sweeps,
    inner_steps,
    learning_rate,
    tolerance,
    verbose,
):
    """Orthogonal nonlinear ALS for the squared-density objective.

    The outer objective is rational-quartic, so the local problem is not a
    linear least-squares solve.  It is nevertheless a genuine alternating
    optimization: one core is minimized by L-BFGS while every fixed-side
    sample and fourth-moment contraction is cached.  QR moves the mixed
    canonical centre between sites and reduces the second moment to a local
    Frobenius norm.
    """
    import torch

    cores = [core.detach().clone() for core in initial_cores]
    _torch_right_orthogonalize(cores)
    total_cells = int(np.prod(modes, dtype=np.int64))
    history = FitHistory(optimizer="als")
    started = time.perf_counter()

    def global_record() -> float:
        with torch.no_grad():
            loss, h2, z = _torch_density_objective(
                cores, indices, weights, gamma
            )
        lv, h2v, zv = float(loss), float(h2), float(z)
        history.loss.append(lv)
        history.l2_norm_sq.append(h2v)
        history.normalization.append(zv)
        history.chi2_to_reference.append(max(0.0, h2v - 1.0))
        return lv

    previous = global_record()
    for sweep in range(int(sweeps)):
        # Left-to-right: optimize the current centre, then move it by QR.
        centres = list(range(len(cores)))
        # Right-to-left: QR the old centre first, then optimize the new one.
        reverse_centres = list(range(len(cores) - 1, 0, -1))

        for centre in centres:
            sample_left, sample_right = _torch_sample_interfaces(
                cores, indices, centre
            )
            fourth_left, fourth_right = _torch_fourth_interfaces(cores, centre)
            parameter = torch.nn.Parameter(cores[centre].detach().clone())
            local = torch.optim.LBFGS(
                [parameter],
                lr=learning_rate,
                max_iter=int(inner_steps),
                tolerance_grad=tolerance,
                tolerance_change=tolerance,
                line_search_fn="strong_wolfe",
            )

            def closure():
                local.zero_grad(set_to_none=True)
                value = _torch_als_local_objective(
                    parameter,
                    centre=centre,
                    modes=modes,
                    total_cells=total_cells,
                    indices=indices,
                    weights=weights,
                    gamma=gamma,
                    sample_left=sample_left,
                    sample_right=sample_right,
                    fourth_left=fourth_left,
                    fourth_right=fourth_right,
                )
                value.backward()
                history.function_calls += 1
                return value

            local.step(closure)
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("ALS produced a non-finite TT core")
            cores[centre] = parameter.detach()
            if centre < len(cores) - 1:
                _torch_move_center_right(cores, centre)

        for old_centre in reverse_centres:
            _torch_move_center_left(cores, old_centre)
            centre = old_centre - 1
            sample_left, sample_right = _torch_sample_interfaces(
                cores, indices, centre
            )
            fourth_left, fourth_right = _torch_fourth_interfaces(cores, centre)
            parameter = torch.nn.Parameter(cores[centre].detach().clone())
            local = torch.optim.LBFGS(
                [parameter],
                lr=learning_rate,
                max_iter=int(inner_steps),
                tolerance_grad=tolerance,
                tolerance_change=tolerance,
                line_search_fn="strong_wolfe",
            )

            def closure_reverse():
                local.zero_grad(set_to_none=True)
                value = _torch_als_local_objective(
                    parameter,
                    centre=centre,
                    modes=modes,
                    total_cells=total_cells,
                    indices=indices,
                    weights=weights,
                    gamma=gamma,
                    sample_left=sample_left,
                    sample_right=sample_right,
                    fourth_left=fourth_left,
                    fourth_right=fourth_right,
                )
                value.backward()
                history.function_calls += 1
                return value

            local.step(closure_reverse)
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("ALS produced a non-finite TT core")
            cores[centre] = parameter.detach()

        current = global_record()
        history.epochs = sweep + 1
        if verbose:
            print(
                f"sweep {sweep + 1:4d}: loss={current:.7e}, "
                f"chi2={history.chi2_to_reference[-1]:.3e}, "
                f"Z={history.normalization[-1]:.3e}"
            )
        scale = max(1.0, abs(previous), abs(current))
        if abs(current - previous) <= tolerance * scale:
            history.converged = True
            break
        previous = current

    history.wall_time = time.perf_counter() - started
    return cores, history


def _fit_tt_riemannian_stochastic(
    initial_root,
    *,
    all_indices_np,
    full_indices,
    full_weights,
    gamma,
    iterations,
    batch_size,
    learning_rate,
    momentum_decay,
    second_moment_decay,
    retraction_method,
    seed,
    verbose,
):
    """Minibatch Riemannian momentum with vector transport.

    The stochasticity occurs only in ``-E[h(V)]``.  The normalization and
    fourth-moment terms remain exact TT contractions on every iteration.  The
    first moment is a tangent vector and is transported after every retraction;
    the second moment is a gauge-invariant scalar tangent norm.
    """
    import torch

    from ..algs.autodiff import riemannian_grad
    from ..algs.riemannian import (retract, tangent_inner, tangent_to_tt,
                                   transport)

    x = initial_root
    rng = np.random.default_rng(seed + 1)
    history = FitHistory(optimizer="riemannian-sgd")
    velocity = None
    squared_norm_average = 0.0
    checked = False
    started = time.perf_counter()
    count = all_indices_np.shape[0]
    batch_size = min(int(batch_size), count)

    for iteration in range(int(iterations)):
        chosen = rng.choice(count, size=batch_size, replace=False)
        batch_np, batch_weights_np = _compress_empirical_cells(
            all_indices_np[chosen]
        )
        batch_indices = torch.as_tensor(
            batch_np, dtype=torch.long, device=x.cores[0].device
        )
        batch_weights = torch.as_tensor(
            batch_weights_np, dtype=x.cores[0].dtype, device=x.cores[0].device
        )

        def minibatch_objective(cores):
            return _torch_density_objective(
                cores, batch_indices, batch_weights, gamma
            )[0]

        value, gradient, frames_ = riemannian_grad(
            minibatch_objective, x, runtime_check=not checked
        )
        checked = True
        history.function_calls += 1
        gradient_norm_sq = float(abs(tangent_inner(gradient, gradient)))
        gradient_norm = gradient_norm_sq ** 0.5
        history.loss.append(value)
        history.gradient_norm.append(gradient_norm)

        if velocity is None:
            velocity = [(1.0 - momentum_decay) * core for core in gradient]
        else:
            velocity = [
                momentum_decay * old + (1.0 - momentum_decay) * new
                for old, new in zip(velocity, gradient)
            ]
        squared_norm_average = (
            second_moment_decay * squared_norm_average
            + (1.0 - second_moment_decay) * gradient_norm_sq
        )
        step_number = iteration + 1
        velocity_correction = 1.0 - momentum_decay ** step_number
        norm_correction = 1.0 - second_moment_decay ** step_number
        rms = (squared_norm_average / norm_correction) ** 0.5
        scale = learning_rate / max(rms, 1e-12) / velocity_correction
        direction = tangent_to_tt(
            x, [-scale * core for core in velocity], frames_=frames_
        )
        x_new = retract(x, direction, method=retraction_method)
        if momentum_decay > 0.0:
            velocity, _ = transport(velocity, x, x_new)
        else:
            # With no momentum the next gradient replaces the state entirely;
            # transporting it would be mathematically redundant.
            velocity = None
        x = x_new

        if verbose and (iteration == 0 or (iteration + 1) % 25 == 0):
            print(
                f"iteration {iteration + 1:4d}: batch_loss={value:.7e}, "
                f"|grad|={gradient_norm:.3e}, step={scale:.3e}"
            )

    with torch.no_grad():
        final_loss, final_h2, final_z = _torch_density_objective(
            list(x.cores), full_indices, full_weights, gamma
        )
    history.loss.append(float(final_loss))
    history.l2_norm_sq.append(float(final_h2))
    history.normalization.append(float(final_z))
    history.chi2_to_reference.append(max(0.0, float(final_h2) - 1.0))
    history.function_calls += 1
    history.epochs = int(iterations)
    history.wall_time = time.perf_counter() - started
    return x, history


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
    optimizer: str = "adam",
    als_inner_steps: int = 8,
    riemannian_retraction: str = "svd",
    riemannian_momentum: float = 0.9,
    riemannian_second_moment: float = 0.99,
    initialization: str = "uniform",
    initialization_noise: float = 2e-2,
    initialization_coarse_bins: int = 2,
    initialization_pseudocount: float = 0.5,
    verbose: bool = False,
) -> tuple[SquaredTTDensity, FitHistory]:
    """Fit a positive TT density from samples by the exact-L2 ratio loss.

    Args:
        samples: Array of shape ``(N, d)`` in the unit cube.  These are samples
            from the residual law ``F_k#nu_{k+1}``.
        modes: Number of equal-width cells in each coordinate.
        rank: Fixed internal TT rank of the square root ``g``.
        gamma: Positive density floor before normalisation.
        epochs: Adam/RGD iterations or complete nonlinear ALS sweeps.
        learning_rate: Adam rate, RGD initial Armijo step, or local ALS
            L-BFGS rate.
        batch_size: Optional size of the empirical linear-term minibatch.  The
            quadratic L2 term remains an exact full TT contraction at every step.
        seed: Reproducible initialisation and minibatch seed.
        device, dtype: Torch training device and precision.
        tolerance: Relative loss/gradient stopping threshold. Adam compares
            two 25-step windows; RGD uses the tangent-gradient norm and ALS
            compares complete sweeps.
        optimizer: ``"adam"``, deterministic ``"riemannian"``, stochastic
            ``"riemannian-sgd"`` or orthogonal nonlinear ``"als"``.  The
            latter is nonlinear because the normalized squared TT objective is
            rational-quartic in one core.
        als_inner_steps: L-BFGS iterations per core update and sweep direction.
        riemannian_retraction: Fixed-rank retraction, ``"svd"`` or the
            orthogonal projector-splitting ``"psa"`` sweep.
        riemannian_momentum, riemannian_second_moment: Decays for transported
            tangent momentum and its gauge-invariant scalar RMS in stochastic
            Riemannian optimization.
        initialization: ``"uniform"`` for the perturbed constant path or
            ``"coarse"`` for a sample-only coarse-grid TT followed by a
            one-dimensional empirical lift to the fine grid.
        initialization_noise: Standard deviation of the dense perturbation of
            the uniform path. Small residual-ratio layers generally need a
            much smaller value than a one-shot fit.
        initialization_coarse_bins, initialization_pseudocount: Resolution and
            smoothing of the coarse-to-fine initialiser.

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
    optimizer = str(optimizer).lower().replace("_", "-")
    allowed_optimizers = ("adam", "riemannian", "riemannian-sgd", "als")
    if optimizer not in allowed_optimizers:
        raise ValueError(
            "optimizer must be 'adam', 'riemannian', 'riemannian-sgd' or 'als'"
        )
    if als_inner_steps < 1:
        raise ValueError("als_inner_steps must be positive")
    if riemannian_retraction not in ("svd", "psa"):
        raise ValueError("riemannian_retraction must be 'svd' or 'psa'")
    if not 0.0 <= riemannian_momentum < 1.0:
        raise ValueError("riemannian_momentum must lie in [0, 1)")
    if not 0.0 <= riemannian_second_moment < 1.0:
        raise ValueError("riemannian_second_moment must lie in [0, 1)")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("uniform", "coarse"):
        raise ValueError("initialization must be 'uniform' or 'coarse'")
    if initialization_noise < 0.0 or not np.isfinite(initialization_noise):
        raise ValueError("initialization_noise must be finite and non-negative")
    full_batch = batch_size is None or batch_size >= points.shape[0]
    if optimizer in ("riemannian", "als") and not full_batch:
        raise ValueError(f"{optimizer} requires a deterministic full-batch loss")

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if initialization == "coarse":
        initial_root = _coarse_to_fine_initial_root(
            points,
            modes_array,
            rank=rank,
            coarse_bins=initialization_coarse_bins,
            pseudocount=initialization_pseudocount,
        )
        params = [
            torch.nn.Parameter(
                torch.as_tensor(core, dtype=torch_dtype, device=device)
            )
            for core in initial_root.cores
        ]
    else:
        ranks = [1] + [int(rank)] * (d - 1) + [1]
        params = []
        for k, n in enumerate(modes_array):
            noise = initialization_noise * torch.randn(
                (ranks[k], int(n), ranks[k + 1]), generator=generator,
                dtype=torch_dtype,
            )
            # A constant rank-one path gives h=1 at initialisation; the small
            # dense perturbation gives all rank directions a gradient.
            noise[0, :, 0] += 1.0
            params.append(torch.nn.Parameter(noise.to(device)))

    indices_np = _cell_indices(points, modes_array)
    full_indices_np, full_weights_np = _compress_empirical_cells(indices_np)
    full_indices = torch.as_tensor(full_indices_np, dtype=torch.long, device=device)
    full_weights = torch.as_tensor(
        full_weights_np, dtype=torch_dtype, device=device
    )
    if full_batch:
        fit_indices_np, fit_weights_np = full_indices_np, full_weights_np
    else:
        fit_indices_np = indices_np
        fit_weights_np = np.full(indices_np.shape[0], 1.0 / indices_np.shape[0])
    indices = torch.as_tensor(fit_indices_np, dtype=torch.long, device=device)
    weights = torch.as_tensor(fit_weights_np, dtype=torch_dtype, device=device)
    rng = np.random.default_rng(seed + 1)
    started = time.perf_counter()

    if optimizer == "adam":
        history = FitHistory(optimizer="adam")
        adam = torch.optim.Adam(params, lr=learning_rate)
        window = 25
        for epoch in range(epochs):
            if full_batch:
                batch_indices, batch_weights = indices, weights
            else:
                chosen = rng.choice(points.shape[0], size=batch_size, replace=False)
                batch_np, batch_weight_np = _compress_empirical_cells(
                    indices_np[chosen]
                )
                batch_indices = torch.as_tensor(
                    batch_np, dtype=torch.long, device=device
                )
                batch_weights = torch.as_tensor(
                    batch_weight_np, dtype=torch_dtype, device=device
                )
            adam.zero_grad(set_to_none=True)
            loss, h2, z = _torch_density_objective(
                params, batch_indices, batch_weights, gamma
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            adam.step()
            history.function_calls += 1

            loss_value = float(loss.detach().cpu())
            h2_value = float(h2.detach().cpu())
            z_value = float(z.detach().cpu())
            history.loss.append(loss_value)
            history.l2_norm_sq.append(h2_value)
            history.normalization.append(z_value)
            history.chi2_to_reference.append(max(0.0, h2_value - 1.0))
            history.gradient_norm.append(float(grad_norm))
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
        fitted_cores = [parameter.detach() for parameter in params]

    elif optimizer == "riemannian":
        from ..algs.autodiff import rgd

        x0 = vector.from_list([parameter.detach().clone() for parameter in params])

        def objective_riemannian(cores):
            return _torch_density_objective(cores, indices, weights, gamma)[0]

        fitted_root, rgd_history = rgd(
            objective_riemannian,
            x0,
            maxit=epochs,
            tol=tolerance,
            step0=learning_rate,
            method=riemannian_retraction,
            verbose=verbose,
        )
        history = FitHistory(
            optimizer="riemannian",
            loss=[float(item["f"]) for item in rgd_history.iterations],
            gradient_norm=[
                float(item["gnorm"]) for item in rgd_history.iterations
            ],
            epochs=rgd_history.grad_calls,
            converged=rgd_history.converged,
            function_calls=rgd_history.fun_calls,
        )
        with torch.no_grad():
            final_loss, final_h2, final_z = _torch_density_objective(
                list(fitted_root.cores), indices, weights, gamma
            )
        if not history.loss or history.loss[-1] != float(final_loss):
            history.loss.append(float(final_loss))
        history.l2_norm_sq.append(float(final_h2))
        history.normalization.append(float(final_z))
        history.chi2_to_reference.append(max(0.0, float(final_h2) - 1.0))
        fitted_cores = list(fitted_root.cores)

    elif optimizer == "riemannian-sgd":
        initial_root = vector.from_list(
            [parameter.detach().clone() for parameter in params]
        )
        fitted_root, history = _fit_tt_riemannian_stochastic(
            initial_root,
            all_indices_np=indices_np,
            full_indices=full_indices,
            full_weights=full_weights,
            gamma=gamma,
            iterations=epochs,
            batch_size=batch_size or min(512, points.shape[0]),
            learning_rate=learning_rate,
            momentum_decay=riemannian_momentum,
            second_moment_decay=riemannian_second_moment,
            retraction_method=riemannian_retraction,
            seed=seed,
            verbose=verbose,
        )
        fitted_cores = list(fitted_root.cores)

    else:
        fitted_cores, history = _fit_tt_als(
            [parameter.detach() for parameter in params],
            modes=modes_array,
            indices=indices,
            weights=weights,
            gamma=gamma,
            sweeps=epochs,
            inner_steps=als_inner_steps,
            learning_rate=learning_rate,
            tolerance=tolerance,
            verbose=verbose,
        )

    history.wall_time = time.perf_counter() - started
    fitted = vector.from_list([
        core.detach().cpu().numpy().copy() for core in fitted_cores
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

    def log_density(self, points) -> np.ndarray:
        """Evaluate the density induced by the complete transport composition."""
        value = _validate_points(points, self.dimension).copy()
        result = np.zeros(value.shape[0], dtype=np.float64)
        for layer in self.layers:
            result += layer.log_density(value)
            value = layer.rosenblatt(value)
        return result

    def density(self, points) -> np.ndarray:
        return np.exp(self.log_density(points))

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
