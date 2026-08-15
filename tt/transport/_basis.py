"""Numpy TT contractions and one-dimensional basis utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from tt.core.vector import vector


def _as_numpy_cores(tensor: vector) -> list[np.ndarray]:
    from tt import backend as bk

    return [
        np.asarray(bk.to_numpy(core), dtype=np.float64)
        for core in tensor.cores
    ]


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


def _sample_tt_numpy(cores: Sequence[np.ndarray], indices: np.ndarray) -> np.ndarray:
    """Evaluate an arbitrary scalar TT at a batch of multi-indices."""
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


def _mean_numpy(cores: Sequence[np.ndarray]) -> float:
    """Exact product-uniform mean of a scalar TT."""
    env = np.ones((1,), dtype=np.float64)
    for core in cores:
        env = env @ core.mean(axis=1)
    return float(env[0])


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


def _linear_right_gram_environments(
    cores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Right suffix integrals for nodal linear hat-function TT cores."""
    d = len(cores)
    right: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    # An identity terminal metric also supports a vector-valued/purified TT
    # root, whose density is the sum of squares over its final channel.  The
    # ordinary scalar-root case is the 1x1 specialization.
    right[d] = np.eye(cores[-1].shape[2], dtype=np.float64)
    for k in range(d - 1, -1, -1):
        core = cores[k]
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = np.einsum(
            "aic,cd->aid", core, right[k + 1], optimize=True
        )
        diagonal = np.full(n, 2.0 * h / 3.0)
        diagonal[[0, -1]] = h / 3.0
        weighted = projected * diagonal[None, :, None]
        value = np.einsum(
            "aid,bid->ab", weighted, core, optimize=True
        )
        value += (h / 6.0) * (
            np.einsum(
                "aid,bid->ab", projected[:, :-1], core[:, 1:],
                optimize=True,
            )
            + np.einsum(
                "aid,bid->ab", projected[:, 1:], core[:, :-1],
                optimize=True,
            )
        )
        right[k] = value
    return right


def _linear_left_gram_environments(
    cores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Left prefix integrals for nodal linear hat-function TT cores."""
    d = len(cores)
    left: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    left[0] = np.ones((1, 1), dtype=np.float64)
    for k, core in enumerate(cores):
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = np.einsum(
            "ab,aic->bic", left[k], core, optimize=True
        )
        diagonal = np.full(n, 2.0 * h / 3.0)
        diagonal[[0, -1]] = h / 3.0
        value = np.einsum(
            "bic,bid->cd",
            projected * diagonal[None, :, None],
            core,
            optimize=True,
        )
        value += (h / 6.0) * (
            np.einsum(
                "bic,bid->cd", projected[:, :-1], core[:, 1:],
                optimize=True,
            )
            + np.einsum(
                "bic,bid->cd", projected[:, 1:], core[:, :-1],
                optimize=True,
            )
        )
        left[k + 1] = value
    return left


def _linear_hat_mass_matrix(modes: int) -> np.ndarray:
    """Exact uniform-measure Gram matrix of nodal linear hat functions."""
    modes = int(modes)
    if modes < 2:
        raise ValueError("linear modes must be at least two")
    h = 1.0 / (modes - 1)
    result = np.diag(np.full(modes, 2.0 * h / 3.0))
    result[[0, -1], [0, -1]] = h / 3.0
    adjacent = np.full(modes - 1, h / 6.0)
    result[np.arange(modes - 1), np.arange(1, modes)] = adjacent
    result[np.arange(1, modes), np.arange(modes - 1)] = adjacent
    return result


def _nested_linear_prolongation(old_modes: int, new_modes: int) -> np.ndarray:
    """Nodal evaluation map from a nested coarse grid to a fine grid."""
    old_modes, new_modes = int(old_modes), int(new_modes)
    if (
        old_modes < 2 or new_modes < old_modes
        or (new_modes - 1) % (old_modes - 1) != 0
    ):
        raise ValueError("new linear grid must exactly contain the old grid")
    factor = (new_modes - 1) // (old_modes - 1)
    position = np.arange(new_modes, dtype=np.float64) / factor
    lower = np.minimum(
        np.floor(position).astype(np.int64), old_modes - 2
    )
    fraction = position - lower
    result = np.zeros((new_modes, old_modes), dtype=np.float64)
    rows = np.arange(new_modes)
    result[rows, lower] = 1.0 - fraction
    result[rows, lower + 1] += fraction
    return result


def _linear_terminal_gram(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Exact integral of the outer product of terminal TT amplitudes."""
    left = np.ones((1, 1), dtype=np.float64)
    for core in cores:
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = np.einsum(
            "ab,aic->bic", left, core, optimize=True
        )
        diagonal = np.full(n, 2.0 * h / 3.0)
        diagonal[[0, -1]] = h / 3.0
        value = np.einsum(
            "bic,bid->cd",
            projected * diagonal[None, :, None],
            core,
            optimize=True,
        )
        value += (h / 6.0) * (
            np.einsum(
                "bic,bid->cd", projected[:, :-1], core[:, 1:],
                optimize=True,
            )
            + np.einsum(
                "bic,bid->cd", projected[:, 1:], core[:, :-1],
                optimize=True,
            )
        )
        left = value
    return 0.5 * (left + left.T)


def _local_purified_right_environments(
    cores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Exact integrated suffix Gramians for locally purified linear cores."""
    d = len(cores)
    right: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    right[d] = np.ones((1, 1), dtype=np.float64)
    for k in range(d - 1, -1, -1):
        core = cores[k]
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = np.einsum(
            "aikr,rs->aiks", core, right[k + 1], optimize=True
        )
        diagonal = np.full(n, 2.0 * h / 3.0)
        diagonal[[0, -1]] = h / 3.0
        value = np.einsum(
            "aiks,biks->ab",
            projected * diagonal[None, :, None, None],
            core,
            optimize=True,
        )
        value += (h / 6.0) * (
            np.einsum(
                "aiks,biks->ab", projected[:, :-1], core[:, 1:],
                optimize=True,
            )
            + np.einsum(
                "aiks,biks->ab", projected[:, 1:], core[:, :-1],
                optimize=True,
            )
        )
        right[k] = 0.5 * (value + value.T)
    return right


def _sample_local_purified_numpy(
    cores: Sequence[np.ndarray], points: np.ndarray
) -> np.ndarray:
    """Evaluate ``sum_a psi(points,a)**2`` without enumerating local paths."""
    lower, fraction = _linear_indices_fractions(
        points, np.asarray([core.shape[1] for core in cores])
    )
    left = np.ones((len(points), 1, 1), dtype=np.float64)
    for k, core in enumerate(cores):
        first = np.moveaxis(core[:, lower[:, k], :, :], 1, 0)
        second = np.moveaxis(core[:, lower[:, k] + 1, :, :], 1, 0)
        selected = (
            (1.0 - fraction[:, k])[:, None, None, None] * first
            + fraction[:, k, None, None, None] * second
        )
        contracted = np.einsum(
            "pab,pakr->pbkr", left, selected, optimize=True
        )
        left = np.einsum(
            "pbkr,pbks->prs", contracted, selected, optimize=True
        )
    return np.maximum(left[:, 0, 0], 0.0)


def _linear_hat_integral_weights(mode: int) -> np.ndarray:
    """Exact integrals of uniform nodal hat functions on ``[0, 1]``."""
    if int(mode) < 2:
        raise ValueError("a linear hat basis needs at least two nodes")
    weights = np.full(int(mode), 1.0 / (int(mode) - 1), dtype=np.float64)
    weights[[0, -1]] *= 0.5
    return weights


def _linear_right_mean_environments(
    cores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Exact suffix integrals of a nodal multilinear TT."""
    d = len(cores)
    right: list[np.ndarray] = [np.empty(0) for _ in range(d + 1)]
    right[d] = np.ones(1, dtype=np.float64)
    for k in range(d - 1, -1, -1):
        integrated = np.einsum(
            "i,aib->ab",
            _linear_hat_integral_weights(cores[k].shape[1]),
            cores[k],
            optimize=True,
        )
        right[k] = integrated @ right[k + 1]
    return right


def _linear_indices_fractions(
    points: np.ndarray, modes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    modes = np.asarray(modes, dtype=np.int64)
    position = points * (modes - 1)[None, :]
    lower = np.minimum(
        np.floor(position).astype(np.int64), modes[None, :] - 2
    )
    fraction = np.clip(position - lower, 0.0, 1.0)
    return lower, fraction


def _validate_linear_knots(
    knots: Sequence[np.ndarray], modes: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Validate strictly increasing nodal coordinates on ``[0,1]``."""
    result = [np.asarray(value, dtype=np.float64) for value in knots]
    if not result:
        raise ValueError("knots must be a non-empty sequence")
    if modes is not None and len(result) != len(modes):
        raise ValueError("knots must provide one vector per TT core")
    for k, value in enumerate(result):
        expected = None if modes is None else int(modes[k])
        if value.ndim != 1 or len(value) < 2 or (
            expected is not None and len(value) != expected
        ):
            raise ValueError("every knot vector must match its TT mode")
        if (
            not np.all(np.isfinite(value))
            or value[0] != 0.0 or value[-1] != 1.0
            or np.any(np.diff(value) <= 0.0)
        ):
            raise ValueError(
                "knot vectors must increase strictly from zero to one"
            )
    return [value.copy() for value in result]


def _adaptive_linear_indices_fractions(
    points: np.ndarray, knots: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Nonuniform hat interval indices and local coordinates."""
    lower = np.empty(points.shape, dtype=np.int64)
    fraction = np.empty(points.shape, dtype=np.float64)
    for k, nodes in enumerate(knots):
        index = np.searchsorted(nodes, points[:, k], side="right") - 1
        index = np.clip(index, 0, len(nodes) - 2)
        lower[:, k] = index
        fraction[:, k] = np.clip(
            (points[:, k] - nodes[index]) / (nodes[index + 1] - nodes[index]),
            0.0, 1.0,
        )
    return lower, fraction


def _adaptive_linear_right_gram_environments(
    cores: Sequence[np.ndarray], knots: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Exact suffix Gramians for nonuniform linear hat TT cores."""
    d = len(cores)
    right: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    right[d] = np.eye(cores[-1].shape[2], dtype=np.float64)
    for k in range(d - 1, -1, -1):
        core = cores[k]
        widths = np.diff(knots[k])
        diagonal = np.zeros(core.shape[1], dtype=np.float64)
        diagonal[:-1] += widths / 3.0
        diagonal[1:] += widths / 3.0
        projected = np.einsum(
            "aic,cd->aid", core, right[k + 1], optimize=True
        )
        value = np.einsum(
            "aid,bid->ab",
            projected * diagonal[None, :, None],
            core,
            optimize=True,
        )
        off_diagonal = widths / 6.0
        value += np.einsum(
            "aid,bid->ab",
            projected[:, :-1] * off_diagonal[None, :, None],
            core[:, 1:],
            optimize=True,
        )
        value += np.einsum(
            "aid,bid->ab",
            projected[:, 1:] * off_diagonal[None, :, None],
            core[:, :-1],
            optimize=True,
        )
        right[k] = value
    return right


def _sample_adaptive_linear_tt_numpy(
    cores: Sequence[np.ndarray], knots: Sequence[np.ndarray],
    points: np.ndarray,
) -> np.ndarray:
    lower, fraction = _adaptive_linear_indices_fractions(points, knots)
    left = np.ones((len(points), 1), dtype=np.float64)
    for k, core in enumerate(cores):
        first = np.moveaxis(core[:, lower[:, k], :], 1, 0)
        second = np.moveaxis(core[:, lower[:, k] + 1, :], 1, 0)
        selected = (
            (1.0 - fraction[:, k])[:, None, None] * first
            + fraction[:, k, None, None] * second
        )
        left = np.einsum("pa,pab->pb", left, selected, optimize=True)
    return left[:, 0]


def _sample_linear_tt_numpy(
    cores: Sequence[np.ndarray], points: np.ndarray
) -> np.ndarray:
    return _sample_linear_tt_vector_numpy(cores, points)[:, 0]


def _sample_linear_tt_vector_numpy(
    cores: Sequence[np.ndarray], points: np.ndarray
) -> np.ndarray:
    """Evaluate every terminal channel of a multilinear TT root."""
    modes = np.asarray([core.shape[1] for core in cores], dtype=np.int64)
    lower, fraction = _linear_indices_fractions(points, modes)
    left = np.ones((len(points), 1), dtype=np.float64)
    rows = np.arange(len(points))
    for k, core in enumerate(cores):
        first = np.moveaxis(core[:, lower[:, k], :], 1, 0)
        second = np.moveaxis(core[:, lower[:, k] + 1, :], 1, 0)
        selected = (
            (1.0 - fraction[:, k])[:, None, None] * first
            + fraction[:, k, None, None] * second
        )
        left = np.einsum("pa,pab->pb", left, selected, optimize=True)
    return left


def _quadratic_indices_fractions(
    points: np.ndarray, modes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform cardinal-quadratic B-spline interval and local coordinate."""
    modes = np.asarray(modes, dtype=np.int64)
    intervals = modes - 2
    position = points * intervals[None, :]
    lower = np.minimum(
        np.floor(position).astype(np.int64), intervals[None, :] - 1
    )
    fraction = np.clip(position - lower, 0.0, 1.0)
    return lower, fraction


def _quadratic_basis_weights(fraction: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        0.5 * (1.0 - fraction) ** 2,
        0.5 + fraction - fraction ** 2,
        0.5 * fraction ** 2,
    )


def _sample_quadratic_tt_numpy(
    cores: Sequence[np.ndarray], points: np.ndarray
) -> np.ndarray:
    modes = np.asarray([core.shape[1] for core in cores], dtype=np.int64)
    lower, fraction = _quadratic_indices_fractions(points, modes)
    left = np.ones((len(points), 1), dtype=np.float64)
    for k, core in enumerate(cores):
        weights = _quadratic_basis_weights(fraction[:, k])
        selected = sum(
            weight[:, None, None]
            * np.moveaxis(core[:, lower[:, k] + offset, :], 1, 0)
            for offset, weight in enumerate(weights)
        )
        left = np.einsum("pa,pab->pb", left, selected, optimize=True)
    return left[:, 0]


@lru_cache(maxsize=None)
def _quadratic_gram_bands(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Gram diagonals for uniform cardinal quadratic B-splines."""
    if n < 3:
        raise ValueError("quadratic B-spline modes need at least three functions")
    # Three-point Gauss-Legendre is exact for products of quadratic splines.
    nodes, quadrature = np.polynomial.legendre.leggauss(3)
    fraction = 0.5 * (nodes + 1.0)
    quadrature = 0.5 * quadrature / (n - 2)
    basis = np.stack(_quadratic_basis_weights(fraction), axis=1)
    weighted_basis = quadrature[:, None] * basis
    local = np.einsum(
        "qi,qj->ij", weighted_basis, basis, optimize=True
    )
    diagonal = np.zeros(n, dtype=np.float64)
    first = np.zeros(n - 1, dtype=np.float64)
    second = np.zeros(n - 2, dtype=np.float64)
    for interval in range(n - 2):
        diagonal[interval:interval + 3] += np.diag(local)
        first[interval:interval + 2] += np.diag(local, 1)
        second[interval] += local[0, 2]
    return diagonal, first, second


def _quadratic_right_gram_environments(
    cores: Sequence[np.ndarray],
) -> list[np.ndarray]:
    d = len(cores)
    right: list[np.ndarray] = [np.empty((0, 0)) for _ in range(d + 1)]
    right[d] = np.ones((1, 1), dtype=np.float64)
    for k in range(d - 1, -1, -1):
        core = cores[k]
        projected = np.einsum(
            "aic,cd->aid", core, right[k + 1], optimize=True
        )
        diagonal, first, second = _quadratic_gram_bands(core.shape[1])
        weighted = projected * diagonal[None, :, None]
        value = np.einsum(
            "aid,bid->ab", weighted, core, optimize=True
        )
        for offset, band in ((1, first), (2, second)):
            weighted_left = projected[:, :-offset] * band[None, :, None]
            value += np.einsum(
                "aid,bid->ab", weighted_left, core[:, offset:],
                optimize=True,
            )
            weighted_right = projected[:, offset:] * band[None, :, None]
            value += np.einsum(
                "aid,bid->ab", weighted_right, core[:, :-offset],
                optimize=True,
            )
        right[k] = value
    return right

__all__ = [
    "_as_numpy_cores",
    "_validate_points",
    "_cell_indices",
    "_sample_tt_numpy",
    "_second_moment_numpy",
    "_mean_numpy",
    "_right_gram_environments",
    "_linear_right_gram_environments",
    "_linear_left_gram_environments",
    "_linear_hat_mass_matrix",
    "_nested_linear_prolongation",
    "_linear_terminal_gram",
    "_local_purified_right_environments",
    "_sample_local_purified_numpy",
    "_linear_hat_integral_weights",
    "_linear_right_mean_environments",
    "_linear_indices_fractions",
    "_validate_linear_knots",
    "_adaptive_linear_indices_fractions",
    "_adaptive_linear_right_gram_environments",
    "_sample_adaptive_linear_tt_numpy",
    "_sample_linear_tt_numpy",
    "_sample_linear_tt_vector_numpy",
    "_quadratic_indices_fractions",
    "_quadratic_basis_weights",
    "_sample_quadratic_tt_numpy",
    "_quadratic_gram_bands",
    "_quadratic_right_gram_environments",
]
