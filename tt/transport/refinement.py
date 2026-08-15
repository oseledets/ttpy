"""Mode scoring and end-to-end refinement of Sample-DIRT chains."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _linear_hat_mass_matrix,
    _linear_left_gram_environments,
    _linear_right_gram_environments,
    _nested_linear_prolongation,
    _validate_points,
)
from ._fit_coordinates import _alternating_conditional_twist_stages
from ._torch_als import (
    _torch_linear_right_orthogonalize,
    _torch_right_orthogonalize,
)
from ._torch_density import (
    _torch_block_orthogonal_rotations,
    _torch_probit_orthogonal_map,
    _torch_sample_linear_tt,
    _torch_sample_local_purified,
)
from ._types import FitHistory, ModeRefinementScore
from .coordinates import ProbitOrthogonalTTDensity
from .positive import LocallyPurifiedLinearTTDensity
from .scalar import LinearSquaredTTDensity
from .transport import SampleDIRT, refine_linear_sample_dirt_modes
from .wrappers import PermutedTTDensity

def _torch_linear_right_environments(cores):
    """Differentiable right Gram environments for linear hat TT cores."""
    import torch

    right = [None] * (len(cores) + 1)
    right[-1] = torch.eye(
        cores[-1].shape[2], dtype=cores[0].dtype, device=cores[0].device
    )
    for coordinate in range(len(cores) - 1, -1, -1):
        core = cores[coordinate]
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = torch.einsum(
            "aic,cd->aid", core, right[coordinate + 1]
        )
        diagonal = torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=core.dtype, device=core.device,
        )
        diagonal[0] = diagonal[-1] = h / 3.0
        value = torch.einsum(
            "aid,bid->ab",
            projected * diagonal[None, :, None],
            core,
        )
        value = value + (h / 6.0) * (
            torch.einsum(
                "aid,bid->ab", projected[:, :-1], core[:, 1:]
            )
            + torch.einsum(
                "aid,bid->ab", projected[:, 1:], core[:, :-1]
            )
        )
        right[coordinate] = value
    return right


def _torch_linear_root_sobolev_ratio(cores):
    """Exact scale-free ``H1`` seminorm of a multilinear TT root.

    For the nodal hat interpolant ``f`` this returns

    ``sum_k integral |partial_k f|**2 / (d * integral f**2)``.

    The mass and one-dimensional stiffness matrices are contracted directly
    with the TT cores.  No quadrature samples or squared-rank Hadamard TT are
    introduced, and the expression remains differentiable under autograd.
    """
    import torch

    if not cores:
        raise ValueError("at least one linear TT core is required")
    right = _torch_linear_right_environments(cores)
    mass = torch.clamp_min(
        right[0].reshape(()), torch.finfo(cores[0].dtype).tiny
    )
    left = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    energy = torch.zeros((), dtype=cores[0].dtype, device=cores[0].device)
    for coordinate, core in enumerate(cores):
        n = int(core.shape[1])
        h = 1.0 / (n - 1)
        differences = core[:, 1:, :] - core[:, :-1, :]
        projected_differences = torch.einsum(
            "ab,aic->bic", left, differences
        )
        weighted_differences = torch.einsum(
            "bic,cd->bid", projected_differences,
            right[coordinate + 1],
        )
        energy = energy + torch.einsum(
            "bid,bid->", weighted_differences, differences
        ) / h

        projected = torch.einsum("ab,aic->bic", left, core)
        diagonal = torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=core.dtype, device=core.device,
        )
        diagonal[0] = diagonal[-1] = h / 3.0
        left = torch.einsum(
            "bic,bid->cd", projected * diagonal[None, :, None], core
        )
        left = left + (h / 6.0) * (
            torch.einsum(
                "bic,bid->cd", projected[:, :-1], core[:, 1:]
            )
            + torch.einsum(
                "bic,bid->cd", projected[:, 1:], core[:, :-1]
            )
        )
    return energy / (len(cores) * mass)


def _torch_local_purified_right_environments(cores):
    """Differentiable exact suffix Gramians for locally purified cores."""
    import torch

    right = [None] * (len(cores) + 1)
    right[-1] = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for coordinate in range(len(cores) - 1, -1, -1):
        core = cores[coordinate]
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = torch.einsum(
            "aikr,rs->aiks", core, right[coordinate + 1]
        )
        diagonal = torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=core.dtype, device=core.device,
        )
        diagonal[0] = diagonal[-1] = h / 3.0
        value = torch.einsum(
            "aiks,biks->ab",
            projected * diagonal[None, :, None, None],
            core,
        )
        value = value + (h / 6.0) * (
            torch.einsum(
                "aiks,biks->ab", projected[:, :-1], core[:, 1:]
            )
            + torch.einsum(
                "aiks,biks->ab", projected[:, 1:], core[:, :-1]
            )
        )
        right[coordinate] = value
    return right


def _torch_linear_log_density_rosenblatt(
    cores, points, gamma: float, *, transform: bool, right=None,
):
    """Differentiable exact log density and optional Rosenblatt map."""
    import torch

    if right is None:
        right = _torch_linear_right_environments(cores)
    root_values = _torch_sample_linear_tt(cores, points)
    normalization = gamma + right[0].reshape(())
    log_density = (
        torch.log(gamma + root_values.square())
        - torch.log(normalization)
    )
    if not transform:
        return log_density, None

    result = torch.empty_like(points)
    left = torch.ones(
        (len(points), 1), dtype=points.dtype, device=points.device
    )
    rows = torch.arange(len(points), device=points.device)
    for coordinate, core in enumerate(cores):
        nodes = torch.einsum("pa,aib->pib", left, core)
        first = nodes[:, :-1]
        delta = nodes[:, 1:] - first
        projected_first = torch.einsum(
            "pia,ab->pib", first, right[coordinate + 1]
        )
        projected_delta = torch.einsum(
            "pia,ab->pib", delta, right[coordinate + 1]
        )
        a = torch.einsum("pib,pib->pi", projected_first, first)
        b = 2.0 * torch.einsum(
            "pib,pib->pi", projected_delta, first
        )
        c = torch.einsum("pib,pib->pi", projected_delta, delta)
        h = 1.0 / (core.shape[1] - 1)
        masses = h * (gamma + a + 0.5 * b + c / 3.0)
        masses = torch.clamp_min(masses, torch.finfo(points.dtype).tiny)
        cumulative = torch.cumsum(masses, dim=1)
        position = points[:, coordinate] * (core.shape[1] - 1)
        interval = torch.clamp(
            torch.floor(position).to(torch.long),
            min=0,
            max=core.shape[1] - 2,
        )
        fraction = torch.clamp(position - interval, 0.0, 1.0)
        before = torch.where(
            interval == 0,
            torch.zeros((), dtype=points.dtype, device=points.device),
            cumulative[rows, torch.clamp_min(interval - 1, 0)],
        )
        aa = a[rows, interval]
        bb = b[rows, interval]
        cc = c[rows, interval]
        local = h * (
            (gamma + aa) * fraction
            + 0.5 * bb * fraction.square()
            + (cc / 3.0) * fraction.pow(3)
        )
        result[:, coordinate] = (before + local) / cumulative[:, -1]
        node_first = core[:, interval, :].permute(1, 0, 2)
        node_second = core[:, interval + 1, :].permute(1, 0, 2)
        selected = (
            (1.0 - fraction)[:, None, None] * node_first
            + fraction[:, None, None] * node_second
        )
        left = torch.einsum("pa,pab->pb", left, selected)
    return log_density, torch.clamp(result, 0.0, 1.0)


def _torch_local_purified_log_density_rosenblatt(
    cores, points, gamma: float, *, transform: bool, right=None,
):
    """Differentiable exact local-purification density and Rosenblatt map."""
    import torch

    if right is None:
        right = _torch_local_purified_right_environments(cores)
    signal = _torch_sample_local_purified(cores, points)
    normalization = gamma + right[0].reshape(())
    log_density = torch.log(gamma + signal) - torch.log(normalization)
    if not transform:
        return log_density, None

    result = torch.empty_like(points)
    left = torch.ones(
        (len(points), 1, 1), dtype=points.dtype, device=points.device
    )
    rows = torch.arange(len(points), device=points.device)
    for coordinate, core in enumerate(cores):
        first = core[:, :-1]
        delta = core[:, 1:] - first
        projected_first = torch.einsum(
            "aikr,rs->aiks", first, right[coordinate + 1]
        )
        projected_delta = torch.einsum(
            "aikr,rs->aiks", delta, right[coordinate + 1]
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
        b = 2.0 * torch.einsum("pab,abi->pi", left, cross_gram)
        c = torch.einsum("pab,abi->pi", left, delta_gram)
        h = 1.0 / (core.shape[1] - 1)
        masses = torch.clamp_min(
            h * (gamma + a + 0.5 * b + c / 3.0),
            torch.finfo(points.dtype).tiny,
        )
        cumulative = torch.cumsum(masses, dim=1)
        position = points[:, coordinate] * (core.shape[1] - 1)
        interval = torch.clamp(
            torch.floor(position).to(torch.long),
            min=0, max=core.shape[1] - 2,
        )
        fraction = torch.clamp(position - interval, 0.0, 1.0)
        before = torch.where(
            interval == 0,
            torch.zeros((), dtype=points.dtype, device=points.device),
            cumulative[rows, torch.clamp_min(interval - 1, 0)],
        )
        aa, bb, cc = a[rows, interval], b[rows, interval], c[rows, interval]
        local = h * (
            (gamma + aa) * fraction
            + 0.5 * bb * fraction.square()
            + (cc / 3.0) * fraction.pow(3)
        )
        result[:, coordinate] = (before + local) / cumulative[:, -1]
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
    return log_density, torch.clamp(result, 0.0, 1.0)


def score_linear_sample_dirt_mode_refinement(
    model: SampleDIRT,
    samples,
    modes: int | Sequence[int],
    *,
    layer_index: int,
    coordinate_indices: Sequence[int] | None = None,
    replicates: int = 4,
    batch_size: int = 4096,
    confidence_z: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
) -> list[ModeRefinementScore]:
    """Score genuinely new nested-grid directions without fitting them.

    The selected layer is embedded exactly in the requested finer nodal
    basis, then right-orthogonalized by a gauge-only sweep.  For every refined
    core, the stochastic global-chain NLL gradient is projected off the
    prolongated coarse finite-element space in the exact product metric

    ``L_k (x) H_k (x) R_k``.

    Here ``H_k`` is the exact hat-basis mass matrix and ``L_k, R_k`` are exact
    TT root Gram environments.  A score uses two disjoint minibatches,

    ``<g_A,new, M^-1 g_B,new>``,

    rather than ``||g_batch,new||^2``.  The cross product is an unbiased
    estimator of the squared population natural gradient; ordinary stochastic
    gradient norms have a positive minibatch-variance bias.  Replicate means,
    standard errors and lower confidence bounds can therefore screen physical
    coordinates before an expensive all-core refinement.  All downstream
    Rosenblatt maps remain in the differentiated loss.  A fixed prefix before
    ``layer_index`` is applied to the samples once and removed exactly.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mode-refinement scoring requires the 'torch' extra"
        ) from exc
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    layer_index = int(layer_index)
    if layer_index < 0 or layer_index >= len(model.layers):
        raise ValueError("layer_index must identify a valid layer")
    if replicates < 1 or batch_size < 1:
        raise ValueError("replicates and batch_size must be positive")
    if confidence_z < 0.0 or not np.isfinite(confidence_z):
        raise ValueError("confidence_z must be finite and non-negative")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    training = _validate_points(samples, model.dimension, name="samples")
    if len(training) < 2:
        raise ValueError("at least two samples are required for cross fitting")

    def unwrap_scalar(layer):
        if isinstance(layer, PermutedTTDensity):
            permutation = layer.permutation.copy()
            base = layer.base
        else:
            permutation = np.arange(model.dimension, dtype=np.int64)
            base = layer
        wrapper = base if isinstance(base, ProbitOrthogonalTTDensity) else None
        if wrapper is not None:
            base = wrapper.base
        if type(base) is not LinearSquaredTTDensity:
            raise TypeError(
                "mode-refinement scoring supports scalar linear-squared TT "
                "layers, optionally wrapped by fixed probit maps and "
                "permutations"
            )
        return base, wrapper, permutation

    old_base, _, old_permutation = unwrap_scalar(model.layers[layer_index])
    old_internal_modes = np.asarray(
        [core.shape[1] for core in old_base._cores], dtype=np.int64
    )
    refined = refine_linear_sample_dirt_modes(
        model,
        modes,
        layer_indices=[layer_index],
        coordinate_indices=coordinate_indices,
    )
    new_base, _, _ = unwrap_scalar(refined.layers[layer_index])
    new_internal_modes = np.asarray(
        [core.shape[1] for core in new_base._cores], dtype=np.int64
    )
    refined_internal = np.flatnonzero(
        new_internal_modes > old_internal_modes
    )
    if len(refined_internal) == 0:
        raise ValueError("requested refinement adds no fine-grid directions")

    # A prefix contributes a sample-dependent but parameter-independent log
    # density.  Its exact inverse image is sufficient for every target-layer
    # gradient and avoids repeatedly differentiating through fixed layers.
    working = refined
    target_layer = layer_index
    if layer_index > 0:
        prefix = SampleDIRT(model.dimension, refined.layers[:layer_index])
        training = prefix.inverse(
            training,
            device=device,
            dtype=dtype,
            batch_size=max(int(batch_size), 8192),
        )
        working = SampleDIRT(model.dimension, refined.layers[layer_index:])
        target_layer = 0

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    specifications = []
    core_groups = []
    target_parameters = []
    for local_index, layer in enumerate(working.layers):
        base, wrapper, permutation = unwrap_scalar(layer)
        cores = [
            torch.as_tensor(core, dtype=torch_dtype, device=device).clone()
            for core in base._cores
        ]
        if local_index == target_layer:
            # This is an exact TT gauge transformation.  Besides stabilizing
            # the environment inverses, it makes the score reproducible under
            # arbitrary invertible gauges of the serialized input cores.
            _torch_linear_right_orthogonalize(cores)
            cores = [torch.nn.Parameter(core) for core in cores]
            target_parameters = cores
        core_groups.append(cores)
        specifications.append((
            base.gamma,
            permutation,
            np.argsort(permutation),
            None if wrapper is None else torch.as_tensor(
                wrapper.rotation, dtype=torch_dtype, device=device
            ),
            () if wrapper is None else wrapper.radial_twist_pairs,
            () if wrapper is None else wrapper.radial_twist_coefficients,
            () if wrapper is None else wrapper.conditional_twist_pairs,
            () if wrapper is None else wrapper.conditional_twist_conditioners,
            () if wrapper is None else wrapper.conditional_twist_coefficients,
        ))
    fixed_right = [
        None if index == target_layer
        else _torch_linear_right_environments(cores)
        for index, cores in enumerate(core_groups)
    ]

    def chain_log_density(points_tensor):
        value = points_tensor
        total = torch.zeros(
            len(value), dtype=torch_dtype, device=value.device
        )
        for index, (cores, specification) in enumerate(
            zip(core_groups, specifications)
        ):
            (
                gamma, permutation, inverse_permutation, rotation,
                radial_pairs, radial_coefficients,
                conditional_pairs, conditional_conditioners,
                conditional_coefficients,
            ) = specification
            permutation_tensor = torch.as_tensor(
                permutation, dtype=torch.long, device=value.device
            )
            internal = value[:, permutation_tensor]
            if rotation is not None:
                internal = _torch_probit_orthogonal_map(
                    internal,
                    rotation,
                    radial_twist_pairs=radial_pairs,
                    radial_twist_coefficients=radial_coefficients,
                    conditional_twist_pairs=conditional_pairs,
                    conditional_twist_conditioners=conditional_conditioners,
                    conditional_twist_coefficients=conditional_coefficients,
                )
            log_density, transformed = _torch_linear_log_density_rosenblatt(
                cores,
                internal,
                gamma,
                transform=(index + 1 < len(core_groups)),
                right=fixed_right[index],
            )
            total = total + log_density
            if transformed is not None:
                inverse_tensor = torch.as_tensor(
                    inverse_permutation, dtype=torch.long, device=value.device
                )
                value = transformed[:, inverse_tensor]
        return total

    def gradient(indices):
        batch = torch.as_tensor(
            training[indices], dtype=torch_dtype, device=device
        )
        loss = -chain_log_density(batch).mean()
        gradients = torch.autograd.grad(
            loss, target_parameters, create_graph=False, retain_graph=False
        )
        return [
            value.detach().cpu().numpy().astype(np.float64, copy=False)
            for value in gradients
        ]

    canonical_cores = [
        core.detach().cpu().numpy().astype(np.float64, copy=True)
        for core in target_parameters
    ]
    left = _linear_left_gram_environments(canonical_cores)
    right = _linear_right_gram_environments(canonical_cores)

    def symmetric_pseudoinverse(metric):
        metric = 0.5 * (metric + metric.T)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        scale = max(float(np.max(eigenvalues)), np.finfo(np.float64).tiny)
        inverse = np.where(
            eigenvalues > 1e-11 * scale, 1.0 / eigenvalues, 0.0
        )
        return (eigenvectors * inverse[None, :]) @ eigenvectors.T

    metrics = {}
    for internal in refined_internal:
        old_modes = int(old_internal_modes[internal])
        new_modes = int(new_internal_modes[internal])
        prolongation = _nested_linear_prolongation(old_modes, new_modes)
        fine_mass = _linear_hat_mass_matrix(new_modes)
        coarse_mass = prolongation.T @ fine_mass @ prolongation
        metrics[int(internal)] = (
            prolongation,
            symmetric_pseudoinverse(left[internal]),
            np.linalg.inv(fine_mass),
            np.linalg.inv(coarse_mass),
            symmetric_pseudoinverse(right[internal + 1]),
        )

    def metric_product(first, second, left_inverse, mode_inverse, right_inverse):
        solved = np.einsum(
            "ac,cib->aib", left_inverse, second, optimize=True
        )
        solved = np.einsum(
            "ij,ajb->aib", mode_inverse, solved, optimize=True
        )
        solved = np.einsum(
            "bd,aid->aib", right_inverse, solved, optimize=True
        )
        return float(np.sum(first * solved))

    rng = np.random.default_rng(seed)
    effective_batch = min(int(batch_size), len(training) // 2)
    cross_scores = {int(index): [] for index in refined_internal}
    for _ in range(int(replicates)):
        chosen = rng.choice(
            len(training), size=2 * effective_batch, replace=False
        )
        first_gradients = gradient(chosen[:effective_batch])
        second_gradients = gradient(chosen[effective_batch:])
        for internal in refined_internal:
            internal = int(internal)
            prolongation, left_inverse, fine_inverse, coarse_inverse, right_inverse = (
                metrics[internal]
            )
            first, second = (
                first_gradients[internal], second_gradients[internal]
            )
            full = metric_product(
                first, second, left_inverse, fine_inverse, right_inverse
            )
            first_coarse = np.einsum(
                "ni,anb->aib", prolongation, first, optimize=True
            )
            second_coarse = np.einsum(
                "ni,anb->aib", prolongation, second, optimize=True
            )
            coarse = metric_product(
                first_coarse,
                second_coarse,
                left_inverse,
                coarse_inverse,
                right_inverse,
            )
            cross_scores[internal].append(full - coarse)

    results = []
    for internal in refined_internal:
        internal = int(internal)
        values = np.asarray(cross_scores[internal], dtype=np.float64)
        mean = float(np.mean(values))
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        core = canonical_cores[internal]
        results.append(ModeRefinementScore(
            layer_index=layer_index,
            physical_coordinate=int(old_permutation[internal]),
            internal_coordinate=internal,
            old_modes=int(old_internal_modes[internal]),
            new_modes=int(new_internal_modes[internal]),
            added_parameters=int(
                (new_internal_modes[internal] - old_internal_modes[internal])
                * core.shape[0] * core.shape[2]
            ),
            cross_scores=values.tolist(),
            mean_score=mean,
            standard_error=standard_error,
            lower_confidence_bound=(
                mean - float(confidence_z) * standard_error
            ),
        ))
    return sorted(
        results, key=lambda result: result.lower_confidence_bound, reverse=True
    )


def fine_tune_linear_sample_dirt(
    model: SampleDIRT,
    samples,
    *,
    validation_samples,
    epochs: int = 500,
    learning_rate: float = 1e-4,
    batch_size: int = 4096,
    validation_interval: int = 10,
    patience: int = 25,
    min_delta: float = 1e-4,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
    canonicalize: bool = True,
    optimize_cores: bool = True,
    core_layer_indices: Sequence[int] | None = None,
    optimize_rotations: bool = False,
    optimize_conditional_twists: bool = False,
    conditional_twist_stages: int = 2,
    conditional_twist_basis_count: int = 4,
    conditional_twist_layer_indices: Sequence[int] | None = None,
    conditional_twist_learning_rate: float | None = None,
    orthogonalization_interval: int = 0,
    orthogonalization_metric: str = "hat-mass",
    sobolev_penalty: float = 0.0,
    curvature_plus_samples=None,
    curvature_minus_samples=None,
    curvature_targets=None,
    curvature_weight: float = 0.0,
) -> tuple[SampleDIRT, FitHistory]:
    """Jointly fine-tune every linear TT layer by the global chain NLL.

    Sequential fitting is functional boosting: every appended layer optimizes
    an exact increment of the physical KL objective while earlier transports
    stay fixed.  This routine performs the complementary end-to-end step.  It
    differentiates through every exact Rosenblatt CDF, so downstream losses
    update all preceding TT contractions as well as the last density factor.
    Probit-orthogonal coordinate wrappers are supported as well.  By default
    their rotations stay fixed while the TT cores are optimized through the
    entire rotated Rosenblatt chain.  Alternatively tangent skew blocks can
    update the rotations with the cores fixed, or both blocks can be moved.
    Tiny nonlinear conditional-rotation stages can also be appended to every
    probit layer and optimized end-to-end with the TT cores and old coordinate
    maps fixed or moving.  Validation checkpoint zero is the input model, so a
    rejected coordinate correction is exactly harmless.  Checkpointing always
    uses ordinary mean NLL.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fine_tune_linear_sample_dirt requires the 'torch' extra"
        ) from exc
    if not isinstance(model, SampleDIRT) or not model.layers:
        raise ValueError("model must contain at least one Sample-DIRT layer")
    if epochs < 1 or batch_size < 1 or validation_interval < 1 or patience < 1:
        raise ValueError("invalid joint optimization iteration parameters")
    if orthogonalization_interval < 0:
        raise ValueError("orthogonalization_interval must be non-negative")
    orthogonalization_metric = str(
        orthogonalization_metric
    ).lower().replace("_", "-")
    if orthogonalization_metric not in ("hat-mass", "coefficient-uniform"):
        raise ValueError(
            "orthogonalization_metric must be 'hat-mass' or "
            "'coefficient-uniform'"
        )
    if (
        learning_rate <= 0.0 or min_delta < 0.0
        or sobolev_penalty < 0.0 or curvature_weight < 0.0
    ):
        raise ValueError("learning_rate must be positive and min_delta non-negative")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("invalid joint tail objective parameters")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if (
        not optimize_cores and not optimize_rotations
        and not optimize_conditional_twists
    ):
        raise ValueError("at least one parameter block must be optimized")
    if core_layer_indices is None:
        core_layer_set = (
            set(range(len(model.layers))) if optimize_cores else set()
        )
    else:
        core_layer_set = {int(index) for index in core_layer_indices}
        if not optimize_cores:
            raise ValueError(
                "core_layer_indices requires optimize_cores=True"
            )
        if (
            not core_layer_set
            or len(core_layer_set) != len(core_layer_indices)
            or any(
                index < 0 or index >= len(model.layers)
                for index in core_layer_set
            )
        ):
            raise ValueError("core layer indices must be unique and valid")
    if optimize_conditional_twists and (
        conditional_twist_stages < 1 or conditional_twist_basis_count < 1
    ):
        raise ValueError("invalid conditional twist capacity")
    if (
        conditional_twist_learning_rate is not None
        and conditional_twist_learning_rate <= 0.0
    ):
        raise ValueError("conditional twist learning rate must be positive")
    if conditional_twist_layer_indices is None:
        conditional_layer_set = set(range(len(model.layers)))
    else:
        conditional_layer_set = {
            int(index) for index in conditional_twist_layer_indices
        }
        if (
            len(conditional_layer_set) != len(conditional_twist_layer_indices)
            or any(
                index < 0 or index >= len(model.layers)
                for index in conditional_layer_set
            )
        ):
            raise ValueError("conditional twist layer indices must be unique")

    training = _validate_points(samples, model.dimension, name="samples")
    validation = _validate_points(
        validation_samples, model.dimension, name="validation_samples"
    )
    curvature_enabled = curvature_weight > 0.0
    curvature_arguments = (
        curvature_plus_samples,
        curvature_minus_samples,
        curvature_targets,
    )
    if curvature_enabled and any(value is None for value in curvature_arguments):
        raise ValueError(
            "positive curvature_weight requires plus/minus samples and targets"
        )
    if not curvature_enabled and any(
        value is not None for value in curvature_arguments
    ):
        raise ValueError(
            "curvature samples/targets require positive curvature_weight"
        )
    if curvature_enabled:
        curvature_plus = _validate_points(
            curvature_plus_samples,
            model.dimension,
            name="curvature_plus_samples",
        )
        curvature_minus = _validate_points(
            curvature_minus_samples,
            model.dimension,
            name="curvature_minus_samples",
        )
        curvature_target = np.asarray(curvature_targets, dtype=np.float64)
        if (
            len(curvature_plus) != len(training)
            or len(curvature_minus) != len(training)
            or curvature_target.shape != (len(training),)
            or not np.all(np.isfinite(curvature_target))
        ):
            raise ValueError(
                "curvature probes and targets must match training rows"
            )
    else:
        curvature_plus = curvature_minus = curvature_target = None
    # In a pure core block update, every layer before the first trainable one
    # is an exact fixed map.  Pull the data through that prefix once.  Its
    # pointwise log density is independent of all optimized parameters, so
    # dropping it changes neither gradients nor mean-NLL checkpoint ordering.
    # Tail reweighting and Sobolev averaging are deliberately excluded because
    # their objectives are not invariant to removing a sample-dependent or
    # layer-count-dependent constant.
    first_trainable = min(core_layer_set) if core_layer_set else 0
    if (
        core_layer_indices is not None
        and first_trainable > 0
        and not optimize_rotations
        and not optimize_conditional_twists
        and tail_weight == 0.0
        and sobolev_penalty == 0.0
        and not curvature_enabled
    ):
        prefix_started = time.perf_counter()
        prefix = SampleDIRT(
            model.dimension, model.layers[:first_trainable]
        )
        transform_batch_size = max(int(batch_size), 8192)
        transformed_training = prefix.inverse(
            training,
            device=device,
            dtype=dtype,
            batch_size=transform_batch_size,
        )
        transformed_validation = prefix.inverse(
            validation,
            device=device,
            dtype=dtype,
            batch_size=transform_batch_size,
        )
        prefix_elapsed = time.perf_counter() - prefix_started
        suffix = SampleDIRT(
            model.dimension, model.layers[first_trainable:]
        )
        fitted_suffix, history = fine_tune_linear_sample_dirt(
            suffix,
            transformed_training,
            validation_samples=transformed_validation,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_interval=validation_interval,
            patience=patience,
            min_delta=min_delta,
            tail_fraction=tail_fraction,
            tail_weight=tail_weight,
            seed=seed,
            device=device,
            dtype=dtype,
            canonicalize=canonicalize,
            optimize_cores=optimize_cores,
            core_layer_indices=[
                index - first_trainable for index in core_layer_set
            ],
            optimize_rotations=optimize_rotations,
            optimize_conditional_twists=optimize_conditional_twists,
            conditional_twist_stages=conditional_twist_stages,
            conditional_twist_basis_count=conditional_twist_basis_count,
            conditional_twist_layer_indices=None,
            conditional_twist_learning_rate=conditional_twist_learning_rate,
            orthogonalization_interval=orthogonalization_interval,
            orthogonalization_metric=orthogonalization_metric,
            sobolev_penalty=sobolev_penalty,
        )
        history.wall_time += prefix_elapsed
        history.optimizer += f"[fixed-prefix={first_trainable}]"
        return SampleDIRT(
            model.dimension,
            [*model.layers[:first_trainable], *fitted_suffix.layers],
        ), history
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    specifications = []
    core_groups = []
    trainable_core_groups = []
    angle_groups = []
    conditional_groups = []
    parameters = []
    nonconditional_parameters = []
    for layer_index, layer in enumerate(model.layers):
        if isinstance(layer, PermutedTTDensity):
            base = layer.base
            permutation = layer.permutation.copy()
        else:
            base = layer
            permutation = np.arange(model.dimension, dtype=np.int64)
        rotation = None
        rotation_block_sizes = None
        radial_twist_pairs = ()
        radial_twist_coefficients = ()
        conditional_twist_pairs = ()
        conditional_twist_conditioners = ()
        conditional_twist_coefficients = ()
        new_conditional_pairs = ()
        new_conditional_conditioners = ()
        if isinstance(base, ProbitOrthogonalTTDensity):
            rotation = torch.as_tensor(
                base.rotation, dtype=torch_dtype, device=device
            )
            rotation_block_sizes = base.rotation_block_sizes.copy()
            radial_twist_pairs = tuple(
                pairs.copy() for pairs in base.radial_twist_pairs
            )
            radial_twist_coefficients = tuple(
                coefficients.copy()
                for coefficients in base.radial_twist_coefficients
            )
            conditional_twist_pairs = tuple(
                pairs.copy() for pairs in base.conditional_twist_pairs
            )
            conditional_twist_conditioners = tuple(
                conditioners.copy()
                for conditioners in base.conditional_twist_conditioners
            )
            conditional_twist_coefficients = tuple(
                coefficients.copy()
                for coefficients in base.conditional_twist_coefficients
            )
            if (
                optimize_conditional_twists
                and layer_index in conditional_layer_set
            ):
                (
                    new_conditional_pairs,
                    new_conditional_conditioners,
                ) = _alternating_conditional_twist_stages(
                    model.dimension, int(conditional_twist_stages)
                )
            base = base.base
        if type(base) not in (
            LinearSquaredTTDensity, LocallyPurifiedLinearTTDensity
        ):
            raise TypeError(
                "joint optimization supports scalar or locally purified "
                "linear layers, optionally with fixed probit rotations"
            )
        canonical = [
            torch.as_tensor(core, dtype=torch_dtype, device=device).clone()
            for core in base._cores
        ]
        kind = (
            "linear-squared"
            if type(base) is LinearSquaredTTDensity else "linear-local-purified"
        )
        trainable_core = layer_index in core_layer_set
        if trainable_core and canonicalize and kind == "linear-squared":
            # A gauge-only QR sweep leaves the represented root, density and
            # Rosenblatt map unchanged.  Starting Adam in this gauge avoids
            # wasting global-KL steps on arbitrarily scaled neighbouring
            # cores inherited from the independent layer fits.
            if orthogonalization_metric == "hat-mass":
                _torch_linear_right_orthogonalize(canonical)
            else:
                _torch_right_orthogonalize(
                    canonical, uniform_measure=True
                )
        group = [
            torch.nn.Parameter(core) if trainable_core else core
            for core in canonical
        ]
        core_groups.append(group)
        trainable_core_groups.append(trainable_core)
        if trainable_core:
            parameters.extend(group)
            nonconditional_parameters.extend(group)
        angles = []
        if optimize_rotations and rotation is not None:
            angles = [
                torch.nn.Parameter(torch.zeros(
                    (int(size) * (int(size) - 1) // 2,),
                    dtype=torch_dtype,
                    device=device,
                ))
                for size in rotation_block_sizes
            ]
            parameters.extend(angles)
            nonconditional_parameters.extend(angles)
        angle_groups.append(angles)
        conditional_coefficients = [
            torch.nn.Parameter(torch.zeros(
                (len(stage_pairs), int(conditional_twist_basis_count)),
                dtype=torch_dtype,
                device=device,
            ))
            for stage_pairs in new_conditional_pairs
        ]
        conditional_groups.append(conditional_coefficients)
        parameters.extend(conditional_coefficients)
        specifications.append((
            kind, base.gamma, permutation, np.argsort(permutation),
            rotation, rotation_block_sizes,
            radial_twist_pairs, radial_twist_coefficients,
            conditional_twist_pairs, conditional_twist_conditioners,
            conditional_twist_coefficients,
            new_conditional_pairs, new_conditional_conditioners,
        ))
    if optimize_rotations and not any(angle_groups):
        raise ValueError(
            "optimize_rotations requires at least one probit-orthogonal layer"
        )
    if optimize_conditional_twists and not any(conditional_groups):
        raise ValueError(
            "optimize_conditional_twists requires a probit-orthogonal layer"
        )
    if sobolev_penalty > 0.0 and any(
        specification[0] != "linear-squared"
        for specification in specifications
    ):
        raise ValueError(
            "Sobolev regularization currently requires scalar linear layers"
        )
    fixed_right_environments = []
    for cores, specification, trainable_core in zip(
        core_groups, specifications, trainable_core_groups
    ):
        if trainable_core:
            fixed_right_environments.append(None)
        elif specification[0] == "linear-squared":
            fixed_right_environments.append(
                _torch_linear_right_environments(cores)
            )
        else:
            fixed_right_environments.append(
                _torch_local_purified_right_environments(cores)
            )

    def chain_log_density(points_tensor):
        value = points_tensor
        total = torch.zeros(
            len(value), dtype=torch_dtype, device=value.device
        )
        for index, (cores, angles, new_coefficients, specification) in enumerate(
            zip(
                core_groups, angle_groups, conditional_groups,
                specifications,
            )
        ):
            (
                kind, gamma, permutation, inverse_permutation,
                rotation, _rotation_block_sizes,
                radial_twist_pairs, radial_twist_coefficients,
                conditional_twist_pairs, conditional_twist_conditioners,
                conditional_twist_coefficients,
                new_conditional_pairs, new_conditional_conditioners,
            ) = specification
            permutation_tensor = torch.as_tensor(
                permutation, dtype=torch.long, device=value.device
            )
            internal = value[:, permutation_tensor]
            if rotation is not None:
                effective_rotation = rotation
                if angles:
                    delta = _torch_block_orthogonal_rotations(
                        angles, _rotation_block_sizes
                    )
                    effective_rotation = rotation @ delta
                internal = _torch_probit_orthogonal_map(
                    internal, effective_rotation,
                    radial_twist_pairs=radial_twist_pairs,
                    radial_twist_coefficients=radial_twist_coefficients,
                    conditional_twist_pairs=(
                        *conditional_twist_pairs, *new_conditional_pairs
                    ),
                    conditional_twist_conditioners=(
                        *conditional_twist_conditioners,
                        *new_conditional_conditioners,
                    ),
                    conditional_twist_coefficients=(
                        *conditional_twist_coefficients,
                        *new_coefficients,
                    ),
                )
            evaluator = (
                _torch_linear_log_density_rosenblatt
                if kind == "linear-squared"
                else _torch_local_purified_log_density_rosenblatt
            )
            log_density, transformed = evaluator(
                cores, internal, gamma,
                transform=(index + 1 < len(core_groups)),
                right=fixed_right_environments[index],
            )
            total = total + log_density
            if transformed is not None:
                inverse_tensor = torch.as_tensor(
                    inverse_permutation, dtype=torch.long, device=value.device
                )
                value = transformed[:, inverse_tensor]
        return total

    validation_tensor = torch.as_tensor(
        validation, dtype=torch_dtype, device=device
    )

    def validation_nll() -> float:
        with torch.no_grad():
            values = []
            for start in range(0, len(validation_tensor), batch_size):
                values.append(chain_log_density(
                    validation_tensor[start:start + batch_size]
                ))
            return -float(torch.cat(values).mean())

    if conditional_twist_learning_rate is None:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    else:
        optimizer_groups = []
        if nonconditional_parameters:
            optimizer_groups.append({
                "params": nonconditional_parameters,
                "lr": learning_rate,
            })
        conditional_parameters = [
            coefficient
            for group in conditional_groups
            for coefficient in group
        ]
        if conditional_parameters:
            optimizer_groups.append({
                "params": conditional_parameters,
                "lr": float(conditional_twist_learning_rate),
            })
        optimizer = torch.optim.Adam(optimizer_groups)
    rng = np.random.default_rng(seed)
    optimized_coordinate_blocks = []
    if optimize_rotations:
        optimized_coordinate_blocks.append("rotation")
    if optimize_conditional_twists:
        optimized_coordinate_blocks.append("conditional")
    if optimize_cores and optimized_coordinate_blocks:
        optimizer_name = (
            "joint-linear-probit-core-"
            + "-".join(optimized_coordinate_blocks)
            + "-adam"
        )
    elif optimized_coordinate_blocks:
        optimizer_name = (
            "joint-linear-probit-"
            + "-".join(optimized_coordinate_blocks)
            + "-adam"
        )
    else:
        optimizer_name = (
            "joint-linear-orthogonal-adam"
            if canonicalize and all(
                specification[0] == "linear-squared"
                for specification in specifications
            ) else "joint-linear-adam"
        )
    history = FitHistory(
        optimizer=optimizer_name
    )
    best_validation = validation_nll()
    history.validation_loss.append(best_validation)
    best_cores = [
        [core.detach().clone() for core in group] for group in core_groups
    ]
    best_angles = [
        [angle.detach().clone() for angle in group]
        for group in angle_groups
    ]
    best_conditional_coefficients = [
        [coefficient.detach().clone() for coefficient in group]
        for group in conditional_groups
    ]
    stale = 0
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        chosen = rng.choice(
            len(training), size=min(batch_size, len(training)), replace=False
        )
        batch = torch.as_tensor(
            training[chosen], dtype=torch_dtype, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        if curvature_enabled:
            plus = torch.as_tensor(
                curvature_plus[chosen], dtype=torch_dtype, device=device
            )
            minus = torch.as_tensor(
                curvature_minus[chosen], dtype=torch_dtype, device=device
            )
            target = torch.as_tensor(
                curvature_target[chosen], dtype=torch_dtype, device=device
            )
            combined_scores = chain_log_density(torch.cat(
                (batch, plus, minus), dim=0
            ))
            positive_scores, plus_scores, minus_scores = torch.split(
                combined_scores, len(batch)
            )
        else:
            positive_scores = chain_log_density(batch)
        point_nll = -positive_scores
        loss = point_nll.mean()
        if tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(point_nll.numel())))
            )
            tail = torch.topk(
                point_nll, tail_count, sorted=False
            ).values.mean()
            loss = (1.0 - tail_weight) * loss + tail_weight * tail
        regularization = torch.zeros(
            (), dtype=torch_dtype, device=batch.device
        )
        if sobolev_penalty > 0.0:
            regularization = torch.stack([
                _torch_linear_root_sobolev_ratio(cores)
                for cores, specification in zip(
                    core_groups, specifications
                )
                if specification[0] == "linear-squared"
            ]).mean()
            loss = loss + float(sobolev_penalty) * regularization
        if curvature_enabled:
            second_contrast = (
                plus_scores + minus_scores - 2.0 * positive_scores
            )
            curvature_loss = torch.mean(
                (second_contrast - target).square()
            )
            weighted_curvature = float(curvature_weight) * curvature_loss
            loss = loss + weighted_curvature
            regularization = regularization + curvature_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 100.0)
        optimizer.step()
        if (
            optimize_cores and orthogonalization_interval > 0
            and (epoch + 1) % int(orthogonalization_interval) == 0
        ):
            # A right-QR sweep is an exact gauge change of a scalar TT root.
            # Periodic regauging keeps contractions balanced during a long
            # stochastic chain fit.  Only core Adam moments become invalid;
            # tangent rotations and conditional-coordinate moments remain in
            # the same parameterization and are deliberately retained.
            for group, specification, trainable_core in zip(
                core_groups, specifications, trainable_core_groups
            ):
                if (
                    not trainable_core
                    or specification[0] != "linear-squared"
                ):
                    continue
                canonical = [core.detach().clone() for core in group]
                if orthogonalization_metric == "hat-mass":
                    _torch_linear_right_orthogonalize(canonical)
                else:
                    _torch_right_orthogonalize(
                        canonical, uniform_measure=True
                    )
                with torch.no_grad():
                    for parameter, core in zip(group, canonical):
                        parameter.copy_(core)
                for parameter in group:
                    optimizer.state.pop(parameter, None)
        history.loss.append(float(loss.detach()))
        history.regularization.append(float(regularization.detach()))
        history.gradient_norm.append(float(gradient_norm))
        history.function_calls += 1
        check = (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        if check:
            value = validation_nll()
            history.validation_loss.append(value)
            threshold = min_delta * max(1.0, abs(best_validation))
            if best_validation - value > threshold:
                best_validation = value
                best_cores = [
                    [core.detach().clone() for core in group]
                    for group in core_groups
                ]
                best_angles = [
                    [angle.detach().clone() for angle in group]
                    for group in angle_groups
                ]
                best_conditional_coefficients = [
                    [coefficient.detach().clone() for coefficient in group]
                    for group in conditional_groups
                ]
                history.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                break
    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started

    fitted_layers = []
    for cores, angles, new_coefficients, specification in zip(
        best_cores, best_angles, best_conditional_coefficients,
        specifications,
    ):
        (
            kind, gamma, permutation, _, rotation,
            rotation_block_sizes,
            radial_twist_pairs, radial_twist_coefficients,
            conditional_twist_pairs, conditional_twist_conditioners,
            conditional_twist_coefficients,
            new_conditional_pairs, new_conditional_conditioners,
        ) = specification
        numpy_cores = [
            core.detach().cpu().numpy().copy() for core in cores
        ]
        base = (
            LinearSquaredTTDensity(
                vector.from_list(numpy_cores), gamma=gamma
            )
            if kind == "linear-squared"
            else LocallyPurifiedLinearTTDensity(numpy_cores, gamma=gamma)
        )
        if rotation is not None:
            fitted_rotation = rotation.detach().cpu().numpy()
            if angles:
                with torch.no_grad():
                    delta = _torch_block_orthogonal_rotations(
                        angles, rotation_block_sizes
                    ).detach().cpu().numpy()
                fitted_rotation = fitted_rotation @ delta
            # Copying an already orthogonal rotation to float32, as well as a
            # float32 matrix exponential, is only approximately orthogonal.
            # A blockwise polar projection changes the learned map only at
            # roundoff scale and makes the serialized volume-one invariant
            # exact to float64 precision in both core-only and angle phases.
            projected = np.zeros_like(fitted_rotation, dtype=np.float64)
            start = 0
            for size in rotation_block_sizes:
                stop = start + int(size)
                u, _, vh = np.linalg.svd(
                    fitted_rotation[start:stop, start:stop].astype(np.float64)
                )
                block = u @ vh
                if np.linalg.det(block) < 0.0:
                    u[:, -1] *= -1.0
                    block = u @ vh
                projected[start:stop, start:stop] = block
                start = stop
            fitted_rotation = projected
            fitted_new_conditional = [
                (
                    pairs,
                    conditioners,
                    coefficient.detach().cpu().numpy().astype(np.float64),
                )
                for pairs, conditioners, coefficient in zip(
                    new_conditional_pairs,
                    new_conditional_conditioners,
                    new_coefficients,
                )
                if torch.any(coefficient != 0.0).item()
            ]
            base = ProbitOrthogonalTTDensity(
                base,
                fitted_rotation,
                rotation_block_sizes=rotation_block_sizes,
                radial_twist_pairs=radial_twist_pairs,
                radial_twist_coefficients=radial_twist_coefficients,
                conditional_twist_pairs=(
                    *conditional_twist_pairs,
                    *[
                        pairs
                        for pairs, _, _ in fitted_new_conditional
                    ],
                ),
                conditional_twist_conditioners=(
                    *conditional_twist_conditioners,
                    *[
                        conditioners
                        for _, conditioners, _ in fitted_new_conditional
                    ],
                ),
                conditional_twist_coefficients=(
                    *conditional_twist_coefficients,
                    *[
                        coefficient
                        for _, _, coefficient in fitted_new_conditional
                    ],
                ),
            )
        if np.array_equal(permutation, np.arange(model.dimension)):
            fitted_layers.append(base)
        else:
            fitted_layers.append(PermutedTTDensity(base, permutation))
    return SampleDIRT(model.dimension, fitted_layers), history

__all__ = [
    "_torch_linear_right_environments",
    "_torch_linear_root_sobolev_ratio",
    "_torch_local_purified_right_environments",
    "_torch_linear_log_density_rosenblatt",
    "_torch_local_purified_log_density_rosenblatt",
    "score_linear_sample_dirt_mode_refinement",
    "fine_tune_linear_sample_dirt",
]
