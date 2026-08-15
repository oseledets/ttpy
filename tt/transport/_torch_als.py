"""Torch environments and local ALS/Kaczmarz subproblems."""

from __future__ import annotations

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _as_numpy_cores,
    _cell_indices,
    _mean_numpy,
    _sample_tt_numpy,
    _second_moment_numpy,
)
from ._torch_density import (
    _torch_fourth_left_step,
    _torch_fourth_right_step,
    _torch_sample_tt,
)

def _torch_kernel_als_sweep(
    cores,
    target_feature_sets,
    target_weights,
    kernel_matrix_sets,
    *,
    regularization: float,
    relaxation: float,
    forward: bool,
) -> float:
    """One exact block-ALS sweep for the separable kernel objective."""
    import torch

    dimension = len(cores)
    kernel_count = len(kernel_matrix_sets)
    dtype, device = cores[0].dtype, cores[0].device

    def right_interfaces(features, matrices):
        quadratic = [None] * (dimension + 1)
        uniform = [None] * (dimension + 1)
        target = [None] * (dimension + 1)
        quadratic[dimension] = torch.ones((1, 1), dtype=dtype, device=device)
        uniform[dimension] = torch.ones((1,), dtype=dtype, device=device)
        target[dimension] = torch.ones(
            (target_weights.shape[0], 1), dtype=dtype, device=device
        )
        for k in range(dimension - 1, -1, -1):
            core, matrix, feature = cores[k], matrices[k], features[k]
            quadratic_core = torch.einsum(
                "bjd,cd->bjc", core, quadratic[k + 1]
            )
            quadratic_kernel = torch.einsum(
                "ij,bjc->bic", matrix, quadratic_core
            )
            quadratic[k] = torch.einsum(
                "aic,bic->ab", core, quadratic_kernel
            )
            uniform_core = torch.einsum(
                "aib,b->ai", core, uniform[k + 1]
            )
            uniform[k] = torch.einsum(
                "ai,i->a", uniform_core, matrix.sum(dim=1)
            )
            target_core = torch.einsum(
                "aic,pc->pai", core, target[k + 1]
            )
            target[k] = torch.einsum("pai,pi->pa", target_core, feature)
        return quadratic, uniform, target

    def left_interfaces(features, matrices):
        quadratic = [None] * (dimension + 1)
        uniform = [None] * (dimension + 1)
        target = [None] * (dimension + 1)
        quadratic[0] = torch.ones((1, 1), dtype=dtype, device=device)
        uniform[0] = torch.ones((1,), dtype=dtype, device=device)
        target[0] = torch.ones(
            (target_weights.shape[0], 1), dtype=dtype, device=device
        )
        for k in range(dimension):
            core, matrix, feature = cores[k], matrices[k], features[k]
            quadratic_core = torch.einsum(
                "ab,aic->bic", quadratic[k], core
            )
            quadratic_kernel = torch.einsum(
                "bic,ij->bjc", quadratic_core, matrix
            )
            quadratic[k + 1] = torch.einsum(
                "bjc,bjd->cd", quadratic_kernel, core
            )
            uniform_core = torch.einsum("a,aib->ib", uniform[k], core)
            uniform[k + 1] = torch.einsum(
                "ib,i->b", uniform_core, matrix.sum(dim=1)
            )
            target_core = torch.einsum("pa,aic->pic", target[k], core)
            target[k + 1] = torch.einsum(
                "pic,pi->pc", target_core, feature
            )
        return quadratic, uniform, target

    if forward:
        opposite = [
            right_interfaces(features, matrices)
            for features, matrices in zip(
                target_feature_sets, kernel_matrix_sets
            )
        ]
        active = [
            (
                torch.ones((1, 1), dtype=dtype, device=device),
                torch.ones((1,), dtype=dtype, device=device),
                torch.ones(
                    (target_weights.shape[0], 1), dtype=dtype, device=device
                ),
            )
            for _ in range(kernel_count)
        ]
        centers = range(dimension)
    else:
        opposite = [
            left_interfaces(features, matrices)
            for features, matrices in zip(
                target_feature_sets, kernel_matrix_sets
            )
        ]
        active = [
            (
                torch.ones((1, 1), dtype=dtype, device=device),
                torch.ones((1,), dtype=dtype, device=device),
                torch.ones(
                    (target_weights.shape[0], 1), dtype=dtype, device=device
                ),
            )
            for _ in range(kernel_count)
        ]
        centers = range(dimension - 1, -1, -1)

    update_norm_sq = 0.0
    for centre in centers:
        shape = cores[centre].shape
        local_size = int(np.prod(shape))
        hessian = torch.zeros(
            (local_size, local_size), dtype=dtype, device=device
        )
        source = torch.zeros(shape, dtype=dtype, device=device)
        for kernel_index, (features, matrices) in enumerate(zip(
            target_feature_sets, kernel_matrix_sets
        )):
            active_quadratic, active_uniform, active_target = active[
                kernel_index
            ]
            opposite_quadratic, opposite_uniform, opposite_target = opposite[
                kernel_index
            ]
            if forward:
                left_q, right_q = (
                    active_quadratic, opposite_quadratic[centre + 1]
                )
                left_u, right_u = (
                    active_uniform, opposite_uniform[centre + 1]
                )
                left_t, right_t = (
                    active_target, opposite_target[centre + 1]
                )
            else:
                left_q, right_q = (
                    opposite_quadratic[centre], active_quadratic
                )
                left_u, right_u = (
                    opposite_uniform[centre], active_uniform
                )
                left_t, right_t = (
                    opposite_target[centre], active_target
                )
            matrix = matrices[centre]
            feature = features[centre]
            left_hessian = torch.einsum("ab,ij->aibj", left_q, matrix)
            hessian += torch.einsum(
                "aibj,cd->aicbjd", left_hessian, right_q
            ).reshape(local_size, local_size) / kernel_count
            weighted_left = left_t * target_weights[:, None]
            target_left = torch.einsum("pa,pi->pai", weighted_left, feature)
            target_coefficient = torch.einsum(
                "pai,pc->aic", target_left, right_t
            )
            uniform_left = torch.einsum(
                "a,i->ai", left_u, matrix.sum(dim=1)
            )
            uniform_coefficient = torch.einsum(
                "ai,c->aic", uniform_left, right_u
            )
            source += (
                target_coefficient - uniform_coefficient
            ) / kernel_count
        hessian = 0.5 * (hessian + hessian.T)
        diagonal_scale = torch.clamp_min(
            torch.diagonal(hessian).abs().mean(),
            torch.finfo(dtype).tiny,
        )
        # The sweep is kept in the product-uniform mixed canonical gauge.
        # Hence E_mu[a**2] is ||G_c||_F**2 / n_c at the active core, and this
        # is an exact L2/Tikhonov term rather than a gauge-dependent parameter
        # ridge.  A separate machine-scale ridge only stabilizes the solve.
        identity = torch.eye(local_size, dtype=dtype, device=device)
        hessian = hessian + (
            float(regularization) / shape[1]
            + 10.0 * torch.finfo(dtype).eps * diagonal_scale
        ) * identity
        candidate = torch.linalg.solve(hessian, source.reshape(-1)).reshape(shape)
        old = cores[centre]
        updated = (
            (1.0 - relaxation) * old + relaxation * candidate
        ).detach()
        update_norm_sq += float((updated - old).square().sum())
        cores[centre] = updated
        if forward and centre + 1 < dimension:
            _torch_move_center_right(
                cores, centre, uniform_measure=True
            )
            updated = cores[centre]
        elif not forward and centre > 0:
            _torch_move_center_left(
                cores, centre, uniform_measure=True
            )
            updated = cores[centre]

        for kernel_index, (features, matrices) in enumerate(zip(
            target_feature_sets, kernel_matrix_sets
        )):
            matrix, feature = matrices[centre], features[centre]
            q, u, t = active[kernel_index]
            if forward:
                quadratic_core = torch.einsum("ab,aic->bic", q, updated)
                quadratic_kernel = torch.einsum(
                    "bic,ij->bjc", quadratic_core, matrix
                )
                q = torch.einsum("bjc,bjd->cd", quadratic_kernel, updated)
                uniform_core = torch.einsum("a,aib->ib", u, updated)
                u = torch.einsum(
                    "ib,i->b", uniform_core, matrix.sum(dim=1)
                )
                target_core = torch.einsum("pa,aic->pic", t, updated)
                t = torch.einsum("pic,pi->pc", target_core, feature)
            else:
                quadratic_core = torch.einsum("bjd,cd->bjc", updated, q)
                quadratic_kernel = torch.einsum(
                    "ij,bjc->bic", matrix, quadratic_core
                )
                q = torch.einsum("aic,bic->ab", updated, quadratic_kernel)
                uniform_core = torch.einsum("aib,b->ai", updated, u)
                u = torch.einsum(
                    "ai,i->a", uniform_core, matrix.sum(dim=1)
                )
                target_core = torch.einsum("aic,pc->pai", updated, t)
                t = torch.einsum("pai,pi->pa", target_core, feature)
            active[kernel_index] = (q, u, t)
    return float(np.sqrt(update_norm_sq))


def _torch_two_sample_correction_objective(
    cores,
    denominator_indices,
    denominator_weights,
    numerator_indices,
    numerator_weights,
):
    """Empirical Pearson loss for ``a = p_numerator/p_denominator - 1``."""
    denominator_values = _torch_sample_tt(cores, denominator_indices)
    numerator_values = _torch_sample_tt(cores, numerator_indices)
    denominator_mean = (denominator_weights * denominator_values).sum()
    denominator_second = (
        denominator_weights * denominator_values.square()
    ).sum()
    numerator_mean = (numerator_weights * numerator_values).sum()
    loss = 0.5 * denominator_second - numerator_mean + denominator_mean
    return loss, denominator_second, denominator_mean


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
    compression_tolerance: float | None = None,
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
    if compression_tolerance is None:
        compression_tolerance = min(
            0.1, max(1e-3, points.shape[0] ** -0.5)
        )
    if not 0.0 < compression_tolerance < 1.0:
        raise ValueError("compression_tolerance must lie in (0, 1)")

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
    # Resolving an empirical histogram to machine precision only stores sample
    # noise.  The default is the parametric N^{-1/2} statistical scale; callers
    # may choose a problem-specific validation-selected tolerance.
    coarse_root = vector(
        np.sqrt(coarse_density), eps=compression_tolerance, rmax=rank
    )

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


def _coarse_to_fine_initial_correction(
    points: np.ndarray,
    modes: np.ndarray,
    *,
    rank: int,
    coarse_bins: int,
    pseudocount: float,
    tolerance: float,
) -> vector:
    """Estimate ``density_ratio - 1`` and truncate sample noise.

    The coarse joint histogram retains global dependence.  Conditional
    one-dimensional histograms lift it to the requested fine cells.  Unlike
    the squared-root initializer, this routine represents the *direct*
    centered ratio, so its L2 objective needs only second-order contractions.
    """
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
    if not 0.0 < tolerance < 1.0:
        raise ValueError("initialization_tolerance must lie in (0, 1)")

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
    # Do not reproduce every noisy histogram cell.  The coarse density itself
    # gets a slightly tighter tolerance because the meaningful object for rank
    # selection is the centered correction after lifting.
    coarse_tt = vector(
        coarse_density,
        eps=max(1e-6, 0.25 * tolerance),
        rmax=rank,
    )

    fine = _cell_indices(points, modes)
    lifted = []
    for k, (core, mode) in enumerate(zip(coarse_tt.cores, modes)):
        width = int(mode // coarse_bins)
        basis = np.zeros((coarse_bins, int(mode)), dtype=np.float64)
        for coarse_cell in range(coarse_bins):
            start = coarse_cell * width
            selected = fine[coarse[:, k] == coarse_cell, k] - start
            local_counts = np.bincount(selected, minlength=width).astype(np.float64)
            local_counts += pseudocount
            local_probability = local_counts / local_counts.sum()
            basis[coarse_cell, start:start + width] = width * local_probability
        lifted.append(np.einsum("asb,si->aib", core, basis, optimize=True))

    ratio = vector.from_list(lifted)
    correction = (ratio - 1.0).round(eps=tolerance, rmax=rank)
    # The exact minimizer has zero reference mean.  Removing the empirical
    # offset explicitly keeps the constant rank-one direction out of the
    # correction TT and improves conditioning.
    correction = (
        correction - _mean_numpy(_as_numpy_cores(correction))
    ).round(eps=tolerance, rmax=rank)
    return correction


def _product_marginal_initial_correction(
    points: np.ndarray,
    modes: np.ndarray,
    *,
    rank: int,
    pseudocount: float,
    tolerance: float,
) -> vector:
    """Scalable sample initializer from empirical one-dimensional marginals.

    A dense ``coarse_bins**d`` joint histogram is useful in small dimension but
    impossible for the 21--64 dimensional tabular examples.  The product of
    smoothed marginal histograms is a rank-one density TT; subtracting the
    analytic uniform density gives a centered correction of rank at most two.
    Dependence is then learned by the optimizer rather than materialized in an
    exponentially large initializer.
    """
    if pseudocount <= 0.0 or not np.isfinite(pseudocount):
        raise ValueError("initialization_pseudocount must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("initialization_tolerance must lie in (0, 1)")
    indices = _cell_indices(points, modes)
    cores = []
    for k, mode in enumerate(modes):
        counts = np.bincount(
            indices[:, k], minlength=int(mode)
        ).astype(np.float64)
        counts += pseudocount
        marginal_density = int(mode) * counts / counts.sum()
        cores.append(marginal_density.reshape(1, int(mode), 1))
    ratio = vector.from_list(cores)
    correction = (ratio - 1.0).round(eps=tolerance, rmax=rank)
    correction = (
        correction - _mean_numpy(_as_numpy_cores(correction))
    ).round(eps=tolerance, rmax=rank)
    return correction


def _enrich_correction_ranks(
    correction: vector,
    modes: np.ndarray,
    *,
    rank: int,
    noise: float,
    seed: int,
) -> vector:
    """Add a small connected random TT so fixed-rank optimization is real.

    Product-marginal initializers have rank at most two. Merely passing a
    larger ``rmax`` to the final rounding cannot create missing tangent
    directions, so Adam/one-site ALS would silently remain rank two. A
    right-orthogonal random TT supplies all feasible channels at controlled
    L2 scale while leaving the sample-derived initializer dominant.
    """
    if noise == 0.0:
        return correction
    dimension = len(modes)
    ranks = [1] * (dimension + 1)
    capacity = 1
    for k in range(1, dimension):
        capacity = min(int(rank), capacity * int(modes[k - 1]))
        ranks[k] = capacity
    capacity = 1
    for k in range(dimension - 1, 0, -1):
        capacity = min(int(rank), capacity * int(modes[k]))
        ranks[k] = min(ranks[k], capacity)
    if np.array_equal(correction.r, np.asarray(ranks)):
        return correction
    current_ranks = correction.r.astype(int).tolist()
    gaps = [ranks[k] - current_ranks[k] for k in range(1, dimension)]
    additive_ranks = (
        [1, *gaps, 1]
        if all(gap > 0 for gap in gaps)
        else ranks
    )
    rng = np.random.default_rng(seed)
    random_cores = [
        rng.standard_normal((
            additive_ranks[k], int(modes[k]), additive_ranks[k + 1]
        )) / np.sqrt(max(1, int(modes[k]) * additive_ranks[k]))
        for k in range(dimension)
    ]
    # Zero mean in one physical direction makes the entire TT mean zero
    # exactly, without adding a rank-one constant and consuming rank budget.
    random_cores[0] -= random_cores[0].mean(axis=1, keepdims=True)
    perturbation = vector.from_list(random_cores).orthogonalize(center=0)
    scale = np.sqrt(_second_moment_numpy(_as_numpy_cores(perturbation)))
    if not np.isfinite(scale) or scale == 0.0:
        raise FloatingPointError("rank enrichment produced a zero/invalid TT")
    perturbation = perturbation * (float(noise) / scale)
    enriched = correction + perturbation
    if all(gap > 0 for gap in gaps):
        return enriched
    return enriched.round(eps=1e-14, rmax=rank)


def _coarse_to_fine_initial_ratio_correction(
    denominator: np.ndarray,
    numerator: np.ndarray,
    modes: np.ndarray,
    *,
    rank: int,
    coarse_bins: int,
    pseudocount: float,
    tolerance: float,
    ratio_clip: float,
) -> vector:
    """Multiscale histogram initializer for ``numerator/denominator - 1``."""
    d = denominator.shape[1]
    if numerator.shape[1] != d:
        raise ValueError("denominator and numerator dimensions must agree")
    if coarse_bins < 2 or np.any(modes % coarse_bins != 0):
        raise ValueError("coarse_bins must divide every mode and be at least two")
    if coarse_bins ** d > 1_048_576:
        raise ValueError("coarse ratio initializer is too large")
    if pseudocount <= 0.0 or not np.isfinite(pseudocount):
        raise ValueError("initialization_pseudocount must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("initialization_tolerance must lie in (0, 1)")
    if ratio_clip <= 1.0 or not np.isfinite(ratio_clip):
        raise ValueError("ratio_clip must be finite and greater than one")

    def coarse_data(points):
        cells = np.minimum(
            np.floor(points * coarse_bins).astype(np.int64), coarse_bins - 1
        )
        strides = coarse_bins ** np.arange(d - 1, -1, -1, dtype=np.int64)
        counts = np.bincount(
            cells @ strides, minlength=coarse_bins ** d
        ).astype(np.float64)
        counts += pseudocount
        probabilities = counts / counts.sum()
        return cells, probabilities.reshape([coarse_bins] * d)

    denominator_coarse, denominator_probability = coarse_data(denominator)
    numerator_coarse, numerator_probability = coarse_data(numerator)
    ratio = numerator_probability / denominator_probability
    ratio = np.clip(ratio, 1.0 / ratio_clip, ratio_clip)
    ratio /= float(np.sum(denominator_probability * ratio))
    ratio_tt = vector(
        ratio,
        eps=max(1e-6, 0.25 * tolerance),
        rmax=rank,
    )

    denominator_fine = _cell_indices(denominator, modes)
    numerator_fine = _cell_indices(numerator, modes)
    lifted = []
    for k, (core, mode) in enumerate(zip(ratio_tt.cores, modes)):
        width = int(mode // coarse_bins)
        basis = np.zeros((coarse_bins, int(mode)), dtype=np.float64)
        for coarse_cell in range(coarse_bins):
            start = coarse_cell * width
            denominator_local = (
                denominator_fine[denominator_coarse[:, k] == coarse_cell, k]
                - start
            )
            numerator_local = (
                numerator_fine[numerator_coarse[:, k] == coarse_cell, k]
                - start
            )
            denominator_counts = np.bincount(
                denominator_local, minlength=width
            ).astype(np.float64) + pseudocount
            numerator_counts = np.bincount(
                numerator_local, minlength=width
            ).astype(np.float64) + pseudocount
            local_ratio = (
                numerator_counts / numerator_counts.sum()
            ) / (denominator_counts / denominator_counts.sum())
            local_ratio = np.clip(
                local_ratio, 1.0 / ratio_clip, ratio_clip
            )
            local_ratio /= np.sum(
                denominator_counts / denominator_counts.sum() * local_ratio
            )
            basis[coarse_cell, start:start + width] = local_ratio
        lifted.append(np.einsum("asb,si->aib", core, basis, optimize=True))

    correction = (vector.from_list(lifted) - 1.0).round(
        eps=tolerance, rmax=rank
    )
    denominator_values = _sample_tt_numpy(
        _as_numpy_cores(correction), _cell_indices(denominator, modes)
    )
    correction = (correction - float(denominator_values.mean())).round(
        eps=tolerance, rmax=rank
    )
    return correction


def _torch_right_orthogonalize(cores, *, uniform_measure: bool = False) -> None:
    """Put a torch TT into a uniform-measure canonical gauge in place.

    The functional inner product of cellwise-constant cores contains ``1/n``
    in every mode.  Standard Euclidean QR would move ``sqrt(n)**(d-1)`` into
    the first core, which overflows float32 already for ``n=64, d=43``.  We
    therefore scale each orthogonal frame so that ``Q Q.T / n = I`` and move
    the reciprocal scale into the neighbouring triangular factor.  The TT
    tensor is unchanged exactly.
    """
    import torch

    for k in range(len(cores) - 1, 0, -1):
        r1, n, r2 = cores[k].shape
        q, r = torch.linalg.qr(cores[k].reshape(r1, n * r2).T, mode="reduced")
        if q.shape[1] != r1:
            raise ValueError(
                f"TT rank {r1} at bond {k} is not attainable for mode {n}; "
                "ALS orthogonalization would lower the requested fixed rank"
            )
        scale = float(n) ** 0.5 if uniform_measure else 1.0
        cores[k] = (scale * q.T).reshape(r1, n, r2).detach()
        cores[k - 1] = torch.einsum(
            "aib,bc->aic", cores[k - 1], r.T / scale
        ).detach()


def _torch_linear_right_orthogonalize(cores) -> None:
    """Put a nodal linear TT in the exact hat-mass right-canonical gauge.

    For the one-dimensional hat Gram matrix ``H=L L^T``, QR is applied to
    ``G_k L`` rather than to raw nodal slices.  The new core consequently
    satisfies ``G_k (H (x) I) G_k^T = I``.  The triangular factor is absorbed
    into the preceding core, so the represented multilinear function is
    unchanged; unlike the older ``I/n`` scaling this is the actual functional
    ``L2([0,1])`` gauge.
    """
    import torch

    for k in range(len(cores) - 1, 0, -1):
        r1, n, r2 = cores[k].shape
        h = 1.0 / (n - 1)
        mass = torch.diag(torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=cores[k].dtype,
            device=cores[k].device,
        ))
        mass[0, 0] = mass[-1, -1] = h / 3.0
        adjacent = torch.full(
            (n - 1,), h / 6.0,
            dtype=cores[k].dtype,
            device=cores[k].device,
        )
        index = torch.arange(n - 1, device=cores[k].device)
        mass[index, index + 1] = adjacent
        mass[index + 1, index] = adjacent
        chol = torch.linalg.cholesky(mass)
        weighted = torch.einsum("aic,ij->ajc", cores[k], chol)
        q, r = torch.linalg.qr(
            weighted.reshape(r1, n * r2).T, mode="reduced"
        )
        if q.shape[1] != r1:
            raise ValueError(
                f"TT rank {r1} at bond {k} is not attainable for mode {n}; "
                "linear orthogonalization would lower the fixed rank"
            )
        weighted_frame = q.T.reshape(r1, n, r2)
        inverse_chol = torch.linalg.solve_triangular(
            chol,
            torch.eye(n, dtype=chol.dtype, device=chol.device),
            upper=False,
        )
        cores[k] = torch.einsum(
            "ajc,ji->aic", weighted_frame, inverse_chol
        ).detach()
        cores[k - 1] = torch.einsum(
            "aib,bc->aic", cores[k - 1], r.T
        ).detach()


def _torch_move_center_right(
    cores, k: int, *, uniform_measure: bool = False
) -> None:
    import torch

    r1, n, r2 = cores[k].shape
    q, r = torch.linalg.qr(cores[k].reshape(r1 * n, r2), mode="reduced")
    if q.shape[1] != r2:
        raise ValueError("left orthogonalization lowered the requested TT rank")
    scale = float(n) ** 0.5 if uniform_measure else 1.0
    cores[k] = (scale * q).reshape(r1, n, r2).detach()
    cores[k + 1] = torch.einsum(
        "ab,bic->aic", r / scale, cores[k + 1]
    ).detach()


def _torch_move_center_left(
    cores, k: int, *, uniform_measure: bool = False
) -> None:
    import torch

    r1, n, r2 = cores[k].shape
    q, r = torch.linalg.qr(cores[k].reshape(r1, n * r2).T, mode="reduced")
    if q.shape[1] != r1:
        raise ValueError("right orthogonalization lowered the requested TT rank")
    scale = float(n) ** 0.5 if uniform_measure else 1.0
    cores[k] = (scale * q.T).reshape(r1, n, r2).detach()
    cores[k - 1] = torch.einsum(
        "aib,bc->aic", cores[k - 1], r.T / scale
    ).detach()


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


def _torch_centered_als_core(
    cores,
    indices,
    weights,
    centre: int,
    total_cells: int,
    regularization: float,
    *,
    uniform_measure: bool = False,
):
    """Closed-form one-site ALS minimizer for the centered quadratic loss.

    With ordinary Euclidean frames
    ``E_mu[a**2] = ||G_c||_F**2 / total_cells``. With frames orthonormal in
    the product uniform measure it is instead ``||G_c||_F**2 / n_c``. The
    latter avoids factors exponential in dimension and is used by centered
    ALS. The remaining terms are linear in the active core, so no iterative
    local solver or fourth-order environment is needed.
    """
    import torch

    left, right = _torch_sample_interfaces(cores, indices, centre)
    weighted_left = weights[:, None] * left
    source = torch.einsum("pa,pb->apb", weighted_left, right)
    sample_coefficient = torch.zeros_like(cores[centre])
    sample_coefficient.index_add_(1, indices[:, centre], source)

    left_mean = torch.ones(
        (1,), dtype=cores[0].dtype, device=cores[0].device
    )
    for k in range(centre):
        left_mean = left_mean @ cores[k].mean(dim=1)
    right_mean = torch.ones(
        (1,), dtype=cores[0].dtype, device=cores[0].device
    )
    for k in range(len(cores) - 1, centre, -1):
        right_mean = cores[k].mean(dim=1) @ right_mean
    uniform_coefficient = torch.einsum(
        "a,b->ab", left_mean, right_mean
    ).unsqueeze(1).expand(-1, cores[centre].shape[1], -1)
    uniform_coefficient = uniform_coefficient / cores[centre].shape[1]
    measure_scale = (
        cores[centre].shape[1] if uniform_measure else float(total_cells)
    )
    return (
        measure_scale
        / (1.0 + regularization)
        * (sample_coefficient - uniform_coefficient)
    )


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


def _torch_kaczmarz_local_nll(
    core,
    *,
    centre: int,
    mode: int,
    indices,
    weights,
    gamma: float,
    floor_mass: float | None,
    sample_left,
    sample_right,
    anchor,
    proximal_weight: float,
):
    """Mass-orthogonal one-site proximal block projection for NLL."""
    import torch

    # Both interfaces are orthonormal for the product probability measure.
    m2 = core.square().sum() / mode
    selected = core[:, indices[:, centre], :].permute(1, 0, 2)
    partial = torch.einsum("pa,pab->pb", sample_left, selected)
    values = (partial * sample_right).sum(dim=1)
    if floor_mass is None:
        nll = torch.log(gamma + m2) - (
            weights * torch.log(gamma + values.square())
        ).sum()
    else:
        normalized_square = values.square() / torch.clamp_min(
            m2, torch.finfo(m2.dtype).tiny
        )
        density = floor_mass + (1.0 - floor_mass) * normalized_square
        nll = -(weights * torch.log(density)).sum()
    proximity = 0.5 * proximal_weight * (core - anchor).square().mean()
    return nll + proximity

__all__ = [
    "_torch_kernel_als_sweep",
    "_torch_two_sample_correction_objective",
    "_compress_empirical_cells",
    "_coarse_to_fine_initial_root",
    "_coarse_to_fine_initial_correction",
    "_product_marginal_initial_correction",
    "_enrich_correction_ranks",
    "_coarse_to_fine_initial_ratio_correction",
    "_torch_right_orthogonalize",
    "_torch_linear_right_orthogonalize",
    "_torch_move_center_right",
    "_torch_move_center_left",
    "_torch_sample_interfaces",
    "_torch_centered_als_core",
    "_torch_fourth_interfaces",
    "_torch_als_local_objective",
    "_torch_kaczmarz_local_nll",
]
