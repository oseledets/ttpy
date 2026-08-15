"""Differentiable Torch contractions and density objectives."""

from __future__ import annotations

import numpy as np

from ._basis import _quadratic_gram_bands

def _torch_sample_tt(cores, indices):
    """Evaluate an arbitrary scalar TT at a batch of multi-indices."""
    import torch

    left = torch.ones((indices.shape[0], 1), dtype=cores[0].dtype,
                      device=cores[0].device)
    for k, core in enumerate(cores):
        selected = core[:, indices[:, k], :].permute(1, 0, 2)
        left = torch.einsum("pa,pab->pb", left, selected)
    return left[:, 0]


def _torch_sample_linear_tt(cores, points):
    """Evaluate a nodal multilinear TT at continuous cube points."""
    return _torch_sample_linear_tt_vector(cores, points)[:, 0]


def _torch_sample_adaptive_linear_tt(cores, knots, points):
    """Evaluate a nodal multilinear TT on differentiable nonuniform knots."""
    import torch

    left = torch.ones(
        (points.shape[0], 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for k, (core, nodes) in enumerate(zip(cores, knots)):
        lower = torch.clamp(
            torch.searchsorted(
                nodes, points[:, k].contiguous(), right=True
            ) - 1,
            min=0, max=core.shape[1] - 2,
        )
        width = nodes[lower + 1] - nodes[lower]
        fraction = torch.clamp(
            (points[:, k] - nodes[lower]) / width, 0.0, 1.0
        )
        first = core[:, lower, :].permute(1, 0, 2)
        second = core[:, lower + 1, :].permute(1, 0, 2)
        selected = (
            (1.0 - fraction)[:, None, None] * first
            + fraction[:, None, None] * second
        )
        left = torch.einsum("pa,pab->pb", left, selected)
    return left[:, 0]


def _torch_adaptive_linear_root_second_moment(cores, knots):
    """Exact differentiable nonuniform hat-Gram contraction."""
    import torch

    environment = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for core, nodes in zip(cores, knots):
        widths = nodes[1:] - nodes[:-1]
        projected = torch.einsum("ac,aib->cib", environment, core)
        diagonal = torch.zeros(
            (core.shape[1],), dtype=core.dtype, device=core.device
        )
        # Avoid in-place writes to quantities depending on the knot logits;
        # scatter_add keeps the full gradient with respect to every width.
        indices = torch.arange(
            core.shape[1] - 1, dtype=torch.long, device=core.device
        )
        diagonal = diagonal.scatter_add(0, indices, widths / 3.0)
        diagonal = diagonal.scatter_add(0, indices + 1, widths / 3.0)
        environment = torch.einsum(
            "cib,cid->bd",
            projected * diagonal[None, :, None],
            core,
        )
        off_diagonal = widths / 6.0
        environment = environment + torch.einsum(
            "cib,cid->bd",
            projected[:, :-1] * off_diagonal[None, :, None],
            core[:, 1:],
        )
        environment = environment + torch.einsum(
            "cib,cid->bd",
            projected[:, 1:] * off_diagonal[None, :, None],
            core[:, :-1],
        )
    return environment.reshape(())


def _torch_adaptive_knots(logits, minimum_width: float):
    """Map unconstrained interval logits to ordered knots on ``[0,1]``."""
    import torch

    result = []
    for value in logits:
        interval_count = int(value.numel())
        available = 1.0 - interval_count * float(minimum_width)
        widths = float(minimum_width) + available * torch.softmax(value, dim=0)
        zero = torch.zeros((1,), dtype=value.dtype, device=value.device)
        # The final cumulative sum is one analytically.  Replacing it by an
        # exact constant avoids endpoint search errors from floating rounding.
        interior = torch.cumsum(widths, dim=0)[:-1]
        one = torch.ones((1,), dtype=value.dtype, device=value.device)
        result.append(torch.cat((zero, interior, one)))
    return result


def _torch_block_orthogonal_rotations(angle_blocks, block_sizes):
    """Exponentiate skew blocks into an exactly orthogonal dense matrix."""
    import torch

    rotations = []
    for angles, size in zip(angle_blocks, block_sizes):
        size = int(size)
        rows, columns = torch.triu_indices(
            size, size, offset=1, device=angles.device
        )
        skew = torch.zeros(
            (size, size), dtype=angles.dtype, device=angles.device
        )
        skew = skew.index_put((rows, columns), angles)
        skew = skew - skew.T
        rotations.append(torch.matrix_exp(skew))
    return torch.block_diag(*rotations)


def _torch_radial_twist_gaussian(
    gaussian, pairs, coefficients, *, inverse: bool = False,
):
    """Differentiable disjoint-pair radial twists in Gaussian space."""
    import torch

    value = gaussian
    stages = list(zip(
        () if pairs is None else pairs,
        () if coefficients is None else coefficients,
    ))
    if inverse:
        stages.reverse()
    for stage_pairs, stage_coefficients in stages:
        pair_tensor = torch.as_tensor(
            stage_pairs, dtype=torch.long, device=value.device
        )
        coefficient_tensor = torch.as_tensor(
            stage_coefficients, dtype=value.dtype, device=value.device
        )
        first_index, second_index = pair_tensor[:, 0], pair_tensor[:, 1]
        first = value[:, first_index]
        second = value[:, second_index]
        radial_quantile = -torch.expm1(
            -0.5 * (first.square() + second.square())
        )
        frequencies = torch.arange(
            1, coefficient_tensor.shape[1] + 1,
            dtype=value.dtype, device=value.device,
        )
        basis = torch.sin(
            np.pi * radial_quantile[:, :, None]
            * frequencies[None, None, :]
        )
        angles = torch.sum(
            basis * coefficient_tensor[None, :, :], dim=2
        )
        if inverse:
            angles = -angles
        cosine, sine = torch.cos(angles), torch.sin(angles)
        updated = value.clone()
        updated[:, first_index] = cosine * first - sine * second
        updated[:, second_index] = sine * first + cosine * second
        value = updated
    return value


def _torch_conditional_twist_gaussian(
    gaussian, pairs, conditioners, coefficients, *, inverse: bool = False,
):
    """Differentiable conditional pair rotations with triangular inverse."""
    import torch

    value = gaussian
    stages = list(zip(
        () if pairs is None else pairs,
        () if conditioners is None else conditioners,
        () if coefficients is None else coefficients,
    ))
    if inverse:
        stages.reverse()
    for stage_pairs, stage_conditioners, stage_coefficients in stages:
        pair_tensor = torch.as_tensor(
            stage_pairs, dtype=torch.long, device=value.device
        )
        conditioner_tensor = torch.as_tensor(
            stage_conditioners, dtype=torch.long, device=value.device
        )
        coefficient_tensor = torch.as_tensor(
            stage_coefficients, dtype=value.dtype, device=value.device
        )
        first_index, second_index = pair_tensor[:, 0], pair_tensor[:, 1]
        first = value[:, first_index]
        second = value[:, second_index]
        conditioning_uniform = 0.5 * (
            1.0 + torch.erf(
                value[:, conditioner_tensor] / np.sqrt(2.0)
            )
        )
        frequencies = torch.arange(
            1, coefficient_tensor.shape[1] + 1,
            dtype=value.dtype, device=value.device,
        )
        basis = torch.sin(
            np.pi * conditioning_uniform[:, :, None]
            * frequencies[None, None, :]
        )
        angles = torch.sum(
            basis * coefficient_tensor[None, :, :], dim=2
        )
        if inverse:
            angles = -angles
        cosine, sine = torch.cos(angles), torch.sin(angles)
        updated = value.clone()
        updated[:, first_index] = cosine * first - sine * second
        updated[:, second_index] = sine * first + cosine * second
        value = updated
    return value


def _torch_probit_orthogonal_map(
    points, rotation, *, inverse: bool = False,
    radial_twist_pairs=None, radial_twist_coefficients=None,
    conditional_twist_pairs=None,
    conditional_twist_conditioners=None,
    conditional_twist_coefficients=None,
):
    """Differentiable Gaussian-measure-preserving map of the open cube."""
    import torch

    epsilon = 1e-6 if points.dtype == torch.float32 else 1e-12
    gaussian = torch.special.ndtri(torch.clamp(
        points, epsilon, 1.0 - epsilon
    ))
    if inverse:
        rotated = _torch_conditional_twist_gaussian(
            gaussian,
            conditional_twist_pairs,
            conditional_twist_conditioners,
            conditional_twist_coefficients,
            inverse=True,
        )
        rotated = _torch_radial_twist_gaussian(
            rotated,
            radial_twist_pairs,
            radial_twist_coefficients,
            inverse=True,
        ) @ rotation.T
    else:
        rotated = _torch_radial_twist_gaussian(
            gaussian @ rotation,
            radial_twist_pairs,
            radial_twist_coefficients,
        )
        rotated = _torch_conditional_twist_gaussian(
            rotated,
            conditional_twist_pairs,
            conditional_twist_conditioners,
            conditional_twist_coefficients,
        )
    return 0.5 * (
        1.0 + torch.erf(rotated / np.sqrt(2.0))
    )


def _torch_sample_linear_tt_vector(cores, points):
    """Evaluate every terminal channel of a nodal multilinear TT."""
    import torch

    left = torch.ones(
        (points.shape[0], 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for k, core in enumerate(cores):
        position = points[:, k] * (core.shape[1] - 1)
        lower = torch.clamp(
            torch.floor(position).to(torch.long), max=core.shape[1] - 2
        )
        fraction = torch.clamp(position - lower, 0.0, 1.0)
        first = core[:, lower, :].permute(1, 0, 2)
        second = core[:, lower + 1, :].permute(1, 0, 2)
        selected = (
            (1.0 - fraction)[:, None, None] * first
            + fraction[:, None, None] * second
        )
        left = torch.einsum("pa,pab->pb", left, selected)
    return left


def _torch_sample_local_purified(cores, points):
    """Evaluate a locally purified MPS density signal at cube points."""
    import torch

    left = torch.ones(
        (points.shape[0], 1, 1),
        dtype=cores[0].dtype, device=cores[0].device,
    )
    for k, core in enumerate(cores):
        position = points[:, k] * (core.shape[1] - 1)
        lower = torch.clamp(
            torch.floor(position).to(torch.long), max=core.shape[1] - 2
        )
        fraction = torch.clamp(position - lower, 0.0, 1.0)
        first = core[:, lower, :, :].permute(1, 0, 2, 3)
        second = core[:, lower + 1, :, :].permute(1, 0, 2, 3)
        selected = (
            (1.0 - fraction)[:, None, None, None] * first
            + fraction[:, None, None, None] * second
        )
        contracted = torch.einsum(
            "pab,pakr->pbkr", left, selected
        )
        left = torch.einsum(
            "pbkr,pbks->prs", contracted, selected
        )
    return torch.clamp_min(left[:, 0, 0], 0.0)


def _torch_nonnegative_linear_log_integral(log_cores):
    """Log of the exact hat-basis integral of positive log-TT cores."""
    import torch

    log_right = torch.zeros(
        (1,), dtype=log_cores[0].dtype, device=log_cores[0].device
    )
    for core in reversed(log_cores):
        n = core.shape[1]
        weights = torch.full(
            (n,), 1.0 / (n - 1), dtype=core.dtype, device=core.device
        )
        weights[0] *= 0.5
        weights[-1] *= 0.5
        log_integrated = torch.logsumexp(
            core + torch.log(weights)[None, :, None], dim=1
        )
        log_right = torch.logsumexp(
            log_integrated + log_right[None, :], dim=1
        )
    return log_right[0]


def _torch_normalize_nonnegative_linear_log_cores(log_cores):
    """Put a positive TT into an exact suffix-stochastic gauge.

    If ``R_k`` is the exact suffix integral beginning at state ``a_k``, the
    diagonal positive gauge

    ``G_k[a,i,b] <- G_k[a,i,b] R_{k+1}[b] / R_k[a]``

    makes every integrated transition row sum to one and divides the complete
    scalar function by its integral.  Unlike QR, this canonicalization
    preserves corewise nonnegativity.  Log space keeps the operation stable in
    high dimension and makes it differentiable for stochastic NLL fitting.
    """
    import torch

    log_right = [None] * (len(log_cores) + 1)
    log_right[-1] = torch.zeros(
        (1,), dtype=log_cores[0].dtype, device=log_cores[0].device
    )
    for k in range(len(log_cores) - 1, -1, -1):
        core = log_cores[k]
        n = core.shape[1]
        weights = torch.full(
            (n,), 1.0 / (n - 1), dtype=core.dtype, device=core.device
        )
        weights[0] *= 0.5
        weights[-1] *= 0.5
        log_integrated = torch.logsumexp(
            core + torch.log(weights)[None, :, None], dim=1
        )
        log_right[k] = torch.logsumexp(
            log_integrated + log_right[k + 1][None, :], dim=1
        )
    normalized = [
        core
        + log_right[k + 1][None, None, :]
        - log_right[k][:, None, None]
        for k, core in enumerate(log_cores)
    ]
    return normalized


def _torch_sample_nonnegative_linear_log_tt(log_cores, points):
    """Stable batched log evaluation of a positive multilinear TT."""
    import torch

    log_left = torch.zeros(
        (points.shape[0], 1),
        dtype=log_cores[0].dtype,
        device=log_cores[0].device,
    )
    negative_infinity = torch.full(
        (), -torch.inf, dtype=log_cores[0].dtype,
        device=log_cores[0].device,
    )
    for k, core in enumerate(log_cores):
        position = points[:, k] * (core.shape[1] - 1)
        lower = torch.clamp(
            torch.floor(position).to(torch.long), max=core.shape[1] - 2
        )
        fraction = torch.clamp(position - lower, 0.0, 1.0)
        first = core[:, lower, :].permute(1, 0, 2)
        second = core[:, lower + 1, :].permute(1, 0, 2)
        log_first_weight = torch.where(
            fraction < 1.0, torch.log1p(-fraction), negative_infinity
        )
        log_second_weight = torch.where(
            fraction > 0.0, torch.log(fraction), negative_infinity
        )
        selected = torch.logaddexp(
            first + log_first_weight[:, None, None],
            second + log_second_weight[:, None, None],
        )
        log_left = torch.logsumexp(
            log_left[:, :, None] + selected, dim=1
        )
    return log_left[:, 0]


def _torch_linear_hat_cholesky(mode: int, *, dtype, device):
    """Cholesky factor of the exact uniform-grid hat-basis Gram matrix."""
    import torch

    n = int(mode)
    if n < 2:
        raise ValueError("a linear hat basis needs at least two nodes")
    h = 1.0 / (n - 1)
    diagonal = torch.full((n,), 2.0 * h / 3.0, dtype=dtype, device=device)
    diagonal[0] = diagonal[-1] = h / 3.0
    gram = torch.diag(diagonal)
    off_diagonal = torch.full(
        (n - 1,), h / 6.0, dtype=dtype, device=device
    )
    gram = gram + torch.diag(off_diagonal, diagonal=1)
    gram = gram + torch.diag(off_diagonal, diagonal=-1)
    return torch.linalg.cholesky(gram)


def _torch_linear_mass_factors(cores):
    """Return ``(L, L^-1)`` for every physical mode of linear TT cores."""
    import torch

    factors = []
    inverses = []
    for core in cores:
        factor = _torch_linear_hat_cholesky(
            core.shape[1], dtype=core.dtype, device=core.device
        )
        inverse = torch.linalg.solve_triangular(
            factor,
            torch.eye(
                core.shape[1], dtype=core.dtype, device=core.device
            ),
            upper=False,
        )
        factors.append(factor)
        inverses.append(inverse)
    return factors, inverses


def _torch_linear_to_mass_cores(cores, factors):
    """Map nodal cores ``C`` to mass-orthonormal cores ``A = C L``."""
    import torch

    return [
        torch.einsum("aib,ij->ajb", core, factor)
        for core, factor in zip(cores, factors)
    ]


def _torch_linear_from_mass_cores(cores, inverse_factors):
    """Map mass-orthonormal cores back to nodal cores ``C = A L^-1``."""
    import torch

    return [
        torch.einsum("ajb,ji->aib", core, inverse)
        for core, inverse in zip(cores, inverse_factors)
    ]


def _torch_tt_frobenius_sq(cores):
    """Exact squared Frobenius norm of a torch TT coefficient tensor."""
    import torch

    environment = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for core in cores:
        projected = torch.einsum("ac,aib->cib", environment, core)
        environment = torch.einsum("cib,cid->bd", projected, core)
    return environment.reshape(())


def _torch_sample_quadratic_tt(cores, points):
    """Evaluate a uniform cardinal-quadratic B-spline TT."""
    import torch

    left = torch.ones(
        (points.shape[0], 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for k, core in enumerate(cores):
        intervals = core.shape[1] - 2
        position = points[:, k] * intervals
        lower = torch.clamp(
            torch.floor(position).to(torch.long), max=intervals - 1
        )
        fraction = torch.clamp(position - lower, 0.0, 1.0)
        weights = (
            0.5 * (1.0 - fraction).square(),
            0.5 + fraction - fraction.square(),
            0.5 * fraction.square(),
        )
        selected = sum(
            weight[:, None, None]
            * core[:, lower + offset, :].permute(1, 0, 2)
            for offset, weight in enumerate(weights)
        )
        left = torch.einsum("pa,pab->pb", left, selected)
    return left[:, 0]


def _torch_linear_root_second_moment(cores):
    """Exact hat-basis Gram contraction of a multilinear root square."""
    import torch

    environment = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for core in cores:
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = torch.einsum("ac,aib->cib", environment, core)
        diagonal = torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=core.dtype, device=core.device,
        )
        diagonal[0] = diagonal[-1] = h / 3.0
        weighted = projected * diagonal[None, :, None]
        environment = torch.einsum("cib,cid->bd", weighted, core)
        environment = environment + (h / 6.0) * (
            torch.einsum(
                "cib,cid->bd", projected[:, :-1], core[:, 1:]
            )
            + torch.einsum(
                "cib,cid->bd", projected[:, 1:], core[:, :-1]
            )
        )
    return (
        environment.reshape(())
        if environment.numel() == 1 else torch.trace(environment)
    )


def _torch_local_purified_integral(cores):
    """Differentiable exact hat-Gram norm of locally purified cores."""
    import torch

    environment = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for core in cores:
        n = core.shape[1]
        h = 1.0 / (n - 1)
        projected = torch.einsum("ab,aikr->bikr", environment, core)
        diagonal = torch.full(
            (n,), 2.0 * h / 3.0,
            dtype=core.dtype, device=core.device,
        )
        diagonal[0] = diagonal[-1] = h / 3.0
        environment = torch.einsum(
            "bikr,biks->rs",
            projected * diagonal[None, :, None, None],
            core,
        )
        environment = environment + (h / 6.0) * (
            torch.einsum(
                "bikr,biks->rs", projected[:, :-1], core[:, 1:]
            )
            + torch.einsum(
                "bikr,biks->rs", projected[:, 1:], core[:, :-1]
            )
        )
    return environment.reshape(())


def _torch_quadratic_root_second_moment(cores):
    """Exact quadratic B-spline Gram contraction of a TT root square."""
    import torch

    environment = torch.ones(
        (1, 1), dtype=cores[0].dtype, device=cores[0].device
    )
    for core in cores:
        projected = torch.einsum("ac,aib->cib", environment, core)
        bands = [
            torch.as_tensor(band, dtype=core.dtype, device=core.device)
            for band in _quadratic_gram_bands(core.shape[1])
        ]
        weighted = projected * bands[0][None, :, None]
        environment = torch.einsum("cib,cid->bd", weighted, core)
        for offset, band in ((1, bands[1]), (2, bands[2])):
            weighted_left = projected[:, :-offset] * band[None, :, None]
            environment = environment + torch.einsum(
                "cib,cid->bd", weighted_left, core[:, offset:]
            )
            weighted_right = projected[:, offset:] * band[None, :, None]
            environment = environment + torch.einsum(
                "cib,cid->bd", weighted_right, core[:, :-offset]
            )
    return environment.reshape(())


def _torch_root_second_moment(cores):
    """Exact uniform second moment of a root TT."""
    import torch

    environment = torch.ones(
        (), dtype=cores[0].dtype, device=cores[0].device
    ).reshape(1, 1)
    for core in cores:
        projected = torch.einsum("ac,aib->cib", environment, core)
        environment = (
            torch.einsum("cib,cid->bd", projected, core) / core.shape[1]
        )
    return environment.reshape(())


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
    values = _torch_sample_tt(cores, indices)
    target_mean = (weights * (gamma + values * values) / z).sum()
    return 0.5 * h2 - target_mean, h2, z


def _torch_nll_objective(
    cores, indices, weights, gamma: float, floor_mass: float | None = None,
):
    """Exact-normalization negative log likelihood for a squared root TT.

    Only the sample expectation is stochastic.  In contrast to the Pearson
    objective, this needs the second moment but not the expensive fourth
    moment, making high-resolution and higher-rank fits substantially faster.
    """
    import torch

    m2 = _torch_root_second_moment(cores)
    values = _torch_sample_tt(cores, indices)
    if floor_mass is None:
        normalization = gamma + m2
        loss = torch.log(normalization) - (
            weights * torch.log(gamma + values.square())
        ).sum()
    else:
        normalized_square = values.square() / torch.clamp_min(
            m2, torch.finfo(m2.dtype).tiny
        )
        density = floor_mass + (1.0 - floor_mass) * normalized_square
        normalization = m2
        loss = -(weights * torch.log(density)).sum()
    return loss, normalization


def _torch_correction_moments(cores):
    """Exact uniform mean and second moment of a direct correction TT."""
    import torch

    mean = torch.ones((1,), dtype=cores[0].dtype, device=cores[0].device)
    gram = torch.ones((1, 1), dtype=cores[0].dtype, device=cores[0].device)
    for core in cores:
        mean = mean @ core.mean(dim=1)
        projected = torch.einsum("ab,aic->bic", gram, core)
        gram = torch.einsum("bic,bid->cd", projected, core) / core.shape[1]
    return mean.reshape(()), gram.reshape(())


def _torch_correction_objective(cores, indices, weights):
    """Quadratic L2 loss whose minimizer is ``a = density_ratio - 1``.

    For residual samples ``V ~ r dmu``, the population objective is

    ``0.5 E_mu[a**2] - E_r[a] + E_mu[a]``.

    Both uniform terms are exact TT contractions.  Only the target linear
    expectation is empirical, so no fourth-order TT contraction is needed.
    """
    mean, second = _torch_correction_moments(cores)
    values = _torch_sample_tt(cores, indices)
    target_mean = (weights * values).sum()
    return 0.5 * second - target_mean + mean, second, mean


def _rbf_cell_kernel_matrix(modes: int, bandwidth: float) -> np.ndarray:
    """Exact cell-cell integrals of a one-dimensional Gaussian RBF.

    Entry ``(i, j)`` is the Lebesgue integral of
    ``exp(-(u-v)^2 / (2 bandwidth^2))`` over cells ``i`` and ``j`` of the
    equal-width partition of ``[0, 1]``.  Product kernels therefore remain
    rank-one operators across dimensions and can be contracted with a TT
    without sampling from the fitted density.
    """
    from scipy.special import erf

    modes = int(modes)
    bandwidth = float(bandwidth)
    if modes < 2:
        raise ValueError("modes must be at least two")
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")
    edges = np.linspace(0.0, 1.0, modes + 1, dtype=np.float64)
    left = edges[:-1]
    right = edges[1:]
    scale = np.sqrt(2.0) * bandwidth

    def second_antiderivative(value):
        value = np.asarray(value, dtype=np.float64)
        return (
            bandwidth * np.sqrt(np.pi / 2.0) * value * erf(value / scale)
            + bandwidth * bandwidth * np.exp(
                -0.5 * (value / bandwidth) ** 2
            )
        )

    # Rectangle inclusion-exclusion for a kernel that depends on u-v.
    result = (
        second_antiderivative(right[:, None] - left[None, :])
        - second_antiderivative(left[:, None] - left[None, :])
        - second_antiderivative(right[:, None] - right[None, :])
        + second_antiderivative(left[:, None] - right[None, :])
    )
    # Roundoff can create tiny asymmetric/negative entries at large bandwidth.
    result = 0.5 * (result + result.T)
    return np.maximum(result, 0.0)


def _rbf_cell_target_features(
    values: np.ndarray, modes: int, bandwidth: float,
) -> np.ndarray:
    """Exact cell integrals ``int_cell k(u, value) du`` for an RBF."""
    from scipy.special import erf

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    modes = int(modes)
    bandwidth = float(bandwidth)
    if modes < 2:
        raise ValueError("modes must be at least two")
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("target values must be finite")
    edges = np.linspace(0.0, 1.0, modes + 1, dtype=np.float64)
    scale = np.sqrt(2.0) * bandwidth
    factor = bandwidth * np.sqrt(np.pi / 2.0)
    return factor * (
        erf((edges[None, 1:] - values[:, None]) / scale)
        - erf((edges[None, :-1] - values[:, None]) / scale)
    )


def _torch_kernel_correction_objective(
    cores,
    target_feature_sets,
    target_weights,
    kernel_matrix_sets,
):
    """Exact product-kernel MMD objective for ``q=(1+a) dmu``.

    Constants independent of ``a`` are omitted.  For every separable kernel,
    the returned term is

    ``0.5 <a, K a>_mu - <a, K (nu-mu)>_mu``.

    The quadratic and uniform cross terms are exact TT contractions.  Only
    the expectation over target samples is empirical.  Averaging several RBF
    bandwidths gives a multiscale characteristic-kernel objective while
    preserving the same contraction structure.
    """
    import torch

    if len(target_feature_sets) != len(kernel_matrix_sets):
        raise ValueError("feature and kernel sets must have the same length")
    if not target_feature_sets:
        raise ValueError("at least one kernel is required")
    total = torch.zeros((), dtype=cores[0].dtype, device=cores[0].device)
    quadratic_total = torch.zeros_like(total)
    uniform_total = torch.zeros_like(total)
    for features, matrices in zip(target_feature_sets, kernel_matrix_sets):
        if len(features) != len(cores) or len(matrices) != len(cores):
            raise ValueError("each kernel must contain one factor per TT core")
        gram = torch.ones(
            (1, 1), dtype=cores[0].dtype, device=cores[0].device
        )
        uniform = torch.ones(
            (1,), dtype=cores[0].dtype, device=cores[0].device
        )
        target = torch.ones(
            (target_weights.shape[0], 1),
            dtype=cores[0].dtype,
            device=cores[0].device,
        )
        for core, feature, matrix in zip(cores, features, matrices):
            gram_core = torch.einsum("ab,aic->bic", gram, core)
            gram_kernel = torch.einsum("bic,ij->bjc", gram_core, matrix)
            gram = torch.einsum("bjc,bjd->cd", gram_kernel, core)
            uniform_core = torch.einsum("a,aib->ib", uniform, core)
            uniform = torch.einsum(
                "ib,i->b", uniform_core, matrix.sum(dim=1)
            )
            target_core = torch.einsum("ba,aic->bic", target, core)
            target = torch.einsum("bic,bi->bc", target_core, feature)
        quadratic = gram.reshape(())
        uniform_cross = uniform.reshape(())
        target_cross = (target_weights * target[:, 0]).sum()
        total = total + 0.5 * quadratic - target_cross + uniform_cross
        quadratic_total = quadratic_total + quadratic
        uniform_total = uniform_total + uniform_cross
    scale = 1.0 / len(target_feature_sets)
    return total * scale, quadratic_total * scale, uniform_total * scale

__all__ = [
    "_torch_sample_tt",
    "_torch_sample_linear_tt",
    "_torch_sample_adaptive_linear_tt",
    "_torch_adaptive_linear_root_second_moment",
    "_torch_adaptive_knots",
    "_torch_block_orthogonal_rotations",
    "_torch_radial_twist_gaussian",
    "_torch_conditional_twist_gaussian",
    "_torch_probit_orthogonal_map",
    "_torch_sample_linear_tt_vector",
    "_torch_sample_local_purified",
    "_torch_nonnegative_linear_log_integral",
    "_torch_normalize_nonnegative_linear_log_cores",
    "_torch_sample_nonnegative_linear_log_tt",
    "_torch_linear_hat_cholesky",
    "_torch_linear_mass_factors",
    "_torch_linear_to_mass_cores",
    "_torch_linear_from_mass_cores",
    "_torch_tt_frobenius_sq",
    "_torch_sample_quadratic_tt",
    "_torch_linear_root_second_moment",
    "_torch_local_purified_integral",
    "_torch_quadratic_root_second_moment",
    "_torch_root_second_moment",
    "_torch_root_moments",
    "_torch_fourth_left_step",
    "_torch_fourth_right_step",
    "_torch_density_objective",
    "_torch_nll_objective",
    "_torch_correction_moments",
    "_torch_correction_objective",
    "_rbf_cell_kernel_matrix",
    "_rbf_cell_target_features",
    "_torch_kernel_correction_objective",
]
