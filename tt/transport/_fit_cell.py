"""Fitters for cellwise squared and centered TT densities."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _as_numpy_cores,
    _cell_indices,
    _mean_numpy,
    _sample_tt_numpy,
    _second_moment_numpy,
    _validate_points,
)
from ._optimizers import (
    _fit_tt_als,
    _fit_tt_block_kaczmarz,
    _fit_tt_riemannian_stochastic,
)
from ._torch_als import (
    _coarse_to_fine_initial_correction,
    _coarse_to_fine_initial_ratio_correction,
    _coarse_to_fine_initial_root,
    _compress_empirical_cells,
    _enrich_correction_ranks,
    _product_marginal_initial_correction,
    _torch_centered_als_core,
    _torch_kernel_als_sweep,
    _torch_move_center_left,
    _torch_move_center_right,
    _torch_right_orthogonalize,
    _torch_two_sample_correction_objective,
)
from ._torch_density import (
    _rbf_cell_kernel_matrix,
    _rbf_cell_target_features,
    _torch_correction_moments,
    _torch_correction_objective,
    _torch_density_objective,
    _torch_kernel_correction_objective,
    _torch_nll_objective,
)
from ._types import FitHistory
from .polynomial import DirectTTDensity
from .scalar import SquaredTTDensity

def fit_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 3,
    gamma: float = 1e-4,
    floor_mass: float | None = None,
    epochs: int = 500,
    learning_rate: float = 3e-2,
    batch_size: int | None = None,
    validation_samples=None,
    validation_metric: str = "objective",
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    tolerance: float = 1e-8,
    optimizer: str = "adam",
    objective: str = "l2",
    als_inner_steps: int = 8,
    riemannian_retraction: str = "svd",
    riemannian_momentum: float = 0.9,
    riemannian_second_moment: float = 0.99,
    initialization: str = "uniform",
    initialization_noise: float = 2e-2,
    initialization_coarse_bins: int = 2,
    initialization_pseudocount: float = 0.5,
    initialization_compression_tolerance: float | None = None,
    verbose: bool = False,
) -> tuple[SquaredTTDensity, FitHistory]:
    """Fit a positive TT density from samples with exact normalization.

    Args:
        samples: Array of shape ``(N, d)`` in the unit cube.  These are samples
            from the residual law ``F_k#nu_{k+1}``.
        modes: Number of equal-width cells in each coordinate.
        rank: Fixed internal TT rank of the square root ``g``.
        gamma: Positive density floor before normalisation.
        floor_mass: Optional fixed mixture weight of the uniform reference,
            giving ``floor_mass + (1-floor_mass) * root**2 / E[root**2]``.
            Unlike ``gamma``, this lower bound is invariant to rescaling the
            TT root. It is available with ``objective="nll"``.
        epochs: Adam/RGD iterations or complete nonlinear ALS sweeps.
        learning_rate: Adam rate, RGD initial Armijo step, or local ALS
            L-BFGS rate.
        batch_size: Optional size of the empirical linear-term minibatch.  The
            quadratic L2 term remains an exact full TT contraction at every step.
        validation_interval: Number of Adam minibatches between held-out
            validation checks. ``patience`` counts checks, not optimizer steps.
        seed: Reproducible initialisation and minibatch seed.
        device, dtype: Torch training device and precision.
        tolerance: Relative loss/gradient stopping threshold. Adam compares
            two 25-step windows; RGD uses the tangent-gradient norm and ALS
            compares complete sweeps.
        optimizer: ``"adam"``, deterministic ``"riemannian"``, stochastic
            ``"riemannian-sgd"``, mass-orthogonal stochastic ``"kaczmarz"``
            or orthogonal nonlinear ``"als"``.  The latter is nonlinear
            because the normalized squared TT objective is rational-quartic in
            one core.
        objective: ``"l2"`` for the exact Pearson density-ratio loss or
            ``"nll"`` for exact-normalization maximum likelihood.  NLL avoids
            the fourth-order contraction on every optimizer step and is the
            scalable choice for high-resolution tabular fits.
        als_inner_steps: L-BFGS iterations per core update and sweep direction.
        riemannian_retraction: Fixed-rank retraction, ``"svd"`` or the
            orthogonal projector-splitting ``"psa"`` sweep.
        riemannian_momentum, riemannian_second_moment: Decays for transported
            tangent momentum and its gauge-invariant scalar RMS in stochastic
            Riemannian optimization.
        initialization: ``"uniform"`` for the perturbed constant path,
            ``"product"`` for one smoothed empirical marginal square-root
            path, ``"mixture"`` for a connected rank-many mixture of such
            paths, or ``"coarse"`` for a sample-only coarse-grid TT followed
            by a one-dimensional empirical lift to the fine grid.
        initialization_noise: Standard deviation of the dense perturbation of
            the uniform path. Small residual-ratio layers generally need a
            much smaller value than a one-shot fit.
        initialization_coarse_bins, initialization_pseudocount: Resolution and
            smoothing of the coarse-to-fine initialiser.
        initialization_compression_tolerance: Relative coarse TT tolerance.
            The default is the statistical ``max(1e-3, N**-0.5)`` scale, not
            machine precision.

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
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    optimizer = str(optimizer).lower().replace("_", "-")
    allowed_optimizers = (
        "adam", "riemannian", "riemannian-sgd", "kaczmarz", "als"
    )
    if optimizer not in allowed_optimizers:
        raise ValueError(
            "optimizer must be 'adam', 'riemannian', 'riemannian-sgd', "
            "'kaczmarz' or 'als'"
        )
    objective = str(objective).lower().replace("_", "-")
    if objective not in ("l2", "nll"):
        raise ValueError("objective must be 'l2' or 'nll'")
    if floor_mass is not None and objective != "nll":
        raise ValueError("floor_mass requires objective='nll'")
    if optimizer == "kaczmarz" and objective != "nll":
        raise ValueError("Kaczmarz currently supports only objective='nll'")
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
    if patience < 1 or validation_interval < 1 or min_delta < 0.0:
        raise ValueError(
            "patience and validation_interval must be positive and "
            "min_delta non-negative"
        )
    validation_metric = str(validation_metric).lower().replace("_", "-")
    if validation_metric not in ("objective", "log-likelihood"):
        raise ValueError(
            "validation_metric must be 'objective' or 'log-likelihood'"
        )
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("uniform", "product", "mixture", "coarse"):
        raise ValueError(
            "initialization must be 'uniform', 'product', 'mixture' or 'coarse'"
        )
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
            compression_tolerance=initialization_compression_tolerance,
        )
        params = [
            torch.nn.Parameter(
                torch.as_tensor(core, dtype=torch_dtype, device=device)
            )
            for core in initial_root.cores
        ]
    else:
        # A constant requested cap is not attainable close to a boundary when
        # it exceeds the product of the adjacent mode sizes.  Build the maximal
        # feasible tapered profile instead of failing during QR.
        ranks = [1] * (d + 1)
        capacity = 1
        for k in range(1, d):
            capacity = min(int(rank), capacity * int(modes_array[k - 1]))
            ranks[k] = capacity
        capacity = 1
        for k in range(d - 1, 0, -1):
            capacity = min(int(rank), capacity * int(modes_array[k]))
            ranks[k] = min(ranks[k], capacity)
        params = []
        marginal_roots = []
        if initialization in ("product", "mixture"):
            product_indices = _cell_indices(points, modes_array)
            for k, n in enumerate(modes_array):
                counts = np.bincount(
                    product_indices[:, k], minlength=int(n)
                ).astype(np.float64)
                density = (
                    (counts + initialization_pseudocount)
                    / (points.shape[0] + initialization_pseudocount * int(n))
                    * int(n)
                )
                marginal_roots.append(np.sqrt(density))
        for k, n in enumerate(modes_array):
            noise = initialization_noise * torch.randn(
                (ranks[k], int(n), ranks[k + 1]), generator=generator,
                dtype=torch_dtype,
            )
            # A constant rank-one path gives h=1 at initialisation; the small
            # dense perturbation gives all rank directions a gradient.
            if initialization == "product":
                noise[0, :, 0] += torch.as_tensor(
                    marginal_roots[k], dtype=torch_dtype
                )
            elif initialization == "mixture":
                # A rank-one tensor embedded in large zero-padded cores has
                # exponentially vanishing gradients in its unused channels.
                # Keep every channel connected by starting from a CP-like
                # mixture of product roots. Small dense off-diagonal noise can
                # then learn transitions between mixture components with an
                # O(noise), rather than O(noise**d), gradient.
                base = torch.as_tensor(marginal_roots[k], dtype=torch_dtype)
                component_noise = max(2e-2, 5.0 * initialization_noise)
                if k == 0:
                    for right in range(ranks[k + 1]):
                        jitter = torch.exp(component_noise * torch.randn(
                            int(n), generator=generator, dtype=torch_dtype
                        ))
                        noise[0, :, right] += base * jitter / ranks[k + 1]
                elif k == d - 1:
                    for left in range(ranks[k]):
                        jitter = torch.exp(component_noise * torch.randn(
                            int(n), generator=generator, dtype=torch_dtype
                        ))
                        noise[left, :, 0] += base * jitter
                else:
                    for channel in range(min(ranks[k], ranks[k + 1])):
                        jitter = torch.exp(component_noise * torch.randn(
                            int(n), generator=generator, dtype=torch_dtype
                        ))
                        noise[channel, :, channel] += base * jitter
            else:
                noise[0, :, 0] += 1.0
            params.append(torch.nn.Parameter(noise.to(device)))

    # Put the same TT tensor into a mixed-canonical gauge before any loss
    # contraction.  Long products of unbalanced cores otherwise overflow in
    # float32 and make both Euclidean and Riemannian gradients badly scaled.
    # This is a pure gauge transform, performed before optimizer state exists.
    canonical = [parameter.detach().clone() for parameter in params]
    _torch_right_orthogonalize(
        canonical, uniform_measure=(optimizer in ("adam", "kaczmarz"))
    )
    params = [torch.nn.Parameter(core) for core in canonical]
    initial_cores = [parameter.detach().clone() for parameter in params]

    indices_np = _cell_indices(points, modes_array)
    full_indices_np, full_weights_np = _compress_empirical_cells(indices_np)
    full_indices = torch.as_tensor(full_indices_np, dtype=torch.long, device=device)
    full_weights = torch.as_tensor(
        full_weights_np, dtype=torch_dtype, device=device
    )
    validation_indices = validation_weights = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )
        validation_indices_np, validation_weights_np = _compress_empirical_cells(
            _cell_indices(validation, modes_array)
        )
        validation_indices = torch.as_tensor(
            validation_indices_np, dtype=torch.long, device=device
        )
        validation_weights = torch.as_tensor(
            validation_weights_np, dtype=torch_dtype, device=device
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

    def heldout_value(cores) -> float | None:
        if validation_indices is None:
            return None
        with torch.no_grad():
            if validation_metric == "log-likelihood":
                return float(_torch_nll_objective(
                    cores, validation_indices, validation_weights, gamma,
                    floor_mass,
                )[0])
            if objective == "nll":
                return float(_torch_nll_objective(
                    cores, validation_indices, validation_weights, gamma,
                    floor_mass,
                )[0])
            return float(_torch_density_objective(
                cores, validation_indices, validation_weights, gamma
            )[0])

    initial_validation = heldout_value(params)

    if optimizer == "adam":
        history = FitHistory(optimizer="adam")
        adam = torch.optim.Adam(params, lr=learning_rate)
        window = 25
        best_validation = initial_validation
        best_cores = [core.clone() for core in initial_cores]
        stale = 0
        if initial_validation is not None:
            history.validation_loss.append(initial_validation)
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
            if objective == "nll":
                loss, z = _torch_nll_objective(
                    params, batch_indices, batch_weights, gamma, floor_mass
                )
                h2 = None
            else:
                loss, h2, z = _torch_density_objective(
                    params, batch_indices, batch_weights, gamma
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            adam.step()
            history.function_calls += 1

            loss_value = float(loss.detach().cpu())
            h2_value = None if h2 is None else float(h2.detach().cpu())
            z_value = float(z.detach().cpu())
            history.loss.append(loss_value)
            history.normalization.append(z_value)
            if h2_value is not None:
                history.l2_norm_sq.append(h2_value)
                history.chi2_to_reference.append(max(0.0, h2_value - 1.0))
            history.gradient_norm.append(float(grad_norm))
            should_validate = (
                validation_indices is not None
                and (
                    (epoch + 1) % validation_interval == 0
                    or epoch + 1 == epochs
                )
            )
            validation_value = heldout_value(params) if should_validate else None
            if validation_value is not None:
                history.validation_loss.append(validation_value)
                improvement = best_validation - validation_value
                threshold = min_delta * max(1.0, abs(best_validation))
                if improvement > threshold:
                    best_validation = validation_value
                    best_cores = [parameter.detach().clone() for parameter in params]
                    history.best_epoch = epoch + 1
                    stale = 0
                else:
                    stale += 1
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                diagnostic = (
                    "n/a" if h2_value is None
                    else f"{max(0.0, h2_value - 1.0):.3e}"
                )
                print(
                    f"epoch {epoch + 1:4d}: loss={loss_value:.7e}, "
                    f"chi2={diagnostic}, Z={z_value:.3e}"
                )
            if should_validate and stale >= patience:
                history.converged = True
                break
            if validation_value is None and len(history.loss) >= 2 * window:
                old = np.mean(history.loss[-2 * window:-window])
                new = np.mean(history.loss[-window:])
                scale = max(1.0, abs(old), abs(new))
                if abs(new - old) <= tolerance * scale:
                    history.converged = True
                    break
        history.epochs = len(history.loss)
        fitted_cores = (
            best_cores
            if initial_validation is not None
            else [parameter.detach() for parameter in params]
        )

    elif optimizer == "riemannian":
        from tt.algs.autodiff import rgd

        x0 = vector.from_list([parameter.detach().clone() for parameter in params])

        def objective_riemannian(cores):
            if objective == "nll":
                return _torch_nll_objective(
                    cores, indices, weights, gamma, floor_mass
                )[0]
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
            if objective == "nll":
                final_loss, final_z = _torch_nll_objective(
                    list(fitted_root.cores), indices, weights, gamma,
                    floor_mass,
                )
                final_h2 = None
            else:
                final_loss, final_h2, final_z = _torch_density_objective(
                    list(fitted_root.cores), indices, weights, gamma
                )
        if not history.loss or history.loss[-1] != float(final_loss):
            history.loss.append(float(final_loss))
        history.normalization.append(float(final_z))
        if final_h2 is not None:
            history.l2_norm_sq.append(float(final_h2))
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
            floor_mass=floor_mass,
            objective=objective,
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

    elif optimizer == "kaczmarz":
        fitted_cores, history = _fit_tt_block_kaczmarz(
            [parameter.detach() for parameter in params],
            modes=modes_array,
            all_indices_np=indices_np,
            full_indices=full_indices,
            full_weights=full_weights,
            validation_indices=validation_indices,
            validation_weights=validation_weights,
            gamma=gamma,
            floor_mass=floor_mass,
            sweeps=epochs,
            batch_size=batch_size or min(1024, points.shape[0]),
            learning_rate=learning_rate,
            inner_steps=als_inner_steps,
            patience=patience,
            seed=seed,
            verbose=verbose,
        )

    else:
        fitted_cores, history = _fit_tt_als(
            [parameter.detach() for parameter in params],
            modes=modes_array,
            indices=indices,
            weights=weights,
            gamma=gamma,
            floor_mass=floor_mass,
            objective=objective,
            sweeps=epochs,
            inner_steps=als_inner_steps,
            learning_rate=learning_rate,
            tolerance=tolerance,
            verbose=verbose,
        )

    if optimizer != "adam" and initial_validation is not None:
        final_validation = heldout_value(fitted_cores)
        history.validation_loss.extend([initial_validation, final_validation])
        threshold = min_delta * max(1.0, abs(initial_validation))
        if final_validation < initial_validation - threshold:
            history.best_epoch = history.epochs
        else:
            fitted_cores = initial_cores
            history.best_epoch = 0

    history.wall_time = time.perf_counter() - started
    fitted = vector.from_list([
        core.detach().cpu().numpy().copy() for core in fitted_cores
    ])
    fitted_second = _second_moment_numpy(_as_numpy_cores(fitted))
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    return SquaredTTDensity(fitted, gamma=fitted_gamma), history


def _positive_density_from_correction(
    correction: vector,
    *,
    gamma: float,
    tolerance: float,
    rank: int,
    sweeps: int,
    seed: int,
) -> SquaredTTDensity:
    """Project ``1 + correction`` to a positive squared TT density.

    A smooth positive-part function avoids a nonsmooth clipping surface.  Its
    square root is compressed by TT cross and then squared analytically by
    :class:`SquaredTTDensity`, exactly as in SIRT.
    """
    from tt.algs.multifuncrs import multifuncrs2

    ratio = (correction + 1.0).round(
        eps=max(1e-6, 0.25 * tolerance), rmax=rank
    )

    def positive_square_root(values):
        raw = values[:, 0]
        positive = 0.5 * (raw + np.sqrt(raw * raw + 4.0 * gamma))
        return np.sqrt(positive)

    root = multifuncrs2(
        [ratio],
        positive_square_root,
        eps=tolerance,
        nswp=sweeps,
        kickrank=min(5, rank),
        rmax=rank,
        verb=0,
        seed=seed,
    )
    return SquaredTTDensity(root, gamma=gamma)


def _direct_density_from_correction(
    correction: vector,
    *,
    tolerance: float,
    rank: int,
    conditional_floor: float,
) -> DirectTTDensity:
    """Store only ``correction``; add the constant in TT contractions."""
    if int(np.max(correction.r)) > rank:
        correction = correction.round(eps=tolerance, rmax=rank)
    return DirectTTDensity(correction, conditional_floor=conditional_floor)


def _scale_correction_for_positive_conditionals(
    correction: vector,
    points: np.ndarray,
    *,
    margin: float,
) -> tuple[vector, float]:
    """Mix a direct correction with the uniform density until it is positive.

    For ``r = 1 + a`` the convex path ``r_tau = 1 + tau * a`` preserves the
    TT ranks, centering, and exact first/second-moment contractions.  We choose
    the largest ``tau <= 1`` for which every discrete conditional numerator
    encountered along the supplied sample prefixes is at least ``margin``.
    The check includes every possible next cell, not only the observed cell.
    """
    if not 0.0 < margin < 1.0 or not np.isfinite(margin):
        raise ValueError("positivity margin must lie in (0, 1)")
    points = _validate_points(points, correction.d, name="positivity_points")
    cores = _as_numpy_cores(correction)
    modes = correction.n.astype(np.int64)
    indices = _cell_indices(points, modes)
    right = DirectTTDensity._right_mean_environments(cores)
    left = np.ones((points.shape[0], 1), dtype=np.float64)
    minimum = np.inf
    for k, core in enumerate(cores):
        extended = np.einsum("pa,aib->pib", left, core, optimize=True)
        values = np.einsum(
            "pib,b->pi", extended, right[k + 1], optimize=True
        )
        minimum = min(minimum, float(values.min()))
        selected = np.moveaxis(core[:, indices[:, k], :], 1, 0)
        left = np.einsum("pa,pab->pb", left, selected, optimize=True)
    scale = 1.0
    if minimum < margin - 1.0:
        scale = min(1.0, (1.0 - margin) / (-minimum))
    if scale == 1.0:
        return correction, scale
    scaled_cores = [core.copy() for core in cores]
    scaled_cores[0] *= scale
    return vector.from_list(scaled_cores), float(scale)


def fit_centered_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 8,
    gamma: float = 1e-8,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    objective: str = "l2",
    kernel_bandwidths: Sequence[float] | None = None,
    als_relaxation: float = 1.0,
    als_regularization: float = 0.0,
    batch_size: int | None = None,
    validation_samples=None,
    validation_fraction: float = 0.2,
    patience: int = 10,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "auto",
    initial_correction: vector | None = None,
    initialization_coarse_bins: int = 2,
    initialization_pseudocount: float = 0.5,
    initialization_tolerance: float = 5e-2,
    rank_enrichment_noise: float = 1e-3,
    projection_tolerance: float = 1e-2,
    projection_rank: int | None = None,
    projection_sweeps: int = 8,
    representation: str = "direct",
    conditional_floor: float = 1e-12,
    positivity_margin: float | None = None,
    verbose: bool = False,
) -> tuple[DirectTTDensity | SquaredTTDensity, FitHistory]:
    """Fit the centered exact ratio ``a = dnu/dmu - 1`` from samples.

    With ``objective='l2'``, the optimized population objective is

    ``0.5 * E_mu[a**2] - E_nu[a] + E_mu[a]``.

    ``objective='kernel'`` replaces the identity operator by an average of
    separable Gaussian RBF kernels.  Its quadratic and reference terms are
    exact TT contractions of analytic cell integrals; only the target linear
    expectation is estimated from samples.  This is MMD squared between the
    represented density and the target, up to an additive target-only
    constant.  It changes the fitting geometry without adding a coordinate
    map, flow, or density mixture.

    Unlike the normalized squared-TT objective, it is quadratic in the
    represented function and uses only first- and second-order exact TT
    contractions.  A held-out split selects the optimization checkpoint.  The
    fitted layer stores ``1 + a`` directly by default, so no square root or
    fourth-order contraction is introduced.  ``representation='squared'`` is
    retained as a robustness baseline for layers whose fitted ratio is signed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without torch
        raise ImportError("fit_centered_tt_density requires the 'torch' extra") from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("samples must have shape (N, d) with N > 0")
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
    if rank < 1 or (projection_rank is not None and projection_rank < 1):
        raise ValueError("rank and projection_rank must be positive")
    if gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and positive")
    if epochs < 0 or learning_rate < 0.0:
        raise ValueError("epochs and learning_rate must be non-negative")
    optimizer = str(optimizer).lower().replace("_", "-")
    if optimizer not in ("adam", "als"):
        raise ValueError("optimizer must be 'adam' or 'als'")
    objective = str(objective).lower().replace("_", "-")
    if objective not in ("l2", "kernel"):
        raise ValueError("objective must be 'l2' or 'kernel'")
    if kernel_bandwidths is not None:
        kernel_bandwidths = tuple(float(value) for value in kernel_bandwidths)
        if not kernel_bandwidths or any(
            value <= 0.0 or not np.isfinite(value)
            for value in kernel_bandwidths
        ):
            raise ValueError("kernel bandwidths must be finite and positive")
    if objective == "l2" and kernel_bandwidths is not None:
        raise ValueError("kernel bandwidths require objective='kernel'")
    if not 0.0 < als_relaxation <= 1.0:
        raise ValueError("als_relaxation must lie in (0, 1]")
    if als_regularization < 0.0 or not np.isfinite(als_regularization):
        raise ValueError("als_regularization must be finite and non-negative")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if patience < 1 or validation_interval < 1 or min_delta < 0.0:
        raise ValueError(
            "patience and validation_interval must be positive and "
            "min_delta non-negative"
        )
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("auto", "coarse", "product", "uniform"):
        raise ValueError(
            "initialization must be 'auto', 'coarse', 'product' or 'uniform'"
        )
    if rank_enrichment_noise < 0.0 or not np.isfinite(rank_enrichment_noise):
        raise ValueError("rank_enrichment_noise must be finite and non-negative")
    if not 0.0 < projection_tolerance < 1.0:
        raise ValueError("projection_tolerance must lie in (0, 1)")
    if projection_sweeps < 1:
        raise ValueError("projection_sweeps must be positive")
    representation = str(representation).lower().replace("_", "-")
    if representation not in ("direct", "squared"):
        raise ValueError("representation must be 'direct' or 'squared'")
    if conditional_floor <= 0.0 or not np.isfinite(conditional_floor):
        raise ValueError("conditional_floor must be finite and positive")
    if positivity_margin is not None and (
        not 0.0 < positivity_margin < 1.0
        or not np.isfinite(positivity_margin)
    ):
        raise ValueError("positivity_margin must be None or lie in (0, 1)")

    rng = np.random.default_rng(seed)
    if validation_samples is None:
        permutation = rng.permutation(points.shape[0])
        validation_count = max(1, int(round(validation_fraction * points.shape[0])))
        if validation_count >= points.shape[0]:
            raise ValueError("at least two samples are needed for a validation split")
        validation = points[permutation[:validation_count]]
        training = points[permutation[validation_count:]]
    else:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )
        training = points

    if initial_correction is not None:
        if not isinstance(initial_correction, vector):
            raise TypeError("initial_correction must be a tt.vector")
        if initial_correction.d != d or not np.array_equal(
            initial_correction.n.astype(np.int64), modes_array
        ):
            raise ValueError(
                "initial_correction dimension and modes must match the fit"
            )
        if int(np.max(initial_correction.r)) > rank:
            raise ValueError("initial_correction ranks must not exceed rank")
        initial = vector.from_list(_as_numpy_cores(initial_correction))
        initial = (
            initial - _mean_numpy(_as_numpy_cores(initial))
        ).round(eps=initialization_tolerance, rmax=rank)
    else:
        coarse_cells = int(initialization_coarse_bins) ** d
        use_coarse = initialization == "coarse" or (
            initialization == "auto" and coarse_cells <= 1_048_576
        )
    if initial_correction is not None:
        pass
    elif initialization == "uniform":
        # The represented function is the centered correction a=r-1, so the
        # exact uniform-density start is a=0.  Rank enrichment below adds a
        # zero-mean O(noise) tangent TT, retaining the requested optimization
        # channels without multiplying noisy one-dimensional marginals.  The
        # latter can have exponentially large chi2 in high dimension even
        # when the residual density is operationally close to uniform.
        initial = vector.from_list([
            np.zeros((1, int(mode), 1), dtype=np.float64)
            for mode in modes_array
        ])
    elif use_coarse:
        initial = _coarse_to_fine_initial_correction(
            training,
            modes_array,
            rank=rank,
            coarse_bins=initialization_coarse_bins,
            pseudocount=initialization_pseudocount,
            tolerance=initialization_tolerance,
        )
    else:
        initial = _product_marginal_initial_correction(
            training,
            modes_array,
            rank=rank,
            pseudocount=initialization_pseudocount,
            tolerance=initialization_tolerance,
        )
    if initial_correction is None:
        initial = _enrich_correction_ranks(
            initial,
            modes_array,
            rank=rank,
            noise=rank_enrichment_noise,
            seed=seed + 37,
        )
    initial = initial.orthogonalize(center=0)

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    params = [
        torch.nn.Parameter(torch.as_tensor(core, dtype=torch_dtype, device=device))
        for core in _as_numpy_cores(initial)
    ]

    train_indices_np = _cell_indices(training, modes_array)
    train_unique = train_weights = None
    validation_unique = validation_weights = None
    train_kernel_features = validation_kernel_features = None
    kernel_matrices = None
    if objective == "l2":
        train_unique_np, train_weights_np = _compress_empirical_cells(
            train_indices_np
        )
        validation_unique_np, validation_weights_np = _compress_empirical_cells(
            _cell_indices(validation, modes_array)
        )
        train_unique = torch.as_tensor(
            train_unique_np, dtype=torch.long, device=device
        )
        train_weights = torch.as_tensor(
            train_weights_np, dtype=torch_dtype, device=device
        )
        validation_unique = torch.as_tensor(
            validation_unique_np, dtype=torch.long, device=device
        )
        validation_weights = torch.as_tensor(
            validation_weights_np, dtype=torch_dtype, device=device
        )
    else:
        if kernel_bandwidths is None:
            pair_count = min(4096, max(256, len(training)))
            first = rng.integers(0, len(training), size=pair_count)
            second = rng.integers(0, len(training), size=pair_count)
            distances = np.linalg.norm(
                training[first] - training[second], axis=1
            )
            positive = distances[distances > 0.0]
            median = float(np.median(positive)) if len(positive) else 1.0
            kernel_bandwidths = (
                0.5 * median, median, 2.0 * median,
            )
        kernel_matrices = []
        train_kernel_features = []
        validation_kernel_features = []
        for bandwidth in kernel_bandwidths:
            matrix_factors = []
            train_factors = []
            validation_factors = []
            for coordinate, mode in enumerate(modes_array):
                matrix_factors.append(torch.as_tensor(
                    _rbf_cell_kernel_matrix(int(mode), bandwidth),
                    dtype=torch_dtype,
                    device=device,
                ))
                train_factors.append(torch.as_tensor(
                    _rbf_cell_target_features(
                        training[:, coordinate], int(mode), bandwidth
                    ),
                    dtype=torch_dtype,
                    device=device,
                ))
                validation_factors.append(torch.as_tensor(
                    _rbf_cell_target_features(
                        validation[:, coordinate], int(mode), bandwidth
                    ),
                    dtype=torch_dtype,
                    device=device,
                ))
            kernel_matrices.append(matrix_factors)
            train_kernel_features.append(train_factors)
            validation_kernel_features.append(validation_factors)
    full_batch = batch_size is None or batch_size >= training.shape[0]
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if optimizer == "als" and not full_batch:
        raise ValueError("centered ALS requires a deterministic full-batch loss")

    history = FitHistory(
        optimizer=(
            f"centered-{optimizer}"
            if objective == "l2" else f"centered-kernel-{optimizer}"
        )
    )
    history.initial_ranks = initial.r.astype(int).tolist()
    torch_optimizer = (
        torch.optim.Adam(params, lr=learning_rate)
        if optimizer == "adam" else None
    )
    started = time.perf_counter()

    def kernel_value(feature_sets, weights):
        assert kernel_matrices is not None
        return _torch_kernel_correction_objective(
            params, feature_sets, weights, kernel_matrices
        )

    def record() -> tuple[float, float]:
        with torch.no_grad():
            if objective == "l2":
                train_value = _torch_correction_objective(
                    params, train_unique, train_weights
                )[0]
                validation_value = _torch_correction_objective(
                    params, validation_unique, validation_weights
                )[0]
            else:
                train_full_weights = torch.full(
                    (len(training),), 1.0 / len(training),
                    dtype=torch_dtype, device=device,
                )
                validation_full_weights = torch.full(
                    (len(validation),), 1.0 / len(validation),
                    dtype=torch_dtype, device=device,
                )
                train_value = kernel_value(
                    train_kernel_features, train_full_weights
                )[0]
                validation_value = kernel_value(
                    validation_kernel_features, validation_full_weights
                )[0]
            mean, second = _torch_correction_moments(params)
        history.loss.append(float(train_value))
        history.validation_loss.append(float(validation_value))
        history.l2_norm_sq.append(float(second))
        history.normalization.append(1.0 + float(mean))
        return float(train_value), float(validation_value)

    if optimizer == "als":
        params = [parameter.detach() for parameter in params]
        _torch_right_orthogonalize(params, uniform_measure=True)

    _, best_validation = record()
    best_cores = [parameter.detach().cpu().numpy().copy() for parameter in params]
    best_epoch = 0
    stale = 0
    total_cells = float(np.prod(modes_array.astype(np.float64)))
    for epoch in range(int(epochs)):
        if optimizer == "adam":
            if objective == "l2" and full_batch:
                fit_indices, fit_weights = train_unique, train_weights
                fit_kernel_features = None
            elif objective == "kernel" and full_batch:
                fit_indices = None
                fit_weights = torch.full(
                    (len(training),), 1.0 / len(training),
                    dtype=torch_dtype, device=device,
                )
                fit_kernel_features = train_kernel_features
            else:
                chosen = rng.choice(
                    training.shape[0], size=batch_size, replace=False
                )
                if objective == "l2":
                    batch_np, weight_np = _compress_empirical_cells(
                        train_indices_np[chosen]
                    )
                    fit_indices = torch.as_tensor(
                        batch_np, dtype=torch.long, device=device
                    )
                    fit_weights = torch.as_tensor(
                        weight_np, dtype=torch_dtype, device=device
                    )
                    fit_kernel_features = None
                else:
                    chosen_torch = torch.as_tensor(
                        chosen, dtype=torch.long, device=device
                    )
                    fit_indices = None
                    fit_weights = torch.full(
                        (len(chosen),), 1.0 / len(chosen),
                        dtype=torch_dtype, device=device,
                    )
                    fit_kernel_features = [
                        [factor.index_select(0, chosen_torch) for factor in factors]
                        for factors in train_kernel_features
                    ]
            torch_optimizer.zero_grad(set_to_none=True)
            if objective == "l2":
                value = _torch_correction_objective(
                    params, fit_indices, fit_weights
                )[0]
            else:
                value = kernel_value(fit_kernel_features, fit_weights)[0]
            value.backward()
            gradient_norm = float(np.sqrt(sum(
                float(parameter.grad.square().sum()) for parameter in params
            )))
            torch_optimizer.step()
            history.function_calls += 1
        elif objective == "kernel":
            train_full_weights = torch.full(
                (len(training),), 1.0 / len(training),
                dtype=torch_dtype, device=device,
            )
            forward_update = _torch_kernel_als_sweep(
                params,
                train_kernel_features,
                train_full_weights,
                kernel_matrices,
                regularization=als_regularization,
                relaxation=als_relaxation,
                forward=True,
            )
            backward_update = _torch_kernel_als_sweep(
                params,
                train_kernel_features,
                train_full_weights,
                kernel_matrices,
                regularization=als_regularization,
                relaxation=als_relaxation,
                forward=False,
            )
            _torch_right_orthogonalize(params, uniform_measure=True)
            gradient_norm = float(np.hypot(forward_update, backward_update))
            history.function_calls += 2 * d
        else:
            update_norm_sq = 0.0
            for centre in range(d):
                candidate = _torch_centered_als_core(
                    params,
                    train_unique,
                    train_weights,
                    centre,
                    total_cells,
                    als_regularization,
                    uniform_measure=True,
                )
                old = params[centre]
                updated = (
                    (1.0 - als_relaxation) * old
                    + als_relaxation * candidate
                ).detach()
                update_norm_sq += float((updated - old).square().sum())
                params[centre] = updated
                history.function_calls += 1
                if centre + 1 < d:
                    _torch_move_center_right(
                        params, centre, uniform_measure=True
                    )
            if d > 1:
                _torch_move_center_left(
                    params, d - 1, uniform_measure=True
                )
            for centre in range(d - 2, -1, -1):
                candidate = _torch_centered_als_core(
                    params,
                    train_unique,
                    train_weights,
                    centre,
                    total_cells,
                    als_regularization,
                    uniform_measure=True,
                )
                old = params[centre]
                updated = (
                    (1.0 - als_relaxation) * old
                    + als_relaxation * candidate
                ).detach()
                update_norm_sq += float((updated - old).square().sum())
                params[centre] = updated
                history.function_calls += 1
                if centre > 0:
                    _torch_move_center_left(
                        params, centre, uniform_measure=True
                    )
            gradient_norm = float(np.sqrt(update_norm_sq))
        history.gradient_norm.append(gradient_norm)
        history.epochs = epoch + 1
        should_validate = (
            (epoch + 1) % validation_interval == 0
            or epoch + 1 == int(epochs)
        )
        if not should_validate:
            continue
        train_value, validation_value = record()
        improvement = best_validation - validation_value
        threshold = min_delta * max(1.0, abs(best_validation))
        if improvement > threshold:
            best_validation = validation_value
            best_cores = [
                parameter.detach().cpu().numpy().copy() for parameter in params
            ]
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if verbose:
            print(
                f"epoch {epoch + 1:4d}: train={train_value:.7e}, "
                f"validation={validation_value:.7e}, |grad|={gradient_norm:.3e}"
            )
        if stale >= patience:
            history.converged = True
            break

    history.best_epoch = best_epoch
    correction = vector.from_list(best_cores)
    correction = (
        correction - _mean_numpy(_as_numpy_cores(correction))
    ).round(eps=initialization_tolerance, rmax=rank)
    history.correction_ranks = correction.r.astype(int).tolist()

    projection_rank = int(rank if projection_rank is None else projection_rank)
    if representation == "direct":
        if positivity_margin is not None:
            positivity_points = np.vstack([
                training,
                validation,
                rng.random((
                    min(10_000, max(1_000, validation.shape[0])), d
                )),
            ])
            correction, history.positivity_scale = (
                _scale_correction_for_positive_conditionals(
                    correction, positivity_points, margin=positivity_margin
                )
            )
        density = _direct_density_from_correction(
            correction,
            tolerance=projection_tolerance,
            rank=projection_rank,
            conditional_floor=conditional_floor,
        )
    else:
        density = _positive_density_from_correction(
            correction,
            gamma=gamma,
            tolerance=projection_tolerance,
            rank=projection_rank,
            sweeps=projection_sweeps,
            seed=seed + 97,
        )
    history.projected_ranks = density.ranks.astype(int).tolist()
    history.chi2_to_reference.append(density.chi2_to_reference)

    diagnostic = rng.random((min(10_000, max(1_000, validation.shape[0])), d))
    raw_ratio = 1.0 + _sample_tt_numpy(
        _as_numpy_cores(correction), _cell_indices(diagnostic, modes_array)
    )
    projected = density.density(diagnostic)
    history.negative_ratio_fraction = float(np.mean(raw_ratio < 0.0))
    if isinstance(density, DirectTTDensity):
        normalized_ratio = raw_ratio / max(
            float(raw_ratio.mean()), np.finfo(float).tiny
        )
        history.projection_rmse = float(np.sqrt(
            np.mean((normalized_ratio - projected) ** 2)
        ))
        history.conditional_clipping_fraction = (
            density.conditional_clipping_fraction(diagnostic)
        )
    else:
        # This smooth positivity projection belongs only to the explicitly
        # requested legacy squared-density representation.
        positive = 0.5 * (
            raw_ratio + np.sqrt(raw_ratio * raw_ratio + 4.0 * gamma)
        )
        positive /= max(float(positive.mean()), np.finfo(float).tiny)
        history.projection_rmse = float(np.sqrt(
            np.mean((positive - projected) ** 2)
        ))
    history.wall_time = time.perf_counter() - started
    return density, history


def fit_centered_tt_ratio(
    denominator_samples,
    numerator_samples,
    modes: int | Sequence[int] = 16,
    rank: int = 6,
    gamma: float = 1e-8,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int | None = None,
    validation_denominator=None,
    validation_numerator=None,
    validation_fraction: float = 0.2,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization_coarse_bins: int = 2,
    initialization_pseudocount: float = 0.5,
    initialization_tolerance: float = 5e-2,
    ratio_clip: float = 20.0,
    projection_tolerance: float = 1e-2,
    projection_rank: int | None = None,
    projection_sweeps: int = 8,
    representation: str = "direct",
    conditional_floor: float = 1e-12,
    verbose: bool = False,
) -> tuple[DirectTTDensity | SquaredTTDensity, FitHistory]:
    """Fit the nominal centered ratio from denominator/numerator samples.

    The population loss

    ``0.5 E_den[a**2] - E_num[a] + E_den[a]``

    is minimized by ``a = p_num / p_den - 1``.  Applying the same invertible
    current transport to both sample sets leaves this density ratio invariant,
    which makes the routine a sample-only analogue of DIRT's approximate-ratio
    construction.  Previous transport error is not folded into this TT layer.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("fit_centered_tt_ratio requires the 'torch' extra") from exc

    denominator = np.asarray(denominator_samples, dtype=np.float64)
    numerator = np.asarray(numerator_samples, dtype=np.float64)
    if denominator.ndim != 2 or numerator.ndim != 2:
        raise ValueError("sample arrays must have shape (N, d)")
    if denominator.shape[0] == 0 or numerator.shape[0] == 0:
        raise ValueError("both sample arrays must be non-empty")
    d = denominator.shape[1]
    denominator = _validate_points(denominator, d, name="denominator_samples")
    numerator = _validate_points(numerator, d, name="numerator_samples")
    if numerator.shape[1] != d:
        raise ValueError("sample dimensions must agree")
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
        if modes_array.shape != (d,):
            raise ValueError(f"modes must contain {d} entries")
    if np.any(modes_array < 2):
        raise ValueError("each mode must contain at least two cells")
    if rank < 1 or (projection_rank is not None and projection_rank < 1):
        raise ValueError("rank and projection_rank must be positive")
    if gamma <= 0.0 or epochs < 0 or learning_rate < 0.0:
        raise ValueError("gamma must be positive; epochs and rate non-negative")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if patience < 1 or min_delta < 0.0:
        raise ValueError("patience must be positive and min_delta non-negative")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    representation = str(representation).lower().replace("_", "-")
    if representation not in ("direct", "squared"):
        raise ValueError("representation must be 'direct' or 'squared'")
    if conditional_floor <= 0.0 or not np.isfinite(conditional_floor):
        raise ValueError("conditional_floor must be finite and positive")

    rng = np.random.default_rng(seed)
    if validation_denominator is None or validation_numerator is None:
        if (validation_denominator is None) != (validation_numerator is None):
            raise ValueError("provide both validation sample arrays or neither")

        def split(points):
            permutation = rng.permutation(points.shape[0])
            count = max(1, int(round(validation_fraction * points.shape[0])))
            if count >= points.shape[0]:
                raise ValueError("at least two samples are needed per split")
            return points[permutation[count:]], points[permutation[:count]]

        denominator, denominator_validation = split(denominator)
        numerator, numerator_validation = split(numerator)
    else:
        denominator_validation = _validate_points(
            validation_denominator, d, name="validation_denominator"
        )
        numerator_validation = _validate_points(
            validation_numerator, d, name="validation_numerator"
        )

    initial = _coarse_to_fine_initial_ratio_correction(
        denominator,
        numerator,
        modes_array,
        rank=rank,
        coarse_bins=initialization_coarse_bins,
        pseudocount=initialization_pseudocount,
        tolerance=initialization_tolerance,
        ratio_clip=ratio_clip,
    ).orthogonalize(center=0)
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    params = [
        torch.nn.Parameter(torch.as_tensor(core, dtype=torch_dtype, device=device))
        for core in _as_numpy_cores(initial)
    ]

    def compressed(points):
        indices, weights = _compress_empirical_cells(
            _cell_indices(points, modes_array)
        )
        return (
            torch.as_tensor(indices, dtype=torch.long, device=device),
            torch.as_tensor(weights, dtype=torch_dtype, device=device),
        )

    denominator_indices, denominator_weights = compressed(denominator)
    numerator_indices, numerator_weights = compressed(numerator)
    validation_denominator_indices, validation_denominator_weights = compressed(
        denominator_validation
    )
    validation_numerator_indices, validation_numerator_weights = compressed(
        numerator_validation
    )
    denominator_cell_indices = _cell_indices(denominator, modes_array)
    numerator_cell_indices = _cell_indices(numerator, modes_array)
    full_batch = (
        batch_size is None
        or batch_size >= min(denominator.shape[0], numerator.shape[0])
    )

    history = FitHistory(optimizer="centered-ratio-adam")
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    started = time.perf_counter()

    def objective(di, dw, ni, nw):
        return _torch_two_sample_correction_objective(params, di, dw, ni, nw)

    def record():
        with torch.no_grad():
            train_value, second, mean = objective(
                denominator_indices,
                denominator_weights,
                numerator_indices,
                numerator_weights,
            )
            validation_value = objective(
                validation_denominator_indices,
                validation_denominator_weights,
                validation_numerator_indices,
                validation_numerator_weights,
            )[0]
        history.loss.append(float(train_value))
        history.validation_loss.append(float(validation_value))
        history.l2_norm_sq.append(float(second))
        history.normalization.append(1.0 + float(mean))
        return float(train_value), float(validation_value)

    _, best_validation = record()
    best_cores = [parameter.detach().cpu().numpy().copy() for parameter in params]
    best_epoch = 0
    stale = 0
    for epoch in range(int(epochs)):
        if full_batch:
            fit_di, fit_dw = denominator_indices, denominator_weights
            fit_ni, fit_nw = numerator_indices, numerator_weights
        else:
            denominator_choice = rng.choice(
                denominator.shape[0], size=batch_size, replace=False
            )
            numerator_choice = rng.choice(
                numerator.shape[0], size=batch_size, replace=False
            )
            denominator_np, denominator_weight_np = _compress_empirical_cells(
                denominator_cell_indices[denominator_choice]
            )
            numerator_np, numerator_weight_np = _compress_empirical_cells(
                numerator_cell_indices[numerator_choice]
            )
            fit_di = torch.as_tensor(
                denominator_np, dtype=torch.long, device=device
            )
            fit_dw = torch.as_tensor(
                denominator_weight_np, dtype=torch_dtype, device=device
            )
            fit_ni = torch.as_tensor(numerator_np, dtype=torch.long, device=device)
            fit_nw = torch.as_tensor(
                numerator_weight_np, dtype=torch_dtype, device=device
            )
        optimizer.zero_grad(set_to_none=True)
        value = objective(fit_di, fit_dw, fit_ni, fit_nw)[0]
        value.backward()
        gradient_norm = float(np.sqrt(sum(
            float(parameter.grad.square().sum()) for parameter in params
        )))
        optimizer.step()
        history.function_calls += 1
        history.gradient_norm.append(gradient_norm)
        train_value, validation_value = record()
        history.epochs = epoch + 1
        improvement = best_validation - validation_value
        threshold = min_delta * max(1.0, abs(best_validation))
        if improvement > threshold:
            best_validation = validation_value
            best_cores = [
                parameter.detach().cpu().numpy().copy() for parameter in params
            ]
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(
                f"epoch {epoch + 1:4d}: train={train_value:.7e}, "
                f"validation={validation_value:.7e}, |grad|={gradient_norm:.3e}"
            )
        if stale >= patience:
            history.converged = True
            break

    history.best_epoch = best_epoch
    correction = vector.from_list(best_cores)
    denominator_values = _sample_tt_numpy(
        _as_numpy_cores(correction), _cell_indices(denominator, modes_array)
    )
    correction = (
        correction - float(denominator_values.mean())
    ).round(eps=initialization_tolerance, rmax=rank)
    history.correction_ranks = correction.r.astype(int).tolist()
    projection_rank = int(rank if projection_rank is None else projection_rank)
    if representation == "direct":
        density = _direct_density_from_correction(
            correction,
            tolerance=projection_tolerance,
            rank=projection_rank,
            conditional_floor=conditional_floor,
        )
    else:
        density = _positive_density_from_correction(
            correction,
            gamma=gamma,
            tolerance=projection_tolerance,
            rank=projection_rank,
            sweeps=projection_sweeps,
            seed=seed + 97,
        )
    history.projected_ranks = density.ranks.astype(int).tolist()
    history.chi2_to_reference.append(density.chi2_to_reference)

    diagnostic = rng.random((min(10_000, max(1_000, numerator_validation.shape[0])), d))
    raw_ratio = 1.0 + _sample_tt_numpy(
        _as_numpy_cores(correction), _cell_indices(diagnostic, modes_array)
    )
    projected = density.density(diagnostic)
    history.negative_ratio_fraction = float(np.mean(raw_ratio < 0.0))
    if isinstance(density, DirectTTDensity):
        normalized_ratio = raw_ratio / max(
            float(raw_ratio.mean()), np.finfo(float).tiny
        )
        history.projection_rmse = float(np.sqrt(
            np.mean((normalized_ratio - projected) ** 2)
        ))
        history.conditional_clipping_fraction = (
            density.conditional_clipping_fraction(diagnostic)
        )
    else:
        positive = 0.5 * (
            raw_ratio + np.sqrt(raw_ratio * raw_ratio + 4.0 * gamma)
        )
        positive /= max(float(positive.mean()), np.finfo(float).tiny)
        history.projection_rmse = float(np.sqrt(
            np.mean((positive - projected) ** 2)
        ))
    history.wall_time = time.perf_counter() - started
    return density, history

__all__ = [
    "fit_squared_tt_density",
    "_positive_density_from_correction",
    "_direct_density_from_correction",
    "_scale_correction_for_positive_conditionals",
    "fit_centered_tt_density",
    "fit_centered_tt_ratio",
]
