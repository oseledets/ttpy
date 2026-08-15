"""Fitters for linear, adaptive, and orthogonally rotated TT roots."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from tt.core.vector import vector

from ._basis import (
    _adaptive_linear_right_gram_environments,
    _as_numpy_cores,
    _linear_right_gram_environments,
    _validate_points,
)
from ._torch_als import (
    _torch_linear_right_orthogonalize,
    _torch_right_orthogonalize,
)
from ._torch_density import (
    _torch_adaptive_knots,
    _torch_adaptive_linear_root_second_moment,
    _torch_block_orthogonal_rotations,
    _torch_linear_from_mass_cores,
    _torch_linear_mass_factors,
    _torch_linear_root_second_moment,
    _torch_linear_to_mass_cores,
    _torch_probit_orthogonal_map,
    _torch_sample_adaptive_linear_tt,
    _torch_sample_linear_tt,
    _torch_tt_frobenius_sq,
)
from ._types import FitHistory
from .coordinates import ProbitOrthogonalTTDensity
from .scalar import AdaptiveLinearSquaredTTDensity, LinearSquaredTTDensity

def fit_linear_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 32,
    rank: int = 4,
    gamma: float = 1e-5,
    floor_mass: float | None = None,
    epochs: int = 500,
    learning_rate: float = 3e-3,
    batch_size: int = 4096,
    validation_samples=None,
    mixture_background_log_ratio=None,
    validation_mixture_background_log_ratio=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "mixture",
    initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    optimizer: str = "adam",
    orthogonalization_interval: int = 25,
    riemannian_retraction: str = "psa",
    riemannian_momentum: float = 0.9,
    riemannian_second_moment: float = 0.99,
    verbose: bool = False,
) -> tuple[LinearSquaredTTDensity, FitHistory]:
    """Fit a continuous linear-basis squared TT by exact-normalization NLL.

    If ``floor_mass`` is supplied, the fitted density is parameterized as

    ``floor_mass + (1-floor_mass) * root**2 / integral(root**2)``.

    This is invariant to the arbitrary scale of the TT root and gives a true
    lower bound relative to the uniform reference.  A fixed ``gamma`` does
    not: its normalized weight is ``gamma / (gamma + integral(root**2))``.

    ``tail_weight > 0`` adds a minibatch CVaR term over the lowest-density
    ``tail_fraction`` of observations.  Checkpoint selection deliberately
    remains ordinary held-out mean NLL, so tail emphasis can help escape
    density holes but cannot silently change the reported KL objective.

    ``mixture_background_log_ratio`` switches the empirical term to the
    exact fixed-background component-replacement objective.  If ``b_i`` is
    ``log(m_{-j}(x_i) / (pi_j q_j(x_i)))`` and ``s`` is this normalized local
    TT density, the optimized pointwise log likelihood, up to a constant, is

    ``log(exp(b_i) + s(z_i))``.

    This directly optimizes all TT contractions for their contribution inside
    a mixture instead of fitting an EM surrogate or a stand-alone density.

    ``orthogonal-adam`` and ``riemannian-adam`` work in mass-orthonormal
    coefficient coordinates.  If ``M = L L.T`` is the exact hat-basis Gram
    matrix and ``C`` are nodal coefficients, these coordinates are ``A=C L``.
    Consequently the Euclidean TT metric is exactly the functional ``L2``
    metric, including the boundary hats.  Orthogonal Adam periodically restores
    a canonical TT gauge; Riemannian Adam keeps a transported tangent first
    moment and one gauge-invariant second moment per tangent block.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_linear_squared_tt_density requires the 'torch' extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    optimizer = str(optimizer).lower().replace("_", "-")
    if (
        rank < 1 or gamma <= 0.0 or epochs < 1 or batch_size < 1
        or validation_interval < 1 or orthogonalization_interval < 1
    ):
        raise ValueError("invalid rank, gamma, epochs or batch_size")
    if optimizer not in ("adam", "orthogonal-adam", "riemannian-adam"):
        raise ValueError(
            "linear squared optimizer must be adam, orthogonal-adam or "
            "riemannian-adam"
        )
    if riemannian_retraction not in ("svd", "psa"):
        raise ValueError("riemannian_retraction must be 'svd' or 'psa'")
    if not 0.0 <= riemannian_momentum < 1.0:
        raise ValueError("riemannian_momentum must lie in [0, 1)")
    if not 0.0 <= riemannian_second_moment < 1.0:
        raise ValueError("riemannian_second_moment must lie in [0, 1)")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1] and tail_weight in [0, 1]")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    initialization = str(initialization).lower().replace("_", "-")
    if initialization not in ("uniform", "product", "mixture"):
        raise ValueError("linear basis supports uniform, product or mixture initialization")
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )
    background = None
    if mixture_background_log_ratio is not None:
        background = np.asarray(
            mixture_background_log_ratio, dtype=np.float64
        )
        if background.shape != (len(points),) or np.any(np.isnan(background)):
            raise ValueError(
                "mixture_background_log_ratio must match samples and contain "
                "no NaNs"
            )
    validation_background = None
    if validation_mixture_background_log_ratio is not None:
        validation_background = np.asarray(
            validation_mixture_background_log_ratio, dtype=np.float64
        )
        if (
            validation is None
            or validation_background.shape != (len(validation),)
            or np.any(np.isnan(validation_background))
        ):
            raise ValueError(
                "validation_mixture_background_log_ratio must match "
                "validation_samples and contain no NaNs"
            )
    if (background is None) != (validation_background is None):
        raise ValueError(
            "training and validation mixture background ratios must be "
            "provided together"
        )

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ranks = [1] * (d + 1)
    capacity = 1
    for k in range(1, d):
        capacity = min(int(rank), capacity * int(modes_array[k - 1]))
        ranks[k] = capacity
    capacity = 1
    for k in range(d - 1, 0, -1):
        capacity = min(int(rank), capacity * int(modes_array[k]))
        ranks[k] = min(ranks[k], capacity)

    marginal_roots = []
    if initialization in ("product", "mixture"):
        for k, n_value in enumerate(modes_array):
            n = int(n_value)
            counts = np.histogram(
                points[:, k], bins=n - 1, range=(0.0, 1.0)
            )[0].astype(np.float64)
            cell = (
                counts + initialization_pseudocount
            ) / (
                len(points) + initialization_pseudocount * (n - 1)
            ) * (n - 1)
            nodal = np.empty(n, dtype=np.float64)
            nodal[[0, -1]] = cell[[0, -1]]
            if n > 2:
                nodal[1:-1] = 0.5 * (cell[:-1] + cell[1:])
            integral = (
                0.5 * nodal[0] + nodal[1:-1].sum() + 0.5 * nodal[-1]
            ) / (n - 1)
            marginal_roots.append(np.sqrt(nodal / integral))

    params = []
    for k, n_value in enumerate(modes_array):
        n = int(n_value)
        core = initialization_noise * torch.randn(
            (ranks[k], n, ranks[k + 1]),
            generator=generator, dtype=torch_dtype,
        )
        if initialization == "uniform":
            core[0, :, 0] += 1.0
        elif initialization == "product":
            core[0, :, 0] += torch.as_tensor(
                marginal_roots[k], dtype=torch_dtype
            )
        else:
            base = torch.as_tensor(marginal_roots[k], dtype=torch_dtype)
            component_noise = max(2e-2, 5.0 * initialization_noise)
            if k == 0:
                for right in range(ranks[k + 1]):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[0, :, right] += base * jitter / ranks[k + 1]
            elif k == d - 1:
                for left in range(ranks[k]):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[left, :, 0] += base * jitter
            else:
                for channel in range(min(ranks[k], ranks[k + 1])):
                    jitter = torch.exp(component_noise * torch.randn(
                        n, generator=generator, dtype=torch_dtype
                    ))
                    core[channel, :, channel] += base * jitter
        params.append(torch.nn.Parameter(core.to(device)))

    mass_coordinates = optimizer != "adam"
    factors, inverse_factors = _torch_linear_mass_factors(params)
    canonical = [parameter.detach().clone() for parameter in params]
    if mass_coordinates:
        canonical = _torch_linear_to_mass_cores(canonical, factors)
        _torch_right_orthogonalize(canonical)
    else:
        _torch_linear_right_orthogonalize(canonical)
    params = [torch.nn.Parameter(core) for core in canonical]
    initial_cores = [parameter.detach().clone() for parameter in params]
    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )
    background_tensor = (
        None if background is None else torch.as_tensor(
            background, dtype=torch_dtype, device=device
        )
    )
    validation_background_tensor = (
        None if validation_background is None else torch.as_tensor(
            validation_background, dtype=torch_dtype, device=device
        )
    )

    def nodal_cores(cores):
        if not mass_coordinates:
            return cores
        return _torch_linear_from_mass_cores(cores, inverse_factors)

    def root_second_moment(cores):
        if mass_coordinates:
            return _torch_tt_frobenius_sq(cores)
        return _torch_linear_root_second_moment(cores)

    def nll(cores, batch, batch_background=None, *, tail_aware: bool):
        second = root_second_moment(cores)
        values = _torch_sample_linear_tt(nodal_cores(cores), batch)
        if floor_mass is None:
            normalization = gamma + second
            point_nll = (
                torch.log(normalization)
                - torch.log(gamma + values.square())
            )
        else:
            normalized_square = values.square() / torch.clamp_min(
                second, torch.finfo(second.dtype).tiny
            )
            density = (
                floor_mass + (1.0 - floor_mass) * normalized_square
            )
            normalization = second
            point_nll = -torch.log(density)
        if batch_background is not None:
            log_density = -point_nll
            point_nll = -torch.logaddexp(batch_background, log_density)
        empirical = point_nll.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(point_nll.numel())))
            )
            tail = torch.topk(point_nll, tail_count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization

    def heldout(cores):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(
                cores,
                validation_tensor,
                validation_background_tensor,
                tail_aware=False,
            )[0])

    history = FitHistory(optimizer=f"linear-{optimizer}")
    rng = np.random.default_rng(seed + 1)
    best_validation = heldout(params)
    patience_validation = best_validation
    best_cores = initial_cores
    stale = 0
    if best_validation is not None:
        history.validation_loss.append(best_validation)
    started = time.perf_counter()
    count = len(points)
    batch_size = min(int(batch_size), count)

    def training_batch():
        if batch_size == count:
            return training_tensor, background_tensor
        chosen = rng.choice(count, size=batch_size, replace=False)
        chosen_tensor = torch.as_tensor(
            chosen, dtype=torch.long, device=device
        )
        batch = training_tensor[chosen_tensor]
        batch_background = (
            None if background_tensor is None
            else background_tensor[chosen_tensor]
        )
        return batch, batch_background

    def record_validation(epoch, cores):
        nonlocal best_validation, patience_validation, best_cores, stale
        check_validation = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0 or epoch + 1 == epochs)
        )
        validation_value = heldout(cores) if check_validation else None
        if check_validation:
            history.validation_loss.append(validation_value)
            if validation_value < best_validation:
                best_validation = validation_value
                best_cores = [p.detach().clone() for p in cores]
                history.best_epoch = epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - validation_value > threshold:
                patience_validation = validation_value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                return True
        return False

    if optimizer in ("adam", "orthogonal-adam"):
        adam_options = {}
        if torch.device(device).type == "cuda":
            # The fused implementation removes dozens of tiny per-core CUDA
            # launches; it has identical Adam semantics.
            adam_options["fused"] = True
        try:
            adam = torch.optim.Adam(
                params, lr=learning_rate, **adam_options
            )
        except TypeError:  # older torch without the fused keyword
            adam = torch.optim.Adam(params, lr=learning_rate)
        for epoch in range(int(epochs)):
            batch, batch_background = training_batch()
            adam.zero_grad(set_to_none=True)
            loss, normalization = nll(
                params, batch, batch_background, tail_aware=True
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(params, 100.0)
            adam.step()
            if (
                optimizer == "orthogonal-adam"
                and (epoch + 1) % int(orthogonalization_interval) == 0
            ):
                canonical = [parameter.detach().clone() for parameter in params]
                _torch_right_orthogonalize(canonical)
                with torch.no_grad():
                    for parameter, core in zip(params, canonical):
                        parameter.copy_(core)
                # A dense gauge change has no elementwise transformation law
                # for Adam's second moment.  A warm restart is the exact,
                # representation-independent choice.
                adam.state.clear()
            history.function_calls += 1
            history.loss.append(float(loss.detach()))
            history.normalization.append(float(normalization.detach()))
            history.gradient_norm.append(float(gradient_norm))
            if record_validation(epoch, params):
                break
            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(
                    f"epoch {epoch + 1:5d}: linear_nll={float(loss):.7e}, "
                    f"Z={float(normalization):.4e}"
                )
        terminal_cores = [p.detach() for p in params]
    else:
        from tt.algs.autodiff import riemannian_grad
        from tt.algs.riemannian import (
            frames, project_delta, retract, tangent_inner, tangent_to_tt,
        )

        point = vector.from_list([p.detach().clone() for p in params])
        point_frames = frames(point, check_rank=False)
        first_moment = None
        second_moments = [0.0] * d
        beta1 = float(riemannian_momentum)
        beta2 = float(riemannian_second_moment)
        epsilon = 1e-8 if torch_dtype == torch.float32 else 1e-12
        for epoch in range(int(epochs)):
            batch, batch_background = training_batch()

            def minibatch_objective(cores):
                return nll(
                    cores, batch, batch_background, tail_aware=True
                )[0]

            loss_value, gradient, gradient_frames = riemannian_grad(
                minibatch_objective,
                point,
                runtime_check=False,
                frames_=point_frames,
            )
            gradient_norm_sq = float(tangent_inner(gradient, gradient))
            if first_moment is None:
                first_moment = [
                    (1.0 - beta1) * core for core in gradient
                ]
            else:
                first_moment = [
                    beta1 * old + (1.0 - beta1) * new
                    for old, new in zip(first_moment, gradient)
                ]
            direction_cores = []
            step_number = epoch + 1
            first_correction = 1.0 - beta1 ** step_number
            second_correction = 1.0 - beta2 ** step_number
            for k, (moment, grad) in enumerate(zip(first_moment, gradient)):
                block_mean_square = float(
                    torch.mean(grad.square()).detach()
                )
                second_moments[k] = (
                    beta2 * second_moments[k]
                    + (1.0 - beta2) * block_mean_square
                )
                denominator = (
                    second_moments[k] / second_correction
                ) ** 0.5 + epsilon
                direction_cores.append(
                    -learning_rate * moment / first_correction / denominator
                )
            tangent_step = tangent_to_tt(
                point, direction_cores, frames_=gradient_frames
            )
            new_point = retract(
                point, tangent_step, method=riemannian_retraction
            )
            new_frames = frames(new_point, check_rank=False)
            if beta1 > 0.0:
                old_moment_tt = tangent_to_tt(
                    point, first_moment, frames_=gradient_frames
                )
                first_moment, _ = project_delta(
                    new_point, old_moment_tt, frames_=new_frames
                )
            else:
                first_moment = None
            point = new_point
            point_frames = new_frames
            with torch.no_grad():
                normalization = root_second_moment(list(point.cores))
            history.function_calls += 1
            history.loss.append(float(loss_value))
            history.normalization.append(float(normalization))
            history.gradient_norm.append(gradient_norm_sq ** 0.5)
            if record_validation(epoch, list(point.cores)):
                break
            if verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
                print(
                    f"epoch {epoch + 1:5d}: linear_nll={loss_value:.7e}, "
                    f"|grad|={gradient_norm_sq ** 0.5:.3e}"
                )
        terminal_cores = [core.detach() for core in point.cores]

    history.epochs = len(history.loss)
    history.wall_time = time.perf_counter() - started
    fitted_cores = (
        best_cores if validation_tensor is not None
        else terminal_cores
    )
    if mass_coordinates:
        fitted_cores = _torch_linear_from_mass_cores(
            fitted_cores, inverse_factors
        )
    fitted = vector.from_list([
        core.cpu().numpy().copy() for core in fitted_cores
    ])
    fitted_second = float(_linear_right_gram_environments(
        _as_numpy_cores(fitted)
    )[0][0, 0])
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    return LinearSquaredTTDensity(fitted, gamma=fitted_gamma), history


def fit_adaptive_linear_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 16,
    rank: int = 4,
    gamma: float = 1e-5,
    floor_mass: float | None = None,
    epochs: int = 250,
    warm_start_epochs: int = 500,
    learning_rate: float = 1e-3,
    knot_learning_rate: float = 1e-4,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    knot_initialization: str = "uniform",
    optimize_knots: bool = True,
    minimum_knot_width: float = 1e-3,
    initialization: str = "mixture",
    warm_start_optimizer: str = "adam",
    initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    orthogonalization_interval: int = 25,
    verbose: bool = False,
) -> tuple[AdaptiveLinearSquaredTTDensity, FitHistory]:
    """Fit a squared TT jointly with its one-dimensional hat knots.

    A conventional uniform-grid :func:`fit_linear_squared_tt_density` model is
    fitted first.  With held-out samples that exact model is checkpoint zero,
    so the additional knot optimization has a strict no-regression fallback.
    Knot intervals are ``minimum_knot_width + c * softmax(logits)`` and hence
    stay ordered without projection.  Both the normalizer and its gradients
    use the exact nonuniform hat Gram matrices.

    ``knot_initialization="quantile"`` starts from empirical marginal
    quantiles and resamples the warm TT core functions at those nodes.
    ``optimize_knots=False`` gives the corresponding fixed-quantile basis
    control while still fine-tuning every TT contraction.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_adaptive_linear_squared_tt_density requires the 'torch' extra"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    d = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(d, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (d,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, d, name="validation_samples"
        )
    knot_initialization = str(knot_initialization).lower().replace("_", "-")
    if knot_initialization not in ("uniform", "quantile"):
        raise ValueError("knot_initialization must be uniform or quantile")
    warm_start_optimizer = str(warm_start_optimizer).lower().replace("_", "-")
    if warm_start_optimizer not in ("adam", "orthogonal-adam"):
        raise ValueError(
            "warm_start_optimizer must be adam or orthogonal-adam"
        )
    if (
        rank < 1 or gamma <= 0.0 or epochs < 1 or warm_start_epochs < 1
        or batch_size < 1 or patience < 1 or validation_interval < 1
        or learning_rate <= 0.0 or knot_learning_rate <= 0.0
        or orthogonalization_interval < 1
    ):
        raise ValueError("invalid adaptive linear optimizer option")
    interval_counts = modes_array - 1
    if (
        not np.isfinite(minimum_knot_width)
        or minimum_knot_width < 0.0
        or np.any(minimum_knot_width * interval_counts >= 1.0)
    ):
        raise ValueError(
            "minimum_knot_width must be nonnegative and leave free interval mass"
        )
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0,1] and tail_weight in [0,1]")

    started = time.perf_counter()
    warm, warm_history = fit_linear_squared_tt_density(
        points,
        modes=modes_array,
        rank=rank,
        gamma=gamma,
        floor_mass=floor_mass,
        epochs=warm_start_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_samples=validation,
        patience=patience,
        validation_interval=validation_interval,
        min_delta=min_delta,
        seed=seed,
        device=device,
        dtype=dtype,
        initialization=initialization,
        initialization_noise=initialization_noise,
        initialization_pseudocount=initialization_pseudocount,
        tail_fraction=tail_fraction,
        tail_weight=tail_weight,
        optimizer=warm_start_optimizer,
        orthogonalization_interval=orthogonalization_interval,
        verbose=verbose,
    )
    uniform_knots = [
        np.linspace(0.0, 1.0, int(mode), dtype=np.float64)
        for mode in modes_array
    ]
    initial_knots = [value.copy() for value in uniform_knots]
    initial_cores = [core.copy() for core in warm._cores]
    interval_probabilities = [
        np.full(int(mode) - 1, 1.0 / (int(mode) - 1), dtype=np.float64)
        for mode in modes_array
    ]
    if knot_initialization == "quantile":
        for k, mode_value in enumerate(modes_array):
            mode = int(mode_value)
            raw_nodes = np.quantile(
                points[:, k], np.linspace(0.0, 1.0, mode)
            )
            raw_nodes[[0, -1]] = (0.0, 1.0)
            raw_widths = np.maximum(np.diff(raw_nodes), 1e-12)
            probabilities = raw_widths / raw_widths.sum()
            available = 1.0 - (mode - 1) * minimum_knot_width
            widths = minimum_knot_width + available * probabilities
            nodes = np.concatenate(([0.0], np.cumsum(widths)))
            nodes[-1] = 1.0
            initial_knots[k] = nodes
            interval_probabilities[k] = probabilities

            # Resample every matrix-valued warm core function.  This is the
            # natural nodal prolongation; checkpoint zero below still retains
            # the exact uniform model in case this initialization is worse.
            core = warm._cores[k]
            position = nodes * (mode - 1)
            lower = np.minimum(
                np.floor(position).astype(np.int64), mode - 2
            )
            fraction = np.clip(position - lower, 0.0, 1.0)
            initial_cores[k] = (
                (1.0 - fraction)[None, :, None] * core[:, lower, :]
                + fraction[None, :, None] * core[:, lower + 1, :]
            )

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    core_parameters = [
        torch.nn.Parameter(torch.as_tensor(
            core, dtype=torch_dtype, device=device
        ))
        for core in initial_cores
    ]
    knot_logits = [
        torch.nn.Parameter(torch.as_tensor(
            np.log(probabilities), dtype=torch_dtype, device=device
        ))
        for probabilities in interval_probabilities
    ]
    fixed_knots = [
        torch.as_tensor(value, dtype=torch_dtype, device=device)
        for value in initial_knots
    ]
    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )

    def current_knots():
        if optimize_knots:
            return _torch_adaptive_knots(knot_logits, minimum_knot_width)
        return fixed_knots

    def nll(cores, knots, batch, *, tail_aware):
        second = _torch_adaptive_linear_root_second_moment(cores, knots)
        values = _torch_sample_adaptive_linear_tt(cores, knots, batch)
        if floor_mass is None:
            normalization = gamma + second
            point_nll = (
                torch.log(normalization)
                - torch.log(gamma + values.square())
            )
        else:
            density = floor_mass + (1.0 - floor_mass) * (
                values.square()
                / torch.clamp_min(second, torch.finfo(second.dtype).tiny)
            )
            normalization = second
            point_nll = -torch.log(density)
        empirical = point_nll.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(point_nll.numel())))
            )
            tail = torch.topk(point_nll, tail_count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization

    def heldout(cores, knots):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(
                cores, knots, validation_tensor, tail_aware=False
            )[0])

    history = FitHistory(
        optimizer=(
            "linear-adaptive-knots-adam[uniform-warm-start]"
            if optimize_knots else
            "linear-fixed-adaptive-adam[uniform-warm-start]"
        )
    )
    history.function_calls = warm_history.function_calls
    uniform_core_tensors = [
        torch.as_tensor(core, dtype=torch_dtype, device=device)
        for core in warm._cores
    ]
    uniform_knot_tensors = [
        torch.as_tensor(value, dtype=torch_dtype, device=device)
        for value in uniform_knots
    ]
    best_validation = heldout(uniform_core_tensors, uniform_knot_tensors)
    best_cores = [core.detach().clone() for core in uniform_core_tensors]
    best_knots = [value.detach().clone() for value in uniform_knot_tensors]
    best_epoch = int(warm_history.best_epoch)
    if best_validation is not None:
        history.validation_loss.append(best_validation)
        initialized_validation = heldout(core_parameters, current_knots())
        history.validation_loss.append(initialized_validation)
        if initialized_validation < best_validation:
            best_validation = initialized_validation
            best_cores = [core.detach().clone() for core in core_parameters]
            best_knots = [value.detach().clone() for value in current_knots()]
            best_epoch = int(warm_history.epochs)

    parameter_groups = [{"params": core_parameters, "lr": learning_rate}]
    if optimize_knots:
        parameter_groups.append({
            "params": knot_logits, "lr": knot_learning_rate
        })
    adam_options = {}
    if torch.device(device).type == "cuda":
        adam_options["fused"] = True
    try:
        optimizer = torch.optim.Adam(parameter_groups, **adam_options)
    except TypeError:  # older torch without fused Adam
        optimizer = torch.optim.Adam(parameter_groups)

    rng = np.random.default_rng(seed + 17)
    count = len(points)
    batch_size = min(int(batch_size), count)
    stale = 0
    patience_validation = best_validation
    terminal_knots = current_knots()
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            chosen = torch.as_tensor(
                rng.choice(count, size=batch_size, replace=False),
                dtype=torch.long, device=device,
            )
            batch = training_tensor[chosen]
        optimizer.zero_grad(set_to_none=True)
        knots = current_knots()
        loss, normalization = nll(
            core_parameters, knots, batch, tail_aware=True
        )
        loss.backward()
        optimized_parameters = core_parameters + (
            knot_logits if optimize_knots else []
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            optimized_parameters, 100.0
        )
        optimizer.step()
        if (epoch + 1) % int(orthogonalization_interval) == 0:
            canonical = [core.detach().clone() for core in core_parameters]
            _torch_right_orthogonalize(canonical)
            with torch.no_grad():
                for parameter, core in zip(core_parameters, canonical):
                    parameter.copy_(core)
            # The gauge change is exact for the represented function, but
            # elementwise Adam moments have no gauge-covariant transform.
            optimizer.state.clear()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(normalization.detach()))
        history.gradient_norm.append(float(gradient_norm))
        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            knots = current_knots()
            value = heldout(core_parameters, knots)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_cores = [core.detach().clone() for core in core_parameters]
                best_knots = [node.detach().clone() for node in knots]
                best_epoch = int(warm_history.epochs) + epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - value > threshold:
                patience_validation = value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                terminal_knots = knots
                break
        terminal_knots = current_knots()
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            minimum_width = min(
                float(torch.min(node[1:] - node[:-1]).detach())
                for node in current_knots()
            )
            print(
                f"epoch {epoch + 1:5d}: adaptive_nll={float(loss):.7e}, "
                f"Z={float(normalization):.4e}, min_width={minimum_width:.3e}"
            )

    history.epochs = int(warm_history.epochs) + len(history.loss)
    history.best_epoch = best_epoch
    history.wall_time = time.perf_counter() - started
    if validation_tensor is None:
        best_cores = [core.detach().clone() for core in core_parameters]
        best_knots = [node.detach().clone() for node in terminal_knots]
    fitted = vector.from_list([
        core.cpu().numpy().copy() for core in best_cores
    ])
    fitted_knots = [
        node.cpu().numpy().copy() for node in best_knots
    ]
    fitted_second = float(_adaptive_linear_right_gram_environments(
        _as_numpy_cores(fitted), fitted_knots
    )[0][0, 0])
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    return AdaptiveLinearSquaredTTDensity(
        fitted, fitted_knots, gamma=fitted_gamma
    ), history


def fit_probit_rotated_linear_squared_tt_density(
    samples,
    modes: int | Sequence[int] = 8,
    rank: int = 8,
    gamma: float = 1e-5,
    floor_mass: float | None = None,
    epochs: int = 500,
    warm_start_epochs: int = 1000,
    learning_rate: float = 1e-3,
    rotation_learning_rate: float = 1e-3,
    rotation_block_size: int = 4,
    optimize_cores: bool = True,
    core_refinement_epochs: int = 0,
    core_refinement_learning_rate: float | None = None,
    batch_size: int = 4096,
    validation_samples=None,
    patience: int = 25,
    validation_interval: int = 1,
    min_delta: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float64",
    initialization: str = "mixture",
    warm_start_optimizer: str = "adam",
    initialization_noise: float = 1e-2,
    initialization_pseudocount: float = 0.5,
    tail_fraction: float = 1.0,
    tail_weight: float = 0.0,
    orthogonalization_interval: int = 25,
    verbose: bool = False,
) -> tuple[ProbitOrthogonalTTDensity, FitHistory]:
    """Jointly fit a scalar TT and exact block-orthogonal cube coordinates.

    The coordinate layer is ``Phi(Phi^-1(u) R)`` with block-diagonal
    orthogonal ``R``.  It is exactly volume preserving, so the scalar TT Gram
    normalizer is unchanged.  Identity rotation plus the independently fitted
    scalar TT is retained as held-out checkpoint zero.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fit_probit_rotated_linear_squared_tt_density requires torch"
        ) from exc

    points = np.asarray(samples, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("samples must be a non-empty matrix")
    points = _validate_points(points, points.shape[1], name="samples")
    dimension = points.shape[1]
    if isinstance(modes, (int, np.integer)):
        modes_array = np.full(dimension, int(modes), dtype=np.int64)
    else:
        modes_array = np.asarray(list(modes), dtype=np.int64)
    if modes_array.shape != (dimension,) or np.any(modes_array < 2):
        raise ValueError("modes must provide at least two nodes per dimension")
    validation = None
    if validation_samples is not None:
        validation = _validate_points(
            validation_samples, dimension, name="validation_samples"
        )
    warm_start_optimizer = str(warm_start_optimizer).lower().replace("_", "-")
    if warm_start_optimizer not in ("adam", "orthogonal-adam"):
        raise ValueError(
            "warm_start_optimizer must be adam or orthogonal-adam"
        )
    if (
        rank < 1 or gamma <= 0.0 or epochs < 1 or warm_start_epochs < 1
        or learning_rate <= 0.0 or rotation_learning_rate <= 0.0
        or rotation_block_size < 2 or batch_size < 1 or patience < 1
        or core_refinement_epochs < 0
        or validation_interval < 1 or orthogonalization_interval < 1
    ):
        raise ValueError("invalid probit-rotation optimizer option")
    if core_refinement_learning_rate is None:
        core_refinement_learning_rate = learning_rate
    if core_refinement_learning_rate <= 0.0:
        raise ValueError("core_refinement_learning_rate must be positive")
    if floor_mass is not None and not 0.0 < floor_mass < 1.0:
        raise ValueError("floor_mass must lie strictly between zero and one")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if not 0.0 < tail_fraction <= 1.0 or not 0.0 <= tail_weight <= 1.0:
        raise ValueError("tail_fraction must be in (0,1] and tail_weight in [0,1]")

    started = time.perf_counter()
    warm, warm_history = fit_linear_squared_tt_density(
        points,
        modes=modes_array,
        rank=rank,
        gamma=gamma,
        floor_mass=floor_mass,
        epochs=warm_start_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_samples=validation,
        patience=patience,
        validation_interval=validation_interval,
        min_delta=min_delta,
        seed=seed,
        device=device,
        dtype=dtype,
        initialization=initialization,
        initialization_noise=initialization_noise,
        initialization_pseudocount=initialization_pseudocount,
        tail_fraction=tail_fraction,
        tail_weight=tail_weight,
        optimizer=warm_start_optimizer,
        orthogonalization_interval=orthogonalization_interval,
        verbose=verbose,
    )
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    core_parameters = [
        torch.nn.Parameter(
            torch.as_tensor(core, dtype=torch_dtype, device=device),
            requires_grad=bool(optimize_cores),
        )
        for core in warm._cores
    ]
    block_sizes = []
    remaining = dimension
    while remaining:
        size = min(int(rotation_block_size), remaining)
        block_sizes.append(size)
        remaining -= size
    angle_parameters = [
        torch.nn.Parameter(torch.zeros(
            (size * (size - 1) // 2,),
            dtype=torch_dtype, device=device,
        ))
        for size in block_sizes
    ]
    rotation_parameter_count = int(sum(
        size * (size - 1) // 2 for size in block_sizes
    ))
    training_tensor = torch.as_tensor(
        points, dtype=torch_dtype, device=device
    )
    validation_tensor = (
        None if validation is None else torch.as_tensor(
            validation, dtype=torch_dtype, device=device
        )
    )

    def rotation():
        return _torch_block_orthogonal_rotations(
            angle_parameters, block_sizes
        )

    def nll(cores, matrix, batch, *, tail_aware):
        transformed = _torch_probit_orthogonal_map(batch, matrix)
        second = _torch_linear_root_second_moment(cores)
        values = _torch_sample_linear_tt(cores, transformed)
        if floor_mass is None:
            normalization = gamma + second
            point_nll = (
                torch.log(normalization)
                - torch.log(gamma + values.square())
            )
        else:
            density = floor_mass + (1.0 - floor_mass) * (
                values.square()
                / torch.clamp_min(second, torch.finfo(second.dtype).tiny)
            )
            normalization = second
            point_nll = -torch.log(density)
        empirical = point_nll.mean()
        if tail_aware and tail_weight > 0.0 and tail_fraction < 1.0:
            tail_count = max(
                1, int(np.ceil(tail_fraction * int(point_nll.numel())))
            )
            tail = torch.topk(point_nll, tail_count, sorted=False).values.mean()
            empirical = (1.0 - tail_weight) * empirical + tail_weight * tail
        return empirical, normalization

    identity = torch.eye(
        dimension, dtype=torch_dtype, device=device
    )

    def heldout(cores, matrix):
        if validation_tensor is None:
            return None
        with torch.no_grad():
            return float(nll(
                cores, matrix, validation_tensor, tail_aware=False
            )[0])

    history = FitHistory(
        optimizer="linear-probit-block-orthogonal-adam[scalar-warm-start]"
    )
    history.function_calls = warm_history.function_calls
    # Record the actual number of independent skew coordinates for audits.
    history.initial_ranks = [rotation_parameter_count]
    best_validation = heldout(core_parameters, identity)
    best_cores = [core.detach().clone() for core in core_parameters]
    best_rotation = identity.detach().clone()
    best_epoch = int(warm_history.best_epoch)
    patience_validation = best_validation
    if best_validation is not None:
        history.validation_loss.append(best_validation)

    groups = []
    if optimize_cores:
        groups.append({"params": core_parameters, "lr": learning_rate})
    groups.append({
        "params": angle_parameters, "lr": rotation_learning_rate
    })
    adam_options = {}
    if torch.device(device).type == "cuda":
        adam_options["fused"] = True
    try:
        optimizer = torch.optim.Adam(groups, **adam_options)
    except TypeError:  # older torch without fused Adam
        optimizer = torch.optim.Adam(groups)
    rng = np.random.default_rng(seed + 31)
    count = len(points)
    batch_size = min(int(batch_size), count)
    stale = 0
    terminal_rotation = identity
    for epoch in range(int(epochs)):
        if batch_size == count:
            batch = training_tensor
        else:
            chosen = torch.as_tensor(
                rng.choice(count, size=batch_size, replace=False),
                dtype=torch.long, device=device,
            )
            batch = training_tensor[chosen]
        optimizer.zero_grad(set_to_none=True)
        matrix = rotation()
        loss, normalization = nll(
            core_parameters, matrix, batch, tail_aware=True
        )
        loss.backward()
        optimized_parameters = (
            core_parameters if optimize_cores else []
        ) + angle_parameters
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            optimized_parameters, 100.0
        )
        optimizer.step()
        if (
            optimize_cores
            and (epoch + 1) % int(orthogonalization_interval) == 0
        ):
            canonical = [core.detach().clone() for core in core_parameters]
            _torch_linear_right_orthogonalize(canonical)
            with torch.no_grad():
                for parameter, core in zip(core_parameters, canonical):
                    parameter.copy_(core)
            optimizer.state.clear()
        history.function_calls += 1
        history.loss.append(float(loss.detach()))
        history.normalization.append(float(normalization.detach()))
        history.gradient_norm.append(float(gradient_norm))
        check = (
            validation_tensor is not None
            and ((epoch + 1) % int(validation_interval) == 0
                 or epoch + 1 == epochs)
        )
        if check:
            matrix = rotation()
            value = heldout(core_parameters, matrix)
            history.validation_loss.append(value)
            if value < best_validation:
                best_validation = value
                best_cores = [
                    core.detach().clone() for core in core_parameters
                ]
                best_rotation = matrix.detach().clone()
                best_epoch = int(warm_history.epochs) + epoch + 1
            threshold = min_delta * max(1.0, abs(patience_validation))
            if patience_validation - value > threshold:
                patience_validation = value
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                history.converged = True
                terminal_rotation = matrix
                break
        terminal_rotation = rotation()
        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            distance = float(torch.linalg.norm(
                terminal_rotation - identity
            ).detach())
            print(
                f"epoch {epoch + 1:5d}: rotated_nll={float(loss):.7e}, "
                f"rotation_distance={distance:.3e}"
            )

    rotation_steps = len(history.loss)
    if core_refinement_epochs > 0:
        # Alternating minimization: freeze the selected volume-preserving
        # coordinates, then let every TT contraction readapt.  The best
        # rotation-only checkpoint remains the fallback.
        fixed_rotation = best_rotation.detach().clone()
        refinement_cores = [
            torch.nn.Parameter(core.detach().clone()) for core in best_cores
        ]
        try:
            refinement_optimizer = torch.optim.Adam(
                refinement_cores,
                lr=float(core_refinement_learning_rate),
                **adam_options,
            )
        except TypeError:
            refinement_optimizer = torch.optim.Adam(
                refinement_cores, lr=float(core_refinement_learning_rate)
            )
        stale = 0
        patience_validation = best_validation
        for refinement_epoch in range(int(core_refinement_epochs)):
            if batch_size == count:
                batch = training_tensor
            else:
                chosen = torch.as_tensor(
                    rng.choice(count, size=batch_size, replace=False),
                    dtype=torch.long, device=device,
                )
                batch = training_tensor[chosen]
            refinement_optimizer.zero_grad(set_to_none=True)
            loss, normalization = nll(
                refinement_cores, fixed_rotation, batch, tail_aware=True
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                refinement_cores, 100.0
            )
            refinement_optimizer.step()
            if (
                (refinement_epoch + 1)
                % int(orthogonalization_interval) == 0
            ):
                canonical = [
                    core.detach().clone() for core in refinement_cores
                ]
                _torch_linear_right_orthogonalize(canonical)
                with torch.no_grad():
                    for parameter, core in zip(refinement_cores, canonical):
                        parameter.copy_(core)
                refinement_optimizer.state.clear()
            history.function_calls += 1
            history.loss.append(float(loss.detach()))
            history.normalization.append(float(normalization.detach()))
            history.gradient_norm.append(float(gradient_norm))
            check = (
                validation_tensor is not None
                and ((refinement_epoch + 1) % int(validation_interval) == 0
                     or refinement_epoch + 1 == core_refinement_epochs)
            )
            if check:
                value = heldout(refinement_cores, fixed_rotation)
                history.validation_loss.append(value)
                if value < best_validation:
                    best_validation = value
                    best_cores = [
                        core.detach().clone() for core in refinement_cores
                    ]
                    best_rotation = fixed_rotation
                    best_epoch = (
                        int(warm_history.epochs) + rotation_steps
                        + refinement_epoch + 1
                    )
                threshold = min_delta * max(
                    1.0, abs(patience_validation)
                )
                if patience_validation - value > threshold:
                    patience_validation = value
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    history.converged = True
                    break

    history.epochs = int(warm_history.epochs) + len(history.loss)
    history.best_epoch = best_epoch
    history.wall_time = time.perf_counter() - started
    if validation_tensor is None:
        best_cores = [core.detach().clone() for core in core_parameters]
        best_rotation = terminal_rotation.detach().clone()
    fitted = vector.from_list([
        core.cpu().numpy().copy() for core in best_cores
    ])
    fitted_second = float(_linear_right_gram_environments(
        _as_numpy_cores(fitted)
    )[0][0, 0])
    fitted_gamma = (
        gamma if floor_mass is None
        else floor_mass * fitted_second / (1.0 - floor_mass)
    )
    base = LinearSquaredTTDensity(fitted, gamma=fitted_gamma)
    rotation_numpy = best_rotation.cpu().numpy().astype(np.float64)
    # Float32 matrix_exp is orthogonal only to float32 precision.  The polar
    # factor restores the exact model invariant before serialization without
    # introducing any additional parameter or likelihood Jacobian.
    left_singular, _, right_singular = np.linalg.svd(rotation_numpy)
    rotation_numpy = left_singular @ right_singular
    return ProbitOrthogonalTTDensity(
        base, rotation_numpy,
        rotation_block_sizes=block_sizes,
    ), history

__all__ = [
    "fit_linear_squared_tt_density",
    "fit_adaptive_linear_squared_tt_density",
    "fit_probit_rotated_linear_squared_tt_density",
]
