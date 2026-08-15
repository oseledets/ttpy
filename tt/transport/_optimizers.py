"""Generic fixed-rank TT optimization drivers."""

from __future__ import annotations

import time

import numpy as np

from ._torch_als import (
    _compress_empirical_cells,
    _torch_als_local_objective,
    _torch_fourth_interfaces,
    _torch_kaczmarz_local_nll,
    _torch_move_center_left,
    _torch_move_center_right,
    _torch_right_orthogonalize,
    _torch_sample_interfaces,
)
from ._torch_density import _torch_density_objective, _torch_nll_objective
from ._types import FitHistory

def _fit_tt_block_kaczmarz(
    initial_cores,
    *,
    modes,
    all_indices_np,
    full_indices,
    full_weights,
    validation_indices,
    validation_weights,
    gamma,
    floor_mass,
    sweeps,
    batch_size,
    learning_rate,
    inner_steps,
    patience,
    seed,
    verbose,
):
    """Stochastic mass-orthogonal ALS/Kaczmarz sweeps for squared-TT NLL."""
    import torch

    cores = [core.detach().clone() for core in initial_cores]
    _torch_right_orthogonalize(cores, uniform_measure=True)
    history = FitHistory(optimizer="kaczmarz")
    rng = np.random.default_rng(seed + 37)
    count = len(all_indices_np)
    batch_size = min(int(batch_size), count)
    proximal_weight = 1.0 / max(float(learning_rate), 1e-12)
    best_validation = float("inf")
    best_cores = [core.clone() for core in cores]
    stale = 0
    started = time.perf_counter()

    def batch_data():
        chosen = rng.choice(count, size=batch_size, replace=False)
        batch_np, weight_np = _compress_empirical_cells(
            all_indices_np[chosen]
        )
        return (
            torch.as_tensor(
                batch_np, dtype=torch.long, device=cores[0].device
            ),
            torch.as_tensor(
                weight_np, dtype=cores[0].dtype, device=cores[0].device
            ),
        )

    def update(centre: int, indices, weights):
        sample_left, sample_right = _torch_sample_interfaces(
            cores, indices, centre
        )
        anchor = cores[centre].detach().clone()
        parameter = torch.nn.Parameter(anchor.clone())
        local = torch.optim.LBFGS(
            [parameter],
            lr=1.0,
            max_iter=int(inner_steps),
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            local.zero_grad(set_to_none=True)
            value = _torch_kaczmarz_local_nll(
                parameter,
                centre=centre,
                mode=int(modes[centre]),
                indices=indices,
                weights=weights,
                gamma=gamma,
                floor_mass=floor_mass,
                sample_left=sample_left,
                sample_right=sample_right,
                anchor=anchor,
                proximal_weight=proximal_weight,
            )
            value.backward()
            history.function_calls += 1
            return value

        local.step(closure)
        cores[centre] = parameter.detach()

    for sweep in range(int(sweeps)):
        # A Kaczmarz block defines one coherent local projection.  Reuse it
        # throughout the complete forward/backward core sweep; resampling at
        # every core would instead be noisy coordinate SGD with a moving
        # objective.
        block_indices, block_weights = batch_data()
        for centre in range(len(cores)):
            update(centre, block_indices, block_weights)
            if centre < len(cores) - 1:
                _torch_move_center_right(
                    cores, centre, uniform_measure=True
                )
        for old_centre in range(len(cores) - 1, 0, -1):
            _torch_move_center_left(
                cores, old_centre, uniform_measure=True
            )
            update(old_centre - 1, block_indices, block_weights)

        with torch.no_grad():
            # Keep the stochastic method independent of the full training
            # count inside a sweep.  The block NLL has the exact TT
            # normalization and only its empirical log term is stochastic.
            loss, normalization = _torch_nll_objective(
                cores, block_indices, block_weights, gamma, floor_mass
            )
            history.loss.append(float(loss))
            history.normalization.append(float(normalization))
            if validation_indices is None:
                validation = float(loss)
                # Losses from different random blocks are not comparable for
                # checkpoint selection.  Without a fixed validation set the
                # stochastic iterate itself is the checkpoint.
                best_cores = [core.clone() for core in cores]
                history.best_epoch = sweep + 1
                stale = 0
            else:
                validation = float(_torch_nll_objective(
                    cores, validation_indices, validation_weights, gamma,
                    floor_mass,
                )[0])
                history.validation_loss.append(validation)
        if validation_indices is not None:
            if validation < best_validation - 1e-5:
                best_validation = validation
                best_cores = [core.clone() for core in cores]
                history.best_epoch = sweep + 1
                stale = 0
            else:
                stale += 1
        if verbose:
            print(
                f"sweep {sweep + 1:3d}: nll={float(loss):.7e}, "
                f"validation={validation:.7e}"
            )
        if validation_indices is not None and stale >= patience:
            history.converged = True
            break

    history.epochs = len(history.loss)
    with torch.no_grad():
        final_loss, final_normalization = _torch_nll_objective(
            best_cores, full_indices, full_weights, gamma, floor_mass
        )
    history.loss.append(float(final_loss))
    history.normalization.append(float(final_normalization))
    history.wall_time = time.perf_counter() - started
    return best_cores, history


def _fit_tt_als(
    initial_cores,
    *,
    modes,
    indices,
    weights,
    gamma,
    floor_mass,
    objective,
    sweeps,
    inner_steps,
    learning_rate,
    tolerance,
    verbose,
):
    """Orthogonal nonlinear ALS for the squared-density objective.

    One core is minimized by L-BFGS while every fixed-side contraction is
    cached.  QR moves the mixed canonical centre between sites and reduces the
    exact second moment to a local Frobenius norm.  The Pearson-L2 objective
    additionally caches fourth-moment environments; NLL needs only sample
    interfaces and the exact second moment.
    """
    import torch

    cores = [core.detach().clone() for core in initial_cores]
    uniform_measure = objective == "nll"
    _torch_right_orthogonalize(
        cores, uniform_measure=uniform_measure
    )
    # Keep the product in floating point: n**d overflows int64 already for
    # modest tabular problems (4**43, 8**21), while the normalized Frobenius
    # factor is perfectly representable in float64.
    total_cells = float(np.prod(np.asarray(modes, dtype=np.float64)))
    history = FitHistory(optimizer="als")
    started = time.perf_counter()

    def global_record() -> float:
        with torch.no_grad():
            if objective == "nll":
                loss, z = _torch_nll_objective(
                    cores, indices, weights, gamma, floor_mass
                )
                h2 = None
            else:
                loss, h2, z = _torch_density_objective(
                    cores, indices, weights, gamma
                )
        lv, zv = float(loss), float(z)
        history.loss.append(lv)
        history.normalization.append(zv)
        if h2 is not None:
            h2v = float(h2)
            history.l2_norm_sq.append(h2v)
            history.chi2_to_reference.append(max(0.0, h2v - 1.0))
        return lv

    def local_value(
        parameter, centre, sample_left, sample_right,
        fourth_left=None, fourth_right=None,
    ):
        if objective == "nll":
            return _torch_kaczmarz_local_nll(
                parameter,
                centre=centre,
                mode=int(modes[centre]),
                indices=indices,
                weights=weights,
                gamma=gamma,
                floor_mass=floor_mass,
                sample_left=sample_left,
                sample_right=sample_right,
                anchor=parameter.detach(),
                proximal_weight=0.0,
            )
        return _torch_als_local_objective(
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
            fourth_left = fourth_right = None
            if objective == "l2":
                fourth_left, fourth_right = _torch_fourth_interfaces(
                    cores, centre
                )
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
                value = local_value(
                    parameter, centre, sample_left, sample_right,
                    fourth_left, fourth_right,
                )
                value.backward()
                history.function_calls += 1
                return value

            local.step(closure)
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("ALS produced a non-finite TT core")
            cores[centre] = parameter.detach()
            if centre < len(cores) - 1:
                _torch_move_center_right(
                    cores, centre, uniform_measure=uniform_measure
                )

        for old_centre in reverse_centres:
            _torch_move_center_left(
                cores, old_centre, uniform_measure=uniform_measure
            )
            centre = old_centre - 1
            sample_left, sample_right = _torch_sample_interfaces(
                cores, indices, centre
            )
            fourth_left = fourth_right = None
            if objective == "l2":
                fourth_left, fourth_right = _torch_fourth_interfaces(
                    cores, centre
                )
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
                value = local_value(
                    parameter, centre, sample_left, sample_right,
                    fourth_left, fourth_right,
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
                if history.chi2_to_reference else
                f"sweep {sweep + 1:4d}: loss={current:.7e}, "
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
    floor_mass,
    objective,
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

    from tt.algs.autodiff import riemannian_grad
    from tt.algs.riemannian import (retract, tangent_inner, tangent_to_tt,
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
            if objective == "nll":
                return _torch_nll_objective(
                    cores, batch_indices, batch_weights, gamma, floor_mass
                )[0]
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
        if objective == "nll":
            final_loss, final_z = _torch_nll_objective(
                list(x.cores), full_indices, full_weights, gamma, floor_mass
            )
            final_h2 = None
        else:
            final_loss, final_h2, final_z = _torch_density_objective(
                list(x.cores), full_indices, full_weights, gamma
            )
    history.loss.append(float(final_loss))
    history.normalization.append(float(final_z))
    if final_h2 is not None:
        history.l2_norm_sq.append(float(final_h2))
        history.chi2_to_reference.append(max(0.0, float(final_h2) - 1.0))
    history.function_calls += 1
    history.epochs = int(iterations)
    history.wall_time = time.perf_counter() - started
    return x, history

__all__ = [
    "_fit_tt_block_kaczmarz",
    "_fit_tt_als",
    "_fit_tt_riemannian_stochastic",
]
