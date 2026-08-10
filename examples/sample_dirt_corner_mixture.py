"""Eight-dimensional Sample-DIRT benchmark from the TTDE paper.

Section 5.2 of Novikov, Panov and Oseledets (UAI 2021) considers an
eight-dimensional equal-weight mixture of 128 Gaussians whose means are
randomly selected corners of the unit cube.  The paper does not publish the
random seed or component variance.  This script fixes both explicitly and
maps the four-sigma bounding box affinely to ``[0, 1]^8``, the current domain
of :class:`tt.transport.SampleDIRT`.

The fit receives samples only.  The selected corner set is used exclusively
for diagnostics after training, never by the TT optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

from sample_dirt_2d_gallery import sliced_wasserstein
from sample_dirt_common import (
    probit_bridge_samples,
    reflected_gaussian_bridge_samples,
)
from tt.transport import SampleDIRT, TruncatedGaussianMixture


def random_corners(
    dimension: int,
    component_count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select reproducible distinct binary corners and return their codes."""
    if dimension < 1 or dimension > 62:
        raise ValueError("dimension must lie in [1, 62]")
    available = 1 << dimension
    if component_count < 1 or component_count > available:
        raise ValueError(f"component_count must lie in [1, {available}]")
    rng = np.random.default_rng(seed)
    codes = np.sort(rng.choice(available, size=component_count, replace=False))
    shifts = np.arange(dimension, dtype=np.int64)
    corners = ((codes[:, None] >> shifts[None, :]) & 1).astype(np.float64)
    return corners, codes.astype(np.int64)


def sample_corner_mixture(
    count: int,
    corners: np.ndarray,
    *,
    sigma: float,
    tail_sigma: float,
    seed: int,
) -> np.ndarray:
    """Draw exactly from the box-truncated corner mixture on the unit cube."""
    if count < 1:
        raise ValueError("count must be positive")
    if sigma <= 0.0 or tail_sigma <= 0.0:
        raise ValueError("sigma and tail_sigma must be positive")
    lower = -tail_sigma * sigma
    upper = 1.0 + tail_sigma * sigma
    mixture = TruncatedGaussianMixture(
        corners, sigma, lower=lower, upper=upper
    )
    return mixture.sample(count, seed=seed)


def corner_codes(samples: np.ndarray) -> np.ndarray:
    """Assign samples to their nearest binary corner."""
    bits = np.asarray(samples >= 0.5, dtype=np.int64)
    shifts = 1 << np.arange(samples.shape[1], dtype=np.int64)
    return bits @ shifts


def corner_metrics(samples: np.ndarray, selected_codes: np.ndarray) -> dict:
    """Discrete mode recovery diagnostics after nearest-corner assignment."""
    available = 1 << samples.shape[1]
    counts = np.bincount(corner_codes(samples), minlength=available)
    probabilities = counts / counts.sum()
    target = np.zeros(available, dtype=np.float64)
    target[selected_codes] = 1.0 / selected_codes.size
    selected_mass = float(probabilities[selected_codes].sum())
    occupied_selected = int(np.count_nonzero(counts[selected_codes]))
    spurious = np.setdiff1d(np.flatnonzero(counts), selected_codes).size
    return {
        "corner_tv": 0.5 * float(np.abs(probabilities - target).sum()),
        "selected_corner_mass": selected_mass,
        "selected_corners_occupied": occupied_selected,
        "spurious_corners_occupied": int(spurious),
        "corner_probabilities": probabilities.tolist(),
    }


def method_options(name: str, args) -> dict:
    common = {
        "modes": args.modes,
        "rank": args.rank,
        "gamma": args.gamma,
        "tolerance": args.tolerance,
        "device": args.device,
        "dtype": args.dtype,
        "initialization": args.initialization,
        "initialization_noise": args.initialization_noise,
        "initialization_coarse_bins": args.initialization_coarse_bins,
        "initialization_pseudocount": args.initialization_pseudocount,
    }
    if name == "adam":
        return common | {
            "optimizer": "adam",
            "epochs": args.adam_steps,
            "learning_rate": args.adam_rate,
        }
    if name == "riemannian":
        return common | {
            "optimizer": "riemannian",
            "epochs": args.riemannian_steps,
            "learning_rate": args.riemannian_rate,
            "riemannian_retraction": "psa",
        }
    if name == "riemannian-sgd":
        return common | {
            "optimizer": "riemannian-sgd",
            "epochs": args.riemannian_sgd_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.riemannian_sgd_rate,
            "riemannian_retraction": "psa",
            "riemannian_momentum": args.riemannian_momentum,
            "riemannian_second_moment": args.riemannian_second_moment,
        }
    return common | {
        "optimizer": "als",
        "epochs": args.als_sweeps,
        "learning_rate": 1.0,
        "als_inner_steps": args.als_inner_steps,
    }


def plot_result(
    path: Path,
    target: np.ndarray,
    generated: np.ndarray,
    selected_codes: np.ndarray,
    *,
    method: str,
) -> None:
    """Plot four two-coordinate projections and all discrete corner masses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    figure = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.72))
    shown = min(4_000, target.shape[0], generated.shape[0])
    for column, (first, second) in enumerate(pairs):
        for row, (points, label) in enumerate(
            ((target, "target"), (generated, f"Sample-DIRT ({method})"))
        ):
            axis = figure.add_subplot(grid[row, column])
            axis.scatter(
                points[:shown, first],
                points[:shown, second],
                s=2.2,
                alpha=0.28,
                linewidths=0,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_title(f"{label}: $x_{first + 1},x_{second + 1}$", fontsize=9)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))

    axis = figure.add_subplot(grid[2, :])
    target_mass = np.zeros(256)
    target_mass[selected_codes] = 1.0 / selected_codes.size
    generated_mass = np.bincount(corner_codes(generated), minlength=256)
    generated_mass = generated_mass / generated_mass.sum()
    axis.vlines(np.arange(256), 0.0, target_mass, color="black", alpha=0.3,
                linewidth=1.0, label="target")
    axis.plot(generated_mass, color="#d95f02", linewidth=1.0,
              label="generated")
    axis.set_xlim(-1, 256)
    axis.set_xlabel("binary corner code")
    axis.set_ylabel("probability")
    axis.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_chain_diagnostics(path: Path, record: dict, *, parameter_name: str) -> None:
    """Show whether the learned composition keeps up with every bridge law."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = record["layers"]
    steps = np.arange(1, len(layers) + 1)
    labels = [f"{layer[parameter_name]:g}" for layer in layers]
    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 3.8), constrained_layout=True
    )

    axis = axes[0]
    axis.plot(
        steps,
        [layer["bridge_selected_corner_mass"] for layer in layers],
        "o-",
        color="black",
        linewidth=1.5,
        markersize=3.5,
        label="bridge target",
    )
    axis.plot(
        steps,
        [layer["selected_corner_mass"] for layer in layers],
        "o-",
        color="#d95f02",
        linewidth=1.5,
        markersize=3.5,
        label="learned chain",
    )
    axis.set_ylim(0.45, 1.015)
    axis.set_xlabel("incremental TT layer")
    axis.set_ylabel("mass on the 128 selected corners")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)

    axis = axes[1]
    axis.plot(
        steps,
        [layer["bridge_swd"] for layer in layers],
        "o-",
        linewidth=1.5,
        markersize=3.5,
        label="to current bridge",
    )
    axis.plot(
        steps,
        [layer["target_swd"] for layer in layers],
        "o-",
        linewidth=1.5,
        markersize=3.5,
        label="to final target",
    )
    axis.set_xlabel("incremental TT layer")
    axis.set_ylabel("sliced Wasserstein distance")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)

    tick_step = max(1, len(steps) // 8)
    for axis in axes:
        chosen = steps[::tick_step]
        axis.set_xticks(chosen, [labels[index - 1] for index in chosen])
    axes[0].set_xlabel(f"bridge parameter {parameter_name}")
    axes[1].set_xlabel(f"bridge parameter {parameter_name}")
    figure.suptitle(
        f"Sample-DIRT chain ({record['method']}): residual layers and error propagation",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_method(
    name: str,
    target_train: np.ndarray,
    target_metric: np.ndarray,
    bridges: list[np.ndarray],
    metric_bridges: list[np.ndarray],
    base: np.ndarray,
    selected_codes: np.ndarray,
    oracle: TruncatedGaussianMixture,
    exact_from_base: np.ndarray,
    exact_cell_from_base: np.ndarray,
    args,
) -> tuple[dict, np.ndarray]:
    model = SampleDIRT(target_train.shape[1])
    layers = []
    started = time.perf_counter()
    stage_count = min(args.stage_metric_count, base.shape[0])
    stage_base = base[:stage_count]
    stage_target = target_metric[:stage_count]
    generated = base.copy()
    for layer_index, (bridge_value, bridge, metric_bridge) in enumerate(
        zip(args.layer_values, bridges, metric_bridges)
    ):
        options = method_options(name, args)
        options["seed"] = args.seed + layer_index
        # Held-out points must be pulled back by T_k before the new residual
        # layer is appended.  This evaluates exactly the same L2 objective as
        # training, with an independent estimate of its linear expectation.
        validation_residual = model.inverse(metric_bridge[:stage_count])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            history = model.fit_layer(bridge, **options)
        fitted_layer = model.layers[-1]
        validation_residual_loss = float(
            0.5 * fitted_layer.model_l2_norm_sq()
            - np.mean(fitted_layer.density(validation_residual))
        )
        stage_generated = model.forward(stage_base)
        stage_metrics = corner_metrics(stage_generated, selected_codes)
        bridge_metrics = corner_metrics(
            metric_bridge[:stage_count], selected_codes
        )
        layer_record = {
            "layer": layer_index + 1,
            "bridge_parameter": bridge_value,
            args.bridge_parameter_name: bridge_value,
            "initial_loss": history.loss[0],
            "final_loss": history.loss[-1],
            "validation_residual_loss": validation_residual_loss,
            "residual_generalization_gap": (
                validation_residual_loss - history.loss[-1]
            ),
            "iterations": history.epochs,
            "function_calls": history.function_calls,
            "time": history.wall_time,
            "chi2": model.layers[-1].chi2_to_reference,
            "tt_ranks": model.layers[-1].ranks.tolist(),
            "maximum_tt_rank": int(model.layers[-1].ranks.max()),
            "stored_parameters": model.stored_parameters,
            "bridge_swd": sliced_wasserstein(
                stage_generated,
                metric_bridge[:stage_count],
                seed=args.seed + 31,
            ),
            "target_swd": sliced_wasserstein(
                stage_generated,
                stage_target,
                seed=args.seed + 31,
            ),
            "estimated_kl_target_model": float(
                np.mean(
                    oracle.log_density(stage_target)
                    - model.log_density(stage_target)
                )
            ),
            "bridge_selected_corner_mass": bridge_metrics[
                "selected_corner_mass"
            ],
            "bridge_corner_tv": bridge_metrics["corner_tv"],
            **stage_metrics,
        }
        layers.append(layer_record)
        print(
            f"{name:15s} layer={layer_index + 1} "
            f"{args.bridge_parameter_name}={bridge_value:.4f} "
            f"loss={history.loss[-1]:.7f} time={history.wall_time:.3f}s "
            f"bridge-SWD={layer_record['bridge_swd']:.6f} "
            f"target-SWD={layer_record['target_swd']:.6f} "
            f"selected-mass={layer_record['selected_corner_mass']:.4f} "
            f"calls={history.function_calls}",
            flush=True,
        )
    generated = model.forward(base)
    elapsed = time.perf_counter() - started
    metrics = corner_metrics(generated, selected_codes)
    record = {
        "method": name,
        "total_time": elapsed,
        "training_time": float(sum(layer["time"] for layer in layers)),
        "swd": sliced_wasserstein(generated, target_metric, seed=args.seed + 31),
        "roundtrip_error": model.roundtrip_error(generated[:100]),
        "analytic_transport_rmse": float(
            np.sqrt(np.mean((generated - exact_from_base) ** 2))
        ),
        "cell_oracle_transport_rmse": float(
            np.sqrt(np.mean((generated - exact_cell_from_base) ** 2))
        ),
        "estimated_kl_target_model": float(
            np.mean(
                oracle.log_density(target_metric)
                - model.log_density(target_metric)
            )
        ),
        "stored_parameters": model.stored_parameters,
        **metrics,
        "layers": layers,
    }
    return record, generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=40_000)
    parser.add_argument("--metric-count", type=int, default=20_000)
    parser.add_argument(
        "--stage-metric-count",
        type=int,
        default=2_000,
        help="prefix used for diagnostics after each bridge layer",
    )
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--tail-sigma", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=1e-8)
    parser.add_argument(
        "--initialization",
        choices=("uniform", "coarse"),
        default="coarse",
    )
    parser.add_argument("--initialization-noise", type=float, default=2e-2)
    parser.add_argument("--initialization-coarse-bins", type=int, default=2)
    parser.add_argument("--initialization-pseudocount", type=float, default=0.1)
    parser.add_argument(
        "--adam-steps",
        type=int,
        default=1,
        help=(
            "refinement steps after the coarse estimator; use held-out metrics "
            "before increasing this because the empirical L2 loss can overfit"
        ),
    )
    parser.add_argument("--adam-rate", type=float, default=3e-3)
    parser.add_argument("--riemannian-steps", type=int, default=100)
    parser.add_argument("--riemannian-rate", type=float, default=1e-1)
    parser.add_argument("--riemannian-sgd-steps", type=int, default=300)
    parser.add_argument("--riemannian-sgd-rate", type=float, default=5e-1)
    parser.add_argument("--riemannian-momentum", type=float, default=0.0)
    parser.add_argument("--riemannian-second-moment", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--als-sweeps", type=int, default=8)
    parser.add_argument("--als-inner-steps", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--corner-seed", type=int, default=20089)
    parser.add_argument(
        "--bridge",
        choices=("probit", "reflected"),
        default="probit",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.60, 1.0),
        help=(
            "strictly increasing probit-bridge strengths ending at one; "
            "the default is the validated two-layer Sample-DIRT chain"
        ),
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=(1.0, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15,
                 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0),
        help="strictly decreasing reflected-diffusion noise ending at zero",
    )
    parser.add_argument(
        "--refinement-layers",
        type=int,
        default=0,
        help="additional exact-residual layers fitted to target samples",
    )
    parser.add_argument(
        "--corrections-per-alpha",
        type=int,
        default=1,
        help="low-rank residual corrections fitted at every bridge level",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("adam", "riemannian", "riemannian-sgd", "als"),
        default=("adam",),
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--png-dir", type=Path)
    args = parser.parse_args()
    if args.bridge == "probit":
        if (
            not args.alphas
            or any(
                left >= right
                for left, right in zip(args.alphas, args.alphas[1:])
            )
            or args.alphas[-1] != 1.0
        ):
            parser.error("--alphas must increase strictly and end at 1.0")
        base_values = list(args.alphas)
        args.bridge_parameter_name = "alpha"
    else:
        if (
            not args.noise_levels
            or any(
                left <= right
                for left, right in zip(
                    args.noise_levels, args.noise_levels[1:]
                )
            )
            or args.noise_levels[-1] != 0.0
        ):
            parser.error(
                "--noise-levels must decrease strictly and end at 0.0"
            )
        base_values = list(args.noise_levels)
        args.bridge_parameter_name = "noise_std"
    if args.refinement_layers < 0:
        parser.error("--refinement-layers must be non-negative")
    if args.corrections_per_alpha < 1:
        parser.error("--corrections-per-alpha must be positive")
    if args.stage_metric_count < 1:
        parser.error("--stage-metric-count must be positive")
    args.layer_values = [
        value
        for value in base_values
        for _ in range(args.corrections_per_alpha)
    ] + [base_values[-1]] * args.refinement_layers

    corners, selected_codes = random_corners(
        8, args.components, seed=args.corner_seed
    )
    oracle = TruncatedGaussianMixture(
        corners,
        args.sigma,
        lower=-args.tail_sigma * args.sigma,
        upper=1.0 + args.tail_sigma * args.sigma,
    )
    target_train = oracle.sample(args.train_count, seed=args.seed)
    target_metric = oracle.sample(args.metric_count, seed=args.seed + 1)
    target_metric_2 = oracle.sample(args.metric_count, seed=args.seed + 2)
    bridge_rng = np.random.default_rng(args.seed + 3)
    bridge_function = (
        probit_bridge_samples
        if args.bridge == "probit"
        else reflected_gaussian_bridge_samples
    )
    bridges = []
    for value in base_values:
        bridge = bridge_function(target_train, value, bridge_rng)
        bridges.extend(
            bridge.copy() for _ in range(args.corrections_per_alpha)
        )
    bridges.extend(target_train.copy() for _ in range(args.refinement_layers))
    metric_bridge_rng = np.random.default_rng(args.seed + 103)
    metric_bridges = []
    for value in base_values:
        metric_bridge = bridge_function(
            target_metric, value, metric_bridge_rng
        )
        metric_bridges.extend(
            metric_bridge.copy() for _ in range(args.corrections_per_alpha)
        )
    metric_bridges.extend(
        target_metric.copy() for _ in range(args.refinement_layers)
    )
    base = np.random.default_rng(args.seed + 4).random((args.metric_count, 8))
    exact_from_base = oracle.inverse_rosenblatt(base)
    exact_cell_from_base = oracle.inverse_cell_rosenblatt(base, args.modes)

    result = {
        "problem": "8D 128-corner Gaussian mixture Sample-DIRT",
        "paper": "Novikov--Panov--Oseledets, UAI 2021, Section 5.2",
        "train_count": args.train_count,
        "unique_target_sample_count": args.train_count,
        "per_method_bridge_sample_exposures": (
            args.train_count * len(args.layer_values)
        ),
        "metric_count": args.metric_count,
        "modes": args.modes,
        "rank": args.rank,
        "components": args.components,
        "sigma": args.sigma,
        "tail_sigma": args.tail_sigma,
        "initialization": args.initialization,
        "bridge": args.bridge,
        "bridge_parameter_name": args.bridge_parameter_name,
        "seed": args.seed,
        "corner_seed": args.corner_seed,
        "bridge_values": args.layer_values,
        "target_swd_floor": sliced_wasserstein(
            target_metric, target_metric_2, seed=args.seed + 31
        ),
        "target_corner_metrics": corner_metrics(target_metric, selected_codes),
        "analytic_roundtrip_error": oracle.roundtrip_error(target_metric[:100]),
        "analytic_corner_metrics": corner_metrics(
            exact_from_base, selected_codes
        ),
        "grid_kl_floor": float(
            np.mean(
                oracle.log_density(target_metric)
                - oracle.cell_log_density(target_metric, args.modes)
            )
        ),
        "grid_transport_rmse": float(
            np.sqrt(np.mean((exact_cell_from_base - exact_from_base) ** 2))
        ),
        "grid_corner_metrics": corner_metrics(
            exact_cell_from_base, selected_codes
        ),
        "methods": [],
    }
    for name in args.methods:
        record, generated = run_method(
            name,
            target_train,
            target_metric,
            bridges,
            metric_bridges,
            base,
            selected_codes,
            oracle,
            exact_from_base,
            exact_cell_from_base,
            args,
        )
        result["methods"].append(record)
        if args.png_dir:
            plot_result(
                args.png_dir / f"corner_mixture_{name}.png",
                target_metric,
                generated,
                selected_codes,
                method=name,
            )
            plot_chain_diagnostics(
                args.png_dir / f"corner_mixture_{name}_stages.png",
                record,
                parameter_name=args.bridge_parameter_name,
            )

    print("\nSummary")
    print(f"target-vs-target SWD floor: {result['target_swd_floor']:.6f}")
    for method in result["methods"]:
        print(
            f"{method['method']:15s} time={method['total_time']:.3f}s "
            f"SWD={method['swd']:.6f} corner-TV={method['corner_tv']:.4f} "
            f"selected-mass={method['selected_corner_mass']:.4f}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
