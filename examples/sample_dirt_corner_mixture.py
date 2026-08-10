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
from sample_dirt_common import probit_bridge_samples
from tt.transport import SampleDIRT


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
    """Draw from the corner mixture and map its bounding box to the unit cube."""
    if count < 1:
        raise ValueError("count must be positive")
    if sigma <= 0.0 or tail_sigma <= 0.0:
        raise ValueError("sigma and tail_sigma must be positive")
    rng = np.random.default_rng(seed)
    component = rng.integers(corners.shape[0], size=count)
    raw = corners[component] + sigma * rng.standard_normal((count, corners.shape[1]))
    lower = -tail_sigma * sigma
    width = 1.0 + 2.0 * tail_sigma * sigma
    unit = (raw - lower) / width
    # Only Gaussian tail mass outside the declared bounding box is clipped.
    return np.clip(unit, 0.0, 1.0)


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


def run_method(
    name: str,
    target_train: np.ndarray,
    target_metric: np.ndarray,
    bridges: list[np.ndarray],
    base: np.ndarray,
    selected_codes: np.ndarray,
    args,
) -> tuple[dict, np.ndarray]:
    model = SampleDIRT(target_train.shape[1])
    layers = []
    started = time.perf_counter()
    for layer_index, (alpha, bridge) in enumerate(zip(args.layer_alphas, bridges)):
        options = method_options(name, args)
        options["seed"] = args.seed + layer_index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            history = model.fit_layer(bridge, **options)
        layers.append({
            "layer": layer_index + 1,
            "alpha": alpha,
            "initial_loss": history.loss[0],
            "final_loss": history.loss[-1],
            "iterations": history.epochs,
            "function_calls": history.function_calls,
            "time": history.wall_time,
            "chi2": model.layers[-1].chi2_to_reference,
        })
        print(
            f"{name:15s} layer={layer_index + 1} alpha={alpha:.3f} "
            f"loss={history.loss[-1]:.7f} time={history.wall_time:.3f}s "
            f"calls={history.function_calls}",
            flush=True,
        )
    generated = model.forward(base)
    elapsed = time.perf_counter() - started
    metrics = corner_metrics(generated, selected_codes)
    record = {
        "method": name,
        "total_time": elapsed,
        "swd": sliced_wasserstein(generated, target_metric, seed=args.seed + 31),
        "roundtrip_error": model.roundtrip_error(generated[:100]),
        "stored_parameters": model.stored_parameters,
        **metrics,
        "layers": layers,
    }
    return record, generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=40_000)
    parser.add_argument("--metric-count", type=int, default=20_000)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--tail-sigma", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=1e-5)
    parser.add_argument("--adam-steps", type=int, default=300)
    parser.add_argument("--adam-rate", type=float, default=3e-2)
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
        "--alphas",
        type=float,
        nargs="+",
        default=(0.25, 0.45, 0.65, 0.80, 0.90, 0.97, 1.0),
        help="strictly increasing probit-bridge strengths ending at one",
    )
    parser.add_argument(
        "--refinement-layers",
        type=int,
        default=0,
        help="additional exact-residual layers fitted to target samples",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("adam", "riemannian", "riemannian-sgd", "als"),
        default=("riemannian", "riemannian-sgd", "als"),
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--png-dir", type=Path)
    args = parser.parse_args()
    if (
        not args.alphas
        or any(left >= right for left, right in zip(args.alphas, args.alphas[1:]))
        or args.alphas[-1] != 1.0
    ):
        parser.error("--alphas must increase strictly and end at 1.0")
    if args.refinement_layers < 0:
        parser.error("--refinement-layers must be non-negative")
    args.layer_alphas = list(args.alphas) + [1.0] * args.refinement_layers

    corners, selected_codes = random_corners(
        8, args.components, seed=args.corner_seed
    )
    target_train = sample_corner_mixture(
        args.train_count,
        corners,
        sigma=args.sigma,
        tail_sigma=args.tail_sigma,
        seed=args.seed,
    )
    target_metric = sample_corner_mixture(
        args.metric_count,
        corners,
        sigma=args.sigma,
        tail_sigma=args.tail_sigma,
        seed=args.seed + 1,
    )
    target_metric_2 = sample_corner_mixture(
        args.metric_count,
        corners,
        sigma=args.sigma,
        tail_sigma=args.tail_sigma,
        seed=args.seed + 2,
    )
    bridge_rng = np.random.default_rng(args.seed + 3)
    bridges = [
        probit_bridge_samples(target_train, alpha, bridge_rng)
        for alpha in args.alphas
    ]
    bridges.extend(target_train.copy() for _ in range(args.refinement_layers))
    base = np.random.default_rng(args.seed + 4).random((args.metric_count, 8))

    result = {
        "problem": "8D 128-corner Gaussian mixture Sample-DIRT",
        "paper": "Novikov--Panov--Oseledets, UAI 2021, Section 5.2",
        "train_count": args.train_count,
        "metric_count": args.metric_count,
        "modes": args.modes,
        "rank": args.rank,
        "components": args.components,
        "sigma": args.sigma,
        "tail_sigma": args.tail_sigma,
        "seed": args.seed,
        "corner_seed": args.corner_seed,
        "alphas": args.layer_alphas,
        "target_swd_floor": sliced_wasserstein(
            target_metric, target_metric_2, seed=args.seed + 31
        ),
        "target_corner_metrics": corner_metrics(target_metric, selected_codes),
        "methods": [],
    }
    for name in args.methods:
        record, generated = run_method(
            name,
            target_train,
            target_metric,
            bridges,
            base,
            selected_codes,
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
