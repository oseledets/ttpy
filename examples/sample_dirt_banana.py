"""Learn a sample-only banana distribution with a Sample-DIRT chain.

The target density is never evaluated during fitting.  Seven diffusion-style
bridges are formed by re-noising the same endpoint samples in probit
coordinates, and one low-rank TT Rosenblatt correction is fitted per bridge.
The analytic banana density is used only after training as an oracle.

Run from the repository root::

    python examples/sample_dirt_banana.py
    python examples/sample_dirt_banana.py \
        --gif docs/media/sample_dirt_banana.gif \
        --json docs/media/sample_dirt_banana.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr, ndtri

from tt.transport import SampleDIRT


SCALE = np.array([1.25, 1.15], dtype=np.float64)
ALPHAS = (0.20, 0.40, 0.60, 0.78, 0.90, 0.97, 1.00)
MODES = (20, 24, 28, 32, 36, 40, 48)
RANKS = (2, 3, 3, 4, 4, 5, 6)


@dataclass
class Stage:
    """One frame of the fitted bridge sequence."""

    layer: int
    alpha: float
    modes: int
    rank_cap: int
    ranks: tuple[int, ...]
    parameters: int
    epochs: int
    bridge: np.ndarray
    generated: np.ndarray
    swd: float


def banana_samples(count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw banana samples and map them smoothly to the unit square."""
    latent = rng.standard_normal((count, 2))
    physical = np.column_stack([
        1.05 * latent[:, 0],
        0.32 * latent[:, 1] + 0.52 * (latent[:, 0] ** 2 - 1.0),
    ])
    return ndtr(physical / SCALE)


def to_physical(points: np.ndarray) -> np.ndarray:
    """Undo the probit chart used to place the problem on ``[0, 1]^2``."""
    eps = np.finfo(np.float64).eps
    return ndtri(np.clip(points, eps, 1.0 - eps)) * SCALE


def banana_log_density(points: np.ndarray) -> np.ndarray:
    """Exact normalized unit-square log density, used only for evaluation."""
    eps = np.finfo(np.float64).eps
    z = ndtri(np.clip(points, eps, 1.0 - eps))
    physical = z * SCALE
    latent_1 = physical[:, 0] / 1.05
    latent_2 = (
        physical[:, 1] - 0.52 * (latent_1 * latent_1 - 1.0)
    ) / 0.32
    log_two_pi = np.log(2.0 * np.pi)
    log_physical = (
        -0.5 * (latent_1 * latent_1 + latent_2 * latent_2)
        - log_two_pi
        - np.log(1.05 * 0.32)
    )
    # dx/dz = SCALE and dz/du = 1 / phi(z).
    log_chart_jacobian = (
        np.log(SCALE).sum() + 0.5 * np.sum(z * z, axis=1) + log_two_pi
    )
    return log_physical + log_chart_jacobian


def probit_bridge_samples(
    target: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the Gaussian/probit bridge using endpoint samples only."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    eps = np.finfo(np.float64).eps
    latent = ndtri(np.clip(target, eps, 1.0 - eps))
    if alpha == 1.0:
        return target.copy()
    noise = rng.standard_normal(latent.shape)
    return ndtr(alpha * latent + np.sqrt(1.0 - alpha * alpha) * noise)


def sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    *,
    projections: int = 128,
    seed: int = 0,
) -> float:
    """Mean projected W2 distance in physical coordinates."""
    count = min(len(first), len(second))
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((projections, first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    first_projected = np.sort(first[:count] @ directions.T, axis=0)
    second_projected = np.sort(second[:count] @ directions.T, axis=0)
    return float(np.mean(np.sqrt(np.mean(
        (first_projected - second_projected) ** 2, axis=0
    ))))


def fit_transport(
    *,
    train_samples: int = 20_000,
    evaluation_samples: int = 20_000,
    plot_samples: int = 900,
    epochs: int = 10,
    seed: int = 731,
    device: str = "cpu",
) -> tuple[SampleDIRT, dict, list[Stage], np.ndarray]:
    """Fit the chain and return its report plus animation checkpoints."""
    if min(train_samples, evaluation_samples, plot_samples) < 2:
        raise ValueError("all sample counts must be at least two")

    rng = np.random.default_rng(seed)
    training = banana_samples(train_samples, rng)
    target = banana_samples(evaluation_samples, rng)
    target_two = banana_samples(evaluation_samples, rng)
    metric_base = rng.random((evaluation_samples, 2))
    plot_base = rng.random((plot_samples, 2))
    plot_target = target[:min(evaluation_samples, 5_000)].copy()
    bridge_rng = np.random.default_rng(seed + 101)

    model = SampleDIRT(2)
    metric_target_physical = to_physical(target)
    initial_swd = sliced_wasserstein(
        to_physical(metric_base), metric_target_physical, seed=seed + 31
    )
    stages = [Stage(
        layer=0,
        alpha=0.0,
        modes=1,
        rank_cap=1,
        ranks=(1, 1, 1),
        parameters=0,
        epochs=0,
        bridge=plot_base.copy(),
        generated=plot_base.copy(),
        swd=initial_swd,
    )]
    layer_reports = []
    started = time.perf_counter()

    for level, (alpha, modes, rank) in enumerate(zip(ALPHAS, MODES, RANKS), 1):
        bridge = probit_bridge_samples(training, alpha, bridge_rng)
        history = model.fit_layer(
            bridge,
            estimator="centered",
            modes=modes,
            rank=rank,
            epochs=epochs,
            optimizer="als",
            patience=4,
            seed=seed + level - 1,
            device=device,
            initialization_coarse_bins=4,
            initialization_tolerance=2e-2,
            projection_tolerance=1e-2,
            projection_rank=rank,
            representation="direct",
        )
        fitted = model.layers[-1]
        generated_metric = model.forward(metric_base)
        swd = sliced_wasserstein(
            to_physical(generated_metric),
            metric_target_physical,
            seed=seed + 31,
        )
        ranks = tuple(int(value) for value in fitted.ranks)
        layer_reports.append({
            "layer": level,
            "alpha": alpha,
            "modes": modes,
            "rank_cap": rank,
            "ranks": list(ranks),
            "parameters": fitted.size,
            "epochs": history.epochs,
            "best_validation_loss": min(history.validation_loss),
            "swd_physical": swd,
        })
        stages.append(Stage(
            layer=level,
            alpha=alpha,
            modes=modes,
            rank_cap=rank,
            ranks=ranks,
            parameters=model.stored_parameters,
            epochs=history.epochs,
            bridge=bridge[:plot_samples].copy(),
            generated=model.forward(plot_base),
            swd=swd,
        ))

    wall_time = time.perf_counter() - started
    generated = model.forward(metric_base)
    combined = np.vstack([target, generated])
    log_ratio = banana_log_density(combined) - model.log_density(combined)
    tv_integrand = np.abs(np.tanh(0.5 * log_ratio))
    target_tv = tv_integrand[:evaluation_samples]
    generated_tv = tv_integrand[evaluation_samples:]
    tv = 0.5 * (target_tv.mean() + generated_tv.mean())
    tv_standard_error = 0.5 * np.sqrt(
        target_tv.var(ddof=1) / evaluation_samples
        + generated_tv.var(ddof=1) / evaluation_samples
    )
    sampling_floor = sliced_wasserstein(
        metric_target_physical,
        to_physical(target_two),
        seed=seed + 31,
    )
    report = {
        "problem": "2D banana, sample-only probit bridge",
        "seed": seed,
        "unique_target_samples": train_samples,
        "bridge_observations": train_samples * len(ALPHAS),
        "optimizer": "orthogonal full-batch ALS",
        "layers": layer_reports,
        "initial_swd_physical": initial_swd,
        "final_swd_physical": stages[-1].swd,
        "target_swd_floor": sampling_floor,
        "estimated_kl_target_model": float(np.mean(log_ratio[:evaluation_samples])),
        "estimated_tv_target_model": float(tv),
        "estimated_tv_standard_error": float(tv_standard_error),
        "stored_parameters": model.stored_parameters,
        "roundtrip_error": model.roundtrip_error(generated[:1000]),
        "fit_seconds": wall_time,
    }
    return model, report, stages, plot_target


def _target_contours(ax, target: np.ndarray, bounds) -> None:
    """Draw a lightly smoothed empirical target-density oracle."""
    from scipy.ndimage import gaussian_filter

    density, x_edges, y_edges = np.histogram2d(
        target[:, 0], target[:, 1], bins=72,
        range=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        density=True,
    )
    density = gaussian_filter(density, 1.35)
    maximum = float(density.max(initial=0.0))
    if maximum > 0.0:
        levels = maximum * np.array([0.08, 0.18, 0.35, 0.60])
        x = 0.5 * (x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(x, y, density.T, levels=np.unique(levels),
                   colors="#343a40", linewidths=0.8, alpha=0.65)


def render_gif(stages: list[Stage], target: np.ndarray, output: Path) -> None:
    """Render the bridge, accumulated transport, and convergence together."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    target_physical = to_physical(target)
    all_points = [target_physical]
    for stage in stages:
        all_points.extend([
            to_physical(stage.bridge),
            to_physical(stage.generated),
        ])
    stacked = np.vstack(all_points)
    low = np.quantile(stacked, 0.003, axis=0)
    high = np.quantile(stacked, 0.997, axis=0)
    padding = 0.04 * (high - low)
    bounds = (
        low[0] - padding[0], high[0] + padding[0],
        low[1] - padding[1], high[1] + padding[1],
    )
    swd_values = np.array([stage.swd for stage in stages])
    images = []

    for frame_index, stage in enumerate(stages):
        fig = plt.figure(figsize=(9.6, 5.1), dpi=100)
        grid = fig.add_gridspec(2, 2, height_ratios=(4.0, 1.25), hspace=0.42)
        bridge_ax = fig.add_subplot(grid[0, 0])
        generated_ax = fig.add_subplot(grid[0, 1])
        metric_ax = fig.add_subplot(grid[1, :])

        bridge_physical = to_physical(stage.bridge)
        generated_physical = to_physical(stage.generated)
        bridge_ax.scatter(
            bridge_physical[:, 0], bridge_physical[:, 1],
            s=5, alpha=0.48, linewidths=0, color="#d97706",
        )
        bridge_ax.set_title(
            rf"samples defining bridge $p_{{\alpha}}$,  $\alpha={stage.alpha:.2f}$",
            fontsize=10,
        )

        _target_contours(generated_ax, target_physical, bounds)
        generated_ax.scatter(
            generated_physical[:, 0], generated_physical[:, 1],
            s=5, alpha=0.56, linewidths=0, color="#087e8b",
            label="transported reference",
        )
        generated_ax.set_title(
            "same reference cloud after accumulated maps\n"
            f"SWD = {stage.swd:.3f}  (target contours in black)",
            fontsize=10,
        )

        for ax in (bridge_ax, generated_ax):
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.grid(alpha=0.12, linewidth=0.5)

        visible = frame_index + 1
        metric_ax.plot(
            np.arange(visible), swd_values[:visible], "o-",
            color="#087e8b", linewidth=2.0, markersize=5,
        )
        metric_ax.scatter(
            [frame_index], [stage.swd], s=80, color="#d97706",
            edgecolor="white", linewidth=1.0, zorder=3,
        )
        metric_ax.set_xlim(-0.2, len(stages) - 0.8)
        metric_ax.set_ylim(0.0, 1.08 * swd_values.max())
        metric_ax.set_xticks(range(len(stages)))
        metric_ax.set_xticklabels(["ref", *[str(i) for i in range(1, len(stages))]])
        metric_ax.set_xlabel("number of learned residual TT transports")
        metric_ax.set_ylabel("sliced $W_2$")
        metric_ax.grid(alpha=0.2, linewidth=0.5)

        if stage.layer == 0:
            status = "uniform reference — no fitted parameters"
        else:
            status = (
                f"layer {stage.layer}/{len(stages) - 1}: "
                f"{stage.modes}×{stage.modes} cells, "
                f"TT rank {max(stage.ranks)}, "
                f"{stage.parameters} cumulative parameters"
            )
        fig.suptitle(
            "Sample-DIRT learns a banana from samples, one residual map at a time\n"
            + status,
            fontsize=12,
        )
        fig.subplots_adjust(top=0.84, bottom=0.11, left=0.08, right=0.98)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        # Keep full RGB frames until the final save.  Independent adaptive
        # palettes can assign the GIF background index differently per frame,
        # which makes changing text acquire transparent holes during playback.
        images.append(Image.open(buffer).convert("RGB"))

    durations = [1400] + [950] * (len(images) - 2) + [3500]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"saved {output} ({len(images)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=20_000)
    parser.add_argument("--evaluation-samples", type=int, default=20_000)
    parser.add_argument("--plot-samples", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    _, report, stages, target = fit_transport(
        train_samples=args.train_samples,
        evaluation_samples=args.evaluation_samples,
        plot_samples=args.plot_samples,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if args.gif is not None:
        render_gif(stages, target, args.gif)


if __name__ == "__main__":
    main()
