"""Sample-DIRT progression on three sample-only two-dimensional targets.

The targets are a banana, two moons, and a spiral.  Their analytic densities
are never evaluated: the fit sees only target samples in the unit square.  A
probit diffusion bridge supplies intermediate sample clouds, and the same
uniform reference points are pushed through every accumulated transport so
that the stages can be compared point-for-point.

Example
-------
python examples/sample_dirt_2d_gallery.py \
    --json sample_dirt_2d_gallery.json --png sample_dirt_2d_gallery.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from sample_dirt_common import probit_bridge_samples
from tt.transport import SampleDIRT


@dataclass(frozen=True)
class Target2D:
    key: str
    label: str
    unit_samples: np.ndarray
    plot_scale: np.ndarray
    axis_labels: tuple[str, str]

    def to_plot(self, unit_points: np.ndarray) -> np.ndarray:
        eps = 1e-5
        return ndtri(np.clip(unit_points, eps, 1.0 - eps)) * self.plot_scale


def _to_unit(points: np.ndarray, scale) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(scale, dtype=np.float64)
    return ndtr(points / scale), scale


def make_banana(count: int, rng: np.random.Generator) -> Target2D:
    latent = rng.standard_normal((count, 2))
    points = np.column_stack(
        [
            1.05 * latent[:, 0],
            0.32 * latent[:, 1] + 0.52 * (latent[:, 0] ** 2 - 1.0),
        ]
    )
    unit, scale = _to_unit(points, (1.25, 1.15))
    return Target2D("banana", "Banana", unit, scale, ("x₁", "x₂"))


def make_two_moons(count: int, rng: np.random.Generator) -> Target2D:
    labels = rng.integers(0, 2, size=count)
    angle = rng.uniform(0.0, np.pi, size=count)
    points = np.empty((count, 2), dtype=np.float64)
    first = labels == 0
    points[first, 0] = np.cos(angle[first])
    points[first, 1] = np.sin(angle[first])
    points[~first, 0] = 1.0 - np.cos(angle[~first])
    points[~first, 1] = 0.48 - np.sin(angle[~first])
    points += 0.055 * rng.standard_normal(points.shape)
    points -= np.array([0.5, 0.24])
    unit, scale = _to_unit(points, (1.15, 0.72))
    return Target2D("moons", "Two moons", unit, scale, ("x₁", "x₂"))


def make_spiral(count: int, rng: np.random.Generator) -> Target2D:
    angle = rng.uniform(0.35, 3.2 * np.pi, size=count)
    radius = 0.12 + 0.105 * angle
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    points += 0.052 * rng.standard_normal(points.shape)
    unit, scale = _to_unit(points, (0.82, 0.82))
    return Target2D("spiral", "Spiral", unit, scale, ("x₁", "x₂"))


def sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    *,
    projections: int = 96,
    seed: int = 0,
) -> float:
    """Average one-dimensional W2 distance over random projections."""
    count = min(first.shape[0], second.shape[0])
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((projections, first.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a = np.sort(first[:count] @ directions.T, axis=0)
    b = np.sort(second[:count] @ directions.T, axis=0)
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=0))))


def fit_progression(
    target: Target2D,
    alphas: np.ndarray,
    *,
    train_count: int,
    plot_count: int,
    modes: int,
    rank: int,
    epochs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    model = SampleDIRT(2)
    reference = rng.random((plot_count, 2))
    metric_reference = rng.random((min(train_count, 5000), 2))
    target_metric = target.unit_samples[: metric_reference.shape[0]]

    def stage_record(label: str, alpha: float, points: np.ndarray, **extra) -> dict:
        metric_points = model.forward(metric_reference)
        record = {
            "label": label,
            "alpha": float(alpha),
            "points": np.round(target.to_plot(points), 5).tolist(),
            "swd": sliced_wasserstein(metric_points, target_metric, seed=seed + 31),
        }
        record.update(extra)
        return record

    stages = [stage_record("Reference", 0.0, reference)]
    for layer_index, alpha in enumerate(alphas):
        bridge = probit_bridge_samples(target.unit_samples[:train_count], alpha, rng)
        history = model.fit_layer(
            bridge,
            modes=modes,
            rank=rank,
            gamma=1e-4,
            epochs=epochs,
            learning_rate=3e-2,
            seed=seed + layer_index,
            tolerance=2e-7,
        )
        generated = model.forward(reference)
        stages.append(
            stage_record(
                f"Layer {layer_index + 1}",
                alpha,
                generated,
                chi2=float(model.layers[-1].chi2_to_reference),
                epochs=int(history.epochs),
                rank=model.layers[-1].ranks.tolist(),
            )
        )
        print(
            f"{target.label:10s} layer={layer_index + 1} alpha={alpha:.2f} "
            f"epochs={history.epochs:3d} chi2={model.layers[-1].chi2_to_reference:.3e} "
            f"SWD={stages[-1]['swd']:.4f}"
        )

    target_plot = target.to_plot(target.unit_samples[:plot_count])
    return {
        "key": target.key,
        "label": target.label,
        "axis_labels": list(target.axis_labels),
        "stages": stages,
        "target_points": np.round(target_plot, 5).tolist(),
        "roundtrip_error": model.roundtrip_error(model.forward(reference[:100])),
        "stored_parameters": model.stored_parameters,
    }


def save_png(result: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["targets"]
    columns = len(rows[0]["stages"]) + 1
    fig, axes = plt.subplots(
        len(rows), columns, figsize=(2.25 * columns, 2.2 * len(rows)), squeeze=False
    )
    for row_index, row in enumerate(rows):
        panels = row["stages"] + [
            {
                "label": "Target samples",
                "points": row["target_points"],
                "swd": 0.0,
            }
        ]
        all_points = np.concatenate(
            [np.asarray(panel["points"]) for panel in panels], axis=0
        )
        lo = np.quantile(all_points, 0.002, axis=0)
        hi = np.quantile(all_points, 0.998, axis=0)
        padding = 0.04 * np.maximum(hi - lo, 1e-6)
        for column_index, panel in enumerate(panels):
            ax = axes[row_index, column_index]
            points = np.asarray(panel["points"])
            ax.scatter(points[:, 0], points[:, 1], s=2.5, alpha=0.55, linewidths=0)
            ax.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
            ax.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(row["axis_labels"][0])
            ax.set_ylabel(row["axis_labels"][1])
            if row_index == 0:
                ax.set_title(panel["label"])
            if column_index == 0:
                ax.text(
                    -0.36,
                    0.5,
                    row["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            if column_index < columns - 1:
                ax.text(
                    0.03,
                    0.04,
                    f"SWD {panel['swd']:.3f}",
                    transform=ax.transAxes,
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
    fig.suptitle("Sample-DIRT: one reference cloud through accumulated transports")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=9000)
    parser.add_argument("--plot-count", type=int, default=700)
    parser.add_argument("--modes", type=int, default=40)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()

    # The last two bridge points are deliberately close to the target.  Thin
    # geometries (moons and spirals) emerge only near the end of the probit
    # diffusion path, so a uniform alpha grid would leave the final layer with
    # nearly all of the topological work.
    alphas = np.array([0.30, 0.58, 0.80, 0.98, 1.0])
    rng = np.random.default_rng(args.seed)
    target_count = max(args.train_count, args.plot_count, 5000)
    targets = [
        make_banana(target_count, rng),
        make_two_moons(target_count, rng),
        make_spiral(target_count, rng),
    ]
    result = {
        "title": "Sample-DIRT on 2D sample-only targets",
        "alphas": alphas.tolist(),
        "train_count": args.train_count,
        "plot_count": args.plot_count,
        "modes": args.modes,
        "rank": args.rank,
        "epochs": args.epochs,
        "seed": args.seed,
        "targets": [
            fit_progression(
                target,
                alphas,
                train_count=args.train_count,
                plot_count=args.plot_count,
                modes=args.modes,
                rank=args.rank,
                epochs=args.epochs,
                seed=args.seed + 100 * target_index,
            )
            for target_index, target in enumerate(targets)
        ],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, separators=(",", ":")))
    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        save_png(result, args.png)
    if not args.json and not args.png:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
