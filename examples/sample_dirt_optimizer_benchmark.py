"""Compare Adam, Riemannian GD and orthogonal ALS on banana Sample-DIRT.

Every method sees exactly the same sample-only probit bridges, uses the same
piecewise-constant TT model, seed, rank and exact-contraction density-ratio
loss.  The budgets differ because an ALS epoch is a complete bidirectional
sweep while an Adam/RGD epoch is one first-order iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

from sample_dirt_2d_gallery import make_banana, sliced_wasserstein
from sample_dirt_common import probit_bridge_samples
from tt.transport import SampleDIRT


def _method_options(name: str, args) -> dict:
    common = {
        "modes": args.modes,
        "rank": args.rank,
        "gamma": 1e-4,
        "tolerance": args.tolerance,
        "device": args.device,
        "dtype": args.dtype,
    }
    if name == "adam":
        return common | {
            "optimizer": "adam",
            "epochs": args.adam_steps,
            "learning_rate": 3e-2,
        }
    if name == "riemannian":
        return common | {
            "optimizer": "riemannian",
            "epochs": args.riemannian_steps,
            "learning_rate": 1e-1,
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


def run_method(name: str, target, bridges, base, args) -> dict:
    model = SampleDIRT(2)
    layer_records = []
    started = time.perf_counter()
    for layer_index, (alpha, bridge) in enumerate(zip(args.alphas, bridges)):
        options = _method_options(name, args)
        options["seed"] = args.seed + layer_index
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            history = model.fit_layer(bridge, **options)
        layer_records.append(
            {
                "layer": layer_index + 1,
                "alpha": alpha,
                "initial_loss": history.loss[0],
                "final_loss": history.loss[-1],
                "loss": history.loss,
                "gradient_norm": history.gradient_norm,
                "iterations": history.epochs,
                "function_calls": history.function_calls,
                "time": history.wall_time,
                "chi2": model.layers[-1].chi2_to_reference,
            }
        )
        print(
            f"{name:10s} layer={layer_index + 1} alpha={alpha:.2f} "
            f"loss={history.loss[-1]:.7f} time={history.wall_time:.3f}s "
            f"calls={history.function_calls}",
            flush=True,
        )
    generated = model.forward(base)
    return {
        "method": name,
        "total_time": time.perf_counter() - started,
        "swd": sliced_wasserstein(generated, target[: base.shape[0]], seed=31),
        "roundtrip_error": model.roundtrip_error(generated[:100]),
        "stored_parameters": model.stored_parameters,
        "layers": layer_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=9000)
    parser.add_argument("--metric-count", type=int, default=5000)
    parser.add_argument("--modes", type=int, default=40)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--adam-steps", type=int, default=320)
    parser.add_argument("--riemannian-steps", type=int, default=100)
    parser.add_argument("--riemannian-sgd-steps", type=int, default=200)
    parser.add_argument("--riemannian-sgd-rate", type=float, default=7e-1)
    parser.add_argument("--riemannian-momentum", type=float, default=0.0)
    parser.add_argument("--riemannian-second-moment", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--als-sweeps", type=int, default=10)
    parser.add_argument("--als-inner-steps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("adam", "riemannian", "riemannian-sgd", "als"),
        default=("adam", "riemannian", "riemannian-sgd", "als"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.alphas = [0.30, 0.58, 0.80, 0.98, 1.0]

    rng = np.random.default_rng(args.seed)
    target = make_banana(max(args.train_count, args.metric_count), rng).unit_samples
    bridge_rng = np.random.default_rng(args.seed + 1)
    bridges = [
        probit_bridge_samples(target[: args.train_count], alpha, bridge_rng)
        for alpha in args.alphas
    ]
    base = np.random.default_rng(args.seed + 2).random((args.metric_count, 2))
    result = {
        "problem": "banana Sample-DIRT",
        "train_count": args.train_count,
        "modes": args.modes,
        "rank": args.rank,
        "alphas": args.alphas,
        "methods": [
            run_method(name, target, bridges, base, args)
            for name in args.methods
        ],
    }
    print("\nSummary")
    for method in result["methods"]:
        print(
            f"{method['method']:10s} time={method['total_time']:.3f}s "
            f"SWD={method['swd']:.6f} roundtrip={method['roundtrip_error']:.2e}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
