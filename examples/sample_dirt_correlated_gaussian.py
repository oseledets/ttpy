"""Sample-DIRT on the correlated Gaussian motivating example of Cui--Dolgov.

The Gaussian is represented as a Gaussian copula on the unit cube so that the
uniform reference used by the piecewise-constant prototype is exact.  The
bridge linearly interpolates the covariance in latent Gaussian coordinates.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.special import ndtr

from sample_dirt_common import print_sample_report
from tt.transport import SampleDIRT


def gaussian_copula_samples(count: int, covariance: np.ndarray, rng) -> np.ndarray:
    return ndtr(rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--samples", type=int, default=12_000)
    parser.add_argument("--test-samples", type=int, default=20_000)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    index = np.arange(args.dimension)
    target_covariance = 0.75 ** np.abs(index[:, None] - index[None, :])
    model = SampleDIRT(args.dimension)
    for layer_index, strength in enumerate((0.25, 0.5, 0.75, 1.0)):
        covariance = (1.0 - strength) * np.eye(args.dimension) + strength * target_covariance
        bridge = gaussian_copula_samples(args.samples, covariance, rng)
        history = model.fit_layer(
            bridge,
            modes=args.bins,
            rank=args.rank,
            gamma=1e-4,
            epochs=args.epochs,
            learning_rate=3e-2,
            seed=args.seed + layer_index,
        )
        layer = model.layers[-1]
        print(
            f"layer={layer_index + 1}, strength={strength:.2f}, "
            f"loss={history.loss[-1]:.5e}, chi2={layer.chi2_to_reference:.3e}"
        )

    target = gaussian_copula_samples(args.test_samples, target_covariance, rng)
    generated = model.sample(args.test_samples, seed=args.seed + 100)
    print_sample_report(generated, target)
    print(f"stored TT parameters: {model.stored_parameters}")
    print(f"roundtrip max error:  {model.roundtrip_error(generated[:200]):.3e}")


if __name__ == "__main__":
    main()

