"""Sample-DIRT test based on the predator--prey posterior in paper Section 6.1.

The model, parameter box, observation times and data are those distributed with
TT-IRT.  Random-walk Metropolis is only an oracle producing target samples;
``fit_diffusion_sample_dirt`` receives the resulting array and never evaluates
the likelihood.
"""

from __future__ import annotations

import argparse

import numpy as np

from sample_dirt_common import (
    fit_diffusion_sample_dirt,
    print_sample_report,
    random_walk_metropolis,
)


LOWER = np.array([30.0, 3.0, 0.36, 60.0, 0.72, 15.0, 0.3, 0.18])
UPPER = np.array([80.0, 8.0, 0.96, 160.0, 1.92, 40.0, 0.8, 0.48])
TRUTH = np.array([50.0, 5.0, 0.6, 100.0, 1.2, 25.0, 0.5, 0.3])
OBSERVATIONS = np.array([
    [51.027970813917953, 2.5795502682328695],
    [80.700617965861198, 6.4103848394239265],
    [84.742366511782151, 9.0130730424638674],
    [77.429057991275116, 13.669289747364965],
    [69.849796955492806, 18.396253174455008],
    [53.037650094715666, 21.442674733780002],
    [36.427352617433813, 25.013654999126562],
    [23.071899751605533, 22.198456733272518],
    [16.777993664328319, 17.740046366416266],
    [32.503301130011160, 15.075966104699976],
    [49.092267816825562, 15.365687554051749],
    [57.644591974583911, 15.305169205381983],
    [57.338041388363010, 20.403709524110930],
])


def forward(parameters: np.ndarray) -> np.ndarray:
    """Fixed-step RK4 at the 13 observation times used in the paper."""
    # TT-IRT stores the interaction parameters as (s, alpha), despite the
    # prose list in the paper displaying them in the opposite order.
    p, q, r, carrying, s, alpha, u, v = parameters
    state = np.array([p, q], dtype=np.float64)
    result = np.empty((13, 2), dtype=np.float64)
    result[0] = state
    dt = 1.0 / 12.0

    def rhs(y):
        prey, predator = y
        interaction = prey * predator / (alpha + prey)
        return np.array([
            r * prey * (1.0 - prey / carrying) - s * interaction,
            u * interaction - v * predator,
        ])

    for step in range(1, 601):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % 50 == 0:
            result[step // 50] = state
    return result


def log_posterior_unit(unit: np.ndarray) -> float:
    parameters = LOWER + unit * (UPPER - LOWER)
    prediction = forward(parameters)
    residual = prediction - OBSERVATIONS
    return -0.5 * float(np.sum(residual * residual)) / (2.0 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6_000)
    parser.add_argument("--burn-in", type=int, default=3_000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--bins", type=int, default=14)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    initial = (TRUTH - LOWER) / (UPPER - LOWER)
    oracle = random_walk_metropolis(
        log_posterior_unit,
        initial,
        args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        proposal_scale=0.006,
        seed=args.seed,
    )
    print(
        f"oracle MCMC acceptance={oracle.acceptance:.3f}, "
        f"final scale={oracle.final_scale:.3e}"
    )
    model, _ = fit_diffusion_sample_dirt(
        oracle.samples,
        alphas=(0.55, 0.75, 0.9, 1.0),
        modes=args.bins,
        rank=args.rank,
        gamma=1e-4,
        epochs=args.epochs,
        learning_rate=3e-2,
        seed=args.seed,
    )
    generated = model.sample(args.samples, seed=args.seed + 100)
    print_sample_report(generated, oracle.samples)
    print(f"stored TT parameters: {model.stored_parameters}")


if __name__ == "__main__":
    main()
