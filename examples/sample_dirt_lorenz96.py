"""Sample-DIRT test based on the Lorenz--96 posterior in paper Section 6.2.

Dimension ten is the quick default used by the accompanying TT-IRT code.  Pass
``--dimension 40`` for the paper configuration.  MCMC supplies an oracle target
sample only; the transport fit is sample-only.
"""

from __future__ import annotations

import argparse

import numpy as np

from sample_dirt_common import (
    fit_diffusion_sample_dirt,
    print_sample_report,
    random_walk_metropolis,
)


def lorenz96_rhs(state: np.ndarray) -> np.ndarray:
    return (
        (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1)
        - state
        + 8.0
    )


def terminal_state(initial: np.ndarray, steps: int = 20) -> np.ndarray:
    state = np.asarray(initial, dtype=np.float64).copy()
    dt = 0.1 / steps
    for _ in range(steps):
        k1 = lorenz96_rhs(state)
        k2 = lorenz96_rhs(state + 0.5 * dt * k1)
        k3 = lorenz96_rhs(state + 0.5 * dt * k2)
        k4 = lorenz96_rhs(state + dt * k3)
        state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def build_problem(dimension: int, seed: int):
    rng = np.random.default_rng(seed)
    truth = 1.0 + 1e-2 * rng.standard_normal(dimension)
    data = terminal_state(truth)[1::2] + 0.1 * rng.standard_normal(dimension // 2)

    def log_posterior_unit(unit):
        initial = -10.0 + 20.0 * unit
        residual = terminal_state(initial)[1::2] - data
        log_likelihood = -0.5 * np.sum((residual / 0.1) ** 2)
        log_prior = -0.5 * np.sum((initial - 1.0) ** 2)
        return float(log_likelihood + log_prior)

    return truth, data, log_posterior_unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--burn-in", type=int, default=3_000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    if args.dimension % 2:
        raise ValueError("dimension must be even")

    truth, _data, log_posterior = build_problem(args.dimension, args.seed)
    initial = (truth + 10.0) / 20.0
    oracle = random_walk_metropolis(
        log_posterior,
        initial,
        args.samples,
        burn_in=args.burn_in,
        thin=args.thin,
        proposal_scale=0.004,
        seed=args.seed + 1,
    )
    print(
        f"oracle MCMC acceptance={oracle.acceptance:.3f}, "
        f"final scale={oracle.final_scale:.3e}"
    )
    model, _ = fit_diffusion_sample_dirt(
        oracle.samples,
        alphas=(0.5, 0.7, 0.85, 1.0),
        modes=args.bins,
        rank=args.rank,
        gamma=1e-4,
        epochs=args.epochs,
        learning_rate=2e-2,
        batch_size=min(args.samples, 4_000),
        seed=args.seed,
    )
    generated = model.sample(args.samples, seed=args.seed + 100)
    print_sample_report(generated, oracle.samples)
    print(f"stored TT parameters: {model.stored_parameters}")


if __name__ == "__main__":
    main()

