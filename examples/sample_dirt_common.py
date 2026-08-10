"""Shared utilities for the Sample-DIRT paper examples.

The posterior examples use a small random-walk Metropolis chain only to create
an oracle sample set.  The transport fit itself receives those samples and has
no access to either the log density or the forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from tt.transport import SampleDIRT


@dataclass
class MetropolisResult:
    samples: np.ndarray
    acceptance: float
    final_scale: float


def random_walk_metropolis(
    log_density,
    initial,
    sample_count: int,
    *,
    burn_in: int = 2_000,
    thin: int = 2,
    proposal_scale: float = 0.02,
    seed: int = 0,
    adapt: bool = True,
) -> MetropolisResult:
    """Random-walk Metropolis on the unit cube, used only as a test oracle."""
    if sample_count < 1 or burn_in < 0 or thin < 1:
        raise ValueError("invalid chain length")
    rng = np.random.default_rng(seed)
    current = np.asarray(initial, dtype=np.float64).copy()
    if current.ndim != 1 or np.any(current <= 0.0) or np.any(current >= 1.0):
        raise ValueError("initial state must be an interior point of the unit cube")
    current_logp = float(log_density(current))
    if not np.isfinite(current_logp):
        raise ValueError("initial state has non-finite log density")
    total = burn_in + sample_count * thin
    output = np.empty((sample_count, current.size), dtype=np.float64)
    accepted = 0
    accepted_window = 0
    stored = 0
    scale = float(proposal_scale)
    for step in range(total):
        proposal = current + scale * rng.standard_normal(current.size)
        proposal_logp = -np.inf
        if np.all((proposal > 0.0) & (proposal < 1.0)):
            proposal_logp = float(log_density(proposal))
        if np.log(rng.random()) < proposal_logp - current_logp:
            current = proposal
            current_logp = proposal_logp
            accepted += 1
            accepted_window += 1
        if adapt and step < burn_in and (step + 1) % 100 == 0:
            rate = accepted_window / 100.0
            # A gentle Robbins-Monro update around the multivariate optimum.
            scale *= np.exp(0.5 * (rate - 0.234))
            scale = float(np.clip(scale, 1e-4, 0.25))
            accepted_window = 0
        if step >= burn_in and (step - burn_in) % thin == 0:
            output[stored] = current
            stored += 1
    return MetropolisResult(output, accepted / total, scale)


def probit_bridge_samples(
    target_unit_samples: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Diffusion-style bridge from uniform to a distribution on the unit cube."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    eps = np.finfo(np.float64).eps
    target = np.clip(np.asarray(target_unit_samples), eps, 1.0 - eps)
    latent = ndtri(target)
    if alpha == 1.0:
        return target.copy()
    mixed = alpha * latent + np.sqrt(1.0 - alpha * alpha) * rng.standard_normal(latent.shape)
    return ndtr(mixed)


def reflected_gaussian_bridge_samples(
    target_unit_samples: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Neumann heat-flow bridge on the cube, available from samples only.

    Reflection modulo two is the exact pathwise construction of reflected
    Brownian motion at a fixed time. Large noise approaches the uniform law;
    zero noise returns the target samples unchanged.
    """
    target = np.asarray(target_unit_samples, dtype=np.float64)
    if target.ndim != 2 or np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_unit_samples must have shape (N, d) in the cube")
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and non-negative")
    if noise_std == 0.0:
        return target.copy()
    value = np.mod(
        target + noise_std * rng.standard_normal(target.shape),
        2.0,
    )
    return np.where(value <= 1.0, value, 2.0 - value)


def fit_diffusion_sample_dirt(
    target_unit_samples: np.ndarray,
    alphas,
    *,
    seed: int = 0,
    verbose: bool = True,
    **fit_options,
) -> tuple[SampleDIRT, list]:
    """Fit successive residual layers along a probit diffusion bridge."""
    target = np.asarray(target_unit_samples, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("target_unit_samples must have shape (N, d)")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target samples must lie in the unit cube")
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a non-empty vector")
    if np.any(np.diff(alphas) <= 0.0) or alphas[-1] != 1.0:
        raise ValueError("alphas must increase strictly and end at one")

    rng = np.random.default_rng(seed)
    model = SampleDIRT(target.shape[1])
    histories = []
    for layer_index, alpha in enumerate(alphas):
        bridge = probit_bridge_samples(target, float(alpha), rng)
        options = dict(fit_options)
        options["seed"] = int(options.get("seed", seed) + layer_index)
        options["verbose"] = bool(options.get("verbose", False))
        history = model.fit_layer(bridge, **options)
        histories.append(history)
        if verbose:
            fitted = model.layers[-1]
            print(
                f"layer={layer_index + 1}, alpha={alpha:.4f}, "
                f"epochs={history.epochs}, rank={fitted.ranks.tolist()}, "
                f"chi2(residual||ref)={fitted.chi2_to_reference:.3e}"
            )
    return model, histories


def covariance_relative_error(samples: np.ndarray, reference: np.ndarray) -> float:
    covariance = np.cov(samples, rowvar=False)
    target_covariance = np.cov(reference, rowvar=False)
    return float(
        np.linalg.norm(covariance - target_covariance)
        / max(np.linalg.norm(target_covariance), np.finfo(float).tiny)
    )


def print_sample_report(generated: np.ndarray, target: np.ndarray) -> None:
    mean_error = np.linalg.norm(generated.mean(0) - target.mean(0))
    covariance_error = covariance_relative_error(generated, target)
    print(f"mean L2 error:          {mean_error:.4e}")
    print(f"covariance rel. error: {covariance_error:.4e}")
