"""Sample-driven tensor-train transport maps."""

from .gaussian_mixture import TruncatedGaussianMixture

from .sample_dirt import (
    FitHistory,
    SampleDIRT,
    SquaredTTDensity,
    fit_squared_tt_density,
)

__all__ = [
    "FitHistory",
    "SampleDIRT",
    "SquaredTTDensity",
    "TruncatedGaussianMixture",
    "fit_squared_tt_density",
]
