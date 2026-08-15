"""Sample-driven density estimation and transport maps in Tensor Train form.

The reusable models, fitting algorithms, and transport logic live in this
namespace. Research experiments and generated artifacts are maintained in the
separate ``sample-dirt`` project.
"""

from .gaussian_mixture import TruncatedGaussianMixture
from ._fit_cell import (
    fit_centered_tt_density,
    fit_centered_tt_ratio,
    fit_squared_tt_density,
)
from ._fit_coordinates import (
    fit_conditional_twists_to_probit_density,
    fit_radial_twists_to_probit_density,
)
from ._fit_positive import (
    fit_locally_purified_linear_tt_density,
    fit_nonnegative_linear_tt_density,
    fit_purified_linear_tt_density,
    fit_quadratic_squared_tt_density,
)
from ._fit_spline import (
    fit_adaptive_linear_squared_tt_density,
    fit_linear_squared_tt_density,
    fit_probit_rotated_linear_squared_tt_density,
)
from ._types import FitHistory, ModeRefinementScore
from .coordinates import ProbitOrthogonalTTDensity
from .polynomial import DirectTTDensity, QuadraticSquaredTTDensity
from .positive import (
    LocallyPurifiedLinearTTDensity,
    NonnegativeLinearTTDensity,
    PurifiedLinearTTDensity,
)
from .refinement import (
    fine_tune_linear_sample_dirt,
    score_linear_sample_dirt_mode_refinement,
)
from .scalar import (
    AdaptiveLinearSquaredTTDensity,
    LinearSquaredTTDensity,
    SquaredTTDensity,
)
from .transport import (
    SampleDIRT,
    enable_linear_sample_dirt_probit_rotations,
    enrich_linear_sample_dirt_ranks,
    refine_linear_sample_dirt_modes,
    truncate_linear_sample_dirt_ranks,
)
from .wrappers import PermutedTTDensity, damp_tt_density

__all__ = [
    "AdaptiveLinearSquaredTTDensity",
    "DirectTTDensity",
    "FitHistory",
    "LinearSquaredTTDensity",
    "LocallyPurifiedLinearTTDensity",
    "ModeRefinementScore",
    "NonnegativeLinearTTDensity",
    "ProbitOrthogonalTTDensity",
    "PurifiedLinearTTDensity",
    "QuadraticSquaredTTDensity",
    "PermutedTTDensity",
    "SampleDIRT",
    "SquaredTTDensity",
    "TruncatedGaussianMixture",
    "damp_tt_density",
    "enable_linear_sample_dirt_probit_rotations",
    "enrich_linear_sample_dirt_ranks",
    "fine_tune_linear_sample_dirt",
    "fit_adaptive_linear_squared_tt_density",
    "fit_centered_tt_density",
    "fit_centered_tt_ratio",
    "fit_conditional_twists_to_probit_density",
    "fit_linear_squared_tt_density",
    "fit_locally_purified_linear_tt_density",
    "fit_nonnegative_linear_tt_density",
    "fit_probit_rotated_linear_squared_tt_density",
    "fit_purified_linear_tt_density",
    "fit_quadratic_squared_tt_density",
    "fit_radial_twists_to_probit_density",
    "fit_squared_tt_density",
    "refine_linear_sample_dirt_modes",
    "score_linear_sample_dirt_mode_refinement",
    "truncate_linear_sample_dirt_ranks",
]
