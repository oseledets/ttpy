"""Diagnostics and result records shared by fitting algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FitHistory:
    """Diagnostics recorded while fitting one Sample-DIRT layer."""

    loss: list[float] = field(default_factory=list)
    l2_norm_sq: list[float] = field(default_factory=list)
    normalization: list[float] = field(default_factory=list)
    chi2_to_reference: list[float] = field(default_factory=list)
    regularization: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    epochs: int = 0
    best_epoch: int = 0
    converged: bool = False
    optimizer: str = ""
    function_calls: int = 0
    wall_time: float = 0.0
    initial_ranks: list[int] = field(default_factory=list)
    correction_ranks: list[int] = field(default_factory=list)
    projected_ranks: list[int] = field(default_factory=list)
    negative_ratio_fraction: float = 0.0
    projection_rmse: float = 0.0
    conditional_clipping_fraction: float = 0.0
    positivity_scale: float = 1.0


@dataclass
class ModeRefinementScore:
    """Cross-fitted natural-gradient evidence for one refined TT core."""

    layer_index: int
    physical_coordinate: int
    internal_coordinate: int
    old_modes: int
    new_modes: int
    added_parameters: int
    cross_scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    standard_error: float = 0.0
    lower_confidence_bound: float = 0.0

__all__ = [
    "FitHistory",
    "ModeRefinementScore",
]
