"""ttpy 2 — the Tensor Train toolbox, pure Python.

No Fortran, no f2py, no compilation: the numerics run on numpy or (optionally)
torch, selected with :func:`tt.set_backend`.

    >>> import tt
    >>> a = tt.qlaplace_dd([6, 6, 6])          # 3D Laplacian in QTT
    >>> b = tt.ones(2, 18)
    >>> x = tt.amen_solve(a, b, b, 1e-8)
    >>> (tt.matvec(a, x) - b).norm() / b.norm() < 1e-8
    True
"""

from __future__ import annotations

from .backend import get_backend, set_backend
from .core.matrix import matrix
from .core.tools import (IpaS, Toeplitz, col, concatenate, cos, delta, diag,
                         dot, eye, kron, level_major_order, linspace, matvec,
                         mkron, ones, permute, qdiff, qlaplace_dd, qlaplace_dn,
                         qshift, qtri_ones, rand, reshape, sin,
                         shift, stepfun, sum, unit, xfun, zaffine, zeros, zkron,
                         zkronv, zmeshgrid)
from .core.vector import tensor, vector

__version__ = "2.0.0rc3"

__all__ = [
    "vector", "tensor", "matrix", "set_backend", "get_backend",
    "matvec", "col", "kron", "dot", "diag", "mkron", "zkron", "zkronv",
    "zmeshgrid", "zaffine", "concatenate", "sum", "ones", "zeros", "rand",
    "eye", "Toeplitz", "qlaplace_dd", "qlaplace_dn", "qdiff", "qtri_ones",
    "level_major_order", "xfun", "linspace", "sin", "cos",
    "delta", "stepfun", "qshift", "shift", "unit", "IpaS", "reshape", "permute",
    "multifuncrs", "multifuncrs2", "GMRES", "amen_solve", "lobpcg_solve",
    "qtt_divgrad", "qtt_divgrad_from_faces", "qtt_fft1", "amen_mv", "tamen",
    "cross", "rect_cross", "dmrg_cross", "greedy_cross", "maxvol",
    "eigb", "ksl", "ksl_deim", "DirectTTDensity", "PermutedTTDensity", "SampleDIRT",
    "SquaredTTDensity", "LinearSquaredTTDensity",
    "AdaptiveLinearSquaredTTDensity", "QuadraticSquaredTTDensity",
    "LocallyPurifiedLinearTTDensity",
    "NonnegativeLinearTTDensity",
    "ProbitOrthogonalTTDensity",
    "PurifiedLinearTTDensity",
    "damp_tt_density",
    "enrich_linear_sample_dirt_ranks",
    "truncate_linear_sample_dirt_ranks",
    "fit_centered_tt_density", "fit_centered_tt_ratio",
    "fit_squared_tt_density", "fit_linear_squared_tt_density",
    "fit_adaptive_linear_squared_tt_density",
    "fit_conditional_twists_to_probit_density",
    "fit_locally_purified_linear_tt_density",
    "fit_nonnegative_linear_tt_density",
    "fit_probit_rotated_linear_squared_tt_density",
    "fit_radial_twists_to_probit_density",
    "fit_purified_linear_tt_density",
    "fit_quadratic_squared_tt_density", "__version__",
]


# Names that were *modules* in ttpy 1.x (``from tt.amen import amen_solve``) and
# names that were *functions* on the package (``tt.multifuncrs(...)``).  Keeping
# the distinction is what makes old scripts run unchanged.
_LEGACY_MODULES = ("maxvol", "cross", "amen", "eigb", "ksl", "optimize",
                   "riemannian", "completion", "solvers")
_FUNCTIONS = {
    "multifuncrs": ("tt.algs.multifuncrs", "multifuncrs"),
    "multifuncrs2": ("tt.algs.multifuncrs", "multifuncrs2"),
    "GMRES": ("tt.algs.solvers", "GMRES"),
    "amen_solve": ("tt.algs.amen", "amen_solve"),
    "lobpcg_solve": ("tt.algs.lobpcg", "lobpcg_solve"),
    "qtt_divgrad": ("tt.algs.qtt_fd", "qtt_divgrad"),
    "qtt_divgrad_from_faces": ("tt.algs.qtt_fd", "qtt_divgrad_from_faces"),
    "qtt_fft1": ("tt.algs.qtt_fft", "qtt_fft1"),
    "tamen": ("tt.algs.tamen", "tamen"),
    "amen_mv": ("tt.algs.amen_mv", "amen_mv"),
    "eigb_solve": ("tt.algs.eigb", "eigb"),
    "ksl_step": ("tt.algs.ksl", "ksl"),
    "ksl_deim": ("tt.algs.ksl_deim", "ksl_deim"),
    "rect_cross": ("tt.algs.cross", "rect_cross"),
    "dmrg_cross": ("tt.algs.dmrg_cross", "dmrg_cross"),
    "riemannian_grad": ("tt.algs.autodiff", "riemannian_grad"),
    "rgd": ("tt.algs.autodiff", "rgd"),
    "greedy_cross": ("tt.algs.dmrg_cross", "dmrg_cross"),
    "rect_maxvol": ("tt.algs.maxvol", "rect_maxvol"),
    "min_tens": ("tt.algs.optimize", "min_tens"),
    "min_func": ("tt.algs.optimize", "min_func"),
    "DirectTTDensity": ("tt.transport.sample_dirt", "DirectTTDensity"),
    "PermutedTTDensity": ("tt.transport.sample_dirt", "PermutedTTDensity"),
    "SampleDIRT": ("tt.transport.sample_dirt", "SampleDIRT"),
    "SquaredTTDensity": ("tt.transport.sample_dirt", "SquaredTTDensity"),
    "damp_tt_density": ("tt.transport.sample_dirt", "damp_tt_density"),
    "LinearSquaredTTDensity": (
        "tt.transport.sample_dirt", "LinearSquaredTTDensity"
    ),
    "QuadraticSquaredTTDensity": (
        "tt.transport.sample_dirt", "QuadraticSquaredTTDensity"
    ),
    "fit_centered_tt_density": (
        "tt.transport.sample_dirt", "fit_centered_tt_density"
    ),
    "fit_centered_tt_ratio": (
        "tt.transport.sample_dirt", "fit_centered_tt_ratio"
    ),
    "fit_squared_tt_density": (
        "tt.transport.sample_dirt", "fit_squared_tt_density"
    ),
    "fit_linear_squared_tt_density": (
        "tt.transport.sample_dirt", "fit_linear_squared_tt_density"
    ),
    "fit_quadratic_squared_tt_density": (
        "tt.transport.sample_dirt", "fit_quadratic_squared_tt_density"
    ),
}


def __getattr__(name):
    """Resolve the algorithm layer lazily (keeps ``import tt`` cheap).

    Algorithms are imported on first use, so a missing optional dependency or a
    module still under construction cannot break ``import tt``.
    """
    import importlib

    if name in _LEGACY_MODULES:
        return importlib.import_module(f"tt.{name}")
    if name in _FUNCTIONS:
        module, attr = _FUNCTIONS[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'tt' has no attribute {name!r}")
