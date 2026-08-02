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
                         dot, eye, kron, linspace, matvec, mkron, ones,
                         permute, qlaplace_dd, qshift, rand, reshape, sin,
                         shift, stepfun, sum, unit, xfun, zaffine, zeros, zkron,
                         zkronv, zmeshgrid)
from .core.vector import tensor, vector

__version__ = "2.0.0.dev0"

__all__ = [
    "vector", "tensor", "matrix", "set_backend", "get_backend",
    "matvec", "col", "kron", "dot", "diag", "mkron", "zkron", "zkronv",
    "zmeshgrid", "zaffine", "concatenate", "sum", "ones", "zeros", "rand",
    "eye", "Toeplitz", "qlaplace_dd", "xfun", "linspace", "sin", "cos",
    "delta", "stepfun", "qshift", "shift", "unit", "IpaS", "reshape", "permute",
    "multifuncrs", "multifuncrs2", "GMRES", "amen_solve", "amen_mv",
    "cross", "rect_cross", "maxvol", "eigb", "ksl", "__version__",
]


def __getattr__(name):
    """Import the algorithm namespaces lazily (keeps ``import tt`` cheap)."""
    if name in ("multifuncrs", "multifuncrs2"):
        from .algs import multifuncrs as _m
        return getattr(_m, name)
    if name == "GMRES":
        from .algs.solvers import GMRES
        return GMRES
    if name in ("amen_solve", "amen_mv"):
        from .algs import amen as _a
        return getattr(_a, name)
    if name in ("cross", "rect_cross", "greedy_cross"):
        from .algs import cross as _c
        return getattr(_c, name)
    if name in ("maxvol", "rect_maxvol"):
        from .algs.maxvol import maxvol, rect_maxvol
        return {"maxvol": maxvol, "rect_maxvol": rect_maxvol}[name]
    if name == "eigb":
        from .algs.eigb import eigb
        return eigb
    if name in ("ksl", "diag_ksl"):
        from .algs.ksl import diag_ksl, ksl
        return {"ksl": ksl, "diag_ksl": diag_ksl}[name]
    raise AttributeError(f"module 'tt' has no attribute {name!r}")
