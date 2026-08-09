"""Legacy namespace: ``from tt.cross import cross``.

The implementation lives in :mod:`tt.algs.cross`.
"""

from ..algs.cross import cross, rect_cross  # noqa: F401
from ..algs.dmrg_cross import dmrg_cross, greedy_cross  # noqa: F401

__all__ = ["cross", "rect_cross", "dmrg_cross", "greedy_cross"]
