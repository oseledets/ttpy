"""Legacy namespace: ``from tt.cross import cross``.

The implementation lives in :mod:`tt.algs.cross`.
"""

from ..algs.cross import cross, greedy_cross, rect_cross  # noqa: F401

__all__ = ["cross", "rect_cross", "greedy_cross"]
