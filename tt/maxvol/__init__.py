"""Legacy namespace: ``from tt.maxvol import maxvol``.

The implementation lives in :mod:`tt.algs.maxvol`; this package exists so that
scripts written against ttpy 1.x keep importing.
"""

from ..algs.maxvol import maxvol, rect_maxvol  # noqa: F401

__all__ = ["maxvol", "rect_maxvol"]
