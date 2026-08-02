"""Legacy namespace: ``import tt.eigb; tt.eigb.eigb(A, y0, eps)``.

The implementation lives in :mod:`tt.algs.eigb`; this package exists so that
scripts written against ttpy 1.x keep importing.
"""

from ..algs.eigb import EigbHistory, eigb  # noqa: F401

__all__ = ["eigb", "EigbHistory"]
