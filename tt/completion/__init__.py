"""Legacy namespace: ``from tt.completion.als import ttSparseALS``.

The implementation lives in :mod:`tt.algs.completion`.
"""

import sys as _sys

from ..algs import completion as als  # noqa: F401
from ..algs.completion import ttSparseALS  # noqa: F401

_sys.modules[__name__ + ".als"] = als

__all__ = ["ttSparseALS", "als"]
