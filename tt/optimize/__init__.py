"""Legacy namespace: ``from tt.optimize import tt_min``.

The implementation lives in :mod:`tt.algs.optimize`; this package exists so that
scripts written against ttpy 1.x keep importing.  ``tt.optimize.tt_min`` was a
module there, so it is re-exported as this package itself.
"""

import sys as _sys

from ..algs import optimize as tt_min  # noqa: F401
from ..algs.optimize import min_func, min_tens  # noqa: F401

_sys.modules[__name__ + ".tt_min"] = tt_min

__all__ = ["min_tens", "min_func", "tt_min"]
