"""Legacy namespace: ``from tt.amen import amen_solve, amen_mv``.

The implementations live in :mod:`tt.algs.amen` and :mod:`tt.algs.amen_mv`;
this package exists so that scripts written against ttpy 1.x keep importing.
"""

from ..algs.amen import amen_solve  # noqa: F401
from ..algs.amen_mv import amen_mv  # noqa: F401

__all__ = ["amen_solve", "amen_mv"]
