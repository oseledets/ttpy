"""Legacy namespace: ``from tt.riemannian import riemannian``.

The implementation lives in :mod:`tt.algs.riemannian`.
"""

import sys as _sys

from ..algs import riemannian as riemannian  # noqa: F401
from ..algs.riemannian import project, projector_splitting_add, tt_qr  # noqa: F401

_sys.modules[__name__ + ".riemannian"] = riemannian

__all__ = ["project", "projector_splitting_add", "tt_qr", "riemannian"]
