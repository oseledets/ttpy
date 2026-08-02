"""Legacy namespace: ``from tt.solvers import GMRES``.

The implementation lives in :mod:`tt.algs.solvers`.
"""

from .algs.solvers import GMRES, GmresHistory  # noqa: F401

__all__ = ["GMRES", "GmresHistory"]
