"""Legacy namespace: ``import tt.ksl; tt.ksl.ksl(A, y0, tau)``.

The implementation lives in :mod:`tt.algs.ksl`; this package exists so that
scripts written against ttpy 1.x keep importing.
"""

from ..algs.ksl import KslHistory, diag_ksl, expmv_krylov, ksl, tangent_defect  # noqa: F401

__all__ = ["ksl", "diag_ksl", "expmv_krylov", "tangent_defect", "KslHistory"]
