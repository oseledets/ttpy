"""Subpackage interpolate provides usefull types and routines to build an
interpolation and to work with it.
"""

from .chebyshev import (Chebfun, Chebop, chebder, chebdiff,  # noqa: F401
                        chebfit, chebgrid, chebint)
from .grid import GridFun, gridfit, gridint, gridval  # noqa: F401
from .util import (normalize_cheb_domain, normalize_domain,  # noqa: F401
                   normalize_spl_domain)
