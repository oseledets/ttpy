"""Subpackage core defines TT storage class and common types and routines to
work with that types in TT-representation.
"""

from . import utils  # noqa: F401
from .matrix import matrix  # noqa: F401
from .tools import dot  # noqa: F401
from .tools import xfun  # noqa: F401
from .tools import (IpaS, Toeplitz, col, concatenate, cos, delta,  # noqa: F401
                    diag, eye, kron, linspace, matvec, mkron, ones, permute,
                    qlaplace_dd, qshift, rand, reshape, sin, stepfun, sum,
                    unit)
from .vector import tensor, vector  # noqa: F401
