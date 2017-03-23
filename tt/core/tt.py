# Here we import all necessary staff from external files
from __future__ import print_function, absolute_import, division

# main classes
from .matrix import matrix
from .vector import vector, tensor


# tools
from .tools import matvec, col, kron, dot, mkron, concatenate, sum, reshape
from .tools import eye, diag, Toeplitz, qshift, qlaplace_dd, IpaS
from .tools import ones, rand, linspace, sin, cos, delta, stepfun, unit, xfun

# utility
from . import utils





