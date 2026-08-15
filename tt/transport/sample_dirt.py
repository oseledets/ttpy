"""Compatibility facade for the decomposed Sample-DIRT implementation.

New code should import the public API from :mod:`tt.transport`. This module
continues to expose the former combined namespace for downstream code that used
:mod:`tt.transport.sample_dirt` directly.
"""

from . import _basis as _basis_module
from . import _types as _types_module
from . import _torch_density as _torch_density_module
from . import scalar as scalar_module
from . import coordinates as coordinates_module
from . import positive as positive_module
from . import polynomial as polynomial_module
from . import wrappers as wrappers_module
from . import _torch_als as _torch_als_module
from . import _optimizers as _optimizers_module
from . import _fit_spline as _fit_spline_module
from . import _fit_coordinates as _fit_coordinates_module
from . import _fit_positive as _fit_positive_module
from . import _fit_cell as _fit_cell_module
from . import transport as transport_module
from . import refinement as refinement_module

from ._basis import *
from ._types import *
from ._torch_density import *
from .scalar import *
from .coordinates import *
from .positive import *
from .polynomial import *
from .wrappers import *
from ._torch_als import *
from ._optimizers import *
from ._fit_spline import *
from ._fit_coordinates import *
from ._fit_positive import *
from ._fit_cell import *
from .transport import *
from .refinement import *

__all__ = [
    name
    for module in (
    _basis_module,
    _types_module,
    _torch_density_module,
    scalar_module,
    coordinates_module,
    positive_module,
    polynomial_module,
    wrappers_module,
    _torch_als_module,
    _optimizers_module,
    _fit_spline_module,
    _fit_coordinates_module,
    _fit_positive_module,
    _fit_cell_module,
    transport_module,
    refinement_module
    )
    for name in module.__all__
]
