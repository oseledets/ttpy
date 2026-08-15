"""Import shim: the coagulation model lives with its example, not in the package.

``tt.algs.convolution`` holds the reusable primitive; the model
(kernels, right-hand side, predictor-corrector, solve) is
``examples/smoluchowski/solver.py``.  The acceptance tests import it from
there, the same way ``tests/test_examples.py`` imports the other examples.
"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent / "examples" / "smoluchowski"
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from solver import *          # noqa: F401,F403
from solver import (additive_kernel, ballistic_kernel, coagulation_rhs,  # noqa: F401
                    constant_kernel, predictor_corrector_step, solve)
from tt.algs.convolution import (component_sum, trapezoidal_convolution,  # noqa: F401
                                 trapezoidal_weights)
