# TODO: Generalize idct routine.
# TODO: Add vmap routine for function vectorization.

from typing import Literal, Optional

import scipy as sp
import scipy.fft

from tt.core.vector import TensorTrain


def idct(ten, type: Literal[1, 2, 3, 4] = 2, n: Optional[int] = None,
         axis: Optional[int] = None, norm=None):
    if axis is not None:
        raise NotImplementedError('IDCT applied along all axis.')

    def fn(core):
        return sp.fft.idct(core, type=type, n=n, axis=1, norm=norm)
    return TensorTrain.from_list([fn(core) for core in ten.cores])
