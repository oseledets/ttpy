import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_allclose

import tt.fft
from tt.core.vector import TensorTrain


@pytest.mark.parametrize('transform',
                         ['dct', 'dst', 'fft', 'idct', 'idst', 'ifft'])
def test_real_1d(transform: str, norm: str = 'ortho'):
    vec = np.arange(4) + 1
    mat = vec[None, :] * vec[:, None]
    ten = TensorTrain.from_list([vec[None, :, None], vec[None, :, None]])

    sp_transform = getattr(sp.fft, transform)
    tt_transform = getattr(tt.fft, transform)

    actual = tt_transform(ten, norm=norm).full()
    desired = sp_transform(mat, norm=norm)
    assert_allclose(actual, desired, atol=1e-12)


@pytest.mark.parametrize('transform',
                         ['dctn', 'dstn', 'fftn', 'idctn', 'idstn', 'ifftn'])
@pytest.mark.parametrize('norm', ['backward', 'forward', 'ortho'])
def test_real_nd(transform: str, norm: str):
    vec = np.arange(4) + 1
    mat = vec[None, :] * vec[:, None]
    ten = TensorTrain.from_list([vec[None, :, None], vec[None, :, None]])

    sp_transform = getattr(sp.fft, transform)
    tt_transform = getattr(tt.fft, transform)

    actual = tt_transform(ten, norm=norm).full()
    desired = sp_transform(mat, norm=norm)
    assert_allclose(actual, desired, atol=1e-12)
