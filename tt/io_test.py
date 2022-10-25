import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tt import rand
from tt.core.tensor_train import TensorTrain
from tt.io import read, write


class TestIO:

    @pytest.mark.parametrize('is_complex', [False, True])
    @pytest.mark.parametrize('reader', ['python'])
    @pytest.mark.parametrize('writer', ['fortran', 'python'])
    def test_read_write(self, tmpdir, is_complex, reader, writer):
        path = tmpdir / 'tensor-train.data'
        ten = rand((2, 3, 4), 3)
        if not is_complex:
            inp = TensorTrain.from_list(ten.cores)
        else:
            inp = TensorTrain.from_list([core.astype(np.complex128)
                                         for core in ten.cores])

        write(path, inp, native=(writer == 'fortran'))
        out = read(path, native=(reader == 'fortran'))
        assert_array_equal(inp.cores[0], out.cores[0])
