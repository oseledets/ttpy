import numpy as np

from numpy.testing import assert_array_equal

from tt.core.matrix import matrix
from tt.core.tools import ones


class TestMatrix:

    def test_sanity(self):
        cores = [
            np.ones((1, 2, 2, 2)),
            np.ones((2, 3, 3, 2)),
            np.ones((2, 4, 4, 1)),
        ]
        mat = matrix.from_list(cores)
        assert mat.dtype == np.float64
        assert mat.ndim == 2
        assert mat.size == 4 * 9 * 16
        assert mat.shape == (24, 24)
        assert mat.ranks == (1, 2, 2, 1)
        assert_array_equal(mat.cores[0], cores[0])
        assert_array_equal(mat.cores[1], cores[1])
        assert_array_equal(mat.cores[2], cores[2])

    def test_matvec(self):
        cores = [
            np.ones((1, 4, 2, 2)),
            np.ones((2, 3, 3, 2)),
            np.ones((2, 2, 4, 1)),
        ]
        mat = matrix.from_list(cores)
        vec = ones((2, 3, 4))
        res = mat @ vec
        assert res.ndim == 1
        assert res.shape == (24, )
        assert res.ranks == (1, 2, 2, 1)
        assert_array_equal(res.cores[0], 2)
        assert_array_equal(res.cores[1], 3)
        assert_array_equal(res.cores[2], 4)
