import numpy as np

from numpy.testing import assert_array_equal

from tt.core.matrix import matrix


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
