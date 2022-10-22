import numpy as np
import tt

from numpy.testing import assert_array_equal, assert_equal
from tt import vector


class TestVector:

    def test_sanity(self):
        cores = [np.ones((1, 2, 2)), np.ones((2, 3, 2)), np.ones((2, 4, 1))]
        tensor = vector.from_list(cores)
        assert tensor.dtype == np.float64
        assert tensor.ndim == 1
        assert tensor.size == 24
        assert tensor.shape == (24, )
        assert tensor.ranks == (1, 2, 2, 1)
        assert_array_equal(tensor.cores[0], cores[0])
        assert_array_equal(tensor.cores[1], cores[1])
        assert_array_equal(tensor.cores[2], cores[2])

    def test_qtt_vector(self, d=10, eps=1e-14):
        a = np.arange(2 ** d)
        v = tt.reshape(tt.vector(a), [2] * d, eps=eps)
        r = v.r
        assert np.issubdtype(r.dtype, np.int32), \
               'Vector ranks are not integers, expected int32'
        assert_equal(r[1:-1], [2] * (d - 1),
                     'This vector ranks should be exactly 2')
        b = v.full().reshape(-1, order='F')
        assert np.linalg.norm(a - b) < 10 * 2 ** d * eps, \
               'The approximation error is too large.'

    def test_assembly(self):
        # Test direct construction of tt.vector from specified kernels.
        d = 10
        h = [None] * d
        h[0] = np.zeros((1, 2, 2))
        h[-1] = np.zeros((2, 2, 1))

        h[0][:, 0, :] = [[0, 1]]
        h[0][:, 1, :] = [[1, 1]]

        for i in range(1, d - 1):
            h[i] = np.zeros((2, 2, 2))
            h[i][:, 0, :] = np.eye(2)
            h[i][:, 1, :] = [[1, 0], [2 ** i, 1]]

        h[-1][:, 0, :] = [[1], [0]]
        h[-1][:, 1, :] = [[1], [2 ** (d - 1)]]

        v = tt.vector.from_list(h)
        a = v.full().reshape(-1, order='F')
        b = np.arange(2 ** d)
        assert np.linalg.norm(a - b) < 1e-15, \
               'The approximation error is too large.'
