import unittest
import tt
import numpy as np

class TestVector(unittest.TestCase):
    def test_qtt_vector(self):
        d = 10
        eps = 1e-14
        a = np.arange(2**d)
        v = tt.reshape(tt.vector(a), [2]*d, eps=eps)
        r = v.r
        self.assertEqual(r.dtype, np.int32, 'Vector ranks are not integers, expected int32')
        self.assertTrue((r[1:-1] == 2).all(), 'This vector ranks should be exactly 2')
        b = v.full().reshape(-1, order='F')
        self.assertTrue(np.linalg.norm(a - b) < 10 * 2**d * eps, 'The approximation error is too large')
    
    def test_assembly(self):
        # Test direct construction of tt.vector from specified kernels
        d = 10
        h = [None] * d
        h[0] = np.zeros((1, 2, 2))
        h[-1] = np.zeros((2, 2, 1))

        h[0][:, 0, :] = [[0, 1]]
        h[0][:, 1, :] = [[1, 1]]

        for i in range(1, d-1):
            h[i] = np.zeros((2, 2, 2))
            h[i][:, 0, :] = np.eye(2)
            h[i][:, 1, :] = [[1, 0], [2**i, 1]]

        h[-1][:, 0, :] = [[1], [0]]
        h[-1][:, 1, :] = [[1], [2**(d-1)]]

        v = tt.vector.from_list(h)
        a = v.full().reshape(-1, order='F')
        b = np.arange(2**d)
        self.assertTrue(np.linalg.norm(a - b) < 1e-15, 'The approximation error is too large')
