import numpy as np
import pytest
import tt

from copy import deepcopy

from numpy.testing import assert_array_almost_equal
from tt.riemannian import riemannian

try:
    import numba
except (ImportError, ModuleNotFoundError):
    numba = None


@pytest.mark.parametrize('debug', [False, True])
class TestTTLearning:

    def setUp(self):
        np.random.seed(2)  # Reproducibility.

    def test_projector_splitting_add(self, debug):
        Y = tt.rand([5, 2, 3], 3, [1, 2, 3, 1])
        my_res = riemannian.projector_splitting_add(Y.copy(), Y.copy(), debug)
        assert_array_almost_equal(2 * Y.full(), my_res.full())

    @pytest.mark.parametrize('use_jit', [False, True])
    def test_project(self, debug, use_jit):
        if use_jit and not numba:
            pytest.skip('Missing numba package.')

        def random_tanget_space_point(X):
            coresX = tt.tensor.to_list(X)
            point = 0 * tt.ones(X.n)
            for dim in range(X.d):
                curr = deepcopy(coresX)
                curr[dim] = np.random.rand(curr[dim].shape[0],
                                           curr[dim].shape[1],
                                           curr[dim].shape[2])
                point += tt.tensor.from_list(curr)
            return point

        X = tt.rand([4, 4, 4], 3, [1, 4, 4, 1])
        Z = random_tanget_space_point(X)
        PZ = riemannian.project(X, Z, use_jit=use_jit, debug=debug)
        assert_array_almost_equal(Z.full(), PZ.full())

        X = tt.rand([2, 3, 4], 3, [1, 5, 4, 1])
        Z = random_tanget_space_point(X)
        PZ = riemannian.project(X, Z, use_jit=use_jit, debug=debug)
        assert_array_almost_equal(Z.full(), PZ.full())

    @pytest.mark.parametrize('use_jit', [False, True])
    def test_project_sum_equal_ranks(self, debug, use_jit):
        if use_jit and not numba:
            pytest.skip('Missing numba package.')

        X = tt.rand([4, 4, 4], 3, [1, 4, 4, 1])
        Z = [0] * 7
        for idx in range(7):
            Z[idx] = tt.rand([4, 4, 4], 3, [1, 2, 3, 1])
        project_sum = riemannian.project(X, Z, use_jit=use_jit, debug=debug)

        sum_project = X * 0
        for idx in range(len(Z)):
            sum_project += riemannian.project(X, Z[idx], use_jit=use_jit,
                                              debug=debug)
        assert_array_almost_equal(sum_project.full(), project_sum.full())

    @pytest.mark.parametrize('use_jit', [False, True])
    def test_project_sum_different_ranks(self, debug, use_jit):
        if use_jit and not numba:
            pytest.skip('Missing numba package.')

        X = tt.rand([4, 4, 4], 3, [1, 4, 4, 1])
        Z = [0] * 7
        Z[0] = tt.rand([4, 4, 4], 3, [1, 4, 4, 1])
        Z[1] = tt.rand([4, 4, 4], 3, [1, 4, 3, 1])
        Z[2] = tt.rand([4, 4, 4], 3, [1, 2, 3, 1])
        for idx in range(3, 7):
            Z[idx] = tt.rand([4, 4, 4], 3, [1, 2, 2, 1])
        project_sum = riemannian.project(X, Z, use_jit=use_jit, debug=debug)

        sum_project = X * 0
        for idx in range(len(Z)):
            sum_project += riemannian.project(X, Z[idx], use_jit=use_jit,
                                              debug=debug)
        assert_array_almost_equal(sum_project.full(), project_sum.full())
