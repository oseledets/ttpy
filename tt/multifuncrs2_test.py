import numpy as np
import pytest

from tt import rand
from tt.core.vector import TensorTrain
from tt.multifuncrs2 import multifuncrs2


def fn(xs, std=1.0):
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


class TestMultiFuncrs2:

    def test_incorrect(self):
        def fn(xs):
            return xs

        xs = rand((2, 3, 4), 3)

        with pytest.raises(ValueError):
            multifuncrs2([], fn)

        with pytest.raises(ValueError):
            multifuncrs2([xs], [fn, fn])

    @pytest.mark.parametrize('ndim,nonodes', [
        (2, 4),
        (3, 8),
        pytest.param(5, 32, marks=pytest.mark.slow),
    ])
    def test_chebyshev_grid(self, ndim, nonodes):
        grid_shape = (nonodes + 1, ) * ndim
        grid = np.cos(np.pi * np.arange(nonodes + 1) / nonodes)
        grids = (grid, ) * ndim

        # Build input tensor list for multifuncrs2 out of one-dimensional grid.
        core_grid = grid[None, :, None]
        core_ones = np.ones_like(core_grid)
        xs = []
        for axis in range(ndim):
            cores = [core_ones] * axis
            cores += [core_grid]
            cores += [core_ones] * (ndim - axis - 1)
            tensor = TensorTrain.from_list(cores)
            xs.append(tensor)

        # Build a batch of all possible node coordinates on the grid.
        mesh = np.stack(np.meshgrid(*grids))
        mesh = mesh.reshape(ndim, -1).T
        assert mesh.shape[1] == ndim

        # Evaluate function on grid in dense and in TT-format.
        lhs = fn(mesh).reshape(grid_shape)
        rhs = multifuncrs2(xs, fn, verb=False).full()

        # Check relative L2 norm of difference of an dense and TT-tensors.
        norm_l2 = np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)
        assert norm_l2 == pytest.approx(0)
