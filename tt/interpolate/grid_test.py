import numpy as np
import pytest

from tt.core.vector import TensorTrain
from tt.interpolate.grid import gridfit, gridint


def fn(xs, std=1.0):
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
def test_gridfit(ndim: int):
    np.random.seed(42)

    grid_cores = [np.linspace(-1, 1, 11)[None, :, None]] * ndim
    grid_spec = TensorTrain.from_list(grid_cores)
    grid_fn = gridfit(fn, grid_spec)
    assert isinstance(grid_fn.values, TensorTrain)
    assert isinstance(grid_fn.grid, TensorTrain)

    # Check approximation error in the origin.
    origin = np.full((1, ndim), 0.0)
    lhs = fn(origin)
    rhs = grid_fn(origin)
    assert lhs.shape == (1, )
    assert lhs.shape == rhs.shape
    assert lhs.squeeze() == pytest.approx(rhs.squeeze(), 1e-17)


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
def test_gridint(ndim: int):
    np.random.seed(42)

    grid_cores = [np.linspace(-10, 10, 1001)[None, :, None]] * ndim
    grid_spec = TensorTrain.from_list(grid_cores)
    grid_fn = gridfit(fn, grid_spec)

    integral = gridint(grid_fn)
    assert integral == pytest.approx(1.0, 1e-12)
