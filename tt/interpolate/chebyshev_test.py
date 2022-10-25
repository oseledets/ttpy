import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

from tt.core.vector import TensorTrain
from tt.interpolate.chebyshev import chebfit, chebgrid, chebnodes


def fn(xs, std=1.0):
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


def test_chebnodes():
    nodes = chebnodes(0)
    assert nodes.size == 1
    assert nodes[0] == 0.0

    nodes = chebnodes(5)
    assert nodes.size == 6
    assert nodes[0] == 1.0
    assert nodes[1] == np.cos(np.pi / 5)
    assert nodes[5] == -1.0


def test_chebgrid():
    nonodes = 5
    nodes = chebnodes(nonodes)

    grid = chebgrid(nonodes)
    assert isinstance(grid, TensorTrain)
    assert grid.ndim == 1
    assert grid.shape == (nonodes + 1, )
    assert grid.ranks == (1, 1)
    assert_array_equal(grid.cores[0].squeeze(), nodes)

    grid = chebgrid(nonodes, 4)
    assert isinstance(grid, TensorTrain)
    assert grid.ndim == 4
    assert grid.shape == (nonodes + 1, ) * 4
    assert grid.ranks == (1, 1, 1, 1, 1)
    assert_array_equal(grid.cores[0].squeeze(), nodes)
    assert_array_equal(grid.cores[1].squeeze(), nodes)
    assert_array_equal(grid.cores[3].squeeze(), nodes)


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
def test_chebfit(ndim: int):
    np.random.seed(42)

    grid_spec = (20, ) * ndim
    grid_fn = chebfit(fn, grid_spec)

    # Check approximation error in the origin.
    origin = np.full((1, ndim), 0.0)
    lhs = fn(origin).squeeze()
    rhs = grid_fn(origin)
    assert lhs == approx(rhs, 1e-17)


def test_chebint():
    raise NotImplementedError
