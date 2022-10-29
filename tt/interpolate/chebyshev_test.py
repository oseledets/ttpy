import numpy as np
import pytest
import scipy as sp
import scipy.fft
from numpy.polynomial.hermite_e import hermeval
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest import approx

from tt.core.vector import TensorTrain
from tt.interpolate.chebyshev import (Chebop, chebder, chebdiff, chebfit,
                                      chebgrid, chebnodes)


def fn(xs, std=1.0):
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


def fn_grad(xs, std=1.0, order=1):
    weights = np.zeros(order + 1)
    weights[order] = 1
    ys = hermeval(xs / std, weights)
    if order % 2 == 1:
        ys = -ys
    zs = fn(xs, std)
    res = np.prod(ys, axis=1) * zs
    return res


class TestChebop:

    @pytest.mark.parametrize('ndim', [1, 2, 3, 5])
    @pytest.mark.parametrize('order', [1, 2])
    def test_derivative(self, ndim: int, order: int):
        np.random.seed(42)

        grid_spec = (20, ) * ndim
        grid_op = Chebop.derivative(grid_spec)
        cores = grid_op.weights.cores

        grid_fn = chebfit(fn, grid_spec)
        grid_fn_grad = grid_op @ grid_fn
        cores = grid_fn_grad.weights.cores
        assert grid_fn_grad.arity == grid_fn.arity
        assert grid_fn_grad.weights.shape == grid_fn.weights.shape

        xs = np.vstack([np.full(ndim, -1), np.zeros(ndim), np.ones(ndim)])

        # Validate function approximation.
        lhs = grid_fn(xs)
        rhs = fn(xs)
        assert_almost_equal(lhs, rhs)

        # Validate approximation of derivative.
        lhs = grid_fn_grad(xs)
        rhs = fn_grad(xs)
        assert_almost_equal(lhs, rhs)


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


def test_chebdiff(ndim: int = 1, size: int = 10):
    # Build an interpolation for quadratic function.
    xs = chebnodes(size)
    ys = xs ** 2
    zs = sp.fft.idct(ys, type=1)
    zs[1:-1] *= 2
    if size % 2 == 1:
        zs[-1] = -zs[-1]
    zs[np.isclose(zs, 0)] = 0

    # Sanity check: zero-order derivative.
    vals = np.polynomial.chebyshev.chebval([-1, -0.5, 0, 0.5, 1], zs)
    assert_array_equal(vals, [1, 0.25, 0, 0.25, 1])

    # Diff check: first-order derivative.
    D1 = chebdiff(size)
    vals = np.polynomial.chebyshev.chebval([-1, -0.5, 0, 0.5, 1], D1 @ zs)
    assert_array_equal(vals, [-2, -1, 0, 1, 2])

    # Laplace check: second-order derivative.
    D2 = chebdiff(size, 2)
    vals = np.polynomial.chebyshev.chebval([-1, -0.5, 0, 0.5, 1], D2 @ zs)
    assert_array_equal(vals, [2, 2, 2, 2, 2])


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
@pytest.mark.parametrize('order', [1, 2])
def test_chebder(ndim: int, order: int):
    np.random.seed(42)

    grid_spec = (20, ) * ndim
    grid_fn = chebfit(fn, grid_spec)
    grid_fn_grad = chebder(grid_fn)
    assert grid_fn_grad.arity == grid_fn.arity
    assert grid_fn_grad.weights.shape == grid_fn.weights.shape

    xs = np.vstack([np.full(ndim, -1), np.zeros(ndim), np.ones(ndim)])
    lhs = grid_fn_grad(xs)
    rhs = fn_grad(xs)
    assert_almost_equal(lhs, rhs)


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
