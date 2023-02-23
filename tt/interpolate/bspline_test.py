import numpy as np
import pytest
import scipy as sp
import scipy.interpolate
import scipy.stats
from numpy.testing import assert_array_equal

from tt.core.vector import TensorTrain
from tt.interpolate.bspline import BSpline, bsplfit, bsplint


def fn(xs):
    xs = np.asarray(xs)
    if xs.ndim == 1:
        xs = xs[None, :]
    assert xs.ndim == 2
    _, ndim = xs.shape
    mean = np.zeros(ndim)
    cov = np.diag(np.full(ndim, 1.0))
    return sp.stats.multivariate_normal.pdf(xs, mean, cov)


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
@pytest.mark.parametrize('order', [3])
def test_bsplfit(ndim: int, order: int):
    np.random.seed(42)

    grid_size = 1001
    grid_cores = [np.linspace(-10, 10, grid_size)[None, :, None]] * ndim
    grid_spec = TensorTrain.from_list(grid_cores)
    grid_fn = bsplfit(fn, grid_spec, order=order)
    assert isinstance(grid_fn, BSpline)
    assert grid_fn.arity == ndim
    assert_array_equal(grid_fn.domain[:, 0], -10)
    assert_array_equal(grid_fn.domain[:, 1], 10)

    # Check shape of grid tensor train.
    assert grid_fn.grid.ndim == ndim
    assert grid_fn.grid.cores[0].shape == (1, grid_size, 1)

    # Check shape of knots tensor train.
    assert grid_fn.knots.ndim == ndim
    assert grid_fn.knots.cores[0].shape == (1, grid_size + 2 * order - 2, 1)

    # Check approximation error in the origin.
    origin = np.full((1, ndim), 0.0)
    lhs = fn(origin).squeeze()
    rhs = grid_fn(origin)
    assert lhs == pytest.approx(rhs, 1e-17)


@pytest.mark.parametrize('ndim', [1, 2, 3, 5])
@pytest.mark.parametrize('order', [3])
def test_bsplint(ndim: int, order: int):
    np.random.seed(42)

    grid_size = 1001
    grid_cores = [np.linspace(-10, 10, grid_size)[None, :, None]] * ndim
    grid_spec = TensorTrain.from_list(grid_cores)
    grid_fn = bsplfit(fn, grid_spec, order=order)

    integral = bsplint(grid_fn)
    assert integral == pytest.approx(1.0, 1e-12)
