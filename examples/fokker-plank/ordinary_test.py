from functools import partial

import numpy as np
import scipy as sp
from numpy.testing import assert_array_almost_equal

from tt.core.matrix import matrix
from tt.core.vector import TensorTrain
from tt.interpolate import Chebfun, Chebop, chebfit, chebgrid
from tt.interpolate.fft import idct  # TODO


def normal(xs, std=1.0):
    xs = np.asarray(xs)
    if xs.ndim == 1:
        xs = xs[None, :]
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


def chebdiff(size: int) -> np.ndarray:
    """Build discrete Laplace operator on Chebyshev grid.

    Parameters
    ----------
    n : int
        Number of Chebyshev nodes.

    Returns
    -------
    numpy.ndarray
        Chebyshev differential matrix of order 2.

    See Also
    --------
    1. Trefethen L.N. - Spectral Methods in MATLAB. Chapter 6 // 2000.
    """
    if size == 0:
        return np.array([[0]]), np.array([1.0])

    # Build Chebyshev grid.
    ns = np.arange(size + 1)
    grid = np.cos(np.pi * ns / size)

    # Prepare signs in nominator of matrix elements.
    signs = np.ones_like(ns, dtype=np.float64)
    signs[0] = 2
    signs[size] = 2
    signs *= (-1) ** ns
    nomin = signs[:, None] / signs[None, :]

    # Calculate denominators for matrix elements (i.e. x_i - x_j).
    denom = np.eye(size + 1) + grid[:, None] - grid[None, :]

    # Calculate matrix elements and set diagonal elements.
    mat = nomin / denom
    mat -= np.diag(mat.sum(axis=1))
    mat = mat @ mat

    return mat


class Laplace(Chebop):
    """Class Laplace implements a Laplace operator in real domain.
    """

    def __matmul__(self, other) -> Chebfun:
        if isinstance(other, Chebfun):
            # Cast Laplace operator to TT-matrix.
            shape = self.grid.shape
            shapes = [*zip(shape, shape)]
            laplace = matrix.from_train(self.weights, shapes)

            # Get values of on a Chebyshev grid in TT and apply operator.
            values = other[...]
            values = laplace @ values

            # Now, use values to build Chebyshev interpolant. In fact, we could
            # use tt.interpolate.chebyshev.chebfit but it requires grid instead
            # of indicies. So, we have no choice but to repeat once agian.
            weights = idct(values, type=1)

            # Adjust coefficients of Chebyshev series to IDCT-1.
            for core, size in zip(weights.cores, weights.shape):
                core[:, 1:-1, :] *= 2
                if size % 2 == 1:
                    core[:, -1, :] = -core[:, -1, :]

            # Apply Dicrete Cosine Transform to a tensor of values on the grid.
            return Chebfun(weights, values, other.grid)
        else:
            return NotImplemented

    @classmethod
    def dirichlet(cls, spacial_grid, temporal_grid):
        """Create Laplace operator in real domain with zero Direchlet boundary
        conditions.
        """
        def make_core(size: int):
            core = chebdiff(size)
            # Zero Direchlet condition cab be implemented with zeroing elements
            # of differential matrix on the left and on the right sides.
            core[0, :] = 0
            core[-1, :] = 0
            return core
        return cls._from_core_factory(spacial_grid, temporal_grid, make_core)

    @classmethod
    def _from_core_factory(cls, spacial_grid, temporal_grid, core_fn):
        # Assume that temporal_grid grid is uniform.
        time_step = np.diff(temporal_grid)[0]
        scale = time_step / 2

        # Prepare TT-cores.
        cache = {}
        cores = []
        for size in spacial_grid.shape:
            if (core := cache.get(size - 1)) is None:
                core = core_fn(size - 1)
                core = sp.linalg.expm(scale * core)
                core = core.reshape(1, -1, 1, order='F')
                cache[size] = core
            cores.append(core)

        # Create tensor train and Laplace operator in real domain.
        weights = TensorTrain.from_list(cores)
        return Laplace(weights, spacial_grid)


def solve_ivp(initial_fn, drift_fn, drift_grad_fn, spacial_grid,
              temporal_grid, random_state=None):
    """Function solve_ivp solves initial value problem (IVP) for (fractional)
    Fokker-Plank eqution.
    """
    # Prepare differentiation matrices.
    laplace = Laplace.dirichlet(spacial_grid, temporal_grid)

    # Make initialization.
    grid_fn = chebfit(initial_fn, spacial_grid)
    grid_fn = laplace @ grid_fn

    def solve_diffusion(grid_fn):
        # Apply Laplace operator on Chebyshev grid.
        grid_fn = laplace @ grid_fn
        # Optionally, we can truncate tensor train if its rank is
        # overestimated.
        return grid_fn.trim()

    def solve_convection(grid_fn):
        return grid_fn

    # Make splitted steps iteratively.
    for i, time in enumerate(temporal_grid[1:], 1):
        grid_fn = solve_diffusion(grid_fn)
        grid_fn = solve_convection(grid_fn)
        grid_fn = solve_diffusion(grid_fn)

    return grid_fn


def test_diffusion(ndim: int = 1):
    np.random.seed(42)
    np.set_printoptions(linewidth=160)

    grid_fn = solve_ivp(initial_fn=partial(normal, std=0.05),
                        drift_fn=None,
                        drift_grad_fn=None,
                        spacial_grid=chebgrid((50, ) * ndim),
                        temporal_grid=np.linspace(0, 0.01, 31))

    xs = np.linspace(-1, 1, 101)
    lhs = normal(xs[:, None], std=0.01 + np.sqrt(2 * 0.01))
    rhs = grid_fn(xs[:, None])
    assert_array_almost_equal(lhs, rhs, decimal=2)


def test_dumbbell():
    pass


def test_ornstein_uhlenbeck():
    pass
