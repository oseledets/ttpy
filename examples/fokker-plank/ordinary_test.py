from functools import partial

import numpy as np
import pytest
import scipy as sp
import scipy.integrate
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from tt.core.matrix import matrix
from tt.core.vector import TensorTrain
from tt.fft import idctn
from tt.interpolate import (Chebfun, Chebop, chebfit, chebgrid, chebint,
                            normalize_cheb_domain)

RK4_RESULT_RULE = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])


def rk4_step(fun, y0, t_span):
    time = t_span[0]
    step = t_span[1] - time
    ks = np.empty((4, ) + y0.shape)
    ks[0] = fun(time, y0)
    ks[1] = fun(time + 0.5 * step, y0 + 0.5 * ks[0])
    ks[2] = fun(time + 0.5 * step, y0 + 0.5 * ks[1])
    ks[3] = fun(time + 1.0 * step, y0 + 1.0 * ks[2])
    return y0 + step * np.tensordot(RK4_RESULT_RULE, ks, axes=(0, 0))


def odeint(fun, y0, t_span):
    return rk4_step(fun, np.asarray(y0), t_span)


def normal(xs, std=1.0):
    xs = np.asarray(xs)
    if xs.ndim == 1:
        xs = xs[None, :]
    assert xs.ndim == 2
    ndim = xs.shape[1]  # Tensor dimensions.
    prob = np.exp(-0.5 * np.sum(xs * xs, axis=1) / std ** 2)
    norm = (2 * np.pi * std ** 2) ** (ndim / 2)
    return prob / norm


def chebdiff(size: int, domain=None) -> np.ndarray:
    """Build discrete Laplace operator on Chebyshev grid.

    Parameters
    ----------
    n : int
        Number of Chebyshev nodes.
    domain: none, array_like
        Tuple of domain bounds.

    Returns
    -------
    numpy.ndarray
        Chebyshev differential matrix of order 2.

    See Also
    --------
    1. Trefethen L.N. - Spectral Methods in MATLAB. Chapter 6 // 2000.
    """
    if domain is None:
        domain = np.array([-1.0, 1.0])
    domain = np.asarray(domain)
    if domain.shape != (2, ):
        raise ValueError('Domain must be a 2-tuple-like value.')
    scale = 2 / (domain[1] - domain[0])

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
    mat *= scale
    return mat @ mat


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
            weights = idctn(values, type=1)

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
    def dirichlet(cls, spacial_grid, temporal_grid, diffusivity=1.0,
                  domain=(-1, 1)):
        """Create Laplace operator in real domain with zero Direchlet boundary
        conditions.
        """
        def make_core(size: int, subdomain):
            core = chebdiff(size, subdomain)
            # Zero Direchlet condition cab be implemented with zeroing elements
            # of differential matrix on the left and on the right sides.
            core[0, :] = 0
            core[-1, :] = 0
            return diffusivity * core
        return cls._from_factory(make_core, spacial_grid, temporal_grid,
                                 domain)

    @classmethod
    def _from_factory(cls, core_fn, spacial_grid, temporal_grid, domain):
        # Assume that temporal_grid grid is uniform.
        time_step = np.diff(temporal_grid)[0]
        scale = time_step / 2

        # Generate grid.
        if isinstance(spacial_grid, (list, tuple)):
            spacial_grid = chebgrid(spacial_grid)

        # Normalize domain specification (expected ndim x 2 array).
        domain = normalize_cheb_domain(domain, spacial_grid.ndim)

        # Prepare TT-cores.
        cache = {}
        cores = []
        for size, subdomain in zip(spacial_grid.shape, domain):
            if (core := cache.get(size - 1)) is None:
                core = core_fn(size - 1, subdomain)
                core = sp.linalg.expm(scale * core)
                core = core.reshape(1, -1, 1, order='F')
                cache[size] = core
            cores.append(core)

        # Create tensor train and Laplace operator in real domain.
        weights = TensorTrain.from_list(cores)
        return Laplace(weights, spacial_grid)


def solve_ivp(initial_fn, drift_fn, drift_grad_fn, diffusivity, spacial_grid,
              temporal_grid, domain=None, callback_fn=None, random_state=None):
    """Function solve_ivp solves initial value problem (IVP) for (fractional)
    Fokker-Plank eqution.
    """
    if not callable(callback_fn):
        callback_fn = lambda i, t, fn: None  # noqa

    if isinstance(spacial_grid, (list, tuple)):
        arity = len(spacial_grid)
    elif isinstance(spacial_grid, TensorTrain):
        arity = spacial_grid.ndim
    else:
        raise ValueError('Spacial grid should be either tuple, list or '
                         'tensor train.')

    domain = normalize_cheb_domain(domain, arity)
    time_step = np.diff(temporal_grid)[0]

    # Prepare differentiation matrices.
    laplace = Laplace.dirichlet(spacial_grid, temporal_grid, diffusivity,
                                domain)

    # Approximate initial conditions.
    grid_fn = chebfit(initial_fn, spacial_grid, domain)

    def solve_diffusion(grid_fn):
        # Apply Laplace operator on Chebyshev grid.
        grid_fn = laplace @ grid_fn
        # Truncate tensor train if its rank is overestimated.
        return grid_fn.trim()

    def convection_fn(time, input):
        """Right-hand side system of eqs along trajectories.
        """
        xs = input[..., :-1]
        ws = input[..., -1:]
        return np.hstack([
            drift_fn(time, xs),
            -ws * drift_grad_fn(time, xs).sum(axis=-1, keepdims=True),
        ])

    def calc_density(time, xs):
        """Calculate density (solution) at specified spacial point.
        """
        assert time.ndim == 0
        assert xs.ndim == 2

        # Integrate along trajectories back in time.
        sol = odeint(drift_fn, xs, (time, time - time_step))

        # Estimate density on the off-grid points with the interpolant. Also,
        # we should cut off spacial points which are out of domain.
        ys = np.clip(sol, domain[:, 0], domain[:, 1])
        vs = grid_fn(ys)[:, None]

        # Integrate system along trajectory forth in time over density.
        sol = odeint(convection_fn, np.hstack([ys, vs]),
                     (time - time_step, time))

        # Extract density values from solution of ODEs system.
        return sol[:, -1]

    def solve_convection(grid_fn, time):
        grid_fn = chebfit(partial(calc_density, time), spacial_grid, domain)
        grid_fn /= chebint(grid_fn)
        return grid_fn

    # Make splitted steps iteratively.
    callback_fn(0, temporal_grid[0], grid_fn)
    for i, time in enumerate(temporal_grid[1:], 1):
        grid_fn = solve_diffusion(grid_fn)
        grid_fn = solve_convection(grid_fn, time)
        grid_fn = solve_diffusion(grid_fn)
        callback_fn(i, time, grid_fn)
        if i == 100:
            break

    return grid_fn


@pytest.mark.parametrize('ndim,nonodes', [
    (1, 30),
    pytest.param(3, 19, marks=pytest.mark.slow),
    pytest.param(5, 50, marks=pytest.mark.slow),
])
def test_diffusion(ndim: int, nonodes: int, scale: float = 10,
                   time: float = 2.5):
    """Solve diffusion equation with Dirichlet boundary conditions and verify
    convergence.
    """
    np.random.seed(42)
    temporal_grid = np.linspace(0, time, 101)
    errors = np.empty_like(temporal_grid)
    ranks = np.empty_like(temporal_grid, np.int32)

    def analytic_fn(time: float, xs):
        """Analytical solution of the problem.
        """
        mean = np.zeros(ndim)
        std = np.sqrt(1 + 2 * 0.5 * time)
        cov = np.diag(np.full(ndim, std**2))
        return sp.stats.multivariate_normal.pdf(xs, mean, cov)

    def callback_fn(ix: int, time: float, grid_fn):
        """Callback function for monitoring convergence of solution.
        """
        exact_fn = chebfit(fn=lambda xs: analytic_fn(time, xs),
                           grid=(nonodes, ) * ndim,
                           domain=(-scale, scale))
        aerror = (exact_fn.values - grid_fn.values).norm()
        rerror = aerror / exact_fn.values.norm()
        errors[ix] = rerror
        ranks[ix] = max(grid_fn.values.ranks)

    grid_fn = solve_ivp(initial_fn=lambda x: analytic_fn(0, x),
                        drift_fn=lambda t, x: np.zeros_like(x),
                        drift_grad_fn=lambda t, x: np.zeros_like(x),
                        diffusivity=0.5,
                        callback_fn=callback_fn,
                        spacial_grid=(nonodes, ) * ndim,
                        domain=(-scale, scale),
                        temporal_grid=temporal_grid)

    assert ranks[-1] <= 10
    assert_almost_equal(errors[-1], 0.0, decimal=1)

    shape = (11, ) * ndim
    indicies = [np.linspace(-1, 1, size) for size in shape]
    mesh = np.stack(np.meshgrid(*indicies))
    mesh = mesh.reshape(ndim, -1).T

    lhs = analytic_fn(time, mesh)
    lhs = lhs.reshape(shape)

    rhs = grid_fn(mesh)
    rhs = rhs.reshape(shape)

    aerr = np.linalg.norm(rhs - lhs)
    rerr = aerr / np.linalg.norm(lhs)
    assert_almost_equal(rerr, 0.0, decimal=1)
    assert_array_almost_equal(lhs, rhs, decimal=3)


def test_dumbbell():
    pass


def test_ornstein_uhlenbeck():
    pass
