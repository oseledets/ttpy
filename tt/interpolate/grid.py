from dataclasses import dataclass
from functools import wraps
from typing import Optional

import numpy as np
import scipy as sp
import scipy.integrate
from numpy.typing import ArrayLike

from tt.core.vector import TensorTrain
from tt.multifuncrs2 import multifuncrs2


@dataclass
class GridFun:
    """Class GridFun represents a function defined on a grid as a
    multidimensional array stored with tensor train and actions on the
    function.

    Args:
        values: Function values calculated in nodes of grid.
        grid: Set of nodes where function was evaluated. If grid is not
              specified then default grid on unit hypercude will be chosen.
    """

    values: TensorTrain

    grid: Optional[TensorTrain] = None

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.values.ndim

    @property
    def domain(self) -> ArrayLike:
        domain = np.empty((self.arity, 2))
        for i, core in enumerate(self.grid.cores):
            domain[i, 0] = core[0, -1, 0]
            domain[i, 1] = core[0, 0, 0]
        return domain

    def __post_init__(self):
        if self.grid is None:
            nodes = [gridnodes(x)[None, :, None] for x in self.values.shape]
            self.grid = TensorTrain.from_list(nodes)

    def __repr__(self) -> str:
        args = ', '.join([
            f'arity={self.arity}',
            f'nodes={self.values.shape}',
            f'ranks={self.values.ranks}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __call__(self, points: ArrayLike) -> ArrayLike:
        """Evaluate grid function at points on its domain

        Args:
            points: A single or a batch of d-dimensional points.
        """
        points = np.asarray(points)
        if points.ndim == 1:
            batch = points[None, :]
        elif points.ndim == 2:
            batch = points
        else:
            raise ValueError('Exepected one or two dimensional array but '
                             f'actual number of dimensions is {batch.ndim}.')

        nosamples, arity = batch.shape
        if arity != self.values.ndim:
            raise ValueError('Wrong number of arguments: function arity is '
                             f'{self.values.ndim} but not {arity}.')

        # Find the closest indices to points on the left.
        indices = np.empty_like(batch, dtype=np.int32)
        for i, nodes in enumerate(self.grid.cores):
            indices[:, i] = np.searchsorted(nodes.squeeze(), batch[:, i])
            indices[:, i] = np.minimum(indices[:, i], nodes.size - 1)

        # Sequentially calculate grid function values by index points.
        result = np.array([self.values[ix] for ix in indices])
        if points.ndim == 1:
            result = result[0, ...]
        return result

    def __getitem__(self, key):
        """Evaluate grid function in grid nodes.

        Args:
            key: Batch or a single d-dimensional index of a node.
        """
        if key is not Ellipsis:
            raise NotImplementedError
        return self.values

    def __itruediv__(self, other):
        if isinstance(other, (float, int)):
            # TODO: Define __itruediv__ for tensor train.
            self.weights.cores[0] /= other
            if self.values:
                self.values.cores[0] /= other
            return self
        else:
            return NotImplemented

    def trim(self, tol: float = 1e-14, max_rank: int = 1_000_000):
        """Apply TT-round procedure to underlying tensor train.
        """
        values = self.values.round(tol, max_rank)
        return GridFun(values, self.grid)


def gridfit(fn, grid: ArrayLike | TensorTrain, domain=None, tol: float = 1e-6,
            **kwargs) -> GridFun:
    """Evaluate function :fn: on a grid :grid: and represent function values on
    a grid as tensor train.
    """
    kwargs['eps_exit'] = tol
    if 'verb' not in kwargs:
        kwargs['verb'] = False

    if isinstance(grid, TensorTrain):
        if domain is not None:
            raise ValueError('Domain should not be specified if grid has '
                             'already specified as a tensor train.')
        # Assume that grid specified as a tensor train whose cores are one
        # dimensional grids and tensor train is just a tensor product.
        grid_cores = grid.cores
    else:
        raise NotImplementedError('Only grid represented as rank-1 tensor '
                                  'train are supported.')

    @wraps(fn)
    def fn_safe(xs: np.ndarray) -> np.ndarray:
        assert xs.ndim == 2, \
               f'Expected 2-tensor <n x d> but given {xs.ndim}-tensor.'
        ys = fn(xs)
        assert ys.ndim == 1, \
               f'Expected 1-tensor on output but returned {xs.ndim}-tensor.'
        assert ys.size == xs.shape[0], \
               'Number of samples differ for input and returned tensor: ' \
               f'{ys.size} != {xs.shape[0]}.'
        return ys

    # Since grid is a tensor product of one-dimensional grids, we can decompose
    # them and compose tensor trains to evaluate function in the spirit of
    # numpy.meshgrid.
    ones = [np.ones_like(grid) for grid in grid_cores]
    args = []
    for k, grid in enumerate(grid_cores):
        cores = ones[:k] + [grid] + ones[k + 1:]
        args.append(TensorTrain.from_list(cores))

    # Evaluate target function on a grid and return a grid function object.
    values = TensorTrain.from_train(multifuncrs2(args, fn_safe, **kwargs))
    grid = TensorTrain.from_list(grid_cores)
    return GridFun(values, grid)


def gridint(fn: GridFun, method: str = 'trapezoid') -> float:
    """Integrate grid function :fn: on its domain.
    """
    if method == 'trapezoid':
        integrate = sp.integrate.trapezoid
    elif method == 'simpson':
        integrate = sp.integrate.simps
    elif method == 'romb':
        integrate = sp.integrate.romb
    else:
        raise ValueError(f'Unknown integration method: {method}.')

    acc = np.ones(1)
    for core, grid in zip(fn.values.cores, fn.grid.cores):
        margin = integrate(core, grid.squeeze(), axis=1)
        acc = acc @ margin  # Marginalized core over external dim.
    return acc.item()


def gridnodes(nonodes: int) -> np.ndarray:
    """Function gridnodes returns one dimensional default uniform grid on
    standard interal from 0 to 1.
    """
    return np.linspace(0, 1, nonodes)


def gridval(fn: GridFun, xs: ArrayLike) -> ArrayLike:
    """Function gridval evaluates a grid function :fn: on point or batch of
    points :xs:.
    """
    return fn(xs)
