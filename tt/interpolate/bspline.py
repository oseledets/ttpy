from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence, Tuple, TypeVar

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from scipy.interpolate import make_interp_spline

from tt.core.vector import TensorTrain
from tt.interpolate.grid import gridfit

# Type variables for functional transformations on BSpline object.
Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


@dataclass
class BSpline:
    """Class BSpline represents a multi-dimensional B-spline interpolant which
    interpolation coefficients are represent as a tensor train (TT).
    """

    weights: TensorTrain

    grid: TensorTrain

    knots: TensorTrain

    order: int

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.grid.ndim

    @property
    def domain(self) -> np.ndarray:
        domain = np.empty((self.arity, 2))
        for i, core in enumerate(self.grid.cores):
            domain[i, 0] = core[0, 0, 0]
            domain[i, 1] = core[0, -1, 0]
        return domain

    def __repr__(self) -> str:
        args = ', '.join([
            f'arity={self.arity}',
            f'nodes={self.weights.shape}',
            f'ranks={self.weights.ranks}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __call__(self, points: np.ndarray) -> np.ndarray:
        batch = np.asarray(points)
        if batch.ndim == 1:
            batch = batch[None, :]
        elif batch.ndim != 2:
            raise ValueError('Exepected one or two dimensional array but '
                             f'actual number of dimensions is {batch.ndim}.')

        nosamples, arity = batch.shape
        if arity != self.weights.ndim:
            raise ValueError('Wrong number of arguments: function arity is '
                             f'{self.weights.ndim} but not {arity}.')

        def fn(acc: np.ndarray, spline, ranks, xs: np.ndarray) -> np.ndarray:
            ys = spline(xs).reshape(xs.size, *ranks)
            return np.einsum('ni,nij->nj', acc, ys)

        init = np.ones((batch.shape[0], 1))
        acc = self.reduce(fn, init, batch.T)
        return acc[..., 0]

    def __getitem__(self, key):
        raise NotImplementedError

    def reduce(self, fn: Callable[[Carry, Any, Any, X], Carry], init: Carry,
               seq: Iterable[X]) -> Carry:
        """Method reduce applies reduce transformation to each core of tensor
        train which represents multivariate spline.

        .. code-block:: haskell

          reduce :: (c -> BSpline -> Shape -> x -> c) -> c -> [x] -> c
        """
        def extend(carry: Carry, spline, ranks, item: X) -> Tuple[Carry, None]:
            return fn(carry, spline, ranks, item), None
        acc, _ = self.scan(extend, init, seq)
        return acc

    def scan(self, fn: Callable[[Carry, Any, Any, X], Tuple[Carry, Y]],
             init: Carry, seq: Iterable[X]) -> Tuple[Carry, Sequence[Y]]:
        """Method scan applies scan transformation to cores in tensor train
        which represents control points (weights) of the B-spline.

        .. code-block:: haskell

          scan :: (c -> BSpline -> Shape, x -> (c, y)) -> c -> [x] -> (c, [y])
        """
        carry = init
        items = []
        for core, knots, el in zip(self.weights.cores, self.knots.cores, seq):
            # Convert weights core to proper shape and construct temporary
            # univariate BSpline object.
            core = core.swapaxes(0, 1)
            control_points = core.reshape(core.shape[0], -1)
            spline = sp.interpolate.BSpline.construct_fast(
                knots.squeeze(), control_points, self.order)
            # Invoke transformation and accumulate its sequencial output.
            carry, item = fn(carry, spline, core.shape[1:], el)
            items.append(item)
        return carry, items


def bsplfit(fn: Callable[..., np.ndarray], grid, domain=None, order: int = 3,
            bc_type=None, tol: float = 1e-6, **kwargs) -> BSpline:
    """Build multi-variate B-spline interpolant for a function :fn: with
    interpolation coefficients represented as a tensor train.
    """
    # Evaluate target function on a grid for further interpolation.
    grid_fn = gridfit(fn, grid, domain, tol, **kwargs)

    # Build interpolant over external dim for each core of grid function.
    knots_cores = []
    spline_cores = []
    for core, grid in zip(grid_fn.values.cores, grid_fn.grid.cores):
        spline = make_interp_spline(grid.squeeze(), core, order,
                                    bc_type=bc_type, axis=1)
        spline_core = np.moveaxis(spline.c, 0, 1)
        spline_cores.append(spline_core)
        knots_core = spline.t[None, :, None]
        knots_cores.append(knots_core)

    weights = TensorTrain.from_list(spline_cores)
    knots = TensorTrain.from_list(knots_cores)
    return BSpline(weights, grid_fn.grid, knots, order)


def bsplint(grid_fn: BSpline) -> float:
    """Function bsplint computes a definite integral over entire domain of the
    B-spline.
    """
    def integrate(carry, spline, ranks, limits):
        factor = spline \
            .integrate(*limits) \
            .reshape(*ranks)
        return carry @ factor
    limits = grid_fn.domain
    init = np.ones(1)
    return grid_fn.reduce(integrate, init, limits)


def bsplval(grid_fn: BSpline, xs: ArrayLike) -> ArrayLike:
    """Function bsplval evaluates a B-spline interpolant :fn: on a single point
    or batch of points :xs:.
    """
    return grid_fn(xs)
