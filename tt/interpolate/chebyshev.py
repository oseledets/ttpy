from dataclasses import dataclass
from functools import wraps
from typing import Optional

from numpy.typing import ArrayLike
import numpy as np

from tt.core.vector import TensorTrain
from tt.interpolate.fft import idct
from tt.multifuncrs2 import multifuncrs2


@dataclass
class Chebfun:
    """Class Chebfun represents a multi-dimensional Chebyshev polynomial with
    coefficients stored in TT-format.
    """

    coef: TensorTrain
    grid: Optional[TensorTrain] = None

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.coef.ndim

    def __post_init__(self):
        if self.grid is None:
            self.grid = chebgrid([size - 1 for size in self.coef.shape])

    def __repr__(self) -> str:
        args = ', '.join([
            f'arity={self.arity}',
            f'nodes={self.coef.shape}',
            f'ranks={self.coef.ranks}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __call__(self, points: np.ndarray):
        """Evaluate interpolant at points on standard domain :math:`[-1, 1]`.

        Args:
            points: A single or a batch of d-dimensional points.
        """
        batch = np.asarray(points)
        if batch.ndim == 1:
            batch = batch[None, :]
        elif batch.ndim != 2:
            raise ValueError('Exepected one or two dimensional array but '
                             f'actual number of dimensions is {batch.ndim}.')

        nosamples, arity = batch.shape
        if arity != self.coef.ndim:
            raise ValueError('Wrong number of arguments: function arity is '
                             f'{self.coef.ndim} but not {arity}.')

        # Evaluate Chebyshev polynomial basis on the fly and contract it with
        # coefficients in TT-representation from left-to-right.
        zs = np.ones((1, nosamples))
        for xs, core, size in zip(batch.T, self.coef.cores, self.coef.shape):
            # Apply Horner's method for Chebyshev polynomial evaluation.
            ys = np.empty((size, xs.size))
            ys[0] = np.ones_like(xs)
            ys[1] = xs
            for i in range(2, size):
                ys[i] = 2 * xs * ys[i - 1] - ys[i - 2]

            # Adjust coefficients to used variation IDCT.
            ys[1:-1] *= 2
            if size % 2 == 1:
                ys[-1] = -ys[-1]

            # Contract with TT-core.
            zs = np.einsum('in,ijk,jn->kn', zs, core, ys)

        # By the moment, there is only one element along the last axis.
        return zs[..., 0].squeeze()

    def __getitem__(self, key):
        """Evaluate interpolant in grid nodes.

        Args:
            key: Batch or a single d-dimensional index of a node.
        """
        raise NotImplementedError


def chebnodes(nonodes: int):
    if nonodes == 0:
        return np.zeros(1)
    return np.cos(np.pi * np.arange(nonodes + 1) / nonodes)


def chebgrid(nonodes, ndim=None):
    nonodes = np.array(nonodes)
    if nonodes.ndim == 0:
        nonodes = nonodes[None]
    if ndim is None:
        ndim = nonodes.size
    if nonodes.size != ndim:
        nonodes = np.repeat(nonodes, ndim)

    nodes = [chebnodes(x)[None, :, None] for x in nonodes]
    return TensorTrain.from_list(nodes)


def chebfit(fn, grid: ArrayLike, tol: float = 1e-6, **kwargs) -> Chebfun:
    """Approximate function `fn` on multi-dimensional Chebyshev grid `grid`
    with specified precision `tol`. The function leverages cross-approximation
    methods building grid function.
    """
    kwargs['eps_exit'] = tol
    if 'verb' not in kwargs:
        kwargs['verb'] = False

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

    if isinstance(grid, (list, tuple)):
        grid_cores = [chebnodes(nonodes)[None, :, None] for nonodes in grid]
    elif isinstance(grid, TensorTrain):
        # Assume that grid specified as a tensor train whose cores are one
        # dimensional grids and tensor train is just a tensor product.
        grid_cores = chebgrid(grid).cores
    else:
        raise NotImplementedError('Either grid represented as rank-1 tensor '
                                  'train or a sequence of number of Chebyshev '
                                  'nodes are supported.')

    # Since grid is a tensor product of one-dimensional grids, we can decompose
    # them and compose tensor trains to evaluate function in the spirit of
    # numpy.meshgrid.
    ones = [np.ones_like(grid) for grid in grid_cores]
    args = []
    for k, grid in enumerate(grid_cores):
        cores = ones[:k] + [grid] + ones[k + 1:]
        args.append(TensorTrain.from_list(cores))

    # Evaluate target function on a Chebyshev grid for further interpolation.
    values = multifuncrs2(args, fn_safe, **kwargs)
    coefs = idct(values, type=1)

    # Apply Dicrete Cosine Transform to a tensor of values on the grid.
    return Chebfun(coefs)


def chebint():
    pass
