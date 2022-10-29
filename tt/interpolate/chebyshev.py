from dataclasses import dataclass
from functools import wraps
from math import comb, factorial
from typing import Optional

import numpy as np
import scipy as sp
import scipy.sparse
from numpy.typing import ArrayLike

from tt.core.matrix import matrix
from tt.core.vector import TensorTrain, vector
from tt.interpolate.fft import idct
from tt.multifuncrs2 import multifuncrs2


@dataclass
class Chebfun:
    """Class Chebfun represents a multi-dimensional Chebyshev polynomial with
    coefficients stored in TT-format.
    """

    weights: TensorTrain

    grid: Optional[TensorTrain] = None

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.weights.ndim

    def __post_init__(self):
        if self.grid is None:
            self.grid = chebgrid([size - 1 for size in self.weights.shape])

    def __repr__(self) -> str:
        args = ', '.join([
            f'arity={self.arity}',
            f'nodes={self.weights.shape}',
            f'ranks={self.weights.ranks}',
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
        if arity != self.weights.ndim:
            raise ValueError('Wrong number of arguments: function arity is '
                             f'{self.weights.ndim} but not {arity}.')

        # Evaluate Chebyshev polynomial basis on the fly and contract it with
        # coefficients in TT-representation from left-to-right.
        zs = np.ones((1, nosamples))
        for core, xs, size in zip(self.weights.cores, batch.T,
                                  self.weights.shape):
            # Apply Horner's method for Chebyshev polynomial evaluation.
            ys = np.empty((size, xs.size))
            ys[0] = np.ones_like(xs)
            ys[1] = xs
            for i in range(2, size):
                ys[i] = 2 * xs * ys[i - 1] - ys[i - 2]

            # Contract with TT-core.
            zs = np.einsum('in,ijk,jn->kn', zs, core, ys)

        # By the moment, there is only one element along the last axis.
        return zs[0, ...].squeeze()

    def __getitem__(self, key):
        """Evaluate interpolant in grid nodes.

        Args:
            key: Batch or a single d-dimensional index of a node.
        """
        raise NotImplementedError


@dataclass
class Chebop:
    """Class Chebop represents an operator which can be applied to a
    multi-dimensional Chebyshev polynomial with coefficients stored in
    TT-format.
    """

    weights: TensorTrain

    grid: Optional[TensorTrain] = None

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.weights.ndim

    def __post_init__(self):
        if self.grid is None:
            self.grid = chebgrid([size - 1 for size in self.weights.shape])

    def __repr__(self) -> str:
        args = ', '.join([
            f'arity={self.arity}',
            f'nodes={self.weights.shape}',
            f'ranks={self.weights.ranks}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __matmul__(self, other) -> Chebfun:
        if isinstance(other, Chebfun):
            shape = other.weights.shape
            mat = matrix.from_train(self.weights, [*zip(shape, shape)])
            vec = vector.from_train(other.weights)
            res = mat @ vec
            weights = TensorTrain.from_list(res.cores)
            return Chebfun(weights, other.grid)
        else:
            return NotImplemented

    @classmethod
    def derivative(cls, shape, order: int = 1) -> 'Chebop':
        """Create Chebop to calculate derivative of a given order of a Chebfun.
        """
        ops: dict[str, np.ndarray] = {}
        cores = []
        for size in shape:
            if (op := ops.get(size)) is None:
                op = chebdiff(size, order).todense()
                op = op.reshape(1, -1, 1, order='F')
                ops[size] = op
            cores.append(op)
        weights = TensorTrain.from_list(cores)
        return cls(weights)


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


def chebdiff(size: int, order: int = 1) -> sp.sparse.csr_array:
    """Generate differentiation matricies of series pres
    """
    if order == 0:
        return np.eye(size)
    elif order > size:
        return np.zeros((size, size))
    # Generate a matrix for Chebyshev series. See link for explicit formulas.
    # https://en.wikipedia.org/wiki/Chebyshev_polynomials#Differentiation_and_integration
    mult = 2 ** order
    mat = np.zeros((size + 1, size + 1))
    for i in range(mat.shape[0]):
        for j in range(i - order + 1):
            if (j % 2) == (size - order) % 2:
                mat[j, i] = 0.5 if j == 0 else 1.0
                na = (i + order + j) // 2
                nb = (i + order - j) // 2
                nc = (i - order + j) // 2
                nd = (i - order - j) // 2
                mat[j, i] *= mult * i * comb(nb - 1, nd)
                mat[j, i] *= factorial(na - 1) / factorial(nc)
    return sp.sparse.csr_array(mat)


def chebder(fn: Chebfun, order: int = 1) -> Chebfun:
    """Calcualte `order`-th order derivative of Chebyshev function `fn`.
    """
    if not isinstance(order, int):
        raise ValueError('Derivative order expected to be integer value.')
    elif order < 0:
        raise ValueError(f'Derivative order should be non-negative: {order}.')
    elif order == 0:
        return fn

    # Apply diff matrices to each mode with caching.
    ops = {}
    cores = []
    for core, size in zip(fn.weights.cores, fn.weights.shape):
        if (op := ops.get(size)) is None:
            op = chebdiff(size - 1, order)
            ops[size] = op
        # It is important to apply diff matrix on the left since it is CSR.
        prev, _, next = core.shape
        core = np.moveaxis(core, 0, 1).reshape(size, -1)
        core = op.dot(core).reshape(size, prev, next)
        cores.append(np.moveaxis(core, 0, 1))

    weights = TensorTrain.from_list(cores)
    return Chebfun(weights, fn.grid)


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
    weights = idct(values, type=1)

    # Adjust coefficients to IDCT-1.
    for core, size in zip(weights.cores, weights.shape):
        core[:, 1:-1, :] *= 2
        if size % 2 == 1:
            core[:, -1, :] = -core[:, -1, :]

    # Apply Dicrete Cosine Transform to a tensor of values on the grid.
    return Chebfun(weights)


def chebint():
    pass
