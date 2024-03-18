from dataclasses import dataclass
from math import comb, factorial
from typing import Optional

import numpy as np
import scipy as sp
import scipy.sparse
from numpy.typing import ArrayLike

from tt.core.matrix import matrix
from tt.core.vector import TensorTrain, vector
from tt.fft import idctn
from tt.interpolate.grid import gridfit
from tt.interpolate.util import normalize_cheb_domain


@dataclass
class Chebfun:
    """Class Chebfun represents a multi-dimensional Chebyshev polynomial with
    coefficients stored in TT-format.
    """

    weights: TensorTrain

    values: Optional[TensorTrain] = None

    grid: Optional[TensorTrain] = None

    @property
    def arity(self) -> int:
        """Number of arguments which the function accepts.
        """
        return self.weights.ndim

    @property
    def domain(self) -> ArrayLike:
        domain = np.empty((self.arity, 2))
        for i, core in enumerate(self.grid.cores):
            domain[i, 0] = core[0, -1, 0]
            domain[i, 1] = core[0, 0, 0]
        return domain

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
        for core, xs, size, domain in zip(self.weights.cores, batch.T,
                                          self.weights.shape, self.domain):
            # Map points to standard domain.
            add = domain[1] + domain[0]
            sub = domain[1] - domain[0]
            alpha = 2 / sub
            xs = alpha * xs + add / sub

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

    def trim(self, tol: float = 1e-14, max_rank: int = 1000000):
        """Apply TT-round procedure to underlying tensor train.
        """
        weights = self.weights.round(tol, max_rank)
        return Chebfun(weights, self.values, self.grid)


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
            return Chebfun(weights, other.values, other.grid)
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
    return Chebfun(weights, fn.values, fn.grid)


def chebfit(fn, grid: ArrayLike | TensorTrain, domain=None, tol: float = 1e-6,
            **kwargs) -> Chebfun:
    """Approximate function `fn` on multi-dimensional Chebyshev grid `grid`
    with specified precision `tol`. The function leverages cross-approximation
    methods building grid function.
    """
    if isinstance(grid, (list, tuple)):
        # Validate domain an generate transformations.
        domain = normalize_cheb_domain(domain, len(grid))

        # Generate and adjust cores to domain.
        grid_cores = []
        alphas = (domain[:, 1] - domain[:, 0]) / 2
        betas = (domain[:, 0] + domain[:, 1]) / 2
        for nonodes, alpha, beta in zip(grid, alphas, betas):
            grid_core = chebnodes(nonodes)[None, :, None]
            grid_cores.append(alpha * grid_core + beta)

        # Finally, build a grid and remove domain.
        grid = TensorTrain.from_list(grid_cores)
        domain = None
    elif isinstance(grid, TensorTrain):
        pass  # OK. Do nothing.
    else:
        raise NotImplementedError('Either grid represented as rank-1 tensor '
                                  'train or a sequence of number of Chebyshev '
                                  'nodes are supported.')

    # Evaluate target function on a Chebyshev grid for further interpolation.
    grid_fn = gridfit(fn, grid, domain, tol, **kwargs)
    weights = idctn(grid_fn.values, type=1)

    # Adjust coefficients to IDCT-1.
    for core, size in zip(weights.cores, weights.shape):
        core[:, 1:-1, :] *= 2
        if size % 2 == 1:
            core[:, -1, :] = -core[:, -1, :]

    # Apply Dicrete Cosine Transform to a tensor of values on the grid.
    return Chebfun(weights, grid_fn.values, grid)


def chebint(grid_fn: Chebfun) -> float:
    """Integrate Chebfun according to Clenshaw–Curtis quadrature rule.
    """
    acc = np.ones(1)
    cores = grid_fn.weights.cores
    shape = grid_fn.weights.shape
    for core, size, domain in zip(cores, shape, grid_fn.domain):
        weights = np.zeros(size)
        weights[::2] = (domain[1] - domain[0]) / (1 - np.arange(0, size, 2)**2)
        acc = np.einsum('i,j,ijk->k', acc, weights, core)
    return acc.item()
