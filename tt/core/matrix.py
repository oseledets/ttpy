"""TT-matrix.

A TT-matrix is a TT-vector over the merged index ``s = i + n * j`` plus the
splitting ``(n, m)``.  The numbers live in ``self.tt`` (a :class:`tt.vector`),
the splitting in ``self.n`` / ``self.m`` — no third copy.

Core layout, as in legacy ttpy: ``to_list`` gives cores of shape
``(r_k, n_k, m_k, r_{k+1})``.
"""

from __future__ import annotations

from numbers import Number

import numpy as np
from einops import rearrange
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from . import _ops
from .vector import vector


def _merge(cores4):
    """(r,n,m,r) cores -> (r,n*m,r) cores with the legacy index s = i + n*j."""
    return [rearrange(c, "a n m b -> a (m n) b") for c in cores4]


def _split(cores3, n, m):
    """(r,n*m,r) cores -> (r,n,m,r) cores (inverse of :func:`_merge`)."""
    out = []
    for c, nk, mk in zip(cores3, n, m):
        if c.shape[1] != nk * mk:
            raise ValueError(f"mode {c.shape[1]} is not n*m = {nk}*{mk}")
        out.append(rearrange(c, "a (m n) b -> a n m b", n=int(nk), m=int(mk)))
    return out


class matrix(object):
    """A matrix in the TT format."""

    __array_priority__ = 100

    def __init__(self, a=None, eps=1e-14, n=None, m=None, rmax=None):
        self.n = np.array([0], dtype=np.int32)
        self.m = np.array([0], dtype=np.int32)
        self.tt = vector()
        if a is None:
            return
        if isinstance(a, matrix):
            self.n, self.m, self.tt = a.n.copy(), a.m.copy(), a.tt.copy()
            return
        if isinstance(a, vector):
            if n is None or m is None:
                root = np.sqrt(a.n).astype(np.int32)
                if not np.all(root * root == a.n):
                    raise ValueError(
                        "mode sizes are not perfect squares; pass n= and m= "
                        "explicitly to tt.matrix(tt_vector, n=..., m=...)")
                n, m = root, root
            self.n = np.asarray(n, dtype=np.int32).ravel()
            self.m = np.asarray(m, dtype=np.int32).ravel()
            self.tt = a.copy()
            return
        if isinstance(a, (list, tuple)):
            built = matrix.from_list(a)
            self.n, self.m, self.tt = built.n, built.m, built.tt
            return
        arr = bk.asarray(a)
        if arr.ndim % 2:
            raise ValueError(
                f"a dense TT-matrix source needs an even number of axes, got {arr.ndim}")
        d = arr.ndim // 2
        self.n = np.array(arr.shape[:d], dtype=np.int32)
        self.m = np.array(arr.shape[d:], dtype=np.int32)
        # merged mode s_k = i_k + n_k * j_k, so the column axis must come first
        perm = [v for k in range(d) for v in (d + k, k)]  # m1,n1,m2,n2,...
        merged = bk.transpose(arr, tuple(perm)).reshape(
            tuple(int(x) for x in self.n * self.m))
        self.tt = vector(merged, eps, rmax)

    # --- construction --------------------------------------------------------

    @staticmethod
    def from_list(a):
        """Build from cores of shape ``(r, n, m, r)``."""
        cores = [bk.asarray(c) for c in a]
        for k, c in enumerate(cores):
            if c.ndim != 4:
                raise ValueError(
                    f"matrix core {k} has ndim {c.ndim}, expected 4 (r,n,m,r)")
        res = matrix()
        res.n = np.array([c.shape[1] for c in cores], dtype=np.int32)
        res.m = np.array([c.shape[2] for c in cores], dtype=np.int32)
        res.tt = vector.from_list(_merge(cores))
        return res

    @staticmethod
    def to_list(ttmat):
        return _split(ttmat.tt.cores, ttmat.n, ttmat.m)

    @property
    def cores(self):
        """Cores in ``(r, n, m, r)`` layout (derived from ``self.tt``)."""
        return matrix.to_list(self)

    def copy(self):
        c = matrix()
        c.n, c.m, c.tt = self.n.copy(), self.m.copy(), self.tt.copy()
        return c

    def to(self, backend=None, device=None, dtype=None):
        c = matrix()
        c.n, c.m = self.n.copy(), self.m.copy()
        c.tt = self.tt.to(backend, device, dtype)
        return c

    def astype(self, dtype):
        c = matrix()
        c.n, c.m, c.tt = self.n.copy(), self.m.copy(), self.tt.astype(dtype)
        return c

    # --- description ---------------------------------------------------------

    @property
    def r(self):
        return self.tt.r

    @property
    def d(self):
        return self.tt.d

    @property
    def dtype(self):
        return self.tt.dtype

    @property
    def is_complex(self):
        return self.tt.is_complex

    @property
    def erank(self):
        return self.tt.erank

    def rmean(self):
        return self.tt.rmean()

    @property
    def size(self):
        return self.tt.size

    def __repr__(self):
        r, n, m, d = self.tt.r, self.n, self.m, self.tt.d
        head = f"This is a {d}-dimensional matrix \n"
        body = "".join(f"r({i})={r[i]}, n({i})={n[i]}, m({i})={m[i]} \n"
                       for i in range(d))
        return head + body + f"r({d})={r[d]} \n"

    def write(self, fname):
        np.savez(fname, d=np.array([self.d]), n=self.n, m=self.m,
                 **{f"core{k}": bk.to_numpy(c) for k, c in enumerate(self.cores)})

    @staticmethod
    def read(fname):
        with np.load(fname) as z:
            d = int(z["d"][0])
            return matrix.from_list([z[f"core{k}"] for k in range(d)])

    # --- algebra -------------------------------------------------------------

    def round(self, eps=1e-14, rmax=None):
        c = matrix()
        c.n, c.m = self.n.copy(), self.m.copy()
        c.tt = self.tt.round(eps, rmax)
        return c

    def norm(self):
        return self.tt.norm()

    @property
    def T(self):
        """Transpose: swap the row and column index of every core."""
        return matrix.from_list(
            [rearrange(c, "a n m b -> a m n b") for c in self.cores])

    def __add__(self, other):
        if other is None:
            return self
        c = matrix()
        c.n, c.m = self.n.copy(), self.m.copy()
        c.tt = self.tt + other.tt
        return c

    def __radd__(self, other):
        return self if other is None else self.__add__(other)

    def __sub__(self, other):
        c = matrix()
        c.n, c.m = self.n.copy(), self.m.copy()
        c.tt = self.tt - other.tt
        return c

    def __neg__(self):
        return self * (-1.0)

    def __matmul__(self, other):
        """TT-matrix by TT-matrix."""
        from . import tools as _tools
        diff = len(self.n) - len(other.m)
        left = self if diff >= 0 else _tools.kron(
            self, matrix(_tools.ones(1, abs(diff))))
        right = other if diff <= 0 else _tools.kron(
            other, matrix(_tools.ones(1, abs(diff))))
        return matrix.from_list(_ops.matmat_cores(left.cores, right.cores))

    def __mul__(self, other):
        if isinstance(other, matrix):
            return self.__matmul__(other)
        if isinstance(other, vector):
            from . import tools as _tools
            return _tools.matvec(self, other)
        if bk.is_scalar(other):
            c = matrix()
            c.n, c.m = self.n.copy(), self.m.copy()
            c.tt = self.tt * other
            return c
        # dense vector: matrix-by-vector in full format
        x = np.asanyarray(other)
        if x.size != int(np.prod(self.m)):
            raise ValueError(
                f"dense right-hand side has {x.size} entries, matrix needs "
                f"{int(np.prod(self.m))}")
        x = x.reshape(tuple(int(v) for v in self.m[::-1])).T  # F-order multi-index
        curr = x.reshape((1,) + x.shape)
        for k, core in enumerate(self.cores):
            core = bk.to_numpy(core)
            curr = np.tensordot(curr, core, axes=([0, 1], [0, 2]))
            curr = np.moveaxis(curr, -1, 0)
        curr = np.sum(curr, axis=0)
        return curr.T.reshape(-1)

    def __rmul__(self, other):
        if isinstance(other, matrix):
            return other.__matmul__(self)
        if bk.is_scalar(other):
            return self.__mul__(other)
        return NotImplemented

    def __kron__(self, other):
        if other is None:
            return self
        c = matrix()
        c.n = np.concatenate((self.n, other.n))
        c.m = np.concatenate((self.m, other.m))
        c.tt = self.tt.__kron__(other.tt)
        return c

    def __diag__(self):
        """Diagonal of the TT-matrix as a TT-vector."""
        out = []
        for c in self.cores:
            r0, n, m, r1 = c.shape
            if n != m:
                raise ValueError(f"diag needs square modes, got {n}x{m}")
            idx = np.arange(n)
            out.append(c[:, idx, idx, :])  # adjacent advanced indices -> (r0,n,r1)
        return vector.from_list(out)

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2:
            raise IndexError("use m[i, :] for a row or m[:, j] for a column")
        row, col = index
        cores = self.cores
        if isinstance(row, (int, np.integer)) and col == slice(None):
            out = []
            for k in range(self.d):
                out.append(cores[k][:, row % int(self.n[k]), :, :])
                row //= int(self.n[k])
            return vector.from_list(out)
        if isinstance(col, (int, np.integer)) and row == slice(None):
            out = []
            for k in range(self.d):
                out.append(cores[k][:, :, col % int(self.m[k]), :])
                col //= int(self.m[k])
            return vector.from_list(out)
        raise IndexError(
            "only full rows m[i, :] and full columns m[:, j] are supported")

    # --- complex -------------------------------------------------------------

    def real(self):
        c = matrix()
        c.n, c.m, c.tt = self.n.copy(), self.m.copy(), self.tt.real()
        return c

    def imag(self):
        c = matrix()
        c.n, c.m, c.tt = self.n.copy(), self.m.copy(), self.tt.imag()
        return c

    def c2r(self):
        c = matrix()
        c.n = np.concatenate((self.n, [2]))
        c.m = np.concatenate((self.m, [2]))
        c.tt = self.tt._matrix__complex_op("M")
        return c

    def r2c(self):
        c = matrix()
        c.n, c.m = self.n[:-1].copy(), self.m[:-1].copy()
        c.tt = self.tt.r2c()
        return c

    # --- dense ---------------------------------------------------------------

    def full(self):
        """Dense matrix of shape ``(prod(n), prod(m))``, F-ordered multi-indices."""
        d = self.d
        # merged modes s_k = i_k + n_k * j_k  ->  axes (j_1, i_1, ..., j_d, i_d)
        res = _ops.full(self.tt.cores).reshape(
            [int(v) for k in range(d) for v in (self.m[k], self.n[k])])
        rows = [2 * k + 1 for k in range(d - 1, -1, -1)]   # i_d ... i_1
        cols = [2 * k for k in range(d - 1, -1, -1)]       # j_d ... j_1
        res = bk.transpose(res, tuple(rows + cols))
        return res.reshape((int(np.prod(self.n)), int(np.prod(self.m))))
