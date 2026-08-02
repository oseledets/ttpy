"""TT-vector.

The tensor *is* its list of cores (``self.cores``).  Everything the legacy API
exposed — ``core``, ``ps``, ``n``, ``r``, ``d`` — is derived from that list, so
there is exactly one place where the data lives.
"""

from __future__ import annotations

import warnings
from numbers import Number

import numpy as np
from einops import rearrange

from .. import backend as bk
from . import _ops


class vector(object):
    """A tensor in the TT format.

    ``tt.vector(a, eps, rmax)`` compresses a dense array; ``tt.vector()`` makes
    an empty shell that can be filled with :meth:`from_list`.

    Attributes
    ----------
    cores : list of arrays
        The TT cores, ``cores[k]`` of shape ``(r[k], n[k], r[k+1])``.  This is
        the only stored state.
    d, n, r, ps, core, erank : derived, read-only unless stated otherwise.
    """

    __array_priority__ = 100  # so ndarray * vector defers to us

    def __init__(self, a=None, eps=1e-14, rmax=None):
        self.cores = []
        if a is None:
            return
        if isinstance(a, vector):
            self.cores = [c.copy() for c in a.cores]
            return
        if isinstance(a, (list, tuple)) and a and hasattr(a[0], "ndim"):
            self.cores = _ops.check_cores([bk.asarray(c) for c in a])
            return
        arr = bk.asarray(a)
        if arr.ndim == 1:
            raise ValueError(
                "tt.vector(a) needs a d-dimensional array (d >= 2); "
                "reshape a 1-D array first, e.g. a.reshape([2]*d)")
        self.cores = _ops.tt_svd(arr, eps, rmax)

    # --- construction --------------------------------------------------------

    @staticmethod
    def from_list(a, order="F"):
        """Build a TT-vector from a list of ``(r, n, r)`` cores."""
        res = vector()
        res.cores = _ops.check_cores([bk.asarray(c) for c in a])
        return res

    @staticmethod
    def to_list(tt):
        """The list of cores (a copy of the list, not of the arrays)."""
        return list(tt.cores)

    @staticmethod
    def from_flat(core, n, r):
        """Legacy layout -> cores: ``core`` is the F-ordered concatenation."""
        n = np.asarray(n, dtype=np.int64).ravel()
        r = np.asarray(r, dtype=np.int64).ravel()
        d = n.size
        flat = bk.asarray(core).reshape((-1,))
        cores, pos = [], 0
        for k in range(d):
            size = int(r[k] * n[k] * r[k + 1])
            cores.append(rearrange(flat[pos:pos + size], "(b n a) -> a n b",
                                   a=int(r[k]), n=int(n[k]), b=int(r[k + 1])))
            pos += size
        if pos != flat.shape[0]:
            raise ValueError(
                f"flat core buffer has {flat.shape[0]} entries, ranks/modes need {pos}")
        return vector.from_list(cores)

    def copy(self):
        c = vector()
        c.cores = [x.copy() for x in self.cores]
        return c

    def to(self, backend=None, device=None, dtype=None):
        """Move/cast the tensor to another backend. Returns a new tensor."""
        name = backend or bk.backend_of(self.cores[0]).name
        dtype = dtype or self.dtype
        if name == "numpy":
            target = bk.NumpyBackend(dtype)
        elif name == "torch":
            target = bk.TorchBackend(device or "cuda", dtype)
        else:
            raise ValueError(f"unknown backend {name!r}")
        c = vector()
        c.cores = [target.asarray(bk.to_numpy(x), dtype) for x in self.cores]
        return c

    def astype(self, dtype):
        c = vector()
        c.cores = _ops.to_dtype(self.cores, dtype)
        return c

    # --- derived description -------------------------------------------------

    @property
    def d(self):
        return len(self.cores)

    @property
    def n(self):
        return np.array(_ops.modes(self.cores) if self.cores else [0], dtype=np.int32)

    @property
    def r(self):
        return np.array(_ops.ranks(self.cores) if self.cores else [1], dtype=np.int32)

    @property
    def ps(self):
        """Legacy 1-based offsets into :attr:`core`."""
        n, r = self.n.astype(np.int64), self.r.astype(np.int64)
        if not self.cores:
            return np.array([0], dtype=np.int32)
        return np.concatenate(
            ([1], 1 + np.cumsum(n * r[:-1] * r[1:]))).astype(np.int32)

    @property
    def core(self):
        """Legacy flat buffer: F-ordered cores concatenated."""
        if not self.cores:
            return np.array([0.0])
        return np.concatenate(
            [np.asarray(rearrange(bk.to_numpy(c), "a n b -> (b n a)"))
             for c in self.cores])

    @core.setter
    def core(self, value):
        """Legacy assembly path: needs :attr:`n`/:attr:`r` already known."""
        if not self.cores:
            raise AttributeError(
                "cannot set .core on an empty tt.vector; build it with "
                "tt.vector.from_list(cores) or tt.vector.from_flat(core, n, r)")
        self.cores = vector.from_flat(value, self.n, self.r).cores

    @property
    def dtype(self):
        return bk.dtype_of(self.cores[0]) if self.cores else "float64"

    @property
    def backend(self):
        return bk.backend_of(self.cores[0]) if self.cores else bk.get_backend()

    @property
    def is_complex(self):
        return self.dtype.startswith("complex")

    @property
    def erank(self):
        """Effective rank: the rank of an equivalent tensor with constant ranks."""
        d, n, r = self.d, self.n.astype(np.float64), self.r.astype(np.float64)
        if d <= 1:
            return 0.0
        sz = float(np.dot(n * r[:d], r[1:]))
        if sz == 0:
            return 0.0
        b = r[0] * n[0] + n[d - 1] * r[d]
        if d == 2:
            return sz / b
        a = float(np.sum(n[1:d - 1]))
        return float((np.sqrt(b * b + 4 * a * sz) - b) / (2 * a))

    def rmean(self):
        """Mean rank."""
        n, r = self.n.astype(np.float64), self.r.astype(np.float64)
        if not np.all(n):
            return 0.0
        a = float(np.sum(n[1:-1]))
        b = float(n[0] + n[-1])
        c = -float(np.sum(n * r[1:] * r[:-1]))
        if a == 0:
            return -c / b if b else 0.0
        return float(0.5 * (-b + np.sqrt(b * b - 4 * a * c)) / a)

    @property
    def size(self):
        """Number of stored parameters."""
        n, r = self.n.astype(np.int64), self.r.astype(np.int64)
        return int(np.sum(n * r[:-1] * r[1:]))

    def __repr__(self):
        if self.d == 0:
            return "Empty tensor"
        r, n = self.r, self.n
        head = (f"This is a {self.d}-dimensional tensor \n")
        body = "".join(f"r({i})={r[i]}, n({i})={n[i]} \n" for i in range(self.d))
        return head + body + f"r({self.d})={r[self.d]} \n"

    # --- dense / element access ---------------------------------------------

    def full(self, asvector=False):
        """Dense tensor. Beware of the memory this needs."""
        a = _ops.full(self.cores)
        if asvector:  # Fortran-order flattening, backend-agnostic
            a = bk.transpose(a, tuple(reversed(range(a.ndim)))).reshape((-1,))
            # (reversing all axes then a C-order reshape == flatten(order="F"))
        return a

    def __getitem__(self, index):
        """Element or sub-tensor; ints select, slices keep the mode."""
        if isinstance(index, (int, np.integer, slice)):
            index = (index,)
        if len(index) != self.d:
            raise IndexError(
                f"index has {len(index)} entries, tensor has {self.d} modes")
        running = None
        out = []
        for k, idx in enumerate(index):
            c = self.cores[k]
            r0, n, r1 = c.shape
            if isinstance(idx, (int, np.integer)):
                block = c[:, int(idx), :].reshape((r0, 1, r1))
            else:
                block = c[:, idx, :]
            nk = block.shape[1]
            if running is not None:
                block = (running @ block.reshape((r0, nk * r1))).reshape(
                    (running.shape[0], nk, r1))
            if nk == 1:
                running = block.reshape((block.shape[0], r1))
            else:
                out.append(block)
                running = None
        if not out:
            if int(np.prod(running.shape)) == 1:
                return running.reshape(())[()]
            return running
        if running is not None:
            last = out[-1]
            out[-1] = (last.reshape((-1, last.shape[2])) @ running).reshape(
                last.shape[0], last.shape[1], running.shape[1])
        return vector.from_list(out)

    # --- algebra -------------------------------------------------------------

    def round(self, eps=1e-14, rmax=None, method="svd", oversampling=10,
              seed=None, return_error=False):
        """Truncate the ranks to relative Frobenius accuracy ``eps``.

        Args:
            eps: relative Frobenius accuracy (ignored by the randomized method,
                which targets a rank instead).
            rmax: maximal TT rank; required for ``method="randomized"``.
            method: ``"svd"`` (deterministic, quasi-optimal) or ``"randomized"``
                (sketching, no SVD chain — much faster for large ranks and the
                path that actually wins on a GPU; see
                :func:`tt.core._ops.randomized_round`).
            oversampling, seed: passed to the randomized method.
            return_error: with the randomized method, also return the exact
                absolute Frobenius error of the truncation.

        Returns:
            The rounded tensor, or ``(tensor, error)`` if ``return_error``.
        """
        c = vector()
        if method == "svd":
            if return_error:
                raise ValueError(
                    "return_error is only available for method='randomized'; "
                    "for the SVD path compute (x - x.round(eps)).norm()")
            c.cores = _ops.round_cores(self.cores, eps, rmax)
            return c
        if method != "randomized":
            raise ValueError(f"unknown rounding method {method!r}; "
                             "use 'svd' or 'randomized'")
        if rmax is None:
            raise ValueError("method='randomized' needs an explicit rmax: it "
                             "targets a rank, it cannot honour eps by itself")
        res = _ops.randomized_round(self.cores, rmax, oversampling, seed,
                                    return_error)
        if return_error:
            c.cores, err = res
            return c, err
        c.cores = res
        return c

    def orthogonalize(self, center=0):
        c = vector()
        c.cores = _ops.orthogonalize(self.cores, center)
        return c

    def norm(self):
        return _ops.norm(self.cores)

    def __add__(self, other):
        if other is None:
            return self
        if isinstance(other, Number):
            from . import tools
            other = tools.ones(self.n, None) * other
        c = vector()
        c.cores = _ops.add(self.cores, other.cores)
        return c

    def __radd__(self, other):
        if other is None:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0) * other

    def __rsub__(self, other):
        return (-1.0) * self + other

    def __neg__(self):
        return self * (-1.0)

    def __mul__(self, other):
        c = vector()
        if isinstance(other, Number):
            c.cores = _ops.scale(self.cores, other)
        elif isinstance(other, vector):
            c.cores = _ops.hadamard(self.cores, other.cores)
        else:
            return NotImplemented
        return c

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__mul__(other)
        if isinstance(other, vector):
            c = vector()
            c.cores = _ops.hadamard(other.cores, self.cores)
            return c
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.__mul__(1.0 / other)
        return NotImplemented

    def __kron__(self, other):
        if other is None:
            return self
        c = vector()
        c.cores = _ops.kron(self.cores, other.cores)
        return c

    def __dot__(self, other):
        res = _ops.dot(self.cores, other.cores)
        if hasattr(res, "shape") and getattr(res, "shape", ()) != ():
            res = bk.to_numpy(res).flatten("F")
        return res

    def __diag__(self):
        from . import matrix as _matrix
        out = []
        for c in self.cores:
            r0, n, r1 = c.shape
            block = bk.zeros((r0, n, n, r1), dtype=self.dtype, like=c)
            idx = np.arange(n)
            block[:, idx, idx, :] = c  # adjacent advanced indices -> (r0,n,r1)
            out.append(block)
        return _matrix.matrix.from_list(out)

    def __col__(self, k):
        """Select column(s) of the last (block) rank index."""
        cores = list(self.cores)
        last = cores[-1]
        sel = last[:, :, k]
        if sel.ndim == 2:
            sel = sel.reshape((last.shape[0], last.shape[1], 1))
        cores[-1] = sel
        return vector.from_list(cores)

    # --- complex handling ----------------------------------------------------

    def _matrix__complex_op(self, op):
        return self.__complex_op(op)

    def __complex_op(self, op):
        crs = self.cores
        d = self.d
        rdtype = bk.real_dtype(self.dtype)
        newcrs = []
        cr = crs[0]
        rl, n, rr = cr.shape
        newcr = bk.zeros((rl, n, rr * 2), dtype=rdtype, like=cr)
        newcr[:, :, :rr] = cr.real
        newcr[:, :, rr:] = cr.imag
        newcrs.append(newcr)
        for i in range(1, d - 1):
            cr = crs[i]
            rl, n, rr = cr.shape
            newcr = bk.zeros((rl * 2, n, rr * 2), dtype=rdtype, like=cr)
            newcr[:rl, :, :rr] = cr.real
            newcr[rl:, :, rr:] = cr.real
            newcr[:rl, :, rr:] = cr.imag
            newcr[rl:, :, :rr] = -cr.imag
            newcrs.append(newcr)
        cr = crs[-1]
        rl, n, rr = cr.shape
        if op in ("R", "r", "Re"):
            newcr = bk.zeros((rl * 2, n, rr), dtype=rdtype, like=cr)
            newcr[:rl, :, :] = cr.real
            newcr[rl:, :, :] = -cr.imag
        elif op in ("I", "i", "Im"):
            newcr = bk.zeros((rl * 2, n, rr), dtype=rdtype, like=cr)
            newcr[:rl, :, :] = cr.imag
            newcr[rl:, :, :] = cr.real
        elif op in ("A", "B", "all", "both", "M"):
            newcr = bk.zeros((rl * 2, n, 2 * rr), dtype=rdtype, like=cr)
            newcr[:rl, :, :rr] = cr.real
            newcr[rl:, :, :rr] = -cr.imag
            newcr[:rl, :, rr:] = cr.imag
            newcr[rl:, :, rr:] = cr.real
            newcrs.append(newcr)
            if op == "M":
                newcr = bk.zeros((rr * 2, 4, 1), dtype=rdtype, like=cr)
                newcr[:rr, 0, :] = 1.0
                newcr[:rr, 3, :] = 1.0
                newcr[rr:, 1, :] = 1.0
                newcr[rr:, 2, :] = -1.0
            else:
                newcr = bk.zeros((rr * 2, 2, 1), dtype=rdtype, like=cr)
                newcr[:rr, 0, :] = 1.0
                newcr[rr:, 1, :] = 1.0
        else:
            raise ValueError(f"unexpected op {op!r} in tt.vector.__complex_op")
        newcrs.append(newcr)
        return vector.from_list(newcrs)

    def real(self):
        """Real part, as a real TT-tensor."""
        if not self.is_complex:
            return self.copy()
        return self.__complex_op("Re")

    def imag(self):
        if not self.is_complex:
            return self * 0.0
        return self.__complex_op("Im")

    def c2r(self):
        """Stack real and imaginary parts as an extra mode of size 2."""
        return self.__complex_op("both")

    def r2c(self):
        """Inverse of :meth:`c2r`."""
        from . import tools
        tmp = self.astype(bk.complex_dtype(self.dtype))
        cores = list(tmp.cores)
        last = cores[-1].copy()
        last[:, 1, :] = last[:, 1, :] * 1j
        cores[-1] = last
        return tools.sum(vector.from_list(cores), axis=self.d - 1)

    # --- io ------------------------------------------------------------------

    def write(self, fname):
        """Save to ``.npz``. Cores are stored one array per core."""
        np.savez(fname, d=np.array([self.d]),
                 **{f"core{k}": bk.to_numpy(c) for k, c in enumerate(self.cores)})

    @staticmethod
    def read(fname):
        with np.load(fname) as z:
            d = int(z["d"][0])
            return vector.from_list([z[f"core{k}"] for k in range(d)])

    # --- QTT FFT -------------------------------------------------------------

    def qtt_fft1(self, tol, inverse=False, bitReverse=True):
        """1-D (inverse) DFT in the QTT format.

        S. Dolgov, B. Khoromskij, D. Savostyanov, *Superfast Fourier transform
        using QTT approximation*, J. Fourier Anal. Appl. 18(5), 2012.
        """
        d = self.d
        if any(n != 2 for n in _ops.modes(self.cores)):
            raise ValueError("qtt_fft1 needs all mode sizes equal to 2")
        y = [np.asarray(bk.to_numpy(c), dtype=np.complex128) for c in self.cores]
        twiddle = np.exp((1j if inverse else -1j) * np.pi)

        for i in range(d - 1, 0, -1):
            r1, _, r2 = y[i].shape
            crd2 = np.zeros((r1, 2, r2), dtype=complex)
            crd2[:, 0, :] = (y[i][:, 0, :] + y[i][:, 1, :]) / np.sqrt(2)
            crd2[:, 1, :] = (y[i][:, 0, :] - y[i][:, 1, :]) / np.sqrt(2)
            y[i] = np.zeros((r1 * 2, 2, r2), dtype=complex)
            y[i][:r1, 0, :] = crd2[:, 0, :]
            y[i][r1:, 1, :] = crd2[:, 1, :]
            rv = np.ones((1, 1))
            for j in range(i):
                cr = y[j]
                p1, _, p2 = cr.shape
                w = twiddle ** (1.0 / (2 ** (i - j)))
                if j == 0:
                    new = np.zeros((p1, 2, p2 * 2), dtype=complex)
                    new[:, :, :p2] = cr
                    new[:, 0, p2:] = cr[:, 0, :]
                    new[:, 1, p2:] = w * cr[:, 1, :]
                else:
                    new = np.zeros((p1 * 2, 2, p2 * 2), dtype=complex)
                    new[:p1, :, :p2] = cr
                    new[p1:, 0, p2:] = cr[:, 0, :]
                    new[p1:, 1, p2:] = w * cr[:, 1, :]
                q0, _, q2 = new.shape
                new = rv @ new.reshape((q0, 2 * q2))
                q0 = new.shape[0]
                new = new.reshape((2 * q0, q2))
                y[j], rv = np.linalg.qr(new)
                y[j] = y[j].reshape((q0, 2, -1))
            r1, _, r2 = y[i].shape
            y[i] = (rv @ y[i].reshape((r1, 2 * r2))).reshape((rv.shape[0], 2, r2))
            # backward svd sweep
            for j in range(i, 0, -1):
                a0, _, a2 = y[j].shape
                u, s, v = np.linalg.svd(y[j].reshape((a0, 2 * a2)),
                                        full_matrices=False)
                rnew = max(1, _ops.chop(s, np.linalg.norm(s) * tol / np.sqrt(i)))
                u = u[:, :rnew] * s[:rnew]
                y[j] = v[:rnew, :].reshape((rnew, 2, a2))
                b0, _, _ = y[j - 1].shape
                y[j - 1] = (y[j - 1].reshape((-1, a0)) @ u).reshape((b0, 2, rnew))
        y[0] = np.einsum("ij,ajb->aib", np.array([[1, 1], [1, -1]]) / np.sqrt(2), y[0])
        if bitReverse:
            y = [np.transpose(y[d - 1 - i], (2, 1, 0)) for i in range(d)]
        return vector.from_list(y)


class tensor(vector):
    """Deprecated alias of :class:`vector`."""

    def __init__(self, *args, **kwargs):
        super(tensor, self).__init__(*args, **kwargs)
        warnings.warn("tt.tensor is deprecated, use tt.vector instead",
                      DeprecationWarning, stacklevel=2)
