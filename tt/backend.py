"""Array-backend dispatch.

Single owner of the knowledge "how do I allocate / reshape / factorize an array".
Everything else in ttpy2 is written against the functions exported here and works
unchanged on numpy arrays and torch tensors.

Two orthogonal things live here:

* the **default backend** used by constructors (``tt.rand``, ``tt.ones``, ...),
  set with :func:`set_backend`;
* **per-array dispatch** for operations: the backend is taken from the array that
  is passed in, so a numpy tensor keeps working after the default was switched.

Mixing backends inside one TT tensor is an error, not a silent conversion.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "set_backend", "get_backend", "backend_of", "Backend",
    "asarray", "to_numpy", "copy", "zeros", "empty", "eye", "arange", "randn",
    "concatenate", "stack", "transpose", "einsum", "diag", "tril", "triu",
    "svd", "qr", "solve", "lstsq", "eigh", "eig", "expm", "norm", "inv",
    "dtype_of", "is_complex", "result_dtype", "real_dtype", "complex_dtype",
    "eps_of", "device_of", "same_backend",
]

_CANON = {
    "float32": "float32", "float64": "float64",
    "complex64": "complex64", "complex128": "complex128",
    "f4": "float32", "f8": "float64", "c8": "complex64", "c16": "complex128",
}

_REAL_OF = {"float32": "float32", "float64": "float64",
            "complex64": "float32", "complex128": "float64"}
_COMPLEX_OF = {"float32": "complex64", "float64": "complex128",
               "complex64": "complex64", "complex128": "complex128"}
_WIDTH = {"float32": 1, "complex64": 1, "float64": 2, "complex128": 2}


_COMPLEX_NAMES = frozenset({"complex64", "complex128"})


@lru_cache(maxsize=None)
def _canon_cached(dtype) -> str:
    if isinstance(dtype, str):
        name = dtype
    else:
        name = getattr(dtype, "name", None) or str(dtype).replace("torch.", "")
    name = _CANON.get(name)
    if name is None:
        raise TypeError(
            f"unsupported dtype {dtype!r}; ttpy2 works with "
            "float32/float64/complex64/complex128")
    return name


def canon_dtype(dtype) -> str:
    """Map anything dtype-ish to one of the four canonical names.

    Cached: this sits in the inner loop of every algorithm (an AMEn solve called
    it 80k times, 30% of the runtime, all of it string bookkeeping on eight
    distinct dtype objects).
    """
    if dtype is None:
        return None
    try:
        return _canon_cached(dtype)
    except TypeError as exc:
        if "unhashable" not in str(exc):
            raise
        return _canon_cached(str(dtype))


class Backend:
    """Interface every backend implements. Instances are cheap and comparable."""

    name: str

    def __eq__(self, other):
        return isinstance(other, Backend) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"<backend {self.key}>"


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self, dtype="float64"):
        self.dtype = canon_dtype(dtype)
        self.device = "cpu"

    @property
    def key(self):
        return ("numpy", "cpu", self.dtype)

    def _dt(self, dtype):
        return np.dtype(canon_dtype(dtype) if dtype is not None else self.dtype)

    def asarray(self, a, dtype=None):
        out = np.asarray(a)
        if dtype is not None:
            return out.astype(self._dt(dtype), copy=False)
        if out.dtype.kind not in "fc":
            return out.astype(self._dt(None), copy=False)
        canon_dtype(out.dtype)  # raise on float16 & friends
        return out

    def to_numpy(self, a):
        return np.asarray(a)

    @staticmethod
    def copy(a):
        return a.copy()

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=self._dt(dtype))

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=self._dt(dtype))

    def eye(self, n, m=None, dtype=None):
        return np.eye(n, n if m is None else m, dtype=self._dt(dtype))

    def arange(self, n, dtype=None):
        return np.arange(n, dtype=self._dt(dtype))

    def randn(self, shape, dtype=None, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        dt = self._dt(dtype)
        if dt.kind == "c":
            real = _REAL_OF[canon_dtype(dt)]
            return (rng.standard_normal(shape, dtype=real)
                    + 1j * rng.standard_normal(shape, dtype=real)).astype(dt)
        return rng.standard_normal(shape, dtype=dt)

    concatenate = staticmethod(lambda arrays, axis=0: np.concatenate(arrays, axis=axis))
    stack = staticmethod(lambda arrays, axis=0: np.stack(arrays, axis=axis))
    transpose = staticmethod(lambda a, axes: np.transpose(a, axes))
    einsum = staticmethod(np.einsum)
    diag = staticmethod(np.diag)
    tril = staticmethod(np.tril)
    triu = staticmethod(np.triu)
    norm = staticmethod(lambda a: np.linalg.norm(a))
    inv = staticmethod(np.linalg.inv)
    solve = staticmethod(np.linalg.solve)
    eigh = staticmethod(np.linalg.eigh)
    eig = staticmethod(np.linalg.eig)

    @staticmethod
    def svd(a, full_matrices=False):
        try:
            return np.linalg.svd(a, full_matrices=full_matrices)
        except np.linalg.LinAlgError as exc:
            # Two very different failures reach this point and they must not be
            # confused: a genuinely hard matrix (gesdd sometimes fails to
            # converge where the slower gesvd succeeds), and garbage input.
            # Retrying on garbage produces a baffling error from deep inside
            # scipy, so name the real cause here.
            if not np.isfinite(a).all():
                n_bad = int(np.sum(~np.isfinite(a)))
                raise ValueError(
                    f"SVD input contains {n_bad} non-finite entries "
                    f"(inf/NaN) out of {a.size}; the data is broken upstream, "
                    "check for overflow (float32 overflows around 3.4e38)"
                ) from exc
            import scipy.linalg as sla
            return sla.svd(a, full_matrices=full_matrices, lapack_driver="gesvd")

    @staticmethod
    def qr(a):
        return np.linalg.qr(a, mode="reduced")

    @staticmethod
    def lstsq(a, b):
        return np.linalg.lstsq(a, b, rcond=None)[0]

    @staticmethod
    def expm(a):
        import scipy.linalg as sla
        return sla.expm(a)


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device="cuda", dtype="float64"):
        import torch  # noqa: F401  (import error here is the honest failure)
        self.torch = torch
        self.dtype = canon_dtype(dtype)
        self.device = str(device)

    @property
    def key(self):
        return ("torch", self.device, self.dtype)

    def _dt(self, dtype):
        return getattr(self.torch, canon_dtype(dtype) if dtype is not None else self.dtype)

    def asarray(self, a, dtype=None):
        t = self.torch
        if isinstance(a, t.Tensor):
            out = a
        else:
            out = t.as_tensor(np.asarray(a), device=self.device)
        if dtype is not None:
            return out.to(self._dt(dtype))
        if not (out.is_floating_point() or out.is_complex()):
            return out.to(self._dt(None))
        canon_dtype(out.dtype)
        return out

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    @staticmethod
    def copy(a):
        return a.clone()      # torch tensors have no .copy()

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(tuple(shape), dtype=self._dt(dtype), device=self.device)

    def empty(self, shape, dtype=None):
        return self.torch.empty(tuple(shape), dtype=self._dt(dtype), device=self.device)

    def eye(self, n, m=None, dtype=None):
        return self.torch.eye(n, n if m is None else m,
                              dtype=self._dt(dtype), device=self.device)

    def arange(self, n, dtype=None):
        return self.torch.arange(n, dtype=self._dt(dtype), device=self.device)

    def randn(self, shape, dtype=None, rng=None):
        """Gaussian sample; ``rng`` makes it reproducible AND backend-identical.

        With a generator the numbers are drawn by numpy and copied to the
        device, so the same seed gives bit-identical results on both backends —
        which is what a caller passing a seed is asking for.  Without one the
        fast on-device generator is used.  Ignoring ``rng`` (as this did) turns
        every documented ``seed=`` into a silent no-op.
        """
        t = self.torch
        dt = self._dt(dtype)
        if rng is None:
            if dt.is_complex:
                real = getattr(t, _REAL_OF[canon_dtype(dt)])
                re = t.randn(tuple(shape), dtype=real, device=self.device)
                im = t.randn(tuple(shape), dtype=real, device=self.device)
                return (re + 1j * im).to(dt)
            return t.randn(tuple(shape), dtype=dt, device=self.device)
        return self.asarray(
            NumpyBackend(canon_dtype(dt)).randn(shape, dtype, rng=rng), dtype)

    def concatenate(self, arrays, axis=0):
        return self.torch.cat(list(arrays), dim=axis)

    def stack(self, arrays, axis=0):
        return self.torch.stack(list(arrays), dim=axis)

    def transpose(self, a, axes):
        return a.permute(tuple(axes))

    def einsum(self, subscripts, *operands):
        return self.torch.einsum(subscripts, *operands)

    def diag(self, a):
        return self.torch.diag(a)

    def tril(self, a):
        return self.torch.tril(a)

    def triu(self, a):
        return self.torch.triu(a)

    def norm(self, a):
        return self.torch.linalg.norm(a).item()

    def inv(self, a):
        return self.torch.linalg.inv(a)

    def solve(self, a, b):
        return self.torch.linalg.solve(a, b)

    def eigh(self, a):
        return self.torch.linalg.eigh(a)

    def eig(self, a):
        return self.torch.linalg.eig(a)

    def svd(self, a, full_matrices=False):
        return self.torch.linalg.svd(a, full_matrices=full_matrices)

    def qr(self, a):
        return self.torch.linalg.qr(a, mode="reduced")

    def lstsq(self, a, b):
        return self.torch.linalg.lstsq(a, b, driver="gelsd").solution

    def expm(self, a):
        return self.torch.linalg.matrix_exp(a)


_default = NumpyBackend()


def set_backend(name="numpy", device=None, dtype=None):
    """Set the backend used by constructors. Returns the new backend."""
    global _default
    if isinstance(name, Backend):
        _default = name
        return _default
    if name == "numpy":
        _default = NumpyBackend(dtype or "float64")
    elif name == "torch":
        _default = TorchBackend(device or "cuda", dtype or "float64")
    else:
        raise ValueError(f"unknown backend {name!r}; use 'numpy' or 'torch'")
    return _default


def get_backend() -> Backend:
    return _default


def backend_of(a) -> Backend:
    """Backend that owns array ``a`` (dispatch by type, not by global state)."""
    if isinstance(a, np.ndarray):
        return NumpyBackend(canon_dtype(a.dtype))
    mod = type(a).__module__.split(".")[0]
    if mod == "torch":
        return TorchBackend(str(a.device), canon_dtype(a.dtype))
    raise TypeError(f"no ttpy2 backend for array of type {type(a)!r}")


def same_backend(arrays, what="cores"):
    """Assert all arrays live on one backend/device; return it. Loud on mixture."""
    backends = {}
    for a in arrays:
        b = backend_of(a)
        backends.setdefault((b.name, b.device), b)
    if len(backends) > 1:
        raise TypeError(
            f"{what} live on different backends: {sorted(backends)}; "
            "convert them explicitly (x.to(backend=..., device=...))")
    return next(iter(backends.values()))


# --- per-array dispatch ------------------------------------------------------

def asarray(a, dtype=None, backend=None):
    """Convert to a backend array.

    An array that already belongs to a backend keeps it: switching the default
    backend must not silently drag existing tensors to another device.
    """
    if backend is None:
        try:
            backend = backend_of(a)
        except TypeError:
            backend = _default
    return backend.asarray(a, dtype)


def to_numpy(a):
    return backend_of(a).to_numpy(a)


def copy(a):
    """Duplicate an array. numpy spells it .copy(), torch spells it .clone()."""
    return backend_of(a).copy(a)


def zeros(shape, dtype=None, like=None, backend=None):
    return (backend or (backend_of(like) if like is not None else _default)).zeros(shape, dtype)


def empty(shape, dtype=None, like=None, backend=None):
    return (backend or (backend_of(like) if like is not None else _default)).empty(shape, dtype)


def eye(n, m=None, dtype=None, like=None, backend=None):
    return (backend or (backend_of(like) if like is not None else _default)).eye(n, m, dtype)


def arange(n, dtype=None, like=None, backend=None):
    return (backend or (backend_of(like) if like is not None else _default)).arange(n, dtype)


def randn(shape, dtype=None, like=None, backend=None, rng=None):
    bk = backend or (backend_of(like) if like is not None else _default)
    return bk.randn(shape, dtype, rng=rng)


def concatenate(arrays, axis=0):
    return same_backend(arrays, "arrays").concatenate(arrays, axis=axis)


def stack(arrays, axis=0):
    return same_backend(arrays, "arrays").stack(arrays, axis=axis)


def transpose(a, axes):
    return backend_of(a).transpose(a, axes)


def einsum(subscripts, *operands):
    return same_backend(operands, "operands").einsum(subscripts, *operands)


def diag(a):
    return backend_of(a).diag(a)


def tril(a):
    return backend_of(a).tril(a)


def triu(a):
    return backend_of(a).triu(a)


def svd(a, full_matrices=False):
    return backend_of(a).svd(a, full_matrices=full_matrices)


def qr(a):
    return backend_of(a).qr(a)


def solve(a, b):
    return same_backend((a, b), "operands").solve(a, b)


def lstsq(a, b):
    return same_backend((a, b), "operands").lstsq(a, b)


def eigh(a):
    return backend_of(a).eigh(a)


def eig(a):
    return backend_of(a).eig(a)


def expm(a):
    return backend_of(a).expm(a)


def norm(a):
    return backend_of(a).norm(a)


def inv(a):
    return backend_of(a).inv(a)


def dtype_of(a) -> str:
    return canon_dtype(a.dtype)


def device_of(a) -> str:
    return backend_of(a).device


def is_complex(a) -> bool:
    return dtype_of(a) in _COMPLEX_NAMES


def real_dtype(dtype) -> str:
    return _REAL_OF[canon_dtype(dtype)]


def complex_dtype(dtype) -> str:
    return _COMPLEX_OF[canon_dtype(dtype)]


def result_dtype(*dtypes) -> str:
    """Widest of the given canonical dtypes: complex beats real, f8 beats f4."""
    names = [canon_dtype(d) for d in dtypes if d is not None]
    if not names:
        return _default.dtype
    width = max(_WIDTH[n] for n in names)
    cplx = any(n.startswith("complex") for n in names)
    base = "float64" if width == 2 else "float32"
    return _COMPLEX_OF[base] if cplx else base


def eps_of(a_or_dtype) -> float:
    dt = a_or_dtype if isinstance(a_or_dtype, str) else dtype_of(a_or_dtype)
    return float(np.finfo(np.dtype(_REAL_OF[canon_dtype(dt)])).eps)
