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
from numbers import Number

import numpy as np

__all__ = [
    "set_backend", "get_backend", "backend_of", "Backend",
    "asarray", "to_numpy", "copy", "zeros", "empty", "eye", "arange", "randn",
    "concatenate", "stack", "transpose", "einsum", "diag", "tril", "triu",
    "svd", "qr", "solve", "lstsq", "eigh", "eig", "expm", "norm", "inv",
    "dtype_of", "is_complex", "is_scalar", "scalar_dtype",
    "result_dtype", "real_dtype", "complex_dtype",
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

    def ingest_dtype(self, source):
        """Canonical dtype a *host* array takes on when it enters this backend.

        The backend's dtype says what lives on it, so incoming data adopts that
        width -- while staying in the domain it arrived in, since narrowing a
        complex array to a real one would throw away half of every number.  The
        alternative, keeping whatever dtype numpy handed over, means a backend
        whose declared dtype is not a setting (``docs/NUMERICS.md``).
        """
        name = canon_dtype(source)
        return (_COMPLEX_OF if name in _COMPLEX_NAMES else _REAL_OF)[self.dtype]


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
        # ingest_dtype also raises on float16 & friends
        return out.astype(self._dt(self.ingest_dtype(out.dtype)), copy=False)

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
    @staticmethod
    def einsum(subs, *ops):
        """np.einsum through BLAS, with the contraction path cached.

        Two separate costs are at stake.  Without ``optimize`` numpy runs its
        own nested loops and never reaches BLAS (387 us against 76 us on an
        AMEn local matvec, r=34).  With ``optimize=True`` it searches for a
        contraction path on *every* call, which for the tiny cores of a QTT
        problem costs more than the contraction itself.  Caching the path by
        (subscripts, shapes) buys both: the search runs once per shape
        combination and the contraction goes to BLAS every time.
        """
        return np.einsum(subs, *ops,
                         optimize=_einsum_path(subs, tuple(o.shape for o in ops)))
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
        move_to = None
        if isinstance(a, t.Tensor):
            out = a
        else:
            # Cast on the host, then move.  as_tensor(..., device=...) would
            # materialise the *source* dtype on the device first, and a device
            # that does not have it refuses the transfer even when the very next
            # line casts the value away (MPS has no float64; numpy hands us
            # float64).
            out = t.as_tensor(np.asarray(a))
            move_to = self.device
        if dtype is not None:
            out = out.to(self._dt(dtype))
        elif not (out.is_floating_point() or out.is_complex()):
            out = out.to(self._dt(None))
        elif move_to is not None:       # host data adopts the backend's width
            out = out.to(self._dt(self.ingest_dtype(out.dtype)))
        else:
            canon_dtype(out.dtype)      # raise on float16 & friends
        return out if move_to is None else out.to(device=move_to)

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
        # NOT .item(): that detaches the value from the autograd tape, so a
        # functional containing ||x|| differentiates to a silently wrong
        # gradient -- no error, no NaN, just a missing term (measured: a
        # completion functional whose ||x - b|| = 29.4 contributed exactly
        # zero, max |AD - finite differences| = 7.03e-01).  It also made the
        # two backends disagree in return type, since numpy's np.linalg.norm
        # returns a 0-d numpy scalar.  Callers that need a Python float say
        # float(...), and most already did.
        return self.torch.linalg.norm(a)

    def inv(self, a):
        return self.torch.linalg.inv(a)

    def solve(self, a, b):
        return self.torch.linalg.solve(a, b)

    def eigh(self, a):
        return self.torch.linalg.eigh(a)

    def eig(self, a):
        return self.torch.linalg.eig(a)

    def svd(self, a, full_matrices=False):
        t = self.torch
        try:
            return t.linalg.svd(a, full_matrices=full_matrices)
        except Exception as exc:
            # Mirror of NumpyBackend.svd: separate a genuinely hard matrix
            # (gesdd fails to converge where the slower gesvd succeeds --
            # near-repeated singular values do this) from garbage input, and
            # name the real cause for the latter.
            if not bool(t.isfinite(a).all()):
                n_bad = int((~t.isfinite(a)).sum())
                raise ValueError(
                    f"SVD input contains {n_bad} non-finite entries "
                    f"(inf/NaN) out of {a.numel()}; the data is broken "
                    "upstream, check for overflow (float32 overflows around "
                    "3.4e38)") from exc
            import scipy.linalg as sla
            u, sv, vh = sla.svd(a.detach().cpu().numpy(),
                                full_matrices=full_matrices,
                                lapack_driver="gesvd")
            # the fallback goes through the host and does not carry autograd;
            # every consumer of bk.svd (rounding, tt_svd) is outside any tape
            return (t.as_tensor(u, device=a.device, dtype=a.dtype),
                    t.as_tensor(sv, device=a.device),
                    t.as_tensor(vh, device=a.device, dtype=a.dtype))

    def qr(self, a):
        return self.torch.linalg.qr(a, mode="reduced")

    def lstsq(self, a, b):
        return self.torch.linalg.lstsq(a, b, driver="gelsd").solution

    def expm(self, a):
        t = self.torch
        try:
            return t.linalg.matrix_exp(a)
        except NotImplementedError:
            # torch has no matrix_exp kernel on MPS (pytorch#141287); without
            # this fallback a KSL step on the MPS backend dies inside
            # expmv_krylov.  The matrices this package exponentiates are the
            # small Krylov/local blocks of a sweep, so the host round-trip
            # costs microseconds, not the transfer it sounds like.
            return t.linalg.matrix_exp(a.cpu()).to(a.device)


_F64 = np.dtype("float64")
_NUMPY_F64 = NumpyBackend("float64")
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


_BACKEND_CACHE = {}


def backend_of(a) -> Backend:
    """Backend that owns array ``a`` (dispatch by type, not by global state).

    Backend objects are immutable and compare by ``key``, so there is no reason
    to build a fresh one per call -- and every ``bk.norm`` / ``bk.svd`` / ...
    goes through here, hundreds of times per sweep, so caching them is most of
    the dispatch cost of an integrator (``docs/NUMERICS.md``).
    """
    if isinstance(a, np.ndarray):
        dt = a.dtype
        if dt == _F64:
            return _NUMPY_F64
        hit = _BACKEND_CACHE.get(dt)
        if hit is None:
            hit = _BACKEND_CACHE[dt] = NumpyBackend(canon_dtype(dt))
        return hit
    mod = type(a).__module__.split(".")[0]
    if mod == "torch":
        key = (type(a).__module__, str(a.device), a.dtype)
        hit = _BACKEND_CACHE.get(key)
        if hit is None:
            hit = _BACKEND_CACHE[key] = TorchBackend(str(a.device),
                                                     canon_dtype(a.dtype))
        return hit
    raise TypeError(f"no ttpy2 backend for array of type {type(a)!r}")


def same_backend(arrays, what="cores"):
    """Assert all arrays live on one backend/device; return it. Loud on mixture.

    The fast path is a type check: this sits in front of every contraction, and
    building a Backend object per operand to compare them showed up at 5% of an
    AMEn solve.
    """
    arrays = tuple(arrays)
    if all(type(a) is np.ndarray for a in arrays):
        return _NUMPY_F64 if arrays[0].dtype == np.float64 else NumpyBackend(
            canon_dtype(arrays[0].dtype))
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


@lru_cache(maxsize=None)
def _einsum_path(subs: str, shapes: tuple):
    """Contraction path for these subscripts and shapes (computed once).

    ``broadcast_to`` gives arrays of the right shape without allocating, and
    ``einsum_path`` only looks at shapes.
    """
    dummies = [np.broadcast_to(np.zeros((), dtype=np.float64), s) for s in shapes]
    return np.einsum_path(subs, *dummies, optimize="optimal")[0]


@lru_cache(maxsize=None)
def _classic_subscripts(pattern: str) -> str:
    """einops-style pattern -> classic np.einsum subscripts.

    ``"a n m b, i m j -> a i n b j"`` becomes ``"abcd,ecf->aebdf"``.  Axis names
    may be words, not just letters; ``...`` passes through.
    """
    lhs, _, rhs = pattern.partition("->")
    groups = [g.split() for g in lhs.split(",")]
    out = rhs.split()
    pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names: dict[str, str] = {}
    for token in [t for g in groups for t in g] + out:
        if token == "...":
            continue
        if token not in names:
            if len(names) >= len(pool):
                raise ValueError(f"einsum pattern {pattern!r} has too many axes")
            names[token] = pool[len(names)]
    render = lambda g: "".join("..." if t == "..." else names[t] for t in g)
    return ",".join(render(g) for g in groups) + "->" + render(out)


@lru_cache(maxsize=None)
def _binary_plan(pattern: str):
    """Compile a two-operand pattern into transpose + batched matmul.

    Even with a cached contraction path, ``np.einsum`` re-parses and re-validates
    the subscripts on every call, which is a quarter of an AMEn solve
    (``docs/NUMERICS.md``).  A contraction of two operands without repeated or
    diagonal indices is just

        (batch, left, k) @ (batch, k, right)

    after a permutation, so the permutation is worked out once per pattern and
    the call itself becomes two transposes and one matmul.  Patterns that do not
    fit this shape (one operand, three operands, a repeated index) return None
    and fall back to ``einsum``.
    """
    lhs, _, rhs = pattern.partition("->")
    groups = [g.split() for g in lhs.split(",")]
    out = rhs.split()
    if len(groups) != 2 or not rhs:
        return None
    left, right = groups
    if any(len(set(g)) != len(g) for g in (left, right, out)):
        return None            # repeated index: a diagonal, not a contraction
    if "..." in left + right + out:
        return None
    ls, rs, os_ = set(left), set(right), set(out)
    if not os_ <= (ls | rs):
        return None
    batch = [x for x in left if x in rs and x in os_]
    contracted = [x for x in left if x in rs and x not in os_]
    free_l = [x for x in left if x not in rs and x in os_]
    free_r = [x for x in right if x not in ls and x in os_]
    if len(batch) + len(contracted) + len(free_l) != len(left):
        return None
    if len(batch) + len(contracted) + len(free_r) != len(right):
        return None
    if sorted(batch + free_l + free_r) != sorted(out):
        return None
    perm_l = [left.index(x) for x in batch + free_l + contracted]
    perm_r = [right.index(x) for x in batch + contracted + free_r]
    mid = batch + free_l + free_r
    perm_out = [mid.index(x) for x in out]
    ident = lambda p: p == list(range(len(p)))
    return (tuple(perm_l), tuple(perm_r), tuple(perm_out),
            len(batch), len(free_l), len(free_r), len(contracted),
            ident(perm_l), ident(perm_r), ident(perm_out))


def _matmul_numpy(a, b, plan):
    """The numpy fast path: array methods only, no dispatch.

    Going through backend_of + a transpose lambda cost 7.4 us per call, which on
    an AMEn solve was 31% of the runtime -- more than the arithmetic. Here the
    type is already known, so ndarray.transpose/reshape/@ are called directly.
    """
    (perm_l, perm_r, perm_out, nb, nl, nr, nk, id_l, id_r, id_out) = plan
    sa, sb = a.shape, b.shape
    m = k = n = 1
    for i in range(nb, nb + nl):
        m *= sa[perm_l[i]]
    for i in range(nb + nl, len(perm_l)):
        k *= sa[perm_l[i]]
    for i in range(nb + nk, len(perm_r)):
        n *= sb[perm_r[i]]
    kb = 1
    for i in range(nb, nb + nk):
        kb *= sb[perm_r[i]]
    batch = tuple(sa[perm_l[i]] for i in range(nb))
    if batch != tuple(sb[perm_r[i]] for i in range(nb)) or k != kb:
        return None                     # size-1 broadcast: einsum handles it
    at = (a if id_l else a.transpose(perm_l)).reshape(batch + (m, k))
    bt = (b if id_r else b.transpose(perm_r)).reshape(batch + (k, n))
    out = (at @ bt).reshape(
        batch + tuple(sa[perm_l[i]] for i in range(nb, nb + nl))
        + tuple(sb[perm_r[i]] for i in range(nb + nk, len(perm_r))))
    return out if id_out else out.transpose(perm_out)


def _matmul_contract(a, b, plan):
    """Execute a compiled binary contraction, or return None to fall back.

    numpy's einsum broadcasts an axis of size 1 against a larger one carrying
    the same label.  A matmul cannot, so a shape mismatch on a batch or a
    contracted axis sends the call back to einsum instead of being forced.
    """
    (perm_l, perm_r, perm_out, nb, nl, nr, nk,
     id_l, id_r, id_out) = plan
    sa, sb = a.shape, b.shape
    shape_a = [sa[i] for i in perm_l]
    shape_b = [sb[i] for i in perm_r]
    if (shape_a[:nb] != shape_b[:nb]
            or shape_a[nb + nl:] != shape_b[nb:nb + nk]):
        return None                      # size-1 broadcast: einsum handles it
    # plain loops, not np.prod: these lists hold two or three small ints and
    # np.prod on them cost 0.11 s of a 1.0 s AMEn solve (42635 calls).
    batch = tuple(shape_a[:nb])
    m = 1
    for i in range(nb, nb + nl):
        m *= shape_a[i]
    k = 1
    for i in range(nb + nl, len(shape_a)):
        k *= shape_a[i]
    n = 1
    for i in range(nb + nk, len(shape_b)):
        n *= shape_b[i]
    at = (a if id_l else transpose(a, perm_l)).reshape(batch + (m, k))
    bt = (b if id_r else transpose(b, perm_r)).reshape(batch + (k, n))
    out = (at @ bt).reshape(batch + tuple(shape_a[nb:nb + nl])
                            + tuple(shape_b[nb + nk:]))
    return out if id_out else transpose(out, perm_out)


def einsum(*operands_and_pattern):
    """Contract with an einops-style pattern, through BLAS.

    Signature matches ``einops.einsum``: the tensors first, the pattern last.
    einops hands its pattern to ``np.einsum`` without ``optimize``, which keeps
    even a plain binary contraction out of BLAS; and ``np.einsum`` re-parses the
    subscripts on every call.  Two-operand patterns are therefore compiled once
    into a permutation plus a batched matmul; anything else goes to einsum with
    a cached contraction path.
    """
    *operands, pattern = operands_and_pattern
    if not isinstance(pattern, str):
        raise TypeError("the einsum pattern must come last, as in einops.einsum")
    if len(operands) == 2:
        a, b = operands
        plan = _binary_plan(pattern)
        if plan is not None:
            if type(a) is np.ndarray and type(b) is np.ndarray:
                out = _matmul_numpy(a, b, plan)
            else:
                same_backend(operands, "operands")
                out = _matmul_contract(a, b, plan)
            if out is not None:
                return out
    return same_backend(operands, "operands").einsum(
        _classic_subscripts(pattern), *operands)


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


def is_scalar(a) -> bool:
    """True for a Python/numpy number **or** a 0-d array or tensor.

    ``x.norm()`` on the torch backend is a 0-d tensor and must stay one, or the
    gradient through ``||x||`` is silently dropped (see ``TorchBackend.norm``).
    Everything that accepts a scalar therefore has to accept that shape too,
    otherwise the obvious ``x * (1 / x.norm())`` raises on torch and works on
    numpy -- and scaling would break the tape it was kept alive for.
    """
    if isinstance(a, Number):
        return True
    return getattr(a, "ndim", None) == 0 and hasattr(a, "dtype")


def scalar_dtype(a) -> str:
    """Canonical dtype of a scalar accepted by :func:`is_scalar`, or None.

    ``None`` means "no opinion": a plain Python real carries no dtype of its
    own and must not drag a float32 tensor up to float64.
    """
    if isinstance(a, Number):
        return "complex128" if isinstance(a, complex) and a.imag != 0 else None
    return canon_dtype(a.dtype)


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
