"""TT arithmetic on plain lists of cores.

This module is the single owner of the TT math.  ``tt.vector`` and ``tt.matrix``
are thin wrappers around it, so the same code serves vectors, matrices and the
block (r0 > 1 / rd > 1) case.

A *core list* is ``[G_1, ..., G_d]`` with ``G_k`` of shape ``(r_k, n_k, r_{k+1})``
and ``r_1``/``r_{d+1}`` not necessarily equal to one.  The tensor it represents is

    A[i_1, ..., i_d] = G_1[:, i_1, :] @ G_2[:, i_2, :] @ ... @ G_d[:, i_d, :]

All arrays are backend arrays (numpy or torch); every operation dispatches
through :mod:`tt.backend`, nothing here knows which one it got.
"""

from __future__ import annotations

import numpy as np
from einops import einsum, rearrange

from .. import backend as bk

__all__ = [
    "chop", "tt_svd", "full", "orthogonalize", "round_cores", "add", "scale",
    "hadamard", "dot", "norm", "kron", "sub", "ranks", "modes", "check_cores",
    "to_dtype", "matvec_cores", "matmat_cores",
]


# --- helpers -----------------------------------------------------------------

def ranks(cores):
    return [cores[0].shape[0]] + [c.shape[2] for c in cores]


def modes(cores):
    return [c.shape[1] for c in cores]


def check_cores(cores):
    """Validate a core list; raise loudly on the first inconsistency."""
    if not cores:
        raise ValueError("empty core list")
    for k, c in enumerate(cores):
        if c.ndim != 3:
            raise ValueError(f"core {k} has ndim {c.ndim}, expected 3 (r,n,r)")
    for k in range(len(cores) - 1):
        if cores[k].shape[2] != cores[k + 1].shape[0]:
            raise ValueError(
                f"rank mismatch between cores {k} and {k + 1}: "
                f"{cores[k].shape[2]} != {cores[k + 1].shape[0]}")
    bk.same_backend(cores)
    dts = {bk.dtype_of(c) for c in cores}
    if len(dts) > 1:
        raise TypeError(f"cores have mixed dtypes {sorted(dts)}; "
                        "cast the tensor explicitly")
    return cores


def to_dtype(cores, dtype):
    dtype = bk.canon_dtype(dtype)
    if bk.dtype_of(cores[0]) == dtype:
        return cores
    return [bk.asarray(c, dtype, backend=bk.backend_of(c)) for c in cores]


def _unify(a, b):
    """Bring two core lists to a common dtype (never narrowing)."""
    dt = bk.result_dtype(bk.dtype_of(a[0]), bk.dtype_of(b[0]))
    return to_dtype(a, dt), to_dtype(b, dt), dt


def _lq(a):
    """a = L @ Q with Q having orthonormal rows."""
    q, r = bk.qr(rearrange(a.conj(), "i j -> j i"))
    return rearrange(r.conj(), "i j -> j i"), rearrange(q.conj(), "i j -> j i")


def chop(sv, eps):
    """Smallest rank whose discarded tail has 2-norm below ``eps``.

    ``sv`` are singular values in non-increasing order, ``eps`` is *absolute*.
    Same contract as ``my_chop2`` in legacy ttpy.
    """
    sv = np.asarray(bk.to_numpy(sv), dtype=np.float64)
    if eps <= 0.0 or sv.size == 0:
        return sv.size
    tail = np.cumsum(np.abs(sv[::-1]) ** 2)[::-1]
    below = np.flatnonzero(tail < eps ** 2)
    return int(below[0]) if below.size else sv.size


# --- construction ------------------------------------------------------------

def tt_svd(a, eps=1e-14, rmax=None):
    """TT-SVD of a dense array ``a`` of shape ``(n_1, ..., n_d)``."""
    n = list(a.shape)
    d = len(n)
    if d == 0:
        raise ValueError("cannot decompose a 0-dimensional array")
    if d == 1:
        return [a.reshape((1, n[0], 1))]
    rmax = 10 ** 9 if rmax is None else int(rmax)
    frob = bk.norm(a)
    delta = eps * frob / np.sqrt(d - 1) if frob > 0 else 0.0

    cores = []
    c = a.reshape((1, -1))
    r_left = 1
    for k in range(d - 1):
        c = c.reshape((r_left * n[k], -1))
        u, s, vh = bk.svd(c)
        r_new = max(1, min(chop(s, delta), rmax, u.shape[1]))
        u = u[:, :r_new]
        cores.append(u.reshape((r_left, n[k], r_new)))
        c = s[:r_new].reshape((r_new, 1)) * vh[:r_new, :]
        r_left = r_new
    cores.append(c.reshape((r_left, n[d - 1], 1)))
    return cores


def full(cores):
    """Dense tensor of shape ``(r0,) + n + (rd,)`` with boundary ranks squeezed."""
    check_cores(cores)
    n = modes(cores)
    r = ranks(cores)
    res = cores[0]
    for k in range(1, len(cores)):
        res = res.reshape((-1, r[k])) @ cores[k].reshape((r[k], -1))
    res = res.reshape([r[0]] + n + [r[-1]])
    shape = ([r[0]] if r[0] > 1 else []) + n + ([r[-1]] if r[-1] > 1 else [])
    return res.reshape(shape if shape else [1])


# --- orthogonalization and rounding -----------------------------------------

def orthogonalize(cores, center=None, inplace_ok=False):
    """Right-to-left orthogonalization.

    Cores ``center+1 .. d-1`` are made right-orthogonal (``G G^H = I``); all the
    norm is pushed into core ``center`` (default: the first one).
    """
    check_cores(cores)
    d = len(cores)
    center = 0 if center is None else center
    out = list(cores)
    for k in range(d - 1, center, -1):
        r0, nk, r1 = out[k].shape
        lmat, q = _lq(out[k].reshape((r0, nk * r1)))
        rnew = q.shape[0]
        out[k] = q.reshape((rnew, nk, r1))
        p0, pn, _ = out[k - 1].shape
        out[k - 1] = (out[k - 1].reshape((p0 * pn, r0)) @ lmat).reshape((p0, pn, rnew))
    return out


def round_cores(cores, eps=1e-14, rmax=None):
    """TT rounding: right-to-left QR sweep, then truncated left-to-right SVD."""
    check_cores(cores)
    d = len(cores)
    if d == 1:
        return [cores[0].copy() if hasattr(cores[0], "copy") else cores[0]]
    rmax = 10 ** 9 if rmax is None else int(rmax)

    out = orthogonalize(cores, center=0)
    frob = bk.norm(out[0])
    delta = eps * frob / np.sqrt(d - 1) if frob > 0 else 0.0

    for k in range(d - 1):
        r0, nk, r1 = out[k].shape
        u, s, vh = bk.svd(out[k].reshape((r0 * nk, r1)))
        rnew = max(1, min(chop(s, delta), rmax, u.shape[1]))
        u = u[:, :rnew]
        sv = s[:rnew]
        out[k] = u.reshape((r0, nk, rnew))
        tail = sv.reshape((rnew, 1)) * vh[:rnew, :]
        p0, pn, p1 = out[k + 1].shape
        out[k + 1] = (tail @ out[k + 1].reshape((p0, pn * p1))).reshape((rnew, pn, p1))
    return out


# --- arithmetic --------------------------------------------------------------

def add(a, b):
    """Sum of two TT tensors: ranks add up (no rounding)."""
    check_cores(a)
    check_cores(b)
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    if modes(a) != modes(b):
        raise ValueError(f"mode mismatch: {modes(a)} vs {modes(b)}")
    a, b, dt = _unify(a, b)
    d = len(a)
    if d == 1:
        return [a[0] + b[0]]

    ra, rb = ranks(a), ranks(b)
    out = []
    for k in range(d):
        n = a[k].shape[1]
        left_block = (k == 0 and ra[0] == rb[0])
        right_block = (k == d - 1 and ra[-1] == rb[-1])
        r0 = ra[k] if left_block else ra[k] + rb[k]
        r1 = ra[k + 1] if right_block else ra[k + 1] + rb[k + 1]
        c = bk.zeros((r0, n, r1), dtype=dt, like=a[0])
        if left_block and right_block:
            c = a[k] + b[k]
        elif left_block:
            c[:, :, :ra[k + 1]] = a[k]
            c[:, :, ra[k + 1]:] = b[k]
        elif right_block:
            c[:ra[k], :, :] = a[k]
            c[ra[k]:, :, :] = b[k]
        else:
            c[:ra[k], :, :ra[k + 1]] = a[k]
            c[ra[k]:, :, ra[k + 1]:] = b[k]
        out.append(c)
    return out


def scale(cores, alpha):
    """Multiply the tensor by a scalar (applied to the first core)."""
    check_cores(cores)
    dt = bk.result_dtype(bk.dtype_of(cores[0]),
                         "complex128" if isinstance(alpha, complex)
                         and alpha.imag != 0 else bk.dtype_of(cores[0]))
    out = to_dtype(cores, dt)
    out = list(out)
    out[0] = out[0] * alpha
    return out


def sub(a, b):
    return add(a, scale(b, -1.0))


def hadamard(a, b):
    """Elementwise (Hadamard) product; ranks multiply."""
    check_cores(a)
    check_cores(b)
    if modes(a) != modes(b):
        raise ValueError(f"mode mismatch: {modes(a)} vs {modes(b)}")
    a, b, _ = _unify(a, b)
    out = []
    for ca, cb in zip(a, b):
        c = einsum(ca, cb, "i n j, k n l -> i k n j l")
        out.append(rearrange(c, "i k n j l -> (i k) n (j l)"))
    return out


def dot(a, b):
    """<a, b> = sum conj(a) * b.

    Returns a scalar when all boundary ranks are one, otherwise an array of
    shape ``(ra0, rb0, rad, rbd)`` (block TT case).
    """
    check_cores(a)
    check_cores(b)
    if modes(a) != modes(b):
        raise ValueError(f"mode mismatch: {modes(a)} vs {modes(b)}")
    a, b, _ = _unify(a, b)
    ra, rb = ranks(a), ranks(b)
    # phi[ia, ib] accumulated left to right, carrying the left boundary indices
    phi = einsum(a[0].conj(), b[0], "a n i, b n j -> a b i j")
    for k in range(1, len(a)):
        phi = einsum(phi, a[k].conj(), "a b i j, i n p -> a b j n p")
        phi = einsum(phi, b[k], "a b j n p, j n q -> a b p q")
    if ra[0] == rb[0] == ra[-1] == rb[-1] == 1:
        return phi.reshape(())[()] if hasattr(phi, "reshape") else phi
    return phi


def norm(cores):
    """Frobenius norm, computed through an orthogonalization sweep (stable)."""
    check_cores(cores)
    if len(cores) == 1:
        return bk.norm(cores[0])
    out = orthogonalize(cores, center=0)
    return bk.norm(out[0])


def kron(a, b):
    """Kronecker product: concatenation of core lists (needs matching ranks)."""
    if a is None:
        return list(b)
    if b is None:
        return list(a)
    check_cores(a)
    check_cores(b)
    if ranks(a)[-1] != ranks(b)[0]:
        raise ValueError(
            f"kron needs a.r[-1] == b.r[0], got {ranks(a)[-1]} and {ranks(b)[0]}")
    a, b, _ = _unify(a, b)
    return list(a) + list(b)


# --- TT-matrix helpers -------------------------------------------------------

def matvec_cores(acores, bcores):
    """TT-matrix (r,n,m,r) times TT-vector (r,m,r) -> TT-vector (r,n,r)."""
    out = []
    for A, x in zip(acores, bcores):
        if A.shape[2] != x.shape[1]:
            raise ValueError(
                f"matvec mode mismatch: matrix m={A.shape[2]}, vector n={x.shape[1]}")
        c = einsum(A, x, "a n m b, i m j -> a i n b j")
        out.append(rearrange(c, "a i n b j -> (a i) n (b j)"))
    return out


def matmat_cores(acores, bcores):
    """TT-matrix times TT-matrix, cores in (r,n,m,r) layout."""
    out = []
    for A, B in zip(acores, bcores):
        if A.shape[2] != B.shape[1]:
            raise ValueError(f"matmat mode mismatch: {A.shape[2]} vs {B.shape[1]}")
        c = einsum(A, B, "a n k b, i k m j -> a i n m b j")
        out.append(rearrange(c, "a i n m b j -> (a i) n m (b j)"))
    return out
