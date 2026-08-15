"""Constructors and operations on TT tensors (the ``tt.*`` namespace)."""

from __future__ import annotations

import math

import numpy as np
from einops import rearrange
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk
from . import _ops
from .matrix import matrix
from .vector import vector

__all__ = [
    "matvec", "col", "kron", "dot", "diag", "mkron", "zkron", "zkronv",
    "zmeshgrid", "zaffine", "concatenate", "sum", "ones", "zeros", "rand",
    "eye", "Toeplitz", "qlaplace_dd", "xfun", "linspace", "sin", "cos",
    "delta", "stepfun", "qshift", "shift", "unit", "IpaS", "reshape", "permute",
    "qdiff", "qtri_ones", "qlaplace_dn", "level_major_order",
]


def _vec(cores):
    """Build a TT-vector from freshly made cores, honouring the default backend.

    Constructors allocate their (tiny) cores with numpy because that is how the
    formulas read.  They must still return a tensor on whatever backend
    :func:`tt.set_backend` selected — otherwise ``tt.matvec(tt.qlaplace_dd(d), x)``
    with a GPU ``x`` mixes backends and dies somewhere deep instead of working.
    ``vector.from_list`` deliberately keeps the backend of the arrays it is
    given, so the conversion has to happen here.
    """
    target = bk.get_backend()
    return vector.from_list([target.asarray(c) for c in cores])


def _mat(cores4):
    """Same as :func:`_vec` for TT-matrix cores of shape (r, n, m, r)."""
    target = bk.get_backend()
    return matrix.from_list([target.asarray(c) for c in cores4])


def _like(cores, ref):
    """Put freshly computed cores on the backend of an existing tensor.

    Used by the operations that have to drop to numpy internally: an operation
    on a user's tensor must give the answer back where the tensor lives, which
    is not necessarily the global default.
    """
    target = bk.backend_of(ref.cores[0] if hasattr(ref, "cores") else ref)
    return [target.asarray(c) for c in cores]


def _modes(n, d=None):
    """Normalize the (n, d) argument pair used all over the legacy API."""
    if d is None:
        arr = np.asanyarray(n, dtype=np.int64).ravel()
    else:
        arr = np.asanyarray(n, dtype=np.int64).ravel()
        arr = np.repeat(arr, d) if arr.size == 1 else np.tile(arr, d)
    return arr.astype(np.int32)


# --- products ----------------------------------------------------------------

def matvec(a, b, compression=False):
    """TT-matrix by TT-vector product.

    ``compression`` (a tolerance) applies a one-directional truncation while
    the product is built, which keeps the intermediate ranks bounded.
    """
    cores = _ops.matvec_cores(matrix.to_list(a), b.cores)
    res = vector.from_list(cores)
    if compression:
        res = res.round(compression)
    return res


def col(a, k):
    if hasattr(a, "__col__"):
        return a.__col__(k)
    raise ValueError("col expects a TT-vector or a TT-matrix")


def kron(a, b):
    """Kronecker product of two TT-vectors or two TT-matrices."""
    if a is None:
        return b
    if b is None:
        return a
    if hasattr(a, "__kron__"):
        return a.__kron__(b)
    raise ValueError("kron expects TT-vectors or TT-matrices")


def dot(a, b):
    if hasattr(a, "__dot__"):
        return a.__dot__(b)
    raise ValueError("dot expects TT-vectors or TT-matrices")


def diag(a):
    """Diagonal matrix from a vector, or the diagonal of a matrix."""
    if hasattr(a, "__diag__"):
        return a.__diag__()
    raise ValueError("diag expects a TT-vector or a TT-matrix")


def mkron(a, *args):
    """Kronecker product of all arguments (lists are flattened)."""
    items = list(a) if isinstance(a, list) else [a]
    for i in args:
        items.extend(i) if isinstance(i, list) else items.append(i)
    cores = []
    for t in items:
        cores.extend(t.tt.cores if isinstance(t, matrix) else t.cores)
    return vector.from_list(cores)


def zkron(ttA, ttB):
    """Kronecker product of TT-matrices in z-order (arXiv:1802.02839).

    Contributed by Larisa Markeeva (ttpy, 2018), from her work on solving
    equations on complicated domains in the QTT format via z-order curves.
    """
    al, bl = matrix.to_list(ttA), matrix.to_list(ttB)
    out = [np.kron(bk.to_numpy(B), bk.to_numpy(A)) for A, B in zip(al, bl)]
    return matrix.from_list(_like(out, ttA.tt))


def zkronv(ttA, ttB):
    """Kronecker product of TT-vectors in z-order.

    Contributed by Larisa Markeeva (ttpy, 2018); see :func:`zkron`.
    """
    al, bl = ttA.cores, ttB.cores
    out = [np.kron(bk.to_numpy(B), bk.to_numpy(A)) for A, B in zip(al, bl)]
    return vector.from_list(_like(out, ttA))


def zmeshgrid(d):
    """Meshgrid in z-order; ``4**d`` nodes.

    Contributed by Larisa Markeeva (ttpy, 2018); see :func:`zkron`.
    """
    lin, one = xfun(2, d), ones(2, d)
    return zkronv(lin, one), zkronv(one, lin)


def zaffine(c0, c1, c2, d):
    """``c0 + c1*ex + c2*ey`` in z-ordering, ``d`` QTT cores.

    Contributed by Larisa Markeeva (ttpy, 2018); see :func:`zkron`.
    """
    xx, yy = zmeshgrid(d)
    hx = [bk.to_numpy(c).copy() for c in xx.cores]
    hy = [bk.to_numpy(c) for c in yy.cores]
    hs = [c.copy() for c in hx]
    hs[0][:, :, 0] = c1 * hx[0][:, :, 0] + c2 * hy[0][:, :, 0]
    hs[-1][1, :, :] = c1 * hx[-1][1, :, :] + (c0 + c2 * hy[-1][1, :, :])
    for k in range(1, len(hs) - 1):
        hs[k][1, :, 0] = c1 * hx[k][1, :, 0] + c2 * hy[k][1, :, 0]
    return vector.from_list(_like(hs, xx))


def concatenate(*args):
    """Stack tensors along a new leading mode of size ``len(args)``."""
    k = len(args)
    sel = np.zeros((1, k, 1))
    sel[0, 0, 0] = 1.0
    result = kron(vector.from_list([sel]), args[0])
    for i in range(1, k):
        sel = np.zeros((1, k, 1))
        sel[0, i, 0] = 1.0
        result = result + kron(vector.from_list([sel]), args[i])
    return result


def sum(a, axis=-1):
    """Sum a TT-vector over the given axes (all of them by default)."""
    cores = [c for c in (a.tt.cores if isinstance(a, matrix) else a.cores)]
    d = len(cores)
    if axis is None or (isinstance(axis, int) and axis < 0):
        axes = list(range(d))
    elif isinstance(axis, (int, np.integer)):
        axes = [int(axis)]
    else:
        axes = list(axis)
    for ax in sorted(axes)[::-1]:
        summed = cores[ax].sum(axis=1)  # (r_left, r_right)
        if d == 1:
            return summed.reshape(())[()] if int(np.prod(summed.shape)) == 1 else summed
        if ax > 0:
            left = cores[ax - 1]
            cores[ax - 1] = (left.reshape((-1, left.shape[2])) @ summed).reshape(
                left.shape[0], left.shape[1], summed.shape[1])
        else:
            right = cores[ax + 1]
            cores[ax + 1] = (summed @ right.reshape((right.shape[0], -1))).reshape(
                summed.shape[0], right.shape[1], right.shape[2])
        cores.pop(ax)
        d -= 1
    return vector.from_list(cores)


# --- constructors ------------------------------------------------------------

def ones(n, d=None):
    """TT-vector of all ones."""
    n0 = _modes(n, d)
    return _vec([np.ones((1, int(k), 1)) for k in n0])


def zeros(n, d=None):
    """TT-vector of all zeros (rank 1)."""
    n0 = _modes(n, d)
    return _vec([np.zeros((1, int(k), 1)) for k in n0])


def rand(n, d=None, r=2, samplefunc=None):
    """Random TT-vector with ranks ``r``.

    ``samplefunc(size)`` overrides the sampler and must return ``size`` numbers.
    """
    n0 = np.asanyarray(n, dtype=np.int64).ravel()
    if d is None:
        d = n0.size
    if n0.size == 1:
        n0 = np.repeat(n0, d)
    r0 = np.asanyarray(r, dtype=np.int64).ravel()
    if r0.size == 1:
        r0 = np.full(d + 1, int(r0[0]), dtype=np.int64)
        r0[0] = r0[d] = 1
    cores = []
    for k in range(d):
        shape = (int(r0[k]), int(n0[k]), int(r0[k + 1]))
        if samplefunc is None:
            cores.append(bk.randn(shape))
        else:
            cores.append(bk.asarray(
                np.asarray(samplefunc(int(np.prod(shape)))).reshape(
                    (shape[2], shape[1], shape[0])).transpose(2, 1, 0)))
    return _vec(cores)


def eye(n, d=None):
    """Identity TT-matrix."""
    n0 = _modes(n, d)
    return _mat([np.eye(int(k)).reshape((1, int(k), int(k), 1)) for k in n0])


def xfun(n, d=None):
    """QTT representation of ``0, 1, ..., prod(n)-1``."""
    n0 = _modes(n, d)
    dd = n0.size
    if dd == 1:
        return _vec([np.arange(float(n0[0])).reshape((1, int(n0[0]), 1))])
    cores = []
    first = np.ones((1, int(n0[0]), 2))
    first[0, :, 0] = np.arange(n0[0])
    cores.append(first)
    ni = float(n0[0])
    for i in range(1, dd - 1):
        cur = np.zeros((2, int(n0[i]), 2))
        cur[0, :, 0] = 1.0
        cur[1, :, 1] = 1.0
        cur[1, :, 0] = ni * np.arange(n0[i])
        ni *= float(n0[i])
        cores.append(cur)
    last = np.ones((2, int(n0[-1]), 1))
    last[1, :, 0] = ni * np.arange(n0[-1])
    cores.append(last)
    return _vec(cores)


def linspace(n, d=None, a=0.0, b=1.0, right=True, left=True):
    """QTT representation of a uniform grid on ``[a, b]``."""
    n0 = _modes(n, d)
    t, e = xfun(n0), ones(n0)
    N = int(np.prod(n0.astype(np.int64)))
    if left and right:
        h = (b - a) / (N - 1)
        res = a * e + t * h
    elif left:
        h = (b - a) / N
        res = a * e + t * h
    elif right:
        h = (b - a) / N
        res = a * e + (t + e) * h
    else:
        h = (b - a) / (N - 1)
        res = a * e + (t + e) * h
    return res.round(1e-13)


def sin(d, alpha=1.0, phase=0.0):
    """TT-vector of ``sin(alpha * k + phase)`` on ``k = 0..2^d-1`` (QTT)."""
    cores = []
    first = np.zeros((1, 2, 2))
    first[0, 0, :] = [math.cos(phase), math.sin(phase)]
    first[0, 1, :] = [math.cos(alpha + phase), math.sin(alpha + phase)]
    cores.append(first)
    for i in range(1, d - 1):
        cur = np.zeros((2, 2, 2))
        cur[0, 0, :] = [1.0, 0.0]
        cur[1, 0, :] = [0.0, 1.0]
        cur[0, 1, :] = [math.cos(alpha * 2 ** i), math.sin(alpha * 2 ** i)]
        cur[1, 1, :] = [-math.sin(alpha * 2 ** i), math.cos(alpha * 2 ** i)]
        cores.append(cur)
    last = np.zeros((2, 2, 1))
    last[0, :, 0] = [0.0, math.sin(alpha * 2 ** (d - 1))]
    last[1, :, 0] = [1.0, math.cos(alpha * 2 ** (d - 1))]
    cores.append(last)
    return _vec(cores)


def cos(d, alpha=1.0, phase=0.0):
    """TT-vector of ``cos(alpha * k + phase)``."""
    return sin(d, alpha, phase + math.pi * 0.5)


def delta(n, d=None, center=0):
    """TT-vector of a delta function at flat index ``center``."""
    n0 = _modes(n, d)
    dd = n0.size
    if center < 0:
        cind = [0] * dd
    else:
        cind, rest = [], int(center)
        for i in range(dd):
            cind.append(rest % int(n0[i]))
            rest //= int(n0[i])
        if rest > 0:
            cind = [0] * dd
    cores = []
    for i in range(dd):
        cur = np.zeros((1, int(n0[i]), 1))
        cur[0, cind[i], 0] = 1.0
        cores.append(cur)
    return _vec(cores)


def stepfun(n, d=None, center=1, direction=1):
    """TT-vector of the Heaviside step at flat index ``center``."""
    n0 = _modes(n, d)
    dd = n0.size
    N = int(np.prod(n0.astype(np.int64)))
    if center >= N and direction < 0 or center <= 0 and direction > 0:
        return ones(n0)
    if center <= 0 and direction < 0 or center >= N and direction > 0:
        raise ValueError(
            "Heaviside function with this center and direction is identically zero")
    center = N - center if direction > 0 else center
    cind, rest = [], int(center)
    for i in range(dd):
        cind.append(rest % int(n0[i]))
        rest //= int(n0[i])

    def gen_notx(c, k):
        return [0.0] * (k - c) + [1.0] * c

    def gen_notx_rev(c, k):
        return [1.0] * c + [0.0] * (k - c)

    def gen_x(c, k):
        out = [0.0] * k
        out[k - c - 1] = 1.0
        return out

    def gen_x_rev(c, k):
        out = [0.0] * k
        out[c] = 1.0
        return out

    x, notx = (gen_x, gen_notx) if direction > 0 else (gen_x_rev, gen_notx_rev)

    cores, prevrank = [], 1
    for i in range(dd)[::-1]:
        break_further = max([0] + cind[:i])
        nextrank = 2 if break_further else 1
        cur = np.zeros((nextrank, int(n0[i]), prevrank))
        tempx, tempnotx = x(cind[i], int(n0[i])), notx(cind[i], int(n0[i]))
        one = [1.0] * int(n0[i])
        if not break_further:
            if cind[i]:
                if prevrank > 1:
                    cur[0, :, 0] = one
                    cur[0, :, 1] = tempnotx
                else:
                    cur[0, :, 0] = tempnotx
            else:
                cur[0, :, 0] = one
        else:
            if prevrank > 1:
                cur[0, :, 0] = one
                if cind[i]:
                    cur[0, :, 1] = tempnotx
                cur[1, :, 1] = tempx
            else:
                if cind[i]:
                    cur[0, :, 0] = tempnotx
                    cur[1, :, 0] = tempx
                else:
                    nextrank = 1
                    cur = cur[:1, :, :]
                    cur[0, :, 0] = tempx
        prevrank = nextrank
        cores.append(cur)
    return _vec(cores[::-1])


def unit(n, d=None, j=None, tt_instance=True):
    """``e_j`` in the TT format."""
    if isinstance(n, (int, np.integer)):
        d = 1 if d is None else d
        n = np.full(d, int(n), dtype=np.int64)
    else:
        n = np.asanyarray(n, dtype=np.int64).ravel()
        d = n.size
    j = 0 if j is None else int(j)
    cores, rest = [], j
    for k in range(d):
        cur = np.zeros((1, int(n[k]), 1))
        cur[0, rest % int(n[k]), 0] = 1.0
        rest //= int(n[k])
        cores.append(cur)
    return _vec(cores) if tt_instance else cores


def shift(d, step=-1):
    """QTT shift matrix of size ``2^d``: ``S[i, j] = 1`` iff ``i = j - step``.

    Built from binary addition with a carry: core ``k`` holds bit ``k`` (core 0
    is the least significant one, matching the TT convention that mode 1 runs
    fastest), and the rank index carries the borrow/carry bit.
    """
    if step not in (-1, 1):
        raise ValueError("shift supports step = -1 (down) or +1 (up)")
    cores = []
    for k in range(d):
        cur = np.zeros((2, 2, 2, 2))
        for cin in range(2):
            for j in range(2):
                total = j + cin
                cur[cin, total % 2, j, total // 2] = 1.0
        cores.append(cur)
    cores[0] = cores[0][1:2]        # carry into the lowest bit: add one
    cores[-1] = cores[-1][:, :, :, 0:1]   # no carry out of the highest bit
    S = _mat(cores)
    return S if step == -1 else S.T


def IpaS(d, a, tt_instance=True):
    """``I + a * S_{-1}``: bidiagonal, ones on the diagonal, ``a`` below it."""
    if d == 1:
        M = np.array([[1.0, 0.0], [a, 1.0]]).reshape((1, 2, 2, 1))
        return _mat([M]) if tt_instance else M
    M = (eye(2, d) + a * shift(d, -1)).round(1e-14)
    return M if tt_instance else matrix.to_list(M)


def qlaplace_dd(d):
    """QTT representation of the multidimensional Laplace operator."""
    d0 = list(d)[::-1]
    D = len(d0)
    I = np.eye(2)
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    cr = []
    if D == 1:
        for k in range(1, d0[0] + 1):
            if k == 1:
                cur = np.zeros((1, 2, 2, 3))
                cur[:, :, :, 0] = 2 * I - J - J.T
                cur[:, :, :, 1] = -J
                cur[:, :, :, 2] = -J.T
            elif k == d0[0]:
                cur = np.zeros((3, 2, 2, 1))
                cur[0, :, :, 0] = I
                cur[1, :, :, 0] = J.T
                cur[2, :, :, 0] = J
            else:
                cur = np.zeros((3, 2, 2, 3))
                cur[0, :, :, 0] = I
                cur[1, :, :, 1] = J
                cur[2, :, :, 2] = J.T
                cur[1, :, :, 0] = J.T
                cur[2, :, :, 0] = J
            cr.append(cur)
        return _mat(cr)
    for k in range(D):
        for kappa in range(1, d0[k] + 1):
            if kappa == 1:
                if k == 0:
                    cur = np.zeros((1, 2, 2, 4))
                    cur[:, :, :, 0] = 2 * I - J - J.T
                    cur[:, :, :, 1] = -J
                    cur[:, :, :, 2] = -J.T
                    cur[:, :, :, 3] = I
                elif k == D - 1:
                    cur = np.zeros((2, 2, 2, 3))
                    cur[0, :, :, 0] = 2 * I - J - J.T
                    cur[0, :, :, 1] = -J
                    cur[0, :, :, 2] = -J.T
                    cur[1, :, :, 0] = I
                else:
                    cur = np.zeros((2, 2, 2, 4))
                    cur[0, :, :, 0] = 2 * I - J - J.T
                    cur[0, :, :, 1] = -J
                    cur[0, :, :, 2] = -J.T
                    cur[0, :, :, 3] = I
                    cur[1, :, :, 0] = I
            elif kappa == d0[k]:
                if k == D - 1:
                    cur = np.zeros((3, 2, 2, 1))
                    cur[0, :, :, 0] = I
                    cur[1, :, :, 0] = J.T
                    cur[2, :, :, 0] = J
                else:
                    cur = np.zeros((4, 2, 2, 2))
                    cur[3, :, :, 0] = I
                    cur[0, :, :, 1] = I
                    cur[1, :, :, 1] = J.T
                    cur[2, :, :, 1] = J
            else:
                if k == D - 1:
                    cur = np.zeros((3, 2, 2, 3))
                    cur[0, :, :, 0] = I
                    cur[1, :, :, 1] = J
                    cur[2, :, :, 2] = J.T
                    cur[1, :, :, 0] = J.T
                    cur[2, :, :, 0] = J
                else:
                    cur = np.zeros((4, 2, 2, 4))
                    cur[0, :, :, 0] = I
                    cur[1, :, :, 1] = J
                    cur[2, :, :, 2] = J.T
                    cur[1, :, :, 0] = J.T
                    cur[2, :, :, 0] = J
                    cur[3, :, :, 3] = I
            cr.append(cur)
    return _mat(cr)


def Toeplitz(x, d=None, D=None, kind="F"):
    """Multilevel Toeplitz TT-matrix built from the QTT-vector ``x``.

    ``kind`` per level: ``'F'`` full (``x`` has ``d+1`` cores per level),
    ``'C'`` circulant, ``'L'`` lower, ``'U'`` upper triangular.
    """
    def check_kinds(D, kind):
        if D % len(kind) == 0:
            kind.extend(kind * (D // len(kind) - 1))
        if len(kind) != D:
            raise ValueError("give one kind or exactly D kinds")

    kind = list(kind)
    if not set(kind).issubset({"F", "C", "L", "U"}):
        raise ValueError("Toeplitz kind must be one of F, C, L, U")
    if d is None:
        if D is None:
            D = len(kind)
        if x.d % D:
            raise ValueError("x.d must be divisible by D when d is not given")
        if len(kind) == 1:
            d = np.array([x.d // D - (1 if kind[0] == "F" else 0)] * D, dtype=np.int64)
            kind = kind * D
        else:
            check_kinds(D, kind)
            if set(kind).issubset({"F"}):
                d = np.array([x.d // D - 1] * D, dtype=np.int64)
            elif set(kind).issubset({"C", "L", "U"}):
                d = np.array([x.d // D] * D, dtype=np.int64)
            else:
                raise ValueError(
                    "mixing 'F' with 'C'/'L'/'U' requires an explicit d")
    else:
        d = np.asarray(d, dtype=np.int64).ravel()
        if D is None:
            D = d.size
        elif d.size == 1:
            d = np.array([d[0]] * D, dtype=np.int64)
        if D != d.size:
            raise ValueError("D must equal len(d)")
        check_kinds(D, kind)
        if int(np.sum(d)) + int(np.sum([k == "F" for k in kind])) != x.d:
            raise ValueError("dimension mismatch: x.d != d_1 + ... + d_D (+ 1 per 'F')")

    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    JT = J.T.copy()
    H = np.array([[0.0, 1.0], [1.0, 0.0]])
    # cores below follow the layout of the legacy implementation:
    # axes are (left, i, j, right) after the transposes done there
    S = np.array([[[0.0], [1.0]], [[1.0], [0.0]]]).transpose()   # 1 x 2 x 2
    P = np.zeros((2, 2, 2, 2))
    P[:, :, 0, 0] = I
    P[:, :, 1, 0] = H
    P[:, :, 0, 1] = H
    P[:, :, 1, 1] = I
    P = np.transpose(P)
    Q = np.zeros((2, 2, 2, 2))
    Q[:, :, 0, 0] = I
    Q[:, :, 1, 0] = JT
    Q[:, :, 0, 1] = JT
    Q = np.transpose(Q)
    R = np.zeros((2, 2, 2, 2))
    R[:, :, 1, 0] = J
    R[:, :, 0, 1] = J
    R[:, :, 1, 1] = I
    R = np.transpose(R)
    W = np.zeros([2] * 5)
    W[0, :, :, 0, 0] = W[1, :, :, 1, 1] = I
    W[0, :, :, 1, 0] = W[0, :, :, 0, 1] = JT
    W[1, :, :, 1, 0] = W[1, :, :, 0, 1] = J
    W = np.transpose(W)
    V = np.zeros((2, 2, 2, 2))
    V[0, :, :, 0] = I
    V[0, :, :, 1] = JT
    V[1, :, :, 1] = J
    V = np.transpose(V)

    def f_reshape(a, shape):
        return np.reshape(a, shape, order="F")

    crs = []
    xcrs = [np.asarray(bk.to_numpy(c)) for c in x.cores]
    xr = x.r
    dp = 0
    for j in range(D):
        currd = int(d[j])
        cr = np.tensordot(V, xcrs[dp], (0, 1))
        cr = cr.transpose(3, 0, 1, 2, 4)
        cr = f_reshape(cr, (xr[dp], 2, 2, 2 * xr[dp + 1]))
        dp += 1
        crs.append(cr)
        for _ in range(1, currd - 1):
            cr = np.tensordot(W, xcrs[dp], (1, 1)).transpose([0, 4, 1, 2, 3, 5])
            cr = f_reshape(cr, (2 * xr[dp], 2, 2, 2 * xr[dp + 1]))
            dp += 1
            crs.append(cr)
        if kind[j] == "F":
            cr = np.tensordot(W, xcrs[dp], (1, 1)).transpose([0, 4, 1, 2, 3, 5])
            cr = f_reshape(cr, (2 * xr[dp], 2, 2, 2 * xr[dp + 1]))
            dp += 1
            tmp = np.tensordot(S, xcrs[dp], (1, 1))
            tmp = f_reshape(tmp, (2 * xr[dp], xr[dp + 1]))
            cr = np.tensordot(cr, tmp, (3, 0))
            dp += 1
            crs.append(cr)
        else:
            dotcore = {"C": P, "L": Q, "U": R}[kind[j]]
            cr = np.tensordot(dotcore, xcrs[dp], (1, 1))
            cr = cr.transpose([0, 3, 1, 2, 4])
            cr = f_reshape(cr, (2 * xr[dp], 2, 2, xr[dp + 1]))
            dp += 1
            crs.append(cr)
    # the block tensors above are written with the column index first
    return _mat([rearrange(c, "a n m b -> a m n b") for c in crs])


def qshift(d):
    """QTT down-shift matrix of size ``2^d`` (ones on the first subdiagonal)."""
    return shift(d, -1)


# --- the QTT kit for elliptic problems ---------------------------------------
#
# Three thin constructions that everything in docs/plans/qtt-elliptic-bpx.md is
# assembled from.  They are here rather than in the solver module because they
# are operators, not algorithms, and because a second copy of "what is the
# difference operator" is how the two would drift apart.

def qdiff(d, kind="backward"):
    """QTT difference operator on ``2^d`` nodes, unscaled (no ``h^-1``).

    ``kind='backward'`` gives ``I - S`` with ``S`` the down-shift of
    :func:`qshift`, i.e. ``(Mv)_i = v_i - v_{i-1}``; ``'forward'`` gives its
    transpose.  TT rank 2.

    This is the factor ``M`` of ``A = M^T diag(a) M``: every elliptic operator
    in the QTT kit is built from it, so it has exactly one owner.
    """
    d = int(d)
    m = (eye(2, d) - qshift(d)).round(1e-14)
    if kind == "backward":
        return m
    if kind == "forward":
        return m.T
    raise ValueError(f"kind must be 'backward' or 'forward', got {kind!r}")


def qtri_ones(d, upper=False):
    """QTT lower- (or upper-) triangular all-ones matrix of size ``2^d``.

    This is exactly the inverse of :func:`qdiff`: ``T (I - S) = I``, which is
    why a 1D problem can be solved in closed form rather than iterated (see
    ``tt.algs.qtt_ell.solve_direct_1d``).  TT rank 2.
    """
    d = int(d)
    t = Toeplitz(ones(2, d), d, kind="L")
    return t.T if upper else t


def qlaplace_dn(d, bc="DN", order="dim"):
    """QTT Laplacian on ``2^d`` nodes per dimension with mixed boundaries.

    ``d`` is an int or a list of per-dimension level counts, as in
    :func:`qlaplace_dd`; for ``D > 1`` the result is the Kronecker sum of the 1D
    operators.  ``bc`` is a two-letter string (one letter per end, ``'D'`` or
    ``'N'``) applied to every dimension, or a sequence of ``D`` such strings.
    Unscaled: no ``h^-2`` factor.

    ``order`` decides the index layout for ``D > 1`` and is *not* cosmetic:

    * ``'dim'`` (default) is dimension-major, the layout of :func:`kron` and of
      :func:`qlaplace_dd`, so this function drops into existing code;
    * ``'level'`` is level-major -- the bits of one level of all dimensions
      adjacent -- which is the layout ([BK20] eq. (49)) in which the multilevel
      preconditioner has TT rank ``2^(2D+1)`` independent of the level count.

    The two are the same operator under a permutation of the flat index, but an
    operator and a preconditioner in *different* layouts do not compose, and the
    resulting nonsense is silent. Hence an argument rather than a default: see
    ``docs/plans/qtt-elliptic-bpx.md``.

    ``'DN'`` -- Dirichlet at 0, Neumann at 1 -- is the reason this function
    exists.  It is the only combination with exactly ``2^l`` degrees of freedom
    on every level, so it is the one the multilevel prolongations of [BK20] are
    built for; ``qlaplace_dd`` cannot be used there.  TT ranks are 4 (``D = 1``)
    and 5 (``D > 1``); the operator is checked against its analytic spectrum in
    ``docs/NUMERICS.md``.

    ``'NN'`` is refused: it is singular (constants are in its kernel), it plays
    no part in the [BK20] construction, and returning a singular operator from a
    function whose other outputs are SPD would be a trap.
    """
    dims = [int(d)] if isinstance(d, (int, np.integer)) else [int(v) for v in d]
    ndim = len(dims)
    if isinstance(bc, str):
        bcs = [bc] * ndim
    else:
        bcs = [str(v) for v in bc]
        if len(bcs) != ndim:
            raise ValueError(
                f"bc has {len(bcs)} entries for {ndim} dimensions")

    blocks = []
    for dk, bck in zip(dims, bcs):
        bck = bck.upper()
        if bck == "DD":
            blocks.append(qlaplace_dd([dk]))
            continue
        if bck == "NN":
            raise ValueError(
                "bc='NN' is singular (the constant vector is in its kernel) "
                "and is not part of the multilevel construction; build it "
                "explicitly if you really want it")
        m = qdiff(dk)
        if bck == "DN":
            blocks.append((m.T @ m).round(1e-14))
        elif bck == "ND":
            blocks.append((m @ m.T).round(1e-14))
        else:
            raise ValueError(
                f"bc must be one of 'DD', 'DN', 'ND' per dimension, got {bck!r}")

    if order not in ("dim", "level"):
        raise ValueError(f"order must be 'dim' or 'level', got {order!r}")
    if ndim == 1:
        return blocks[0]

    total = None
    for k in range(ndim):
        term = None
        for j in range(ndim):
            factor = blocks[j] if j == k else eye(2, dims[j])
            term = factor if term is None else kron(term, factor)
        total = term if total is None else total + term
    total = total.round(1e-14)
    if order == "dim":
        return total
    if len(set(dims)) != 1:
        raise ValueError(
            "order='level' needs the same number of levels in every dimension, "
            f"got {dims}")
    return permute(total, level_major_order(dims), 1e-14)


def level_major_order(dims):
    """The permutation taking a dimension-major mode list to level-major.

    ``dims`` is the per-dimension level count. Mode ``k`` of dimension ``j``
    sits at slot ``j * L + k`` in dimension-major order and at ``k * D + j`` in
    level-major, so this returns the index list that :func:`permute` wants.
    Exposed because a preconditioner and its operator must agree on the layout,
    and the only way to check that is to be able to name it.
    """
    dims = [int(v) for v in dims]
    if len(set(dims)) != 1:
        raise ValueError(f"all dimensions must have the same depth, got {dims}")
    ndim, depth = len(dims), dims[0]
    return [j * depth + k for k in range(depth) for j in range(ndim)]


# --- reshape / permute -------------------------------------------------------

def _reverse(cores):
    """Reverse the mode order (mode 1 becomes mode d).

    In the TT flat index ``f = i_1 + n_1 i_2 + ...`` mode 1 is the *fastest*,
    while a C-order reshape treats the first axis as the slowest.  Reversing the
    core list swaps the two conventions, so the streaming reshape below can work
    in plain C order and be reversed back.
    """
    return [rearrange(c, "a n b -> b n a") for c in cores[::-1]]


def _reshape_cores(cores, new_modes, eps, rl=1, rr=1):
    """Stream old cores into new ones, splitting/merging modes as needed.

    Modes are consumed left to right; a new mode is emitted as soon as the
    accumulated block size is divisible by it.  Splits go through a truncated
    SVD, merges are exact.
    """
    new_modes = [int(v) for v in new_modes]
    cores = _ops.orthogonalize(cores, center=0)
    frob = bk.norm(cores[0])
    nsplit = max(len(new_modes) - 1, 1)
    delta = eps * frob / math.sqrt(nsplit) if frob > 0 else 0.0

    n_old = _ops.modes(cores)
    r_old = _ops.ranks(cores)
    total_old = int(np.prod([int(v) for v in n_old])) * r_old[0] * r_old[-1]
    total_new = int(np.prod(new_modes)) * int(rl) * int(rr)
    if total_old != total_new:
        raise ValueError(
            f"reshape changes the number of elements: {total_old} -> {total_new}")

    out = []
    cur = cores[0].reshape((r_old[0] * n_old[0], r_old[1]))
    left = 1                      # rank on the left of the emitted part
    size = r_old[0] * n_old[0]    # elements waiting to be emitted (times `left`)
    consumed = 1
    # `rl` is absorbed into the first emitted mode, `rr` into the last one
    todo = list(new_modes)
    todo[0] *= int(rl)
    todo[-1] *= int(rr)

    for j, m in enumerate(todo):
        while size % m != 0 or size < m:
            if consumed >= len(cores):
                raise ValueError(
                    f"cannot reshape modes {list(n_old)} into {new_modes}: "
                    f"mode groups do not align at new mode {j} (size {m})")
            nxt = cores[consumed]
            cur = (cur @ nxt.reshape((nxt.shape[0], -1))).reshape(
                (left * size * nxt.shape[1], nxt.shape[2]))
            size *= nxt.shape[1]
            consumed += 1
        rest = size // m
        if j == len(todo) - 1:
            while consumed < len(cores):  # absorb whatever is left
                nxt = cores[consumed]
                cur = (cur @ nxt.reshape((nxt.shape[0], -1))).reshape(
                    (left * size * nxt.shape[1], nxt.shape[2]))
                size *= nxt.shape[1]
                consumed += 1
            if size != m:
                raise ValueError(
                    f"cannot reshape modes {list(n_old)} into {new_modes}: "
                    f"{size} elements left for a final mode of size {m}")
            out.append(cur.reshape((left, m, -1)))
            break
        mat = cur.reshape((left * m, -1))
        u, s, vh = bk.svd(mat)
        rnew = max(1, min(_ops.chop(s, delta), u.shape[1]))
        out.append(u[:, :rnew].reshape((left, m, rnew)))
        cur = (s[:rnew].reshape((rnew, 1)) * vh[:rnew, :]).reshape(
            (rnew * rest, -1))
        left, size = rnew, rest
    return out


def _reshape_matrix_cores(cores, pairs, eps):
    """Reshape TT-matrix cores, cutting rows and columns in lockstep.

    Works in "reversed" mode order (first core = slowest index), so plain
    C-order reshapes apply; :func:`reshape` reverses on the way in and out.
    A block is carried as ``(left, R, C, right)`` with the row and the column
    index kept apart — that is the whole point: a matrix reshape may not mix
    row bits with column bits.
    """
    frob = bk.norm(_ops.orthogonalize([rearrange(c, "a n m b -> a (n m) b")
                                       for c in cores], center=0)[0])
    delta = eps * frob / math.sqrt(max(len(pairs) - 1, 1)) if frob > 0 else 0.0

    out = []
    cur = cores[0]
    consumed, left = 1, cores[0].shape[0]

    def absorb(cur):
        nxt = cores[consumed]
        cur = einsum(cur, nxt, "l R C r, r n m q -> l R n C m q")
        return rearrange(cur, "l R n C m q -> l (R n) (C m) q")

    for j, (n_new, m_new) in enumerate(pairs):
        while cur.shape[1] % n_new or cur.shape[2] % m_new:
            if consumed >= len(cores):
                raise ValueError(
                    f"cannot reshape matrix modes into {pairs}: row/column "
                    f"groups do not align at new core {j}")
            cur = absorb(cur)
            consumed += 1
        if j == len(pairs) - 1:
            while consumed < len(cores):
                cur = absorb(cur)
                consumed += 1
            if cur.shape[1] != n_new or cur.shape[2] != m_new:
                raise ValueError(
                    f"cannot reshape matrix modes into {pairs}: "
                    f"{cur.shape[1]}x{cur.shape[2]} left for a {n_new}x{m_new} core")
            out.append(cur)
            break
        rest_n, rest_m = cur.shape[1] // n_new, cur.shape[2] // m_new
        block = rearrange(cur, "l (n p) (m q) r -> (l n m) (p q r)",
                          n=n_new, m=m_new, p=rest_n, q=rest_m)
        u, s, vh = bk.svd(block)
        rnew = max(1, min(_ops.chop(s, delta), u.shape[1]))
        out.append(rearrange(u[:, :rnew], "(l n m) k -> l n m k",
                             l=left, n=n_new, m=m_new))
        cur = rearrange(s[:rnew].reshape((rnew, 1)) * vh[:rnew, :],
                        "k (p q r) -> k p q r", p=rest_n, q=rest_m)
        left = rnew
    return out


def reshape(tt_array, shape, eps=1e-14, rl=1, rr=1):
    """Reshape a TT-vector or TT-matrix into new mode sizes.

    For a TT-matrix ``shape`` is a ``d2 x 2`` array of ``(n, m)`` pairs.
    ``rl`` / ``rr`` attach boundary ranks to the first / last mode.
    """
    if isinstance(tt_array, matrix):
        sz = np.asarray(shape, dtype=np.int64)
        if sz.ndim != 2 or sz.shape[1] != 2:
            raise ValueError("for a TT-matrix, shape must be a d2 x 2 array")
        if int(rl) != 1 or int(rr) != 1:
            raise ValueError(
                "reshape of a TT-matrix with rl/rr != 1 is not supported; "
                "reshape the underlying vector instead (a.tt)")
        pairs = [(int(a), int(b)) for a, b in sz]
        rev = [rearrange(c, "a n m b -> b n m a")
               for c in matrix.to_list(tt_array)[::-1]]
        newc = _reshape_matrix_cores(rev, pairs[::-1], eps)
        return matrix.from_list(
            [rearrange(c, "a n m b -> b n m a") for c in newc[::-1]])
    sz = [int(v) for v in np.asarray(shape, dtype=np.int64).ravel()]
    return vector.from_list(_reverse(
        _reshape_cores(_reverse(tt_array.cores), sz[::-1], eps, rr, rl)))


def permute(x, order, eps=1e-14, return_cores=False):
    """Permute the modes of a TT-vector, or of a TT-matrix, by transpositions.

    A TT-matrix is a TT-vector over the merged mode ``s = i + n j``, so the same
    machinery reorders it; only the ``(n, m)`` bookkeeping has to follow. Having
    one owner here matters because the multilevel constructions of
    ``docs/plans/qtt-elliptic-bpx.md`` need exactly this on operators: their
    rank bounds hold in *level-major* order (all dimensions' bits of one level
    adjacent), while ``kron`` and :func:`qlaplace_dd` produce dimension-major.
    """
    if isinstance(x, matrix):
        if return_cores:
            raise ValueError("return_cores is for TT-vectors, not TT-matrices")
        order = [int(v) for v in np.asarray(order, dtype=np.int64).ravel()]
        out = matrix()
        out.n = np.asarray(x.n, dtype=np.int32)[order].copy()
        out.m = np.asarray(x.m, dtype=np.int32)[order].copy()
        out.tt = permute(x.tt, order, eps)
        return out
    order = [int(v) for v in np.asarray(order, dtype=np.int64).ravel()]
    d = x.d
    if sorted(order) != list(range(d)):
        raise ValueError(f"order must be a permutation of 0..{d - 1}")
    cores = _ops.orthogonalize(x.cores, center=0)
    frob = bk.norm(cores[0])
    # one budget share per swap, plus one for the final recompression below
    nswaps = max(d * (d - 1) // 2, 1)
    delta = eps * frob / math.sqrt(nswaps + 1) if frob > 0 else 0.0

    pos = list(order)  # pos[k] = which original mode sits at slot k (target)
    cur = list(cores)
    # bubble sort the current layout into the requested one
    layout = list(range(d))
    for target_slot in range(d):
        want = pos[target_slot]
        have = layout.index(want)
        for k in range(have, target_slot, -1):
            a, b = cur[k - 1], cur[k]
            r0, n1, _ = a.shape
            _, n2, r2 = b.shape
            merged = rearrange(einsum(a, b, "a n p, p m b -> a m n b"),
                               "a m n b -> (a m) (n b)")
            u, s, vh = bk.svd(merged)
            rnew = max(1, min(_ops.chop(s, delta), u.shape[1]))
            cur[k - 1] = u[:, :rnew].reshape((r0, n2, rnew))
            cur[k] = (s[:rnew].reshape((rnew, 1)) * vh[:rnew, :]).reshape(
                (rnew, n1, r2))
            layout[k - 1], layout[k] = layout[k], layout[k - 1]

    # Recompress once at the end.  Each swap truncates what is negligible
    # *locally*, on two cores, and the intermediate orderings genuinely need
    # higher ranks than the final one; nothing in the sweep ever removes that
    # slack again.  Measured on a 3-peak separable function interleaved into
    # Morton order, d = 15: the swaps alone leave rank 1024 where the tensor's
    # own rank is 102, and a single rounding pass afterwards brings it to 107.
    # A 10x rank for the same tensor is not a detail -- it is quadratic in
    # every downstream contraction.
    # round_cores spends its eps as delta = eps * norm / sqrt(d-1); ask it for
    # the same absolute delta the swaps used, so the total stays within eps
    eps_final = eps * math.sqrt((d - 1) / (nswaps + 1)) if d > 1 else eps
    cur = _ops.round_cores(cur, eps_final)
    if return_cores:
        return cur
    return vector.from_list(cur)
