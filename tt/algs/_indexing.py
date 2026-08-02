"""Index sets and sampling: the bookkeeping every cross-type algorithm repeats.

Two unrelated-looking things live here because they are the same thing seen from
two sides, and because three modules (:mod:`tt.algs.optimize`,
:mod:`tt.algs.completion` and anything that has to evaluate a TT tensor at a
list of positions) would otherwise each grow their own copy:

* **evaluation at a list of multi-indices** -- :func:`sample`,
  :func:`left_product`, :func:`right_product`.  Given ``P`` multi-indices
  stacked in a ``(P, d)`` integer array, the partial products
  ``G_1[:, j_1, :] ... G_{k}[:, j_k, :]`` are computed for all ``P`` rows at
  once, so the cost is ``O(P d r^2)`` of BLAS instead of ``P`` python-level
  tensor lookups.  The left and right partial products are exactly the rows of
  the interface matrices that an ALS/completion sweep needs, which is why they
  are exposed separately and not only through :func:`sample`.

* **the index-set algebra of a sweep** -- :func:`extend_left`,
  :func:`extend_right`, :func:`index_block`.  A cross sweep carries a left index
  set (multi-indices for modes ``0 .. k-1``) and a right index set (modes
  ``k+1 .. d-1``); the block it looks at is their product with the full mode
  ``k``.  Getting the *order* of that product wrong is the classic silent bug of
  cross codes, so it is written down once, here, and the ordering convention is
  stated in each docstring.

Ordering convention
-------------------
Everything is Fortran-ordered, like the rest of ttpy2: a block of shape
``(rL, n, rR)`` is flattened as ``a + rL * j + rL * n * b``, i.e. the *left*
index runs fastest.  :func:`index_block` produces its rows in exactly that
order, so ``values.reshape((rL, n, rR), order='F')`` lines a flat vector of
function values up with the block.

The index arrays themselves are numpy integer arrays: they are pure
bookkeeping, are consumed by numpy fancy indexing everywhere, and never touch a
GPU.  The *cores* stay on whatever backend they came from -- the contractions go
through :mod:`einops` and work unchanged on torch.
"""

from __future__ import annotations

import numpy as np
from ..backend import einsum   # BLAS-routed; einops' own skips optimize=True

from .. import backend as bk

__all__ = [
    "left_step", "right_step", "left_product", "right_product", "sample",
    "extend_left", "extend_right", "index_block",
]


def _ones(shape, like):
    """``ones`` on the backend of ``like`` (:mod:`tt.backend` exposes zeros only)."""
    dtype = bk.dtype_of(like) if like is not None else None
    return bk.zeros(shape, dtype=dtype, like=like) + 1.0


def _check_idx(idx, d, n=None):
    """Validate a ``(P, d)`` multi-index array; raise loudly on anything odd."""
    idx = np.asarray(idx)
    if idx.ndim != 2:
        raise ValueError(f"multi-index array must be 2d (P, d), got shape {idx.shape}")
    if idx.shape[1] != d:
        raise ValueError(
            f"multi-index array has {idx.shape[1]} columns, the tensor has {d} modes")
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"multi-indices must be integers, got dtype {idx.dtype}")
    if n is not None and idx.size:
        bad = np.flatnonzero((idx < 0).any(axis=0) | (idx >= np.asarray(n)).any(axis=0))
        if bad.size:
            k = int(bad[0])
            raise IndexError(
                f"multi-index out of range in mode {k}: values in "
                f"[{idx[:, k].min()}, {idx[:, k].max()}], mode size {n[k]}")
    return idx


def left_step(v, core, col):
    """One step of the left partial product.

    Args:
        v: ``(P, r)`` current left interface rows.
        core: ``(r, n, r')`` core of the mode to absorb.
        col: ``(P,)`` integer array, the index of that mode for each row.

    Returns:
        ``(P, r')`` array ``v[p] @ core[:, col[p], :]``.
    """
    return einsum(v, core[:, col, :], "p a, a p b -> p b")


def right_step(core, col, v):
    """One step of the right partial product; mirror of :func:`left_step`.

    Args:
        core: ``(r, n, r')`` core of the mode to absorb.
        col: ``(P,)`` integer array of that mode's index for each column.
        v: ``(r', P)`` current right interface columns.

    Returns:
        ``(r, P)`` array ``core[:, col[p], :] @ v[:, p]``.
    """
    return einsum(core[:, col, :], v, "a p b, b p -> a p")


def left_product(cores, idx):
    """Left partial products ``G_1[:, j_1, :] ... G_k[:, j_k, :]`` for many indices.

    Args:
        cores: The first ``k`` cores of a TT tensor, ``cores[0]`` with left rank 1.
        idx: ``(P, k)`` integer array of multi-indices for those ``k`` modes.

    Returns:
        ``(P, r_k)`` array, row ``p`` being the row vector of the product taken
        at ``idx[p]``.  With ``k == 0`` this is ``ones((P, 1))``.
    """
    idx = _check_idx(idx, len(cores), [c.shape[1] for c in cores])
    p = idx.shape[0]
    if cores and cores[0].shape[0] != 1:
        raise ValueError(
            f"left_product needs a left boundary rank of 1, got {cores[0].shape[0]}")
    v = _ones((p, 1), cores[0] if cores else None)
    for k, c in enumerate(cores):
        v = left_step(v, c, idx[:, k])
    return v


def right_product(cores, idx):
    """Right partial products ``G_k[:, j_k, :] ... G_d[:, j_d, :]`` for many indices.

    Args:
        cores: The last cores of a TT tensor, ``cores[-1]`` with right rank 1.
        idx: ``(P, len(cores))`` integer array of multi-indices for those modes.

    Returns:
        ``(r_k, P)`` array, column ``p`` being the column vector of the product
        taken at ``idx[p]``.  With no cores this is ``ones((1, P))``.
    """
    idx = _check_idx(idx, len(cores), [c.shape[1] for c in cores])
    p = idx.shape[0]
    if cores and cores[-1].shape[2] != 1:
        raise ValueError(
            f"right_product needs a right boundary rank of 1, got {cores[-1].shape[2]}")
    v = _ones((1, p), cores[-1] if cores else None)
    for k in range(len(cores) - 1, -1, -1):
        v = right_step(cores[k], idx[:, k], v)
    return v


def sample(cores, idx):
    """Values of a TT tensor at ``P`` multi-indices, in one batched contraction.

    Args:
        cores: Full core list, boundary ranks 1.
        idx: ``(P, d)`` integer array.

    Returns:
        ``(P,)`` array of entries, ``out[p] = A[idx[p, 0], ..., idx[p, d-1]]``.
    """
    return left_product(list(cores), idx).reshape((-1,))


# --- index-set algebra of a cross sweep --------------------------------------

def extend_left(left, n):
    """Grow a left index set by one mode.

    Args:
        left: ``(rL, k)`` integer array, multi-indices for modes ``0 .. k-1``.
        n: Size of mode ``k``.

    Returns:
        ``(rL * n, k + 1)`` array whose row ``a + rL * j`` is
        ``[left[a], j]`` -- the left index runs fastest, matching the
        F-ordered flattening of an ``(rL, n)`` block.
    """
    left = np.asarray(left, dtype=np.int64)
    rl = left.shape[0]
    new = np.repeat(np.arange(n, dtype=np.int64), rl).reshape(-1, 1)
    return np.hstack((np.tile(left, (n, 1)), new))


def extend_right(right, n):
    """Grow a right index set by one mode (prepended).

    Args:
        right: ``(rR, k)`` integer array, multi-indices for modes ``j+1 .. d-1``.
        n: Size of mode ``j``.

    Returns:
        ``(n * rR, k + 1)`` array whose row ``j + n * b`` is ``[j, right[b]]``
        -- the mode index runs fastest, matching the F-ordered flattening of an
        ``(n, rR)`` block.
    """
    right = np.asarray(right, dtype=np.int64)
    rr = right.shape[0]
    new = np.tile(np.arange(n, dtype=np.int64), rr).reshape(-1, 1)
    return np.hstack((new, np.repeat(right, n, axis=0)))


def index_block(left, n, right):
    """All multi-indices of the block ``left x mode x right``.

    Args:
        left: ``(rL, kL)`` left index set (modes ``0 .. kL-1``).
        n: Size of the free mode.
        right: ``(rR, kR)`` right index set (the last ``kR`` modes).

    Returns:
        ``(rL * n * rR, kL + 1 + kR)`` array whose row ``a + rL * j + rL * n * b``
        is ``[left[a], j, right[b]]``, i.e. the rows are in the F-order of an
        ``(rL, n, rR)`` block.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    rl, rr = left.shape[0], right.shape[0]
    lcols = np.tile(left, (n * rr, 1))
    mcols = np.tile(np.repeat(np.arange(n, dtype=np.int64), rl), rr).reshape(-1, 1)
    rcols = np.repeat(right, rl * n, axis=0)
    return np.hstack((lcols, mcols, rcols))
