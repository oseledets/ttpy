"""Elementwise (Hadamard) products of TT tensors, formed already compressed.

The Hadamard product ``a (.) b`` has cores that are Kronecker products of the
input core slices, so its ranks are ``r_a r_b``.  The obvious ``(a*b).round()``
therefore builds an ``r_a r_b`` tensor and orthogonalizes *that*, at
``O(d n (r_a r_b)^3)``.

Following

    K. Kormann, "A semi-Lagrangian Vlasov solver in tensor train format",
    SIAM J. Sci. Comput. 37(4):B613-B632, 2015 (Algorithm 3),

this module never forms the inflated tensor.  Two observations do the work:

1. **Orthogonalize the inputs, not the product.**  A QR sweep on ``a`` and on
   ``b`` separately costs ``O(d n r^3)`` each.  With left-orthonormal inputs the
   product's left interfaces are bounded (``||X||_2 <= 1`` entrywise by
   Cauchy-Schwarz), so truncating the product from the right is error-controlled
   even though the product cores are not themselves orthonormal.

2. **Sweep right to left, truncating as you go.**  After the first truncation
   the right bond is the *compressed* rank ``t ~ r``, not ``r_a r_b``, so the
   core to factorize is the tall-thin matrix ``(r_a r_b) x (n t)``.  Its thin
   SVD costs ``O(n^2 r^4)`` -- Kormann's complexity -- and it is one LAPACK call,
   no block bookkeeping.  The core is built by contracting the carry against the
   factors one at a time (a ``tensordot`` then batched matmuls, all GEMM), so the
   ``r_a r_b`` by ``r_a r_b`` core is never materialized either.  Routing those
   contractions through BLAS rather than ``einsum`` matters: the shared, unsummed
   mode index keeps ``einsum`` on its C loop, which measured 12x slower.

A second, left-to-right sweep over the (now right-orthonormal) train truncates
at the requested accuracy; on a right-orthonormal train that truncation is
optimal, which is what recovers the true ranks.

Two generalizations beyond the paper:

* any number of factors, ``hadamard(a, b, c, ...)``;
* a *fused* linear combination of products, :func:`hadamard_sum`, which rounds
  ``sum_j c_j (x_j (.) y_j (.) ...)`` in one sweep instead of rounding each
  product and then adding.
"""

import numpy as np

from ..core.vector import vector

__all__ = ["hadamard", "hadamard_sum"]

def _cores_of(x):
    return [np.asarray(c, dtype=float) for c in vector.to_list(x)]


def _left_orthogonalize(cores):
    """Left-orthonormal cores; the norm is carried into the last core."""
    cores = [c.copy() for c in cores]
    for k in range(len(cores) - 1):
        r0, n, r1 = cores[k].shape
        q, r = np.linalg.qr(cores[k].reshape(r0 * n, r1))
        cores[k] = q.reshape(r0, n, -1)
        cores[k + 1] = np.tensordot(r, cores[k + 1], axes=(1, 0))
    return cores


def _trunc_rank(s, delta, rmax):
    """Largest tail with ``||tail||_2 <= delta``; capped by ``rmax``."""
    if s.size == 0:
        return 0
    tail = np.cumsum(s[::-1] ** 2)[::-1]
    keep = int(np.count_nonzero(tail > delta ** 2))
    keep = max(1, keep)
    return min(keep, int(rmax)) if rmax else keep


def _term_core(factors, k, carry, shapes=None):
    """The k-th core of one product term, already contracted with ``carry``.

    ``carry`` is ``(prod_j p1_j, t)`` (or ``None`` at the last core).  Returns
    the matrix ``(prod_j p0_j) x (n t)`` without ever forming the
    ``prod_j p0_j`` by ``prod_j p1_j`` core.

    Every contraction is a GEMM: the first factor is a plain ``tensordot``, and
    each further factor is a *batched* matmul over the shared mode index.  (An
    ``einsum`` expressing the same thing cannot route the shared, unsummed mode
    to BLAS and falls back on the C loop -- measured 12x slower here.)
    """
    m = len(factors)
    cores = [factors[j][k] for j in range(m)]
    p0 = [c.shape[0] for c in cores]
    n = cores[0].shape[1]
    p1 = [c.shape[2] for c in cores]
    if carry is None:
        carry = np.ones((int(np.prod(p1)), 1))
    t = carry.shape[1]

    # factor 0: (p0_0 n, p1_0) @ (p1_0, rest) -- one GEMM.
    cur = np.tensordot(cores[0], carry.reshape(*p1, t), axes=([2], [0]))
    # dims now: [p0_0, n, p1_1, ..., p1_{m-1}, t]
    for j in range(1, m):
        nd = cur.ndim
        # bring the mode and this factor's right bond to the front:
        # [n, p1_j, (p0_0..p0_{j-1}), (p1_{j+1}..), t]
        perm = [j, j + 1] + list(range(j)) + list(range(j + 2, nd))
        cur = np.ascontiguousarray(cur.transpose(perm))
        rest = cur.shape[2:]
        cur = cur.reshape(n, p1[j], -1)
        B = np.ascontiguousarray(cores[j].transpose(1, 0, 2))   # (n, p0_j, p1_j)
        cur = B @ cur                                           # batched GEMM
        cur = cur.reshape(n, p0[j], *rest)
        # back to [p0_0..p0_{j-1}, p0_j, n, (p1_{j+1}..), t]
        head = list(range(2, 2 + j))
        tail = list(range(2 + j, cur.ndim))
        cur = cur.transpose(head + [1, 0] + tail)
    return np.ascontiguousarray(cur).reshape(int(np.prod(p0)), n * t), n, t


def _right_sweep(terms, coefs, eps, rmax):
    """Right-to-left sweep: build the (fused) product, truncating as we go."""
    d = len(terms[0][0])
    # a tighter tolerance here: this truncation is error-controlled but not
    # rank-optimal, and injecting noise at the target accuracy would inflate the
    # numerical rank that the second (optimal) sweep then cannot remove.
    delta = eps / (10.0 * max(1.0, np.sqrt(d - 1)))
    cores = [None] * d
    carry = None                     # (rows of bond k+1, t), stacked over terms
    for k in range(d - 1, -1, -1):
        blocks, splits = [], []
        offset = 0
        for term in terms:
            width = int(np.prod([f[k].shape[2] for f in term]))
            piece = None if carry is None else carry[offset:offset + width]
            offset += width
            M, n, t = _term_core(term, k, piece, None)
            blocks.append(M)
            splits.append(M.shape[0])
        if k == 0:
            # the sum's first core is a horizontal concatenation: left rank 1,
            # so the term contributions add instead of stacking.
            M = sum(c * blk for c, blk in zip(coefs, blocks))
        else:
            M = np.vstack(blocks)
        norm = np.linalg.norm(M)
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        keep = _trunc_rank(s, delta * norm, rmax)
        cores[k] = vt[:keep].reshape(keep, n, t)      # right-orthonormal
        carry = u[:, :keep] * s[:keep]
    cores[0] = np.tensordot(carry, cores[0], axes=(1, 0))
    return cores


def _left_truncate(cores, eps, rmax):
    """Optimal left-to-right truncation of a right-orthonormal train."""
    d = len(cores)
    if d == 1:
        return cores
    norm = np.linalg.norm(cores[0])
    delta = eps * norm / np.sqrt(d - 1)
    carry = None
    for k in range(d - 1):
        c = cores[k] if carry is None else np.tensordot(carry, cores[k],
                                                        axes=(1, 0))
        r0, n, r1 = c.shape
        u, s, vt = np.linalg.svd(c.reshape(r0 * n, r1), full_matrices=False)
        keep = _trunc_rank(s, delta, rmax)
        cores[k] = u[:, :keep].reshape(r0, n, keep)
        carry = s[:keep, None] * vt[:keep]
    cores[d - 1] = np.tensordot(carry, cores[d - 1], axes=(1, 0))
    return cores


def hadamard(*factors, eps=1e-10, rmax=None):
    """Elementwise product of TT vectors, built already compressed.

    Equivalent to ``(a * b * ...).round(eps, rmax)`` but never forms the
    inflated product: ``O(d n^2 r^4)`` instead of ``O(d n r^6)`` for two
    factors of rank ``r``.  Measured on ``n=4, d=8`` against the explicit
    route, at identical ranks and 1e-14 agreement::

        rank 16    0.034 s -> 0.052 s   (explicit still wins)
        rank 32    0.493 s -> 0.202 s   (2.4x)
        rank 64   11.97  s -> 0.480 s   (25x)
        rank 96   79.6   s -> 0.622 s   (128x)

    Args:
        *factors: two or more :class:`tt.vector` with identical mode sizes.
        eps: Requested relative accuracy.
        rmax: Optional cap on the TT ranks of the result.

    Returns:
        A :class:`tt.vector` approximating the elementwise product.
    """
    if len(factors) < 2:
        raise ValueError("hadamard needs at least two factors")
    return hadamard_sum([factors], eps=eps, rmax=rmax)


def hadamard_sum(terms, coefs=None, eps=1e-10, rmax=None):
    """Round ``sum_j c_j (x_j (.) y_j (.) ...)`` in a single fused sweep.

    Rounding each product separately and then adding them costs one sweep per
    term *and* leaves the sum at the added rank until a final rounding; this
    contracts the whole linear combination against one carry, so the ranks are
    truncated once, on the quantity that is actually wanted.

    Args:
        terms: Sequence of terms; each term is a sequence of :class:`tt.vector`
            factors to be multiplied elementwise.
        coefs: Optional scalar weights, one per term (default all ones).
        eps: Requested relative accuracy.
        rmax: Optional cap on the TT ranks of the result.

    Returns:
        A :class:`tt.vector` approximating the weighted sum of the products.
    """
    terms = [list(term) for term in terms]
    if not terms:
        raise ValueError("hadamard_sum needs at least one term")
    coefs = [1.0] * len(terms) if coefs is None else list(coefs)
    if len(coefs) != len(terms):
        raise ValueError("one coefficient per term is required")
    d = terms[0][0].d
    for term in terms:
        for f in term:
            if f.d != d:
                raise ValueError("all factors must have the same dimension")
    ortho = [[_left_orthogonalize(_cores_of(f)) for f in term]
             for term in terms]
    cores = _right_sweep(ortho, coefs, eps, rmax)
    cores = _left_truncate(cores, eps, rmax)
    return vector.from_list(cores)
