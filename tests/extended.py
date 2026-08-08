"""Double-double arithmetic, for oracles that must be sharper than float64.

Several verification tests measure a residual smaller than the rounding error of
the obvious way to measure it, and an oracle has to be more accurate than the
thing it judges or it is not an oracle.  ``np.longdouble`` is not a portable way
to get that: it is 80-bit x87 on Intel and an alias for float64 on Apple
silicon.  See ``docs/NUMERICS.md``, "The extended-precision oracle".

Double-double (Dekker/Knuth error-free transformations) replaces it.  A number
is a pair of float64s ``(hi, lo)`` with ``hi = fl(hi + lo)``, giving ~106 bits
of significand -- more than x87's 64 -- out of nothing but float64 arithmetic,
so it is bit-identical on every platform numpy runs on.

Costs what it says on the tin: a dot product of length k is a Python loop over
k vector operations, so this is for oracles over a few thousand elements, not
for the library.
"""

import math

import numpy as np

__all__ = ["two_sum", "two_prod", "dd_add", "dd_sub", "dd_mul", "dd_matmul",
           "dd_from", "dd_norm", "dd_to_float", "dd_reshape", "dd_transpose"]

# Dekker's splitting constant for float64: 2^27 + 1.
_SPLIT = float(2 ** 27 + 1)


def two_sum(a, b):
    """``a + b`` as an exact (hi, lo) pair.  Knuth, no ordering assumption."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):
    """Same, valid only when ``|a| >= |b|`` -- three flops instead of six."""
    s = a + b
    return s, b - (s - a)


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """``a * b`` as an exact (hi, lo) pair, without needing an FMA."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_from(a):
    """Lift float64 data to double-double (exactly: the low word is zero)."""
    a = np.asarray(a, dtype=np.float64)
    return a, np.zeros_like(a)


def dd_to_float(x):
    hi, lo = x
    return hi + lo


def dd_add(x, y):
    xh, xl = x
    yh, yl = y
    s, e = two_sum(xh, yh)
    e = e + (xl + yl)
    return _quick_two_sum(s, e)


def dd_sub(x, y):
    return dd_add(x, (-y[0], -y[1]))


def dd_reshape(x, shape):
    return x[0].reshape(shape), x[1].reshape(shape)


def dd_transpose(x, axes):
    return x[0].transpose(axes), x[1].transpose(axes)


def dd_mul(x, y):
    xh, xl = x
    yh, yl = y
    p, e = two_prod(xh, yh)
    e = e + (xh * yl + xl * yh)
    return _quick_two_sum(p, e)


def dd_matmul(a, b):
    """``a @ b`` for double-double matrices, contracting the inner index.

    The accumulation is the whole point, so it runs as an explicit loop over
    the contracted axis: numpy's own matmul would round every partial sum back
    to float64 and throw away exactly what this module exists to keep.
    """
    (ah, al), (bh, bl) = a, b
    m, k = ah.shape
    n = bh.shape[1]
    acc = (np.zeros((m, n)), np.zeros((m, n)))
    for j in range(k):
        term = dd_mul((ah[:, j:j + 1], al[:, j:j + 1]),
                      (bh[j:j + 1, :], bl[j:j + 1, :]))
        acc = dd_add(acc, term)
    return acc


def dd_norm(x):
    """Euclidean norm of a double-double array, as a float.

    The squares are error-free transformations, so the sum of *all* the words
    -- high and low together -- is the exact sum of squares; ``math.fsum``
    then adds them with a single final rounding.  That is one ulp better than
    accumulating in double-double, and much faster than a Python loop over
    the elements.
    """
    sq = dd_mul(x, x)
    words = np.concatenate([np.asarray(sq[0]).reshape(-1),
                            np.asarray(sq[1]).reshape(-1)])
    return float(np.sqrt(math.fsum(words.tolist())))
