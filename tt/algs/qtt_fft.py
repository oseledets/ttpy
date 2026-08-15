"""Superfast Fourier transform of a QTT vector by radix-2 butterflies on cores.

A length-``2**d`` vector stored as a QTT train with modes ``[2]*d`` is
transformed without ever leaving the train: one stage of the classical
radix-2 Cooley--Tukey recursion is applied per TT core.  Stage ``i``
(walking from the last core to the first) puts the +/- butterfly into core
``i``, distributes the stage twiddle factors ``w**(2**-(i-j))`` over cores
``j < i`` (each such core doubles its ranks), re-orthogonalizes the head of
the train by QR and immediately compresses it back by a backward SVD sweep
with per-step threshold ``tol / sqrt(i)``.  The final 2 x 2 Fourier matrix
acts on the first core, and the bit reversal that radix-2 FFTs owe their
output is performed *exactly* by reversing the train (transposing every
core), which costs nothing in QTT.  The result is the algorithm of

    S. Dolgov, B. Khoromskij, D. Savostyanov, "Superfast Fourier transform
    using QTT approximation", J. Fourier Anal. Appl. 18(5), 2012,

with cost ``O(d^2 r^3)`` instead of ``O(d 2**d)`` -- superfast whenever the
intermediate ranks ``r`` stay bounded.

The ``d^2`` is not a typo and not an artifact of this port: stage ``i`` does
not touch core ``i`` alone, it re-orthogonalizes and re-compresses the whole
head of the train, so it costs ``i`` QR factorizations plus ``i`` SVDs, and
the ``d - 1`` stages sum to ``d(d-1)/2`` of each (45 + 45 for ``d = 10``).
Each factorization acts on a core-sized ``2r x r`` matrix, which is where the
``r^3`` comes from; the mode size is fixed at 2 and enters only as a
constant.  Callers sizing this against a dense FFT should read the exponent
of ``d`` as 2, not 1 -- the win over ``O(d 2**d)`` is the ``2**d``, not the
``d``.

That "whenever" is the honest caveat: the stage twiddles double the ranks
before each compression, and it is the tolerance that prunes them back.
A ``tol`` too tight for the actual spectrum (or a signal whose Fourier
image simply is not low-rank -- e.g. white noise) makes the intermediate
ranks grow roughly by a factor of two per stage and the cost degenerates
towards the dense FFT; the transform stays correct, only no longer
superfast.  Conversely, a loose ``tol`` keeps ranks small but the error
accumulates over the ``d - 1`` sweeps.  The per-sweep threshold
``tol / sqrt(i)`` is the original code's budget split, kept verbatim.

Normalization is unitary per stage: every butterfly carries ``1/sqrt(2)``,
so the forward transform equals ``numpy.fft.fft(x) / sqrt(N)`` and the
inverse equals ``numpy.fft.ifft(x) * sqrt(N)`` -- forward followed by
inverse is the identity, and both directions preserve the 2-norm.

The ``inverse`` and ``bitReverse`` options are the contribution of Dishi
Liu (ttpy pull-request commits, 2018): ``inverse`` flips the sign of the
twiddle exponent, turning the transform into the (equally normalized)
inverse DFT; ``bitReverse=False`` skips the final train reversal and
returns the coefficients in bit-reversed order, which is exactly what a
multi-dimensional QTT-FFT needs when this routine runs on each sub-train
of a longer TT.  This port keeps the numerical scheme of the 1.x code
verbatim (including the ``-1 +/- 1.22e-16j`` twiddle base raised to
fractional powers); only the container plumbing was rewritten for the
ttpy2 ``tt.vector``.
"""

from __future__ import annotations

import numpy as np

from ..core import _ops
from ..core.vector import vector

__all__ = ["qtt_fft1"]


def qtt_fft1(x, tol, inverse=False, bitReverse=True):
    """1D (inverse) discrete Fourier transform of a QTT vector.

    Args:
        x: :class:`tt.vector` with all modes equal to 2 (a QTT vector of
            length ``2**d``).
        tol: truncation tolerance; every stage rounds the train back with a
            per-step threshold ``tol / sqrt(i)``.  Too small a value for the
            given signal lets the intermediate ranks double per stage (see
            the module docstring).
        inverse: if ``True``, compute the inverse DFT (positive twiddle
            exponent) instead of the forward one.  Option by Dishi Liu.
        bitReverse: if ``True`` (default), reverse the train at the end so
            the coefficients come out in natural order.  Set to ``False``
            when using this routine as a subroutine of a multi-dimensional
            QTT-FFT: the result is then in bit-reversed order (core ``k`` of
            the output enumerates bit ``d-1-k`` of the frequency index).
            Option by Dishi Liu.

    Returns:
        :class:`tt.vector` with the transform coefficients, scaled
        unitarily: ``fft(full(x)) / sqrt(N)`` for the forward transform,
        ``ifft(full(x)) * sqrt(N)`` for the inverse.
    """
    d = x.d
    n = x.n
    if not np.all(n == 2):
        raise ValueError(f"qtt_fft1 needs all modes equal to 2, got {list(n)}")
    r = x.r.copy()
    y = [np.array(c, dtype=complex, order="F") for c in vector.to_list(x)]

    if inverse:
        twiddle = -1 + 1.22e-16j  # exp(pi*1j)
    else:
        twiddle = -1 - 1.22e-16j  # exp(-pi*1j)

    for i in range(d - 1, 0, -1):
        r1 = y[i].shape[0]  # head rank
        r2 = y[i].shape[2]  # tail rank
        crd2 = np.zeros((r1, 2, r2), order="F", dtype=complex)
        # last block +-
        crd2[:, 0, :] = (y[i][:, 0, :] + y[i][:, 1, :]) / np.sqrt(2)
        crd2[:, 1, :] = (y[i][:, 0, :] - y[i][:, 1, :]) / np.sqrt(2)
        # last block twiddles
        y[i] = np.zeros((r1 * 2, 2, r2), order="F", dtype=complex)
        y[i][0:r1, 0, 0:r2] = crd2[:, 0, :]
        y[i][r1:r1 * 2, 1, 0:r2] = crd2[:, 1, :]
        # 1..i-1 block twiddles and qr
        rv = 1
        for j in range(0, i):
            cr = y[j]
            r1 = cr.shape[0]  # head rank
            r2 = cr.shape[2]  # tail rank
            if j == 0:
                r[j] = r1
                r[j + 1] = r2 * 2
                y[j] = np.zeros((r[j], 2, r[j + 1]), order="F", dtype=complex)
                y[j][0:r1, :, 0:r2] = cr
                y[j][0:r1, 0, r2:r[j + 1]] = cr[:, 0, :]
                y[j][0:r1, 1, r2:r[j + 1]] = \
                    twiddle ** (1.0 / (2 ** (i - j))) * cr[:, 1, :]
            else:
                r[j] = r1 * 2
                r[j + 1] = r2 * 2
                y[j] = np.zeros((r[j], 2, r[j + 1]), order="F", dtype=complex)
                y[j][0:r1, :, 0:r2] = cr
                y[j][r1:r[j], 0, r2:r[j + 1]] = cr[:, 0, :]
                y[j][r1:r[j], 1, r2:r[j + 1]] = \
                    twiddle ** (1.0 / (2 ** (i - j))) * cr[:, 1, :]

            y[j] = np.reshape(y[j], (r[j], 2 * r[j + 1]), order="F")
            y[j] = np.dot(rv, y[j])
            r[j] = y[j].shape[0]
            y[j] = np.reshape(y[j], (2 * r[j], r[j + 1]), order="F")

            y[j], rv = np.linalg.qr(y[j])
            y[j] = np.reshape(y[j], (r[j], 2, rv.shape[0]), order="F")

        y[i] = np.reshape(y[i], (r[i], 2 * r[i + 1]), order="F")
        y[i] = np.dot(rv, y[i])
        r[i] = rv.shape[0]
        # backward svd
        for j in range(i, 0, -1):
            u, s, v = np.linalg.svd(y[j], full_matrices=False)
            rnew = _ops.chop(s, np.linalg.norm(s) * tol / np.sqrt(i))
            u = np.dot(u[:, 0:rnew], np.diag(s[0:rnew]))
            v = v[0:rnew, :]
            y[j] = np.reshape(v, (rnew, 2, r[j + 1]), order="F")
            y[j - 1] = np.reshape(y[j - 1], (r[j - 1] * 2, r[j]), order="F")
            y[j - 1] = np.dot(y[j - 1], u)
            r[j] = rnew
            y[j - 1] = np.reshape(y[j - 1], (r[j - 1], r[j] * 2), order="F")

        y[0] = np.reshape(y[0], (r[0], 2, r[1]), order="F")

    # FFT on the first block
    y[0] = np.transpose(y[0], (1, 0, 2))
    y[0] = np.reshape(y[0], (2, r[0] * r[1]), order="F")
    y[0] = np.dot(np.array([[1, 1], [1, -1]]), y[0]) / np.sqrt(2)
    y[0] = np.reshape(y[0], (2, r[0], r[1]), order="F")
    y[0] = np.transpose(y[0], (1, 0, 2))

    if bitReverse:
        # Reverse the train
        y2 = [None] * d
        for i in range(d):
            y2[d - i - 1] = np.transpose(y[i], (2, 1, 0))
        return vector.from_list(y2)
    # for multi-dimensional qtt_fft
    return vector.from_list(y)
