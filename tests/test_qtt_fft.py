"""QTT-FFT (tt.algs.qtt_fft.qtt_fft1) against the dense numpy.fft oracle.

Normalization convention (verified here, inherited verbatim from the 1.x
code, which spends one ``1/sqrt(2)`` per butterfly stage):

    forward:  qtt_fft1(x, tol)               == fft(full(x))  / sqrt(N)
    inverse:  qtt_fft1(x, tol, inverse=True) == ifft(full(x)) * sqrt(N)

i.e. both directions are unitary and forward-then-inverse is the identity.
``full(asvector=True)`` flattens Fortran-style, so core 0 carries the least
significant bit of the index; with ``bitReverse=False`` the output stays in
bit-reversed order instead.
"""

import numpy as np
import pytest

import tt
from tt.algs.qtt_fft import qtt_fft1


def bitrev_permutation(d):
    """perm[i] = the index whose d-bit binary representation is reversed."""
    idx = np.arange(2 ** d)
    perm = np.zeros_like(idx)
    for _ in range(d):
        perm = (perm << 1) | (idx & 1)
        idx >>= 1
    return perm


def full_vec(x):
    return np.asarray(x.full(asvector=True))


def test_forward_matches_numpy_fft():
    d = 10
    N = 2 ** d
    rng = np.random.default_rng(42)
    x = tt.rand(2, d, 4, samplefunc=rng.standard_normal)
    f = qtt_fft1(x, 1e-12)
    ref = np.fft.fft(full_vec(x)) / np.sqrt(N)
    err = np.linalg.norm(full_vec(f) - ref) / np.linalg.norm(ref)
    assert err < 1e-10


def test_inverse_and_roundtrip():
    d = 10
    N = 2 ** d
    tol = 1e-12
    rng = np.random.default_rng(7)
    x = tt.rand(2, d, 4, samplefunc=rng.standard_normal)
    xv = full_vec(x)
    # inverse alone against the oracle (note the * sqrt(N) convention)
    inv = qtt_fft1(x, tol, inverse=True)
    ref = np.fft.ifft(xv) * np.sqrt(N)
    assert np.linalg.norm(full_vec(inv) - ref) / np.linalg.norm(ref) < 1e-10
    # roundtrip x -> fft -> ifft -> x
    back = qtt_fft1(qtt_fft1(x, tol), tol, inverse=True)
    err = np.linalg.norm(full_vec(back) - xv) / np.linalg.norm(xv)
    assert err < 100 * tol


def test_bitreverse_false_gives_bit_reversed_order():
    d = 10
    N = 2 ** d
    rng = np.random.default_rng(3)
    x = tt.rand(2, d, 4, samplefunc=rng.standard_normal)
    ref = np.fft.fft(full_vec(x)) / np.sqrt(N)
    perm = bitrev_permutation(d)
    raw = full_vec(qtt_fft1(x, 1e-12, bitReverse=False))
    # raw[i] = ref[bitrev(i)]; in particular it is NOT in natural order
    assert np.linalg.norm(raw - ref[perm]) / np.linalg.norm(ref) < 1e-10
    assert np.linalg.norm(raw - ref) / np.linalg.norm(ref) > 1e-2


def test_gaussian_keeps_low_ranks():
    d = 12
    N = 2 ** d
    t = np.linspace(-6, 6, N, endpoint=False)
    g = np.exp(-t ** 2 / 2)
    x = tt.vector(g.reshape([2] * d, order="F"), eps=1e-12)
    f = qtt_fft1(x, 1e-10)
    ref = np.fft.fft(g) / np.sqrt(N)
    assert np.linalg.norm(full_vec(f) - ref) / np.linalg.norm(ref) < 1e-8
    assert max(f.r) < 15


def test_rejects_non_qtt_modes():
    x = tt.rand([2, 3, 2], 3, 2)
    with pytest.raises(ValueError):
        qtt_fft1(x, 1e-10)
