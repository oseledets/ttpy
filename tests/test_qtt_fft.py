"""QTT-FFT (tt.algs.qtt_fft.qtt_fft1) against the dense numpy.fft oracle.

Normalization convention (verified here, inherited verbatim from the 1.x
code, which spends one ``1/sqrt(2)`` per butterfly stage):

    forward:  qtt_fft1(x, tol)               == fft(full(x))  / sqrt(N)
    inverse:  qtt_fft1(x, tol, inverse=True) == ifft(full(x)) * sqrt(N)

i.e. both directions are unitary and forward-then-inverse is the identity.
``full(asvector=True)`` flattens Fortran-style, so core 0 carries the least
significant bit of the index; with ``bitReverse=False`` the output stays in
bit-reversed order instead.

Scope note: this file checks *numerics*, not cost.  The module advertises
``O(d^2 r^3)``, which holds only while the intermediate ranks stay bounded --
and the intermediate ranks, the ones after each stage's compression, are
never exposed by ``qtt_fft1`` and are asserted nowhere below.
``test_gaussian_keeps_low_ranks`` bounds the *output* rank only, which is
evidence that a Gaussian has a low-rank Fourier image, not that the transform
ran in the advertised time.
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
    # the advertised entry point is ``tt.qtt_fft1``, resolved lazily through
    # tt/__init__.py's _FUNCTIONS map; every other test here imports the
    # function directly, so this is the only thing executing that tuple.
    assert tt.qtt_fft1 is qtt_fft1


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


def test_inverse_without_bit_reversal():
    """``inverse=True, bitReverse=False`` -- the multi-dimensional QTT-FFT case.

    The docstring names this combination as the reason ``bitReverse`` exists,
    and the two flags are otherwise only ever exercised one at a time.
    """
    d = 10
    N = 2 ** d
    rng = np.random.default_rng(11)
    x = tt.rand(2, d, 4, samplefunc=rng.standard_normal)
    ref = np.fft.ifft(full_vec(x)) * np.sqrt(N)
    perm = bitrev_permutation(d)
    raw = full_vec(qtt_fft1(x, 1e-12, inverse=True, bitReverse=False))
    assert np.linalg.norm(raw - ref[perm]) / np.linalg.norm(ref) < 1e-10
    assert np.linalg.norm(raw - ref) / np.linalg.norm(ref) > 1e-2


def test_single_core_train():
    """``d == 1``: the stage loop never runs, only the final 2x2 block does.

    ``for i in range(d - 1, 0, -1)`` is empty, so the ranks are whatever
    ``x.r`` was -- no sweep ever recomputes them -- and the whole result comes
    out of the trailing butterfly.  Both flags are degenerate here (a one-core
    train reverses to itself, and no twiddle is ever raised to a power), so
    all four combinations must agree with the 2-point DFT.
    """
    x = tt.rand(2, 1, 1)
    ref = np.fft.fft(full_vec(x)) / np.sqrt(2)
    for inverse in (False, True):
        for bit_reverse in (False, True):
            got = full_vec(qtt_fft1(x, 1e-12, inverse=inverse,
                                    bitReverse=bit_reverse))
            assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-12
    # N = 2 is the one length where the DFT matrix is its own inverse, so the
    # inverse convention ``ifft * sqrt(N)`` must land on the same vector.
    assert np.allclose(np.fft.ifft(full_vec(x)) * np.sqrt(2), ref)


def test_gaussian_keeps_low_ranks():
    d = 12
    N = 2 ** d
    t = np.linspace(-6, 6, N, endpoint=False)
    g = np.exp(-t ** 2 / 2)
    x = tt.vector(g.reshape([2] * d, order="F"), eps=1e-12)
    f = qtt_fft1(x, 1e-10)
    ref = np.fft.fft(g) / np.sqrt(N)
    assert np.linalg.norm(full_vec(f) - ref) / np.linalg.norm(ref) < 1e-8
    # output ranks only -- see the scope note at the top of this file: this
    # says the Fourier image is low-rank, it does not certify the O(d^2 r^3).
    assert max(f.r) < 15


def test_rejects_non_qtt_modes():
    x = tt.rand([2, 3, 2], 3, 2)
    with pytest.raises(ValueError):
        qtt_fft1(x, 1e-10)
