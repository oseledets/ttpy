"""The torch backend must give the same numbers as numpy, not merely run.

Skipped when torch is missing: the package must import and work without it.
"""

import numpy as np
import pytest

import tt
from tt import backend as bk
from tt.core import _ops

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="no CUDA device")


def gpu(dtype="float64"):
    return bk.TorchBackend("cuda", dtype)


def as_np(a):
    return np.asarray(bk.to_numpy(a))


def rel(a, b):
    a, b = as_np(a), as_np(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def to_gpu(x, dtype="float64"):
    return x.to("torch", "cuda", dtype)


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((3, 4, 5, 3))
    return dense, tt.vector(dense, 1e-14)


def test_import_without_touching_the_default_backend():
    assert bk.get_backend().name == "numpy"


def test_full_matches_numpy(sample):
    dense, x = sample
    assert rel(to_gpu(x).full(), dense) < 1e-12


def test_round_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 4, 4, 4))
    x = tt.vector(a, 1e-14)
    for eps in (1e-1, 1e-3, 1e-10):
        cpu = x.round(eps)
        cuda = to_gpu(x).round(eps)
        assert list(cpu.r) == list(cuda.r), f"ranks differ at eps={eps}"
        assert rel(cuda.full(), cpu.full()) < 1e-12


def test_arithmetic_matches_numpy():
    x, y = tt.rand([3, 4, 3], r=3), tt.rand([3, 4, 3], r=3)
    gx, gy = to_gpu(x), to_gpu(y)
    assert rel((gx + gy).full(), (x + y).full()) < 1e-12
    assert rel((gx * gy).full(), (x * y).full()) < 1e-12
    assert rel((2.5 * gx).full(), (2.5 * x).full()) < 1e-12
    assert abs(tt.dot(gx, gy) - tt.dot(x, y)) < 1e-10 * abs(tt.dot(x, y))
    assert abs(gx.norm() - x.norm()) < 1e-10 * x.norm()


def test_matvec_matches_numpy():
    A = tt.qlaplace_dd([6])
    x = tt.rand(2, 6, r=3)
    ref = tt.matvec(A, x).round(1e-10)
    got = tt.matvec(A.to("torch", "cuda"), to_gpu(x)).round(1e-10)
    assert rel(got.full(), ref.full()) < 1e-11


def test_tt_svd_on_gpu():
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((4, 4, 4, 4))
    g = gpu().asarray(dense)
    cores = _ops.tt_svd(g, 1e-10)
    assert rel(_ops.full(cores), dense) < 1e-9


def test_float32_is_accurate_to_float32():
    rng = np.random.default_rng(3)
    dense = rng.standard_normal((4, 4, 4))
    x = tt.vector(dense, 1e-14)
    g = to_gpu(x, "float32").round(1e-5)
    assert rel(g.full(), dense) < 1e-5


def test_complex_on_gpu():
    rng = np.random.default_rng(4)
    dense = rng.standard_normal((3, 3, 3)) + 1j * rng.standard_normal((3, 3, 3))
    x = tt.vector(dense, 1e-14)
    g = to_gpu(x, "complex128")
    assert rel(g.full(), dense) < 1e-12
    assert abs(tt.dot(g, g) - np.sum(np.conj(dense) * dense)) < 1e-10


def test_mixing_backends_fails_loudly():
    x = tt.rand([2, 3], r=2)
    with pytest.raises(TypeError):
        _ops.add(x.cores, to_gpu(x).cores)


def test_set_backend_roundtrip():
    try:
        bk.set_backend("torch", "cuda", "float64")
        y = tt.rand([2, 3, 2], r=2)
        assert y.backend.name == "torch"
        assert rel(y.round(1e-10).full(), y.full()) < 1e-12
    finally:
        bk.set_backend("numpy")
    assert tt.rand([2, 2], r=2).backend.name == "numpy"


def test_copy_works_on_torch_tensors():
    """torch tensors have .clone(), not .copy(); bk.copy() hides the difference.

    Found by the amen_solve agent, which needs to keep a best-so-far iterate.
    """
    x = to_gpu(tt.rand([2, 3, 2], r=2))
    y = x.copy()
    assert y.backend.name == "torch"
    assert rel(y.full(), x.full()) < 1e-14
    y.cores[0] += 1.0                       # a copy must not alias the original
    assert rel(y.full(), x.full()) > 1e-8

    z = tt.vector(x)                        # copy constructor
    assert z.backend.name == "torch"
    assert rel(z.full(), x.full()) < 1e-14

    m = tt.qlaplace_dd([4]).to("torch", "cuda")
    assert rel(m.copy().full(), m.full()) < 1e-14

    single = tt.vector.from_list([bk.TorchBackend("cuda").asarray(
        np.arange(6.0).reshape(1, 6, 1))])
    assert rel(single.round(1e-12).full(), single.full()) < 1e-14


def test_seeded_randn_is_reproducible_and_matches_numpy():
    """A seed that reaches the backend must mean something on both backends.

    It used to be dropped on the floor by TorchBackend.randn, which made every
    documented seed= a silent no-op there.
    """
    shape = (3, 4, 2)
    g = bk.TorchBackend("cuda", "float64")
    a = g.randn(shape, rng=np.random.default_rng(7))
    b = g.randn(shape, rng=np.random.default_rng(7))
    assert rel(a, b) == 0.0, "same seed must give the same numbers"

    c = bk.NumpyBackend("float64").randn(shape, rng=np.random.default_rng(7))
    assert rel(a, c) == 0.0, "and the same numbers as the numpy backend"

    d = g.randn(shape, rng=np.random.default_rng(8))
    assert rel(a, d) > 1e-3, "different seeds must differ"

    # unseeded still works and is not constant
    assert rel(g.randn(shape), g.randn(shape)) > 1e-3
    assert bk.dtype_of(g.randn(shape, "complex128")) == "complex128"
