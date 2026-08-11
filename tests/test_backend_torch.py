"""The torch backend must give the same numbers as numpy, not merely run.

Skipped when torch is missing: the package must import and work without it.
The device and the precision it can work in come from conftest -- on CUDA that
is float64, on MPS float32, and the tolerances follow (see PARITY there).
"""

import numpy as np
import pytest

import tt
from tt import backend as bk
from tt.core import _ops

from conftest import (GPU_CDTYPE, GPU_DEVICE, GPU_DTYPE, PARITY, rel,
                      requires_gpu)

torch = pytest.importorskip("torch")
pytestmark = requires_gpu()


def gpu(dtype=None):
    return bk.TorchBackend(GPU_DEVICE, dtype or GPU_DTYPE)


def to_gpu(x, dtype=None):
    return x.to("torch", GPU_DEVICE, dtype or GPU_DTYPE)


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((3, 4, 5, 3))
    return dense, tt.vector(dense, 1e-14)


def test_import_without_touching_the_default_backend():
    assert bk.get_backend().name == "numpy"


def test_full_matches_numpy(sample):
    dense, x = sample
    assert rel(to_gpu(x).full(), dense) < PARITY


def test_round_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 4, 4, 4))
    x = tt.vector(a, 1e-14)
    for eps in (1e-1, 1e-3, 1e-10):
        cpu = x.round(eps)
        dev = to_gpu(x).round(eps)
        assert list(cpu.r) == list(dev.r), f"ranks differ at eps={eps}"
        assert rel(dev.full(), cpu.full()) < PARITY


def test_arithmetic_matches_numpy():
    x, y = tt.rand([3, 4, 3], r=3), tt.rand([3, 4, 3], r=3)
    gx, gy = to_gpu(x), to_gpu(y)
    assert rel((gx + gy).full(), (x + y).full()) < PARITY
    assert rel((gx * gy).full(), (x * y).full()) < PARITY
    assert rel((2.5 * gx).full(), (2.5 * x).full()) < PARITY
    assert abs(tt.dot(gx, gy) - tt.dot(x, y)) < 100 * PARITY * abs(tt.dot(x, y))
    assert abs(gx.norm() - x.norm()) < 100 * PARITY * x.norm()


def test_matvec_matches_numpy():
    A = tt.qlaplace_dd([6])
    x = tt.rand(2, 6, r=3)
    ref = tt.matvec(A, x).round(1e-10)
    got = tt.matvec(to_gpu(A), to_gpu(x)).round(1e-10)
    assert rel(got.full(), ref.full()) < 10 * PARITY


def test_tt_svd_on_gpu():
    rng = np.random.default_rng(2)
    dense = rng.standard_normal((4, 4, 4, 4))
    g = gpu().asarray(dense, GPU_DTYPE)
    cores = _ops.tt_svd(g, 1e-10 if GPU_DTYPE == "float64" else 1e-5)
    assert rel(_ops.full(cores), dense) < 1000 * PARITY


def test_float32_is_accurate_to_float32():
    rng = np.random.default_rng(3)
    dense = rng.standard_normal((4, 4, 4))
    x = tt.vector(dense, 1e-14)
    g = to_gpu(x, "float32").round(1e-5)
    assert rel(g.full(), dense) < 1e-5


@requires_gpu(dtype="complex128")
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
        bk.set_backend("torch", GPU_DEVICE, GPU_DTYPE)
        y = tt.rand([2, 3, 2], r=2)
        assert y.backend.name == "torch"
        assert rel(y.round(1e-10).full(), y.full()) < PARITY
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
    assert rel(y.full(), x.full()) < PARITY
    y.cores[0] += 1.0                       # a copy must not alias the original
    assert rel(y.full(), x.full()) > 1e-8

    z = tt.vector(x)                        # copy constructor
    assert z.backend.name == "torch"
    assert rel(z.full(), x.full()) < PARITY

    m = to_gpu(tt.qlaplace_dd([4]))
    assert rel(m.copy().full(), m.full()) < PARITY

    single = tt.vector.from_list([gpu().asarray(
        np.arange(6.0).reshape(1, 6, 1), GPU_DTYPE)])
    assert rel(single.round(1e-12).full(), single.full()) < PARITY


def test_seeded_randn_is_reproducible_and_matches_numpy():
    """A seed that reaches the backend must mean something on both backends.

    It used to be dropped on the floor by TorchBackend.randn, which made every
    documented seed= a silent no-op there.
    """
    shape = (3, 4, 2)
    g = gpu()
    a = g.randn(shape, rng=np.random.default_rng(7))
    b = g.randn(shape, rng=np.random.default_rng(7))
    assert rel(a, b) == 0.0, "same seed must give the same numbers"

    c = bk.NumpyBackend(GPU_DTYPE).randn(shape, rng=np.random.default_rng(7))
    assert rel(a, c) == 0.0, "and the same numbers as the numpy backend"

    d = g.randn(shape, rng=np.random.default_rng(8))
    assert rel(a, d) > 1e-3, "different seeds must differ"

    # unseeded still works and is not constant
    assert rel(g.randn(shape), g.randn(shape)) > 1e-3
    if GPU_CDTYPE is not None:
        assert bk.dtype_of(g.randn(shape, GPU_CDTYPE)) == GPU_CDTYPE


def test_expm_falls_back_to_the_host_when_the_device_has_no_kernel(monkeypatch):
    """The MPS gap, reproduced anywhere: matrix_exp raising NotImplementedError.

    torch has no ``matrix_exp`` on MPS (pytorch#141287); the backend computes
    on the host and moves back.  The test makes the native call raise once,
    exactly as the MPS dispatcher does, and checks the fallback still returns
    the exponential -- on any machine, not only an Apple one.
    """
    import scipy.linalg as sla
    be = bk.TorchBackend("cpu", "float64")
    native = torch.linalg.matrix_exp
    calls = {"n": 0}

    def flaky(a):
        calls["n"] += 1
        if calls["n"] == 1:
            raise NotImplementedError(
                "The operator 'aten::linalg_matrix_exp' is not currently "
                "implemented for the MPS device.")
        return native(a)

    monkeypatch.setattr(torch.linalg, "matrix_exp", flaky)
    a = torch.tensor(np.random.default_rng(5).standard_normal((6, 6)))
    got = be.expm(a).numpy()
    ref = sla.expm(a.numpy())
    assert calls["n"] == 2                      # raised once, then the host
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-14
