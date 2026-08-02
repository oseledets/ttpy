"""The algorithm layer under the torch backend.

The core is checked for numpy/torch parity in test_backend_torch.py. This file
asks a narrower question of the algorithms: do they run at all when the default
backend is a GPU, and do they still produce the right numbers?

Where an algorithm deliberately stays on numpy, that is pinned here as a
statement of fact, so the behaviour is documented and checked instead of being
an accident nobody noticed.
"""

import numpy as np
import pytest

import tt
from tt import backend as bk

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="no CUDA device")


@pytest.fixture
def on_cuda():
    """Run the body with the default backend on the GPU, restore afterwards."""
    bk.set_backend("torch", "cuda", "float64")
    try:
        yield
    finally:
        bk.set_backend("numpy")


def rel(a, b):
    a = np.asarray(bk.to_numpy(a))
    b = np.asarray(bk.to_numpy(b))
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def test_cross_runs_on_cuda(on_cuda):
    """cross samples a numpy-valued black box but must answer on the default backend.

    The sampling and the index bookkeeping happen in numpy — shipping a handful
    of fiber values across PCIe to index them on the GPU would be a loss — but
    the tensor it returns has to live where the user's other tensors live, or
    the next tt.matvec mixes backends.
    """
    from tt.algs.cross import cross

    n = [4] * 6
    ref = tt.rand(n, 6, r=3)
    assert ref.backend.name == "torch"
    dense = np.asarray(bk.to_numpy(ref.full()))

    y = cross(lambda idx: dense[tuple(np.asarray(idx, dtype=int).T)], n, eps=1e-10)
    assert rel(y.full(), dense) < 1e-10
    # and it must compose with the GPU tensor it was sampled from
    assert (y.to("torch", "cuda") - ref).norm() / ref.norm() < 1e-9


def test_amen_mv_runs_on_cuda(on_cuda):
    from tt.algs.amen_mv import amen_mv

    A = tt.qlaplace_dd([8]).to("torch", "cuda")
    x = tt.rand(2, 8, r=4)
    y = amen_mv(A, x, 1e-8, verb=0)
    y = y[0] if isinstance(y, tuple) else y
    assert rel(y.full(), tt.matvec(A, x).round(1e-10).full()) < 1e-10


def test_multifuncrs2_runs_on_cuda(on_cuda):
    from tt.algs.multifuncrs import multifuncrs2

    x = tt.ones(2, 8) * 2.0
    y = multifuncrs2([x], lambda v: np.exp(v[:, 0]), eps=1e-8, verb=0)
    values = np.asarray(bk.to_numpy(y.full())).ravel()
    assert np.allclose(values, np.exp(2.0), rtol=1e-8)


def test_the_default_backend_is_restored():
    """A leaked global backend would poison every later test in the session."""
    assert bk.get_backend().name == "numpy"
