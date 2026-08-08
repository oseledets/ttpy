"""The algorithm layer under the torch backend.

The core is checked for numpy/torch parity in test_backend_torch.py. This file
asks a narrower question of the algorithms: do they run at all when the default
backend is a GPU, and do they still produce the right numbers?

Where an algorithm deliberately stays on numpy, that is pinned here as a
statement of fact, so the behaviour is documented and checked instead of being
an accident nobody noticed.

The device and the precision come from conftest; on MPS that is float32, so
what the solvers are asked for (SOLVE_EPS) and what they are held to
(SOLVE_TOL) follow the device rather than being written as float64 constants.
"""

import numpy as np
import pytest

import tt
from tt import backend as bk

from conftest import (GPU_DEVICE, GPU_DTYPE, SOLVE_EPS, SOLVE_TOL, rel,
                      requires_gpu)

torch = pytest.importorskip("torch")
pytestmark = requires_gpu()


@pytest.fixture
def on_gpu():
    """Run the body with the default backend on the GPU, restore afterwards."""
    bk.set_backend("torch", GPU_DEVICE, GPU_DTYPE)
    try:
        yield
    finally:
        bk.set_backend("numpy")


def test_cross_runs_on_gpu(on_gpu):
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

    y = cross(lambda idx: dense[tuple(np.asarray(idx, dtype=int).T)], n,
              eps=SOLVE_EPS)
    assert rel(y.full(), dense) < SOLVE_TOL
    # and it must compose with the GPU tensor it was sampled from
    assert (y.to("torch", GPU_DEVICE) - ref).norm() / ref.norm() < 10 * SOLVE_TOL


def test_amen_mv_runs_on_gpu(on_gpu):
    from tt.algs.amen_mv import amen_mv

    A = tt.qlaplace_dd([8]).to("torch", GPU_DEVICE)
    x = tt.rand(2, 8, r=4)
    y = amen_mv(A, x, SOLVE_EPS, verb=0)
    y = y[0] if isinstance(y, tuple) else y
    assert rel(y.full(), tt.matvec(A, x).round(SOLVE_EPS).full()) < SOLVE_TOL


def test_multifuncrs2_runs_on_gpu(on_gpu):
    from tt.algs.multifuncrs import multifuncrs2

    x = tt.ones(2, 8) * 2.0
    y = multifuncrs2([x], lambda v: np.exp(v[:, 0]), eps=SOLVE_EPS, verb=0)
    assert bk.dtype_of(y.cores[0]) == GPU_DTYPE, (
        "the answer must keep the precision the inputs were in")
    values = np.asarray(bk.to_numpy(y.full())).ravel()
    assert np.allclose(values, np.exp(2.0), rtol=SOLVE_TOL)


def test_the_default_backend_is_restored():
    """A leaked global backend would poison every later test in the session."""
    assert bk.get_backend().name == "numpy"
