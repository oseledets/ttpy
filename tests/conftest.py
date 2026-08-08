"""Which accelerator the GPU tests run on, and what that device can do.

The question a GPU test asks is not "is this CUDA?" but "is there a device, and
does it support what I need?", and it is asked in exactly one place: here.  A
test states what it needs (``requires_gpu(dtype=..., ops=...)``) and is skipped
with a reason naming what is actually missing.

MPS is not a smaller CUDA -- no float64 at all, no complex linear algebra, no
``eigh``/``eig``/``lstsq`` -- so the capability tables below are per device.
They are a measurement; see ``docs/NUMERICS.md``, "The MPS device is
float32-real only".
"""

import os

import numpy as np
import pytest

try:
    import torch
except ImportError:                                  # the package must work without it
    torch = None


def _pick_device():
    # TTPY_NO_GPU=1 forces the no-accelerator path.  Not a convenience: it is
    # how the CPU-only behaviour of the suite gets checked on a machine that
    # does have a GPU, which is otherwise only observable on CI.
    if torch is None or os.environ.get("TTPY_NO_GPU") == "1":
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


GPU_DEVICE = _pick_device()

#: dtypes the device can hold and do linear algebra in.
GPU_DTYPES = {
    None: (),
    "cuda": ("float32", "float64", "complex64", "complex128"),
    "mps": ("float32",),
}[GPU_DEVICE]

#: linalg entry points the device implements, out of the ones the package uses.
GPU_OPS = {
    None: (),
    "cuda": ("qr", "svd", "solve", "eigh", "eig", "lstsq", "expm"),
    "mps": ("qr", "svd", "solve", "expm"),
}[GPU_DEVICE]

#: widest real dtype available on the device, and the widest complex one.
GPU_DTYPE = "float64" if "float64" in GPU_DTYPES else (
    "float32" if GPU_DTYPES else None)
GPU_CDTYPE = "complex128" if "complex128" in GPU_DTYPES else (
    "complex64" if "complex64" in GPU_DTYPES else None)

# What the working precision is worth, per device.  A float64 tolerance is not
# a float32 tolerance divided by nothing: each of these is a separate claim.
#
#   PARITY      how far a torch result may sit from the numpy result of the
#               *same* computation -- pure round-off, no algorithm in between.
#   SOLVE_EPS   the accuracy it is meaningful to ask an iterative solver for.
#   SOLVE_TOL   how far the answer may then be from a well-conditioned
#               reference (10 * SOLVE_EPS).
#   QLAPLACE_D  the qlaplace_dd size the device can actually solve.  This is the
#               one that is easy to get wrong: the reachable residual is
#               kappa * eps, and in float32 that runs out well below the sizes
#               the float64 tests use -- on the numpy backend too, so it is the
#               precision talking and not the device.
#   DENSE_TOL   distance to a dense oracle on that system, which is
#               condition-limited rather than eps-limited.
#
# Both rows are measured; see ``docs/NUMERICS.md``, "GPU test calibration".
_CALIBRATION = {
    "float64": dict(parity=1e-12, solve_eps=1e-10, qlaplace_d=8, dense_tol=1e-7),
    "float32": dict(parity=2e-6, solve_eps=1e-5, qlaplace_d=4, dense_tol=1e-5),
    None: dict(parity=None, solve_eps=None, qlaplace_d=None, dense_tol=None),
}
PARITY = _CALIBRATION[GPU_DTYPE]["parity"]
SOLVE_EPS = _CALIBRATION[GPU_DTYPE]["solve_eps"]
QLAPLACE_D = _CALIBRATION[GPU_DTYPE]["qlaplace_d"]
DENSE_TOL = _CALIBRATION[GPU_DTYPE]["dense_tol"]
SOLVE_TOL = None if SOLVE_EPS is None else 10 * SOLVE_EPS


#: Where a *float64* torch test should run.  Several tests are pinned to float64
#: by their oracle and merely want the fastest device that has it, so they fall
#: back to CPU torch rather than skipping.
TORCH_F64_DEVICE = GPU_DEVICE if "float64" in GPU_DTYPES else "cpu"


def gpu_reason(dtype=None, ops=()):
    """Why this device cannot run the test, or ``None`` if it can.

    Kept as a plain function rather than a mark so that a test needing several
    combinations can ask about each one.
    """
    if torch is None:
        return "torch is not installed"
    if GPU_DEVICE is None:
        return "no GPU device (no CUDA, no MPS)"
    if dtype is not None and dtype not in GPU_DTYPES:
        return f"{GPU_DEVICE} has no {dtype}"
    missing = [o for o in ops if o not in GPU_OPS]
    if missing:
        return f"{GPU_DEVICE} does not implement {', '.join(missing)}"
    return None


def requires_gpu(dtype=None, ops=()):
    """Skip unless there is a device that supports ``dtype`` and ``ops``."""
    reason = gpu_reason(dtype, ops)
    return pytest.mark.skipif(reason is not None, reason=reason or "")


def gpu_backend(dtype=None):
    """A backend object on the device, in ``dtype`` (default: its widest real)."""
    from tt import backend as bk
    return bk.TorchBackend(GPU_DEVICE, dtype or GPU_DTYPE)


def to_gpu(x, dtype=None):
    """Move a tt.vector / tt.matrix to the device."""
    return x.to("torch", GPU_DEVICE, dtype or GPU_DTYPE)


def as_np(a):
    from tt import backend as bk
    return np.asarray(bk.to_numpy(a)) if hasattr(a, "device") else np.asarray(a)


def rel(a, b):
    a, b = as_np(a), as_np(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
