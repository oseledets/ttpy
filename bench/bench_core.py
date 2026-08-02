"""Benchmarks for the TT core.

A number without its regime is worthless, so every row carries backend, device,
dtype, thread count, problem size and the number of repeats.  Timings are the
median of `repeats` runs after one warm-up; GPU runs synchronize before the clock
is stopped.

    python bench/bench_core.py --backends numpy torch --out bench/results/core.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

import tt
from tt import backend as bk
from tt.core import _ops


def _sync(backend):
    if backend.name == "torch":
        backend.torch.cuda.synchronize()


def timeit(fn, backend, repeats=5):
    """Median wall time of `fn` in seconds, after one warm-up run."""
    fn()
    _sync(backend)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(backend)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


# --- workloads ---------------------------------------------------------------

def make_tt(d, n, r, backend, dtype):
    """Random TT with constant ranks, allocated on the given backend."""
    rng = np.random.default_rng(0)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [backend.asarray(rng.standard_normal((ranks[k], n, ranks[k + 1])), dtype)
             for k in range(d)]
    return tt.vector.from_list(cores)


def bench_round(d, n, r, backend, dtype, repeats):
    x = make_tt(d, n, r, backend, dtype)
    return timeit(lambda: _ops.round_cores(x.cores, 1e-8), backend, repeats)


def bench_add_round(d, n, r, backend, dtype, repeats):
    """The workhorse of every TT iteration: add two tensors, truncate back."""
    x = make_tt(d, n, r, backend, dtype)
    y = make_tt(d, n, r, backend, dtype)
    return timeit(lambda: _ops.round_cores(_ops.add(x.cores, y.cores), 1e-8),
                  backend, repeats)


def bench_dot(d, n, r, backend, dtype, repeats):
    x = make_tt(d, n, r, backend, dtype)
    y = make_tt(d, n, r, backend, dtype)
    return timeit(lambda: _ops.dot(x.cores, y.cores), backend, repeats)


def bench_matvec_round(d, n, r, backend, dtype, repeats):
    """QTT Laplacian (rank 3) applied to a rank-r vector, then truncated."""
    A = tt.qlaplace_dd([d]).to(backend.name, backend.device, dtype)
    x = make_tt(d, 2, r, backend, dtype)
    return timeit(lambda: tt.matvec(A, x).round(1e-8), backend, repeats)


def bench_tt_svd(d, n, r, backend, dtype, repeats):
    """Dense -> TT on a tensor with n**d elements."""
    rng = np.random.default_rng(1)
    dense = backend.asarray(rng.standard_normal([n] * d), dtype)
    return timeit(lambda: _ops.tt_svd(dense, 1e-8), backend, repeats)


WORKLOADS = {
    "round": (bench_round, [
        dict(d=30, n=2, r=50), dict(d=30, n=2, r=150), dict(d=60, n=2, r=200),
        dict(d=20, n=4, r=300), dict(d=10, n=8, r=400),
    ]),
    "add+round": (bench_add_round, [
        dict(d=30, n=2, r=50), dict(d=30, n=2, r=100), dict(d=20, n=4, r=150),
    ]),
    "dot": (bench_dot, [
        dict(d=30, n=2, r=100), dict(d=60, n=2, r=200), dict(d=20, n=4, r=300),
    ]),
    "matvec+round": (bench_matvec_round, [
        dict(d=30, n=2, r=50), dict(d=40, n=2, r=100), dict(d=60, n=2, r=150),
    ]),
    "tt_svd": (bench_tt_svd, [
        dict(d=10, n=4, r=0), dict(d=8, n=6, r=0), dict(d=14, n=3, r=0),
    ]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["numpy"],
                    choices=["numpy", "torch"])
    ap.add_argument("--dtypes", nargs="+", default=["float64"],
                    choices=["float32", "float64"])
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = {
        "host": platform.node(),
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cpu_count": os.cpu_count(),
    }

    rows = []
    for backend_name in args.backends:
        for dtype in args.dtypes:
            backend = (bk.NumpyBackend(dtype) if backend_name == "numpy"
                       else bk.TorchBackend(args.device, dtype))
            if backend_name == "torch":
                env["torch"] = backend.torch.__version__
                env["gpu"] = backend.torch.cuda.get_device_name(0)
            for wl in args.workloads:
                fn, cases = WORKLOADS[wl]
                for case in cases:
                    med, best = fn(backend=backend, dtype=dtype,
                                   repeats=args.repeats, **case)
                    rows.append(dict(workload=wl, backend=backend_name,
                                     device=backend.device, dtype=dtype,
                                     repeats=args.repeats, median_s=med,
                                     best_s=best, **case))
                    print(f"{wl:14s} {backend_name:5s} {backend.device:5s} "
                          f"{dtype:8s} d={case['d']:3d} n={case['n']:2d} "
                          f"r={case.get('r', 0):4d} median={med * 1e3:9.2f} ms "
                          f"best={best * 1e3:9.2f} ms", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"env": env, "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
