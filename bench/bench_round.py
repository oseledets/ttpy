"""Deterministic vs randomized TT rounding, CPU vs GPU, fp64 vs fp32.

The question this answers: TT rounding is a chain of small sequential SVDs, so
it is bound by factorization latency.  Randomized rounding replaces the chain by
matrix multiplications.  Does that change which hardware wins?

    python bench/bench_round.py --out bench/results/round_cmp.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone

import numpy as np

from tt import backend as bk
from tt.core import _ops

CASES = [
    dict(d=30, n=2, r=100, rmax=50),
    dict(d=60, n=2, r=200, rmax=100),
    dict(d=20, n=4, r=300, rmax=150),
]


def make_cores(d, n, r, backend, dtype):
    """Random TT whose norm stays O(1).

    Unscaled Gaussian cores multiply up to an astronomical norm: with d=60 and
    r=200 the tensor overflows float32 outright (and the resulting inf reaches
    LAPACK, which is a confusing way to learn that the benchmark data is wrong).
    Scaling each core by 1/sqrt(r) keeps the product bounded.
    """
    rng = np.random.default_rng(0)
    ranks = [1] + [r] * (d - 1) + [1]
    return [backend.asarray(
        rng.standard_normal((ranks[k], n, ranks[k + 1])) / np.sqrt(ranks[k + 1]),
        dtype) for k in range(d)]


def timeit(fn, backend, repeats):
    fn()
    if backend.name == "torch":
        backend.torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if backend.name == "torch":
            backend.torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setups = [("numpy", "cpu", "float64"), ("numpy", "cpu", "float32"),
              ("torch", "cuda", "float64"), ("torch", "cuda", "float32")]
    rows = []
    for name, device, dtype in setups:
        backend = (bk.NumpyBackend(dtype) if name == "numpy"
                   else bk.TorchBackend(device, dtype))
        for case in CASES:
            cores = make_cores(case["d"], case["n"], case["r"], backend, dtype)
            rmax = case["rmax"]
            det, _ = timeit(lambda: _ops.round_cores(cores, 0.0, rmax),
                            backend, args.repeats)
            rnd, _ = timeit(lambda: _ops.randomized_round(cores, rmax, 10, 0),
                            backend, args.repeats)
            # accuracy of both, measured the same way
            y_det = _ops.round_cores(cores, 0.0, rmax)
            y_rnd = _ops.randomized_round(cores, rmax, 10, 0)
            nx = _ops.norm(cores)
            e_det = _ops.norm(_ops.sub(cores, y_det)) / nx
            e_rnd = _ops.norm(_ops.sub(cores, y_rnd)) / nx
            rows.append(dict(backend=name, device=device, dtype=dtype,
                             det_s=det, rand_s=rnd, speedup=det / rnd,
                             err_det=float(e_det), err_rand=float(e_rnd), **case))
            print(f"{name:5s} {device:4s} {dtype:8s} d={case['d']:3d} n={case['n']} "
                  f"r={case['r']:3d}->{rmax:3d}  svd={det * 1e3:8.1f} ms  "
                  f"rand={rnd * 1e3:8.1f} ms  x{det / rnd:5.1f}  "
                  f"err {e_det:.2e} / {e_rnd:.2e}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"env": {"host": platform.node(),
                               "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "threads": {k: os.environ.get(k) for k in
                                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                               "repeats": args.repeats},
                       "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
