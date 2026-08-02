"""Same workloads under the legacy Fortran ttpy and under ttpy2.

Runs in either environment: it only uses the public API both packages share.
Point it at each interpreter in turn and diff the JSON.

    # legacy (python 3.11 + numpy 1.24 + gfortran, see docs/LEGACY_BUILD.md)
    PYTHONPATH=/path/to/ttpy-src python bench/bench_vs_legacy.py --out legacy.json
    # ttpy2
    python bench/bench_vs_legacy.py --out new.json

The tensors are built from explicit cores with a fixed seed, so both packages
get bit-identical input.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

import tt


def which_ttpy():
    return getattr(tt, "__version__", "1.x (legacy, Fortran)")


def timeit(fn, repeats=3):
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def make_tt(d, n, r, seed=0):
    """Identical input for both packages: explicit cores, scaled to keep norm O(1)."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    cores = [rng.standard_normal((ranks[k], n, ranks[k + 1])) / np.sqrt(ranks[k + 1])
             for k in range(d)]
    return tt.vector.from_list(cores)


CASES = [
    ("round", dict(d=30, n=2, r=50)),
    ("round", dict(d=60, n=2, r=100)),
    ("round", dict(d=20, n=4, r=150)),
    ("add+round", dict(d=30, n=2, r=50)),
    ("add+round", dict(d=60, n=2, r=100)),
    ("dot", dict(d=60, n=2, r=100)),
    ("matvec+round", dict(d=30, n=2, r=50)),
    ("matvec+round", dict(d=40, n=2, r=100)),
    ("tt_svd", dict(d=10, n=4, r=0)),
    ("amen_solve", dict(d=12, n=2, r=0)),
]


def run_case(name, d, n, r, repeats):
    if name == "round":
        x = make_tt(d, n, r)
        return timeit(lambda: x.round(1e-8), repeats)
    if name == "add+round":
        x, y = make_tt(d, n, r, 0), make_tt(d, n, r, 1)
        return timeit(lambda: (x + y).round(1e-8), repeats)
    if name == "dot":
        x, y = make_tt(d, n, r, 0), make_tt(d, n, r, 1)
        return timeit(lambda: tt.dot(x, y), repeats)
    if name == "matvec+round":
        A = tt.qlaplace_dd([d])
        x = make_tt(d, 2, r)
        return timeit(lambda: tt.matvec(A, x).round(1e-8), repeats)
    if name == "tt_svd":
        rng = np.random.default_rng(1)
        dense = rng.standard_normal([n] * d)
        return timeit(lambda: tt.vector(dense, 1e-8), repeats)
    if name == "amen_solve":
        A = tt.qlaplace_dd([d])
        f = tt.ones(2, d)
        try:                       # legacy keeps it in a submodule
            from tt.amen import amen_solve
        except ImportError:
            from tt.algs.amen import amen_solve
        return timeit(lambda: amen_solve(A, f, f, 1e-6, verb=0), max(1, repeats - 2))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()

    rows = []
    for name, case in CASES:
        if name in args.skip:
            continue
        try:
            med, best = run_case(name, repeats=args.repeats, **case)
        except Exception as exc:            # a package that cannot do it says so
            rows.append(dict(workload=name, failed=f"{type(exc).__name__}: {exc}",
                             **case))
            print(f"{name:14s} d={case['d']:3d} n={case['n']} r={case['r']:4d}  "
                  f"FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(dict(workload=name, median_s=med, best_s=best, **case))
        print(f"{name:14s} d={case['d']:3d} n={case['n']} r={case['r']:4d}  "
              f"median={med * 1e3:9.2f} ms  best={best * 1e3:9.2f} ms", flush=True)

    env = dict(ttpy=which_ttpy(), python=platform.python_version(),
               numpy=np.__version__, host=platform.node(),
               threads={k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    print("\nenv:", env)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(env=env, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
