#!/usr/bin/env python
"""Robust tensor completion: the loss ALS cannot have, minimised on the manifold.

    python examples/robust_completion.py            # d=5, 2% outliers
    python examples/robust_completion.py 6 8 3      # d, n, rank
    python examples/robust_completion.py 5 6 2 0.1  # ... and outlier fraction

Alternating least squares owes its existence to one structural fact: freezing
all cores but one turns a *quadratic* functional into a linear local problem.
Change the loss and that machinery is gone -- there is no "alternating robust
regression" with a closed-form core update.  Riemannian descent does not care:
:func:`tt.rgd` needs only a differentiable functional, and the gradient comes
from torch autodiff through the tangent parametrization ([NRO22]) without ever
forming the Euclidean gradient, whose TT rank would be the number of samples.

The experiment: recover a rank-``r`` tensor from noisy-free observations of
which a small fraction are *corrupted by large outliers*.

* ``ttSparseALS`` minimises the square loss -- the only loss it can -- and an
  outlier enters that loss with the square of its (huge) magnitude, so the fit
  chases the corruption;
* ``tt.rgd`` with the log-cosh loss (quadratic near zero, linear in the tails,
  everywhere smooth) pays each outlier only linearly and recovers the tensor.

Both methods see identical data.  The oracle is the dense ground truth on the
entries nobody observed.

References
----------
* A. Novikov, M. Rakhuba, I. Oseledets, "Automatic differentiation for
  Riemannian optimization on low-rank matrix and tensor-train manifolds",
  SIAM J. Sci. Comput. 44(2):A843-A869, 2022, arXiv:2103.14974.
* C. Lubich, I. V. Oseledets, B. Vandereycken, "Time integration of tensor
  trains", SIAM J. Numer. Anal. 53(2):917-941, 2015 -- the tangent-space
  machinery in ``tt.algs.riemannian``.
* Reference implementations surveyed for the machinery: ``github.com/Bihaqo/t3f``
  (autodiff; read, not copied); the ALS baseline is this package's own port of
  ttpy's ``ttSparseALS``.
"""

import sys
import time
import warnings

import numpy as np

import tt
import tt.backend as bk
from tt.algs.autodiff import rgd
from tt.algs.completion import ttSparseALS
from tt.algs.cross import element
from tt.core.vector import vector


def run(d=5, n=6, r=2, outlier_frac=0.02, per_dof=40, seed=0):
    rng = np.random.default_rng(seed)
    target = tt.rand([n] * d, r=r)
    dense = np.asarray(target.full())
    scale = float(np.abs(dense).max())

    dof = sum((1 if k == 0 else r) * n * (1 if k == d - 1 else r)
              for k in range(d))
    nobs = per_dof * dof
    idx = np.stack([rng.integers(0, n, nobs) for _ in range(d)], axis=1)
    vals = dense[tuple(idx.T)].copy()
    nbad = int(outlier_frac * nobs)
    bad = rng.choice(nobs, size=nbad, replace=False)
    vals[bad] += 100.0 * scale * rng.choice([-1.0, 1.0], size=nbad)

    held = np.stack([rng.integers(0, n, 50_000) for _ in range(d)], axis=1)
    truth_held = dense[tuple(held.T)]

    def held_out_error(x):
        got = np.asarray(bk.to_numpy(element(x, held))).reshape(-1)
        return float(np.linalg.norm(got - truth_held)
                     / np.linalg.norm(truth_held))

    print(f"rank-{r} target, d={d}, n={n}: {nobs} observations "
          f"({per_dof} per dof), {nbad} of them outliers at 100x the scale\n")

    # --- ALS, square loss: the only loss it can have -------------------------
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_als, info = ttSparseALS({"indices": idx, "values": vals},
                                  [n] * d, ttRank=r, tol=1e-12,
                                  maxnsweeps=100, verbose=False, seed=seed)
    t_als = time.perf_counter() - t0
    print(f"ttSparseALS (square loss):   held-out rel err "
          f"{held_out_error(x_als):9.2e}   {t_als * 1e3:7.0f} ms   "
          f"(fit {info.fit[-1]:.1e} -- it fitted the outliers)")

    # --- Riemannian descent, log-cosh loss ------------------------------------
    x0 = tt.rand([n] * d, r=r).to("torch", "cpu", "float64")
    t = bk.backend_of(x0.cores[0]).torch
    tvals = t.as_tensor(vals)
    width = 0.1 * scale                    # quadratic below, linear above

    def f(cores_list):
        x = vector.from_list(list(cores_list))
        z = (element(x, idx) - tvals) / width
        # log cosh in the overflow-proof form: cosh overflows past |z| ~ 710,
        # and an inf loss autodiffs to a NaN gradient
        az = t.abs(z)
        return (az + t.log1p(t.exp(-2.0 * az)) - np.log(2.0)).sum() * width

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_rgd, h = rgd(f, x0, maxit=500, tol=1e-8)
    t_rgd = time.perf_counter() - t0
    print(f"tt.rgd (log-cosh loss):      held-out rel err "
          f"{held_out_error(x_rgd):9.2e}   {t_rgd * 1e3:7.0f} ms   "
          f"({len(h.iterations)} iterations, {h.stop_reason})")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    run(d, n, r, frac)
