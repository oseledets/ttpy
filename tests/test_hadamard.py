"""Acceptance for the compressed elementwise product (tt.hadamard).

The oracle is the toolbox's own ``(a * b).round(eps)``: forming the inflated
Kronecker product and rounding it is the definition of the answer, so the fused
routine must reproduce both its *value* and its *ranks*.
"""

import numpy as np
import pytest

import tt


def _pair(n, d, ra, rb, seed=0):
    rng = np.random.default_rng(seed)
    a = tt.rand(n, d, r=ra)
    b = tt.rand(n, d, r=rb)
    return a, b


@pytest.mark.parametrize("n,d,ra,rb", [(2, 6, 4, 5), (4, 5, 6, 3),
                                       (3, 7, 5, 5), (2, 10, 8, 6)])
def test_hadamard_matches_value_and_ranks_of_the_rounded_product(n, d, ra, rb):
    a, b = _pair(n, d, ra, rb)
    ref = (a * b).round(1e-8)
    got = tt.hadamard(a, b, eps=1e-8)
    assert (got - ref).norm() / ref.norm() < 1e-10
    # the point of the algorithm: it compresses as well as the explicit route
    assert max(got.r) <= max(ref.r)


def test_hadamard_takes_more_than_two_factors():
    a, b, c = tt.rand(2, 8, r=4), tt.rand(2, 8, r=4), tt.rand(2, 8, r=3)
    ref = (a * b * c).round(1e-8)
    got = tt.hadamard(a, b, c, eps=1e-8)
    assert (got - ref).norm() / ref.norm() < 1e-10
    assert max(got.r) <= max(ref.r)


def test_hadamard_sum_fuses_a_linear_combination_of_products():
    """The advection shape: u*ux + v*uy in one sweep, one truncation."""
    u, ux = tt.rand(2, 8, r=5), tt.rand(2, 8, r=5)
    v, uy = tt.rand(2, 8, r=5), tt.rand(2, 8, r=5)
    ref = (u * ux + v * uy).round(1e-8)
    got = tt.hadamard_sum([[u, ux], [v, uy]], eps=1e-8)
    assert (got - ref).norm() / ref.norm() < 1e-10
    assert max(got.r) <= max(ref.r)


def test_hadamard_sum_honours_coefficients():
    a, b = tt.rand(2, 6, r=4), tt.rand(2, 6, r=4)
    c, e = tt.rand(2, 6, r=3), tt.rand(2, 6, r=3)
    ref = (2.5 * (a * b) - 1.5 * (c * e)).round(1e-8)
    got = tt.hadamard_sum([[a, b], [c, e]], coefs=[2.5, -1.5], eps=1e-8)
    assert (got - ref).norm() / ref.norm() < 1e-10


def test_hadamard_respects_a_rank_cap():
    a, b = tt.rand(2, 10, r=8), tt.rand(2, 10, r=8)
    got = tt.hadamard(a, b, eps=1e-12, rmax=12)
    assert max(got.r) <= 12


def test_hadamard_is_cheaper_than_the_explicit_route_at_high_rank():
    """At rank 64 the explicit product pays r^6 where this pays r^4."""
    import time
    a, b = tt.rand(2, 12, r=64), tt.rand(2, 12, r=64)
    t0 = time.perf_counter()
    ref = (a * b).round(1e-8)
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = tt.hadamard(a, b, eps=1e-8)
    t_fused = time.perf_counter() - t0
    assert (got - ref).norm() / ref.norm() < 1e-10
    assert t_fused < t_naive        # measured ~50x on a laptop
