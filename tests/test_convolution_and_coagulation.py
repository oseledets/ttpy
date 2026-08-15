"""Multicomponent Smoluchowski coagulation in TT (Matveev et al., JCP 2016).

Every check here has an oracle that does not come from this module:

* the lower-triangular trapezoidal convolution against a brute-force dense
  quadrature written from the definition (``1e-10``);
* the ``d = 2`` constant-kernel solution against the paper's analytic
  formula (18) and the exact total density ``1/(1 + t/2)``;
* the observed convergence order against the scheme's advertised
  ``O(h^2 + tau^2)``;
* the additive kernel against the two identities it satisfies exactly on an
  unbounded domain -- mass ``int (v_1 + v_2) n`` is conserved and the total
  density decays as ``N_0 exp(-M_0 t)``.
"""

import numpy as np
import pytest

import tt
from tt.algs.smoluchowski import (additive_kernel, coagulation_rhs,
                                  component_sum, constant_kernel, solve,
                                  trapezoidal_convolution,
                                  trapezoidal_weights)
from tt.core.vector import vector

i0 = pytest.importorskip("scipy.special").i0


# --- helpers -----------------------------------------------------------------

def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def exponential_ic(N, h, a=1.0, b=1.0):
    """``n_0 = a b exp(-a v_1 - b v_2)`` on the grid: rank 1 by construction."""
    x = h * np.arange(N)
    return vector.from_list([(a * np.exp(-a * x)).reshape(1, N, 1),
                             (b * np.exp(-b * x)).reshape(1, N, 1)]), x


def analytic(x, t, a=1.0, b=1.0):
    """Eq. (18) of the paper: ``K == 1``, exponential initial data."""
    v1, v2 = np.meshgrid(x, x, indexing="ij")
    arg = 2.0 * np.sqrt(a * b * v1 * v2 * t / (t + 2.0))
    return a * b * np.exp(-a * v1 - b * v2) / (1.0 + t / 2.0) ** 2 * i0(arg)


def dense_convolution(f, g, h):
    """``C[i] = h^d sum_{j<=i} w_j^{(i)} f[i-j] g[j]``, from the definition.

    Written independently of the module under test: an explicit loop over the
    output index, the product of one-dimensional trapezoidal weight vectors,
    no FFT and no tensor train anywhere.
    """
    f, g = np.asarray(f), np.asarray(g)
    d = f.ndim
    out = np.zeros(f.shape)
    for idx in np.ndindex(*f.shape):
        if min(idx) == 0:                      # int_0^0 = 0
            continue
        rev = tuple(slice(i, None, -1) for i in idx)     # f[i - j]
        fwd = tuple(slice(0, i + 1) for i in idx)        # g[j]
        weight = np.ones([i + 1 for i in idx])
        for axis, i in enumerate(idx):
            w = np.ones(i + 1)
            w[0] = w[i] = 0.5
            weight = weight * w.reshape(
                [-1 if a == axis else 1 for a in range(d)])
        out[idx] = h ** d * np.sum(weight * f[rev] * g[fwd])
    return out


# --- (c) the convolution alone, against a dense quadrature -------------------

def test_convolution_matches_dense_quadrature_1d():
    rng = np.random.default_rng(0)
    N, h = 32, 0.3
    f, g = rng.standard_normal(N), rng.standard_normal(N)
    got = trapezoidal_convolution(vector.from_list([f.reshape(1, N, 1)]),
                                  vector.from_list([g.reshape(1, N, 1)]),
                                  h, eps=1e-14)
    # measured: 2.4e-16
    assert rel(np.asarray(got.full()).reshape(-1),
               dense_convolution(f, g, h)) < 1e-10


@pytest.mark.parametrize("d, N, ranks", [(2, 16, [1, 3, 1]),
                                         (3, 8, [1, 3, 2, 1])])
def test_convolution_matches_dense_quadrature(d, N, ranks):
    """Random low-rank f, g -- the TT path must be exact, not merely close.

    measured: 1.1e-15 (d = 2), 1.5e-15 (d = 3).
    """
    f = tt.rand([N] * d, d, ranks)
    g = tt.rand([N] * d, d, [1] + [2] * (d - 1) + [1])
    h = 0.25
    got = trapezoidal_convolution(f, g, h, eps=1e-14)
    ref = dense_convolution(np.asarray(f.full()), np.asarray(g.full()), h)
    assert rel(np.asarray(got.full()), ref) < 1e-10


def test_convolution_vanishes_on_the_boundary_hyperplanes():
    """``int_0^{v_k}`` is empty when ``v_k = 0``; the discrete rule must agree."""
    N, h = 12, 0.5
    f = tt.rand([N, N], 2, [1, 2, 1])
    g = tt.rand([N, N], 2, [1, 2, 1])
    c = np.asarray(trapezoidal_convolution(f, g, h, eps=1e-14).full())
    assert np.max(np.abs(c[0, :])) < 1e-12
    assert np.max(np.abs(c[:, 0])) < 1e-12


def test_convolution_is_symmetric_in_its_arguments():
    N, h = 20, 0.4
    f = tt.rand([N, N], 2, [1, 3, 1])
    g = tt.rand([N, N], 2, [1, 2, 1])
    a = trapezoidal_convolution(f, g, h, eps=1e-14)
    b = trapezoidal_convolution(g, f, h, eps=1e-14)
    assert (a - b).norm() / a.norm() < 1e-12


# --- (a) the analytic solution, the main oracle ------------------------------

@pytest.mark.parametrize(
    "N, vmax, tau, T, tol_sol, tol_dens",
    [
        # measured on this machine (numpy backend, eps = 1e-10):
        #   N=100 Vmax=10 tau=0.1  T=1 -> solution 1.63e-3, density 1.43e-3, R=9
        (100, 10.0, 0.1, 1.0, 5e-2, 5e-3),
        #   N=100 Vmax=20 tau=0.1  T=1 -> solution 1.36e-3, density 4.87e-4, R=10
        (100, 20.0, 0.1, 1.0, 5e-2, 2e-3),
        #   N=100 Vmax=10 tau=0.1  T=2 -> solution 4.58e-3, density 8.42e-3, R=9
        (100, 10.0, 0.1, 2.0, 5e-2, 2e-2),
        #   N=200 Vmax=10 tau=0.05 T=1 -> solution 7.38e-4, density 1.78e-3, R=9
        (200, 10.0, 0.05, 1.0, 5e-2, 5e-3),
    ],
)
def test_constant_kernel_matches_the_analytic_solution(N, vmax, tau, T,
                                                       tol_sol, tol_dens):
    """Paper's eq. (18) for ``K == 1``, ``a = b = 1``, on ``[0, V_max]^2``.

    The paper reports 1.4e-1 at ``N = 100`` and 2.3e-3 at ``N = 500`` for this
    setup; the numbers this implementation reaches are recorded above the
    parameter list, and the thresholds are deliberately loose around them.
    """
    h = vmax / (N - 1)
    n0, x = exponential_ic(N, h)
    kernel = constant_kernel([N, N])
    n = solve(n0, kernel, h, tau, int(round(T / tau)), eps=1e-10)

    err = rel(np.asarray(n.full()), analytic(x, T))
    assert err < tol_sol, f"solution error {err:.3e}"

    w = trapezoidal_weights([N, N], h)
    dens = float(tt.dot(w, n))
    dens_err = abs(dens - 1.0 / (1.0 + T / 2.0)) * (1.0 + T / 2.0)
    assert dens_err < tol_dens, f"density error {dens_err:.3e}"

    # the paper's R = 7..13 for this problem
    assert int(max(n.r)) <= 20, f"ranks blew up to {list(n.r)}"


def test_total_density_follows_the_exact_law_along_the_trajectory():
    """``dN/dt = -N^2/2`` for ``K == 1`` gives ``N(t) = 1/(1 + t/2)`` exactly.

    The oracle is a continuum identity, so the grid has to be fine enough that
    the trapezoidal quadrature of ``n`` itself is accurate: at ``t = 0`` the
    rule already misses ``2 h^2/12 = 4.2e-4`` here, and on the coarse
    ``N = 100, V_max = 20`` grid it misses 6.8e-3 -- a quadrature error, not a
    solver error, which is why this check uses the finer grid.
    """
    N, vmax, tau = 400, 20.0, 0.05
    h = vmax / (N - 1)
    n0, _ = exponential_ic(N, h)
    w = trapezoidal_weights([N, N], h)
    seen = []

    def record(step, t, n):
        seen.append((t, float(tt.dot(w, n))))

    solve(n0, constant_kernel([N, N]), h, tau, 20, eps=1e-10, callback=record)
    assert len(seen) == 21 and seen[0][0] == 0.0
    for t, dens in seen:
        exact = 1.0 / (1.0 + t / 2.0)
        # measured: max relative deviation 4.19e-4 over t in [0, 1]
        assert abs(dens - exact) / exact < 2e-3, (t, dens, exact)


@pytest.mark.parametrize("d, N, kind", [(2, 12, "constant"), (2, 12, "additive"),
                                        (3, 8, "constant"), (3, 8, "additive")])
def test_rhs_matches_a_dense_assembly_of_the_same_kernel(d, N, kind):
    """``L1 - n L2`` against a dense rebuild -- the kernel bookkeeping itself.

    The convolution is pinned separately above; what this pins is the rest of
    :func:`coagulation_rhs`: which factor of each rank-1 term multiplies ``n``
    inside the gain, and that the sink collapses to one scalar quadrature per
    term.  Agreement measured at 1e-15.
    """
    h = 0.4
    rng = np.random.default_rng(7)
    n = tt.rand([N] * d, d, [1] + [2] * (d - 1) + [1],
                samplefunc=rng.standard_normal)
    kernel = (constant_kernel([N] * d) if kind == "constant"
              else additive_kernel([N] * d, h))

    nd = np.asarray(n.full())
    w = np.asarray(trapezoidal_weights([N] * d, h).full())
    gain = np.zeros_like(nd)
    sink = np.zeros_like(nd)
    for kv, ku in kernel:
        kvd, kud = np.asarray(kv.full()), np.asarray(ku.full())
        gain += 0.5 * dense_convolution(kvd * nd, kud * nd, h)
        sink += kvd * float(np.sum(w * kud * nd))

    assert rel(np.asarray(coagulation_rhs(n, kernel, h, eps=1e-14).full()),
               gain - nd * sink) < 1e-10


def test_rhs_is_zero_for_a_zero_distribution():
    N, h = 24, 0.5
    z = tt.zeros([N, N])
    r = coagulation_rhs(z, constant_kernel([N, N]), h)
    assert r.norm() == 0.0


# --- (b) convergence ---------------------------------------------------------

def test_second_order_convergence_under_joint_refinement():
    """Halve ``h`` and ``tau`` together: a second-order scheme divides the
    error by ~4.  Measured on ``V_max = 20``, ``T = 1`` (large enough that the
    truncation of the domain does not set a floor above the discretization):

        N = 100, tau = 0.10 -> 1.36e-3
        N = 200, tau = 0.05 -> 3.70e-4   (ratio 3.69)
        N = 400, tau = 0.025 -> 9.59e-5  (ratio 3.86)
    """
    vmax, T = 20.0, 1.0
    errs = []
    for N, tau in [(100, 0.1), (200, 0.05), (400, 0.025)]:
        h = vmax / (N - 1)
        n0, x = exponential_ic(N, h)
        n = solve(n0, constant_kernel([N, N]), h, tau,
                  int(round(T / tau)), eps=1e-12)
        errs.append(rel(np.asarray(n.full()), analytic(x, T)))
    assert errs[0] / errs[1] > 2.0, errs
    assert errs[1] / errs[2] > 2.0, errs
    assert errs[2] < 5e-4, errs


# --- (d) the additive kernel -------------------------------------------------

def test_additive_kernel_conserves_mass_and_decays_density():
    """``K = sum_i u_i + sum_i v_i``: rank-2, and two exact identities.

    On an unbounded domain the total mass ``M = int (v_1 + v_2) n`` is
    conserved and ``dN/dt = -M N``, i.e. ``N(t) = N_0 exp(-M_0 t)``.  On the
    truncated box mass leaks out of the top -- the point of the check is that
    it leaks far more slowly than the density decays.

    Measured (``V_max = 40``, ``N = 200``, ``tau = 0.01``, ``T = 0.2``):
    density 0.673604 vs 0.674840 predicted (1.8e-3 relative), mass drift
    3.2e-3 while the density falls by 33%, max rank 14.
    """
    N, vmax, tau, T = 200, 40.0, 0.01, 0.2
    h = vmax / (N - 1)
    n0, _ = exponential_ic(N, h)
    w = trapezoidal_weights([N, N], h)
    s = component_sum([N, N], h)

    dens0 = float(tt.dot(w, n0))
    mass0 = float(tt.dot(w, s * n0))
    assert abs(dens0 - 1.0) < 1e-2 and abs(mass0 - 2.0) < 1e-2

    n = solve(n0, additive_kernel([N, N], h), h, tau,
              int(round(T / tau)), eps=1e-10)
    dens = float(tt.dot(w, n))
    mass = float(tt.dot(w, s * n))

    predicted = dens0 * np.exp(-mass0 * T)
    assert abs(dens - predicted) / predicted < 1e-2, (dens, predicted)
    drift = abs(mass - mass0) / mass0
    decay = abs(dens - dens0) / dens0
    assert drift < 5e-2, drift
    assert drift < 0.1 * decay, (drift, decay)
    assert int(max(n.r)) <= 30, list(n.r)


# --- building blocks ---------------------------------------------------------

def test_component_sum_is_the_sum_of_the_coordinates():
    N, h = 7, 0.3
    s = np.asarray(component_sum([N, N, N], h).full())
    x = h * np.arange(N)
    ref = x[:, None, None] + x[None, :, None] + x[None, None, :]
    assert rel(s, ref) < 1e-13
    assert list(component_sum([N, N, N], h).r) == [1, 2, 2, 1]


def test_trapezoidal_weights_integrate_a_polynomial_exactly():
    """The trapezoidal rule is exact on linear functions, in every dimension."""
    N, vmax = 9, 2.0
    h = vmax / (N - 1)
    w = trapezoidal_weights([N, N], h)
    s = component_sum([N, N], h)
    # int_0^2 int_0^2 (v1 + v2) dv = 2 * 2 * 2 = 8
    assert abs(float(tt.dot(w, s)) - 8.0) < 1e-12


def test_exported_on_the_tt_namespace():
    assert tt.smoluchowski_solve is solve
    assert "smoluchowski_solve" in tt.__all__
