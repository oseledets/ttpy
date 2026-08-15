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
  density decays as ``N_0 exp(-M_0 t)``;
* the ballistic kernel of the paper's eq. (17) against its own closed form
  evaluated node by node (the cross-approximated separable form must
  reproduce it), and the coagulation it drives against Table 5 of the paper.
"""

import numpy as np
import pytest

import tt
from smoluchowski_solver_shim import (additive_kernel, ballistic_kernel,
                                  coagulation_rhs,
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


# --- (e) the ballistic kernel, eq. (17) --------------------------------------

def ballistic_dense(x, floor):
    """Eq. (17) on the full ``d = 2`` grid, straight from the formula.

    Written here from the paper, independently of the module under test: an
    explicit meshgrid over ``(u_1, u_2, v_1, v_2)``, the two component sums
    clipped at ``floor``, no tensor train anywhere.
    """
    u1, u2, v1, v2 = np.meshgrid(x, x, x, x, indexing="ij")
    su = np.maximum(u1 + u2, floor)
    sv = np.maximum(v1 + v2, floor)
    return (su ** (1 / 3) + sv ** (1 / 3)) ** 2 * np.sqrt(1 / su + 1 / sv)


def test_ballistic_bond_cut_reassembles_the_kernel_exactly():
    """``sum_a ku_a(u) kv_a(v) == K(u; v)`` entry by entry.

    This is the check on the *cut*, which is the only step of
    :func:`ballistic_kernel` that could silently be wrong: the cross is an
    approximation to ``eps``, but regrouping its cores at the ``u | v`` bond
    is an identity, so the reassembled pairs must agree with the cross's own
    tensor to round-off and with eq. (17) to the cross's accuracy.

    Measured: 3.2e-16 against the cross's tensor, 7.7e-9 against eq. (17)
    at ``eps = 1e-8`` (worst pointwise 6.6e-8).
    """
    N, vmax, eps = 16, 10.0, 1e-8
    h = vmax / (N - 1)
    x = h * np.arange(N)
    info = {}
    kernel = ballistic_kernel([N, N], h, eps=eps, info=info)
    assert len(kernel) == info["rank"]
    assert info["floor"] == h

    got = np.zeros((N,) * 4)
    for kv, ku in kernel:
        got += np.multiply.outer(np.asarray(ku.full()).reshape(N, N),
                                 np.asarray(kv.full()).reshape(N, N))
    ref = ballistic_dense(x, h)
    assert rel(got, ref) < 100 * eps, rel(got, ref)
    assert np.max(np.abs(got - ref) / ref) < 1e-6
    # the same number, measured by the function itself on nodes it did not use
    assert info["err"] < 100 * eps and info["err_max"] < 1e-6


def test_ballistic_kernel_rank_is_moderate_and_flat_in_N():
    """``K = F(sum u, sum v)`` is a function of two scalars, so the rank of
    the ``u | v`` bond is the epsilon-rank of the two-variable matrix ``F``
    and cannot grow with the grid.

    The oracle is that epsilon-rank, computed independently by a dense SVD of
    ``F(s, t)`` sampled on the range of the sums -- no tensor train, no cross.
    Measured: SVD rank 8 at ``eps = 1e-6`` (and 8 for ``V_max`` from 10 to
    1000), bond rank 6 at ``N = 100`` and 7 at ``N = 1000``, against the
    paper's Table 6 value ``R = 19..23`` at the same accuracy.
    """
    eps, vmax = 1e-6, 100.0
    ranks = {}
    for N in (100, 1000):
        h = vmax / (N - 1)
        info = {}
        ballistic_kernel([N, N], h, eps=eps, info=info)
        ranks[N] = info["rank"]
        # the independent oracle: epsilon-rank of F(s, t) on the sum range
        s = np.unique(np.concatenate([np.geomspace(h, 2 * vmax, 128),
                                      np.linspace(h, 2 * vmax, 128)]))
        f = ((s[:, None] ** (1 / 3) + s[None, :] ** (1 / 3)) ** 2
             * np.sqrt(1 / s[:, None] + 1 / s[None, :]))
        sv = np.linalg.svd(f, compute_uv=False)
        svd_rank = int(np.sum(sv / sv[0] > eps))
        assert svd_rank <= 12, svd_rank
        assert ranks[N] <= svd_rank + 2, (ranks[N], svd_rank)

    assert ranks[100] < 40 and ranks[1000] < 40, ranks
    assert ranks[1000] <= ranks[100] + 5, ranks


def test_ballistic_coagulation_is_much_faster_than_the_constant_kernel():
    """Fig. 2 of the paper: the ballistic kernel drives "much faster dynamics".

    Same grid, same initial datum, same horizon -- only the kernel differs.
    The constant kernel has the exact law ``N(t) = 1/(1 + t/2)``, so at
    ``t = 1`` it is at 0.667 of its start; the ballistic one must be well
    below that, and the ranks must stay where the paper's do (R = 12..18).

    Measured: 0.1839 ballistic vs 0.6667 constant, max rank 11.
    """
    N, vmax, tau, T = 100, 10.0, 0.05, 1.0
    h = vmax / (N - 1)
    n0, _ = exponential_ic(N, h)
    w = trapezoidal_weights([N, N], h)
    nsteps = int(round(T / tau))

    n_const = solve(n0, constant_kernel([N, N]), h, tau, nsteps, eps=1e-6)
    n_ball = solve(n0, ballistic_kernel([N, N], h, eps=1e-6), h, tau, nsteps,
                   eps=1e-6)
    d_const = float(tt.dot(w, n_const))
    d_ball = float(tt.dot(w, n_ball))

    assert abs(d_const - 1.0 / (1.0 + T / 2.0)) < 1e-2, d_const
    assert d_ball < 0.5 * d_const, (d_ball, d_const)
    assert int(max(n_ball.r)) <= 25, list(n_ball.r)


def test_ballistic_density_matches_table_5_of_the_paper():
    """External oracle: Table 5, total density at ``t = 1``, ``tau = 0.05``.

    ``N = 100, V_max = 10 -> 0.1847`` is the cheapest row of that table.  The
    residual gap is not solver error: the paper keeps the kernel finite by
    starting the grid at ``V_min > 0`` and dissipating everything below it
    (its eq. (4)), while :func:`ballistic_kernel` clips the component sums at
    one grid step instead.  Measured here: 0.1839, i.e. 0.46% below theirs,
    which is why the threshold is 2e-2 rather than the 5e-2 a pure
    "same ballpark" check would need.
    """
    N, vmax, tau, T = 100, 10.0, 0.05, 1.0
    h = vmax / (N - 1)
    n0, _ = exponential_ic(N, h)
    n = solve(n0, ballistic_kernel([N, N], h, eps=1e-6), h, tau,
              int(round(T / tau)), eps=1e-6)
    dens = float(tt.dot(trapezoidal_weights([N, N], h), n))
    assert abs(dens - 0.1847) / 0.1847 < 2e-2, dens


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


def test_the_primitive_is_in_the_package_and_the_model_is_not():
    """The split is deliberate: a Volterra convolution is a TT primitive,
    coagulation is an application.

    ``tt.algs.convolution`` ships with the package; the kernels, the
    right-hand side and the time stepping live in
    ``examples/smoluchowski/solver.py`` and are imported from there, like
    every other example.
    """
    from tt.algs import convolution
    assert hasattr(convolution, "trapezoidal_convolution")
    assert not hasattr(tt, "smoluchowski_solve")
    import smoluchowski_solver_shim as model
    assert hasattr(model, "solve") and hasattr(model, "coagulation_rhs")

