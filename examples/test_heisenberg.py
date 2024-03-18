import numpy as np
import pytest
import tt
import tt.eigb

from time import monotonic


def gen_1d(mat, e, i, d):
    w = mat
    for j in range(i):
        w = tt.kron(e, w)
    for j in range(d - i - 1):
        w = tt.kron(w, e)
    return w


def gen_heisen(d):
    sx = [[0, 1], [1, 0]]
    sx = np.array(sx, dtype=np.float)
    sz = [[1, 0], [0, -1]]
    sz = np.array(sz, dtype=np.float)
    sz = 0.5 * sz
    sp = [[0, 1], [0, 0]]
    sp = np.array(sp, dtype=np.float)
    sm = sp.T
    e = np.eye(2)
    sx = tt.matrix(sx, 1e-12)
    sz = tt.matrix(sz, 1e-12)
    sp = tt.matrix(sp, 1e-12)
    sm = tt.matrix(sm, 1e-12)
    e = tt.matrix(e, 1e-12)
    # Generate ssx, ssz.
    ssp = [gen_1d(sp, e, i, d) for i in range(d)]
    ssz = [gen_1d(sz, e, i, d) for i in range(d)]
    ssm = [gen_1d(sm, e, i, d) for i in range(d)]
    A = None
    for i in range(d - 1):
        A = A + 0.5 * (ssp[i] * ssm[i + 1] + ssm[i] * ssp[i + 1]) + (
            ssz[i] * ssz[i + 1])
        A = A.round(1e-8)
    return A


@pytest.mark.slow
def test_heisenberg():
    """Compute minimal eigenvalues for the Heisenberg model.
    """
    d = 60  # The dimension of the problem (number of spins).
    B = 5  # Number of eigenvalues sought.
    eps = 1e-5  # Accuracy of the computations.

    A = gen_heisen(d)
    n = A.n
    d = A.tt.d
    r = [B] * (d + 1)
    r[0] = 1
    r[d] = B
    x0 = tt.rand(n, d, r)
    print('Matrices are done')
    elapsed = -monotonic()
    y, lam = tt.eigb.eigb(A, x0, eps, max_full_size=1000)
    elapsed += monotonic()
    print('Eigenvalues: ', lam)
    print('Elapsed time: {:3.1f}'.format(elapsed))
