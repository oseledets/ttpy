import pytest
import tt
import tt.eigb

from time import monotonic


@pytest.mark.slow
def test_eigb():
    """This code computes many eigenvalus of the Laplacian operator.
    """
    d = 8
    f = 8
    A = tt.qlaplace_dd([d] * f)
    # A = (-1)*A
    # A = tt.eye(2,d)
    n = [2] * (d * f)
    r = [8] * (d * f + 1)
    r[0] = 1
    r[d * f] = 8  # Number of eigenvalues sought
    x = tt.rand(n, d * f, r)
    # x = tt_ones(2,d)
    elapsed = -monotonic()
    y, lam = tt.eigb.eigb(A, x, 1e-6)
    elapsed += monotonic()
    print('Eigenvalues:', lam)
    print('Time is:', elapsed)
