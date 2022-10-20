import tt
from tt.amen import amen_solve


def test_amen(d=12):
    """This program test two subroutines: matrix-by-vector multiplication and
    linear system solution via AMR scheme.
    """
    A = tt.qlaplace_dd([d])
    x = tt.ones(2, d)
    _ = amen_solve(A, x, x, 1e-6)  # y
