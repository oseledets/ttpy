"""Test common matrix operation with QTT-toolbox like matmat-operation and
matvec-operation.
"""

import numpy as np
import tt


def test_matmul():
    x = tt.xfun(2, 3)
    e = tt.ones(2, 2)
    X = tt.matrix(x, n=[2] * 3, m=[1] * 3)  # [0, 1, 2, 3, 4, 5, 6, 7]^T
    E = tt.matrix(e, n=[1] * 2, m=[2] * 2)  # [1, 1, 1, 1]
    # [[ 0.  0.  0.  0.]
    #  [ 1.  1.  1.  1.]
    #  [ 2.  2.  2.  2.]
    #  [ 3.  3.  3.  3.]
    #  [ 4.  4.  4.  4.]
    #  [ 5.  5.  5.  5.]
    #  [ 6.  6.  6.  6.]
    #  [ 7.  7.  7.  7.]]
    print((X * E).full())
    assert np.all((X * E) * np.arange(4) == np.arange(8) * 6.)


def test_matvec():
    A = tt.matrix(tt.xfun(2, 3), n=[1] * 3, m=[2] * 3)
    u = np.arange(8)
    assert A * u == 140.
