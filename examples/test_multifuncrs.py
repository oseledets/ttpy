import numpy as np
import tt


def test_multifuncrs2():
    a = tt.rand([3, 5, 7, 11], 4, [1, 4, 6, 5, 1])
    b = tt.rand([3, 5, 7, 11], 4, [1, 2, 4, 3, 1])
    c = tt.multifuncrs2([a, b], lambda x: np.sum(x, axis=1), eps=1e-6)

    print('Relative error norm:', (c - (a + b)).norm() / (a + b).norm())
