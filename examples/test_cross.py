import numpy as np
import tt


def test_multifuncrs():
    d = 30
    n = 2 ** d
    b = 1E3
    h = b / (n + 1)
    # x = np.arange(n)
    # x = np.reshape(x, [2] * d, order = 'F')
    # x = tt.tensor(x, 1e-12)
    x = tt.xfun(2, d)
    e = tt.ones(2, d)
    x = x + e
    x = x * h

    def fn(x):  # Should be rank 2
        return np.sin(x) / x

    y = tt.multifuncrs([x], fn, 1e-6, y0=tt.ones(2, d))
    # y1 = tt.tensor(fn(x.full()), 1e-8)

    print("pi / 2 ~ ", tt.dot(y, tt.ones(2, d)) * h)
    # print (y - y1).norm() / y.norm()
