import sys
sys.path.append("../")
import tt
#import ctypes as ct
#ct.CDLL("libblas.so.3", ct.RTLD_GLOBAL)
#ct.CDLL("libcblas.so.1", ct.RTLD_GLOBAL)
#ct.CDLL("liblapacke.so", ct.RTLD_GLOBAL)
import numpy as np

a = tt.rand([3, 5, 7, 11], 4, [1, 4, 6, 5, 1])
b = tt.rand([3, 5, 7, 11], 4, [1, 2, 4, 3, 1])

c = tt.multifuncrs2([a, b], lambda x: np.sum(x, axis=1), eps=1E-6)

print "Relative error norm:", (c - (a + b)).norm() / (a + b).norm()









