import sys
sys.path.append('../')
import numpy as np
import tt

d = 20
n = 2 ** d
b = 1e+3
h = b / (n + 1)
x = np.arange(n)
x = np.reshape(x, [2] * d, order = 'F')
x = tt.tensor(x, 1e-12)
#x = tt.xfun(2, d)
e = tt.ones(2, d)
x = x + e
x = x * h 


sf = lambda x : np.sin(x) #Should be rank 2

y = tt.multifuncrs([x], sf, 1e-6)
y1 = tt.tensor(sf(x.full()), 1e-8)

print (y - y1).norm() / y.norm()
