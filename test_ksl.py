import numpy as np
from math import pi,sqrt
import tt
from tt.kls import ksl,kls
import time
from scipy.linalg import expm
d = 6
f = 1
#A = tt.qlaplace_dd([d]*f)
A = tt.eye(2,d * f)
n = [2]*(d*f)
n0 = 2**d

x = np.arange(1,n0+1)*1.0/(n0+1)
x = np.exp(-10*(x-0.5)**2)
#x = np.sin(x)
x = tt.tensor(x.reshape([2]*d,order='F'),1e-2)
#x = tt.ones(2,d*f)
#x = [1,0,0,0]; x = np.array(x).reshape([2,2]); x = tt.tensor(x,1e-8);
y = x.copy()
tau = 1e-3
tf = 1
t = 0
while t <= tf:
    y = ksl(-1j*A,y,tau)
    t += tau
#y2 = kls(-1.0j*A,y,tau)
