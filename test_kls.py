import numpy as np
from math import pi,sqrt
#from kls import *
#from tt_tensor2 import *
import tt
from tt.kls import kls
import time
from scipy.linalg import expm
d = 6
f = 1
A = tt.qlaplace_dd([d]*f)
n = [2]*(d*f)
n0 = 2**d
x = np.arange(1,n0+1)*1.0/(n0+1)
x = np.exp(-10*(x-0.5)**2)
x = tt.tensor(x.reshape([2]*d,order='F'),1e-8)
#x = tt.ones(2,d)
tau = 1
tau1 = 1
y = x.copy()
ns_fin = 8
tau0 = 1.0
tau_ref = tau0/2**ns_fin
for i in xrange(2**ns_fin):  
    y=kls(-1.0*A,y,tau_ref)
yref = y.copy()
tau = 5e-2
res = ""
ns = 2
while ( ns <= ns_fin ):
    tau = tau0/(2**ns) 
    y = x.copy()
    for i in xrange(2**ns):
        y = kls(-1.0*A,y,tau)
    er = (y - yref).norm()/y.norm()
    res += 'tau=%3.1e er=%3.1e ns=%d \n' % (tau,er,2**ns)
    ns += 1
print res
