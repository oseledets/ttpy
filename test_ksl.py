import numpy as np
from math import pi,sqrt
import tt
from tt.ksl import ksl
import time
from scipy.linalg import expm
d = 6
f = 1
A = tt.qlaplace_dd([d]*f)*(2**d+1)**2
#A = tt.eye(2,d)
n = [2]*(d*f)
n0 = 2**d
t = np.arange(1,n0+1)*1.0/(n0+1)
x = np.exp(-10*(t-0.5)**2)
#x = np.sin(pi*t)
x = tt.tensor(x.reshape([2]*d,order='F'),1e-8)
#x = tt.ones(2,d)

tau = 1
tau1 = 1
y = x.copy()
ns_fin = 8
tau0 = 1.0
tau_ref = tau0/2**ns_fin
for i in xrange(2**ns_fin):  
    y=ksl(1j*A,y,tau_ref)
yref = y.copy()
tau = 5e-2
res = ""
ns = 2
while ( ns <= ns_fin ):
    tau = tau0/(2**ns) 
    y = x.copy()
    for i in xrange(2**ns):
        y = ksl(1.0j*A,y,tau)
    er = (y - yref).norm()/y.norm()
    res += 'tau=%3.1e er=%3.1e ns=%d \n' % (tau,er,2**ns)
    ns += 1
print res
