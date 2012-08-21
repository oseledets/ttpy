import numpy as np
from eigb import *
#from tt_tensor2 import *
import tt_tensor2 as tt
from tt_tensor2 import tt_tensor #These two lines seems
import time
d = 8
f = 10
A = tt.qlaplace_dd([d]*f)
#A = (-1)*A
#A = tt.eye(2,d)
n = [2]*(d*f)
r = [5]*(d*f+1)
r[0] = 1
r[d*f] = 3 #Number of eigenvalues sought
x = tt.rand(n,d*f,r)
#x = tt_ones(2,d)
t = time.time()
y,lam=eigb(A,x,1e-6)

core = y.core
rr = y.r
ps = y.ps
n = y.n
d = y.d
last_core = core[y.ps[d-1]-1:y.ps[d]-1]
last_core = last_core.reshape((rr[d-1]*n[d-1],r[d]),order='F')
print np.dot(last_core.T,last_core)
print 'Eigenvalues:', lam
t1 = time.time()
print 'Time is:', t1-t
