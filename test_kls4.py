import numpy as np
from math import pi,sqrt
from numpy.linalg import norm
from kls import *
#from tt_tensor2 import *
import tt_tensor2 as tt
from tt_tensor2 import tt_tensor, tt_matrix #These two lines seems
import time
from scipy.linalg import expm
d = 2
f = 1
A = tt.qlaplace_dd([d]*f)
#A = (-1)*A
n = [2]*(d*f)
#A = tt.eye(n)
#a0 = tt_matrix([[1,0],[0,0.5]],1e-10)
#a1 = tt_matrix([[1,0],[0,0.5]],1e-10)
#B = tt.kron(a0,a1)
#C = tt.kron(a1,a0)
#A = tt.kron(B,C) + tt.kron(C,B)
#A = tt.kron(A,a1)
#r = [5]*(d*f+1)
#r[0] = 1
#r[d*f] = 3 #Number of eigenvalues sought
#x = tt.rand(n,d*f,r)
#x = tt.ones(n) #Starting vector
n0 = 2**d
x = np.arange(1,n0+1)*1.0/(n0+1)
x = np.sin(sqrt(2)*x*pi) #Exact manifold
#x = np.exp(-10*(x-0.5)**2)
#
x = tt_tensor(x.reshape([2]*d,order='F'),1e-8)

#x = tt.ones(2,d)
tau = 1e-2
y = x.copy()
y = y
A = 1j*A
#x0 = x.full().flatten()
x1 = y.full().copy().flatten('F')
#yeul = x0 + tau*np.dot(A.full(),x0)
A1 = A.full().copy()
Z=expm(tau*A1)
yexact = np.dot(Z,x1)
y=kls(A,y,tau)
print y.full().flatten()
er = norm(yexact - y.full().flatten('F'))
print 'er=',er
