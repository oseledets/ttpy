#This is a clone of the MATLAB spectral discretization for the Henon-Heiles potential
#The goal is to compute many eigenfunctions of this operator
import numpy as np
from scipy.linalg import toeplitz
from kls import *
#from tt_tensor2 import *
import tt_tensor2 as tt
from tt_tensor2 import tt_tensor, tt_matrix #These two lines seems
import time
from math import pi


f = 64 #The number of degrees of freedom
L = 7 #The domain is [-L, L], periodic
lm = 0.111803 #The magic constant
#lm = 0 
N = 28 # The size of the spectral discretization

h = 2 * pi/N
x = h * np.arange(1,N+1)
x = L * (x - pi)/pi
column = -0.5*((-1)**(np.arange(1,N)))/(np.sin(h*np.arange(1,N)/2)**2)
column = np.concatenate(([-pi**2/(3*h**2)-1.0/6],column))  
lp = (pi/L)**2*toeplitz(column)
lp = -lp
e = tt.eye([N])

lp = tt_matrix(lp)
#Calculate the kinetic energy (Laplace) operator
lp2 = None
eps = 1e-8
for i in xrange(f):
    w = lp
    for j in xrange(i):
        w = tt.kron(e,w)
    for j in xrange(i+1,f):
        w = tt.kron(w,e)
    lp2 = lp2 + w
    lp2 = lp2.round(eps)


#Now we will compute Henon-Heiles stuff
xx = []
t = tt_tensor(x)
ee = tt.ones([N])
for  i in xrange(f):
    t0 = t
    for j in xrange(i):
        t0 = tt.kron(ee,t0)
    for j in xrange(i+1,f):
        t0 = tt.kron(t0,ee)
    xx.append(t0)

#Harmonic potential
harm = None
for i in xrange(f):
    harm = harm + (xx[i]*xx[i])
    harm = harm.round(eps)

#Henon-Heiles correction
V = None
for s in xrange(f-1):
    V = V + (xx[s]*xx[s]*xx[s+1] - (1.0/3)*xx[s+1]*xx[s+1]*xx[s+1])
    V = V.round(eps)





A = 0.5*lp2 + tt.diag(0.5*harm + lm*V)
A = A.round(eps) 

#Generate the initial condition
psi = None
pp1 = np.exp(-0.5*((x-2)**2))
pp1 = tt_tensor(pp1,1e-12)
for i in xrange(f):
    psi = tt.kron(psi,pp1)

pol = None
for i in xrange(f):
    pol = pol + xx[i]*xx[i]
    pol = pol.round(1e-8)

psi = psi * pol
psi = psi.round(1e-12)


radd = 5
rnd = tt.rand(psi.n,psi.d,radd)

psi = psi + 0*rnd

t = 0
tf = 1
tau = 1e-2
t1 = time.time()
while t <= tf:
    psi = kls(-1.0*A,psi,tau)
    t += tau
t2 = time.time()
print('Total time: %f' % (t2-t1))
#n = A.n
#d = A.tt.d
#r = [3]*(d+1)
#r[0] = 1
#r[d] = B
#x0 = tt.rand(n,d,r)

#The initial approximation is a Gaussian

#Save here 
#q = np.load("test.npz")
#x0.core = q['core']
#x0.ps = q['ps']
