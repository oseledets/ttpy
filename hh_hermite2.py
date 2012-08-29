#This is a clone of the MATLAB spectral discretization for the Henon-Heiles potential
#Using the Hermite-DVR representation
#The goal is to compute many eigenfunctions of this operator
import numpy as np
from numpy.fft import fft
from scipy.linalg import toeplitz
from tt.kls import *
#from tt.tensor2 import *
import tt
import time
from math import pi,sqrt
import quadgauss
import os
import sys
# open 2 fds
# put /dev/null fds on 1 and 2
#import Redirect
#Redirect.start_redirect()
#os.dup2(null_fds[1], 2)
#os.close(null_fds[0])
#os.close(null_fds[1])



f = 18 #The number of degrees of freedom
L = 7 #The domain is [-L, L], periodic
lm = 0.111803 #The magic constant
#lm = 0 #The magic constant

#lm = 
N = 20 # The size of the spectral discretization

x, ws = quadgauss.cdgqf(N,6,0,0.5) #Generation of hermite quadrature
#Generate Laplacian
lp = np.zeros((N,N))
for i in xrange(N):
    for j in xrange(N):
        if i is not j:
            lp[i,j] = (-1)**(i - j)*(2*(x[i] - x[j])**(-2) - 0.5)
        else:
            lp[i,j] = 1.0/6*(4*N - 1 - 2 * x[i]**2)
lp = tt.matrix(lp)
#h = 2 * pi/N
#x = h * np.arange(1,N+1)
#x = L * (x - pi)/pi
#column = -0.5*((-1)**(np.arange(1,N)))/(np.sin(h*np.arange(1,N)/2)**2)
#column = np.concatenate(([-pi**2/(3*h**2)-1.0/6],column))  
#lp = (pi/L)**2*toeplitz(column)
#lp = -lp
e = tt.eye([N])

#lp = tt.matrix(lp)
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
t = tt.tensor(x)
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

H = 1j*A
#Generate the initial Gaussian, which is just shifted

gs = np.exp(-0.5*(x-2)**2)
gs = tt.tensor(gs,1e-8)
start = None
for i in xrange(f):
    start = tt.kron(start,gs)
tau = 1e-2
radd = 3
start = start+0*tt.rand(start.n,start.d,radd)
y = start.copy()
print 'initial value norm:', start.norm()
cf = []
tf = 8.0
nit = 1000
tau = (tf/nit)
i = 0
t = 0
import time
t1 = time.time()
while t <= tf:
    print '%f/%f' % (t,tf)
    y = kls(H,y,tau)
    #cf.append(np.dot(y.full().flatten(),start.full().flatten()))
    cf.append(tt.dot(y,start))
    t += tau
t2 = time.time()
print("Elapsed time: %f" % (t2-t1))
#Redirect.stop_redirect()

zz = np.abs(fft(np.conj(cf)))
lls = np.arange(zz.size)*pi/(0.5*tf)

