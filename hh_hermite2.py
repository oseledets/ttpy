#This is a clone of the MATLAB spectral discretization for the Henon-Heiles potential
#Using the Hermite-DVR representation
#The goal is to compute many eigenfunctions of this operator
import numpy as np
from numpy.fft import fft
from scipy.linalg import toeplitz
from tt.ksl import ksl
import tt
import time
from math import pi,sqrt
import quadgauss
import mctdh
import dvr

f = 2 #The number of degrees of freedom
lm = 0.111803 #The magic constant
#lm = 0 #The magic constant

#lm = 
N = 20 # The size of the spectral discretization

#x, ws = quadgauss.cdgqf(N,6,0,0.5) #Generation of hermite quadrature
#Generate Laplacian
#lp = np.zeros((N,N))
#for i in xrange(N):
#    for j in xrange(N):
#        if i is not j:
#            lp[i,j] = (-1)**(i - j)*(2*(x[i] - x[j])**(-2) - 0.5)
#        else:
#            lp[i,j] = 1.0/6*(4*N - 1 - 2 * x[i]**2)
#lp = tt.matrix(lp)

trafo, x, dif2, dif1 = mctdh.initho(N, 1.0, 0.0, 0.0, 0.0, '')
basis = 1 # HO Basis
rpbaspar = np.zeros(3, dtype = np.int)
rpbaspar[0] = 0.0 #hoxeq
rpbaspar[1] = 1.0 #hofreq = rpbaspar(2) #ho
rpbaspar[2] = 1.0 #homass = rpbaspar(3) #homass
ipbaspar = []
ipbaspar1 = []
lsq = False
lnw = False
weight = dvr.dvrweights(trafo, x, basis, rpbaspar, ipbaspar, ipbaspar1, lsq, lnw)

lp = -dif2
lp = tt.matrix(lp)
e = tt.eye([N])

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

#Compute CAP
import aux
cap = None
#6.0 1.0 3 1
zcapl = [6.0] * f
zcapr = [-6.0] * f
etal = [-1.0] * f
etar = etal
bl = [3] * f
br = bl 
sgnl = [1] * f
sgnr = [-1] * f
for i in xrange(f):
    z = aux.cap(x, zcapl[i], etal[i], bl[i], sgnl[i]) + aux.cap(x, zcapr[i], etar[i], br[i], sgnr[i])
    z = tt.tensor(z)
    for j in xrange(i):
        z = tt.kron(ee, z)
    for j in xrange(i + 1, f):
        z = tt.kron(z, ee)
    cap = cap + z

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

A = 0.5 * lp2 + tt.diag(0.5 * harm + lm * V) #+ tt.diag(cap)
A = A.round(eps) 
#A = tt.eye([N,N])
H = 1j * A
#H1 = H.full()
#d, v = eig(H1)
#import ipdb; ipdb.set_trace()
#Generate the initial Gaussian, which is just shifted
#This is how it is defined in MCTDH
#q1      gauss  2.0  0.0  0.7071
x0 = 2.0
p0 = 0.0
dx = 0.7071

gs = np.exp(-0.25*((x-x0)/dx)**2) * np.exp(1j * p0 * (x - x0))
gs = gs * weight
gs = tt.tensor(gs,1e-8)
start = None
for i in xrange(f):
    start = tt.kron(start,gs)
radd = 8
#start = start+0*tt.rand(start.n,start.d,radd)
start = start*(1.0/start.norm())
y = start.copy()
print 'initial value norm:', start.norm()
cf = []

tf = 10
ns = 2
i = 0
t = 0
import time
t1 = time.time()
#s1 = 0.86053284899368-0.46936702639161j
#s1 = 0.99855056911186 - 0.04996850934329j
#S = expm(tau * H.full())
#y1 = np.dot(S, start.full().flatten('F'))
#s2 = np.dot(np.conj(y1),start.full().flatten('F'))
#s1 = 0.52021138426564-0.80781148446727j
cf = {}
tau = 1e-2
y0 = y.copy()
while t <= tf:    
    print '%f/%f' % (t, tf)
    cf[round(t,3)] = tt.dot(y, start)
    y = ksl(H, y, tau, scheme = 'symm')
    print y.norm()
    t += tau
t2 = time.time()
print("Elapsed time: %f" % (t2-t1))

#Load MCTDH computations
#work/python/henon-heiles/4d


#from scipy.linalg import expm
#H1 = H.full()
#H1 = expm(tf * H1)
#ytrue = np.dot(H1, start.full().flatten("F"))
#er = np.linalg.norm(ytrue - y.full().flatten("F"))
#print "Error:", er
