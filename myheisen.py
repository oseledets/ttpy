
import sys
import os
sys.path.append("/Users/ivan/work/python/ttpy")
sys.path.append("/Users/iv/work/python/ttpy")
import tt
import numpy as np
from tt.eigb import eigb
import time
#This example is about the spin-system example
def gen_1d(mat,e,i,d):
    w = mat
    for j in xrange(i):
        w = tt.kron(e,w)
    for j in xrange(d-i-1):
        w = tt.kron(w,e)
    return w

def gen_heisen(d):
    sx = [[0,1],[1,0]]
    sx = np.array(sx,dtype=np.float)
    sz = [[1,0],[0,-1]]
    sz = np.array(sz,dtype=np.float)
    sz = 0.5 * sz
    sp = [[0,1],[0,0]]; sp =  np.array(sp,dtype=np.float)
    sm = sp.T
    e = np.eye(2)
    sx = tt.matrix(sx,1e-12)
    sz = tt.matrix(sz,1e-12)
    sp = tt.matrix(sp,1e-12)
    sm = tt.matrix(sm,1e-12)
    e = tt.matrix(e,1e-12)
    #Generate ssx, ssz
    ssp = [gen_1d(sp,e,i,d) for i in xrange(d)]
    ssz = [gen_1d(sz,e,i,d) for i in xrange(d)]
    ssm = [gen_1d(sm,e,i,d) for i in xrange(d)]
    A = None
    for i in xrange(d-1):
        A = A + 0.5 * (ssp[i] * ssm[i+1] + ssm[i] * ssp[i+1]) +  (ssz[i] * ssz[i+1])
        A = A.round(1e-8)
    return A
es = []
lm = []
ds = [20]
for d in ds:
    B = 3
    A = gen_heisen(d)
    n = A.n
    d = A.tt.d
    r = [2]*(d+1)
    r[0] = 1
    r[d] = B
    x0 = tt.rand(n,d,r)
    t1 = time.time()
    print 'Matrices are done'
    y, lam = eigb(A,x0,1e-3)
    es.append(lam[0]/d)
    lm.append(d)
    t2 = time.time()
