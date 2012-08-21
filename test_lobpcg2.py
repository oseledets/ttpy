import numpy as np
import blopex

#Callback functions that do nothing




#def bvec(x,bx):
#    print x.shape
#    bx[:,:] = x[:,:]

def full_bvec(x,bx,mat,name):
    #print name
    #print x
    #bx1 = bx.copy()
    bx[:,:] = np.dot(mat,x)
def tvec(x,bx):
    bx[:,:]=x[:,:]

def fun(x):
    x[0] = 20
n = 7
m = 2
X = np.random.randn(n,m)
#X = np.arange(1,n*m+1).reshape((n,m),order='F')
#X = X * X
X, dmp = np.linalg.qr(X)
X = np.asfortranarray(X)


d = np.arange(1,n+1)
d = np.diag(d)

from functools import partial
bvec = partial(full_bvec,mat=d,name='bvec')
avec = partial(full_bvec,mat=d,name='avec')
Y = np.zeros((n,m))
#full_bvec(X,Y,d)
lam = blopex.blopex.lobpcg2(avec,tvec,tvec,X,100,1e-6)
