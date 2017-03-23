from __future__ import print_function, absolute_import, division
from six.moves import xrange
import numpy as np
from . import cross
from .tt_tensor2 import *

class black_box_tensor:
    def __init__(self,sz,f=None,eps=1e-6):
        self.f = f
        self.n = np.array(sz,dtype=np.int32)
        self.eps=eps
    def dmrg(self):
        d = self.n.size
        #r = np.ones((d+1,1),dtype=int32)
        #ps = np.ones((d+1,1), dtype=int32)
        r,ps=cross.cross.tt_cross(self.n,self.f,self.eps)
        tt=tt_tensor()
        tt.n = self.n
        tt.ps = ps
        tt.r = r
        tt.d = d
        tt.core=cross.cross.core.copy()
        cross.cross.cross_dealloc() #Clear the allocated core
        tt.ps = ps.copy()
        self.tt = tt
#This class will receive a 3d function a grid (i.e., f is an f(x,y,z))
class fun_qtt: #Only QTT is assumed, i.e. d is an array [d1,d2,d3,...,] 
    def __init__(self,f,d,a,b,order='F'):
        self.f = f
        self.d = np.array(d,dtype=np.int32)
        self.m = self.d.size
        self.a = np.array(a) #The sizes of the boxes
        self.sz = 2**self.d
        self.h = (np.array(b)-self.a)/np.array((self.sz-1),dtype=float)
        self.sm = np.zeros((self.m,self.d.sum()),dtype=np.int32)
        self.full_sz = np.array([2]*self.d.sum(),dtype=np.int32)
        self.order = order
        start=0
        for i in xrange(self.m):
            for j in xrange(self.d[i]):
                self.sm[i,j+start]=2**(j)
            start = start + self.d[i]
        # ind_tt1 = (i1) + (i2-1)*2 + (i3-1)*4 + ...
        # ind_tt2 = (i2) + (i2-1)*2  +
        # we have self.m 
    def __call__(self,ind):
        #We are given a QTT index ind, have to convert it to a TT index
        if self.order is 'F':
            ind_tt=np.dot(self.sm,(np.array(ind,dtype=np.int32)-1))
        else:
            ind_tt=np.dot(self.sm,(np.array(ind,dtype=np.int32)))

        x = self.a + self.h*ind_tt 
        return self.f(x)

class fun_ind: #No QTT is assumed, just convert index -> float
    def __init__(self,f,n,a,b,order='F'):
        self.f = f
        self.n = np.array(n, dtype = np.int32)
        self.d = self.n.size
        self.a = np.array(a)
        self.h = (np.array(b)-self.a)/np.array((self.n-1),dtype=float)
    def __call__(self,ind):
        if self.order is 'F':
            x = self.a + self.h*(np.array(ind,dtype=np.int32)-1)
        else:
            x = self.a + self.h*np.array(ind,dtype=np.int32)
        return self.f(x)        
