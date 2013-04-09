""" Basic subroutines for ttpy """  
""" They still focus on the linear format for passing the data around, 
    and still convert to list (and back for some simple tasks) """ 
import numpy as np
from numpy import prod, reshape, nonzero, size, sqrt
import math
from math import sqrt
from numbers import Number
import tt_f90
import core_f90





#The main class for working with TT-tensors

class tensor:
    def __init__(self,a=None,eps=1e-14):
        if a is None:
            self.core = 0
            self.d = 0
            self.n = 0
            self.r = 0
            self.ps = 0
            return
        self.d = a.ndim
        self.n = np.array(a.shape,dtype=np.int32)
        r = np.zeros((self.d+1,),dtype=np.int32)
        ps = np.zeros((self.d+1,),dtype=np.int32)
        
        if ( np.iscomplex(a).any() ):
            self.r, self.ps = tt_f90.tt_f90.zfull_to_tt(a.flatten('F'),self.n,self.d,eps)
            self.core = tt_f90.tt_f90.zcore.copy()
        else:
            self.r,self.ps = tt_f90.tt_f90.dfull_to_tt(np.real(a).flatten('F'),self.n,self.d,eps)
            self.core = tt_f90.tt_f90.core.copy()

        tt_f90.tt_f90.tt_dealloc()        
    
    @staticmethod
    def from_list(a,order='F'):
        d = len(a) #Number of cores
        res = tensor()
        n = np.zeros(d,dtype=np.int32)
        r = np.zeros(d+1,dtype=np.int32)
        cr = np.array([])
        for i in xrange(d):
            cr = np.concatenate((cr,a[i].flatten(order)))
            r[i] = a[i].shape[0]
            r[i+1] = a[i].shape[2]
            n[i] = a[i].shape[1]
        res.d = d
        res.n = n
        res.r = r
        res.core = cr
        res.get_ps()
        return res

    @staticmethod
    def to_list(tt):
        d = tt.d
        r = tt.r
        n = tt.n
        ps = tt.ps
        core = tt.core
        res = []
        for i in xrange(d):
            cur_core = core[ps[i]-1:ps[i+1]-1]
            cur_core = cur_core.reshape((r[i],n[i],r[i+1]),order='F')
            res.append(cur_core)
        return res
 

    #Print statement
    def __repr__(self):
        res = "This is a %d-dimensional tensor \n" % self.d
        r = self.r
        d = self.d
        n = self.n
        for i in range(0,d):
            res = res + ("r(%d)=%d, n(%d)=%d \n" % (i, r[i],i,n[i]))
        res = res + ("r(%d)=%d \n" % (d,r[d]))
        return res
    
    def write(self,fname):
        if ( np.iscomplex(self.core).any()):
            pass
        else:
            tt_f90.tt_f90.dtt_write_2(self.n,self.r,self.ps,np.real(self.core),fname)

    def full(self):
        #Generate correct size vector
        sz = self.n.copy()
        if self.r[0] > 1:
            sz = np.concatenate(([self.r[0]],sz))
        if self.r[self.d] > 1:
            sz = np.concatenate(([self.r[self.d]],sz))
        #a = np.zeros(sz,order='F')
        if ( np.iscomplex(self.core).any() ):
            a = tt_f90.tt_f90.ztt_to_full(self.n,self.r,self.ps,self.core,np.prod(sz)) 
        else:
            a = tt_f90.tt_f90.dtt_to_full(self.n,self.r,self.ps,np.real(self.core),np.prod(sz)) 
        a = a.reshape(sz,order='F')
        #import ipdb; ipdb.set_trace()
        return a

    def __add__(self,other):
        if other is None:
            return self
        c = tensor()
        c.r = np.zeros((self.d+1,),dtype=np.int32)
        c.ps = np.zeros((self.d+1,),dtype=np.int32)
        c.n = self.n
        c.d = self.d
        if ( np.iscomplex(self.core).any() or np.iscomplex(other.core).any()):
            c.r,c.ps = tt_f90.tt_f90.ztt_add(self.n,self.r,other.r,self.ps,other.ps,self.core+0j,other.core+0j)
            c.core = tt_f90.tt_f90.zcore.copy()
        else:
             #This could be a real fix in the case we fell to the real world
            c.r,c.ps = tt_f90.tt_f90.dtt_add(self.n,self.r,other.r,self.ps,other.ps,np.real(self.core),np.real(other.core))
            c.core = tt_f90.tt_f90.core.copy()
        tt_f90.tt_f90.tt_dealloc()
        return c

    def __radd__(self,other):
        if other is None:
            return self
        return other + self


    #@profile
    def round(self,eps):
       c=tensor()
       c.n=self.n
       c.d=self.d
       c.r=self.r.copy()
       c.ps=self.ps.copy()
       if ( np.iscomplex(self.core).any() ):
           tt_f90.tt_f90.ztt_compr2(c.n,c.r,c.ps,self.core,eps)
           c.core = tt_f90.tt_f90.zcore.copy()
       else:
           tt_f90.tt_f90.dtt_compr2(c.n,c.r,c.ps,self.core,eps)
           c.core=tt_f90.tt_f90.core.copy()
       tt_f90.tt_f90.tt_dealloc()
       return c

    #@profile
    def norm(self):
        if ( np.iscomplex(self.core).any() ):
            nrm = tt_f90.tt_f90.ztt_nrm(self.n,self.r,self.ps,self.core)
        else:
            nrm=tt_f90.tt_f90.dtt_nrm(self.n,self.r,self.ps,np.real(self.core))
        return nrm
	
    def __rmul__(self,other):
       c = tensor()
       c.d = self.d
       c.n = self.n
       if isinstance(other,Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0]-1:c.ps[1]-1]
            new_core = new_core * other
            c.core = np.array(c.core,dtype=new_core.dtype)
            c.core[c.ps[0]-1:c.ps[1]-1] = new_core
       else:
           c =_hdm(self,other)
       return c
	
    def __mul__(self,other):
        c = tensor()
        c.d = self.d
        c.n = self.n
        if isinstance(other,Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0]-1:c.ps[1]-1]
            new_core = new_core * other
            c.core = np.array(c.core,dtype=new_core.dtype)
            c.core[c.ps[0]-1:c.ps[1]-1] = new_core
        else: 
            c = _hdm(other,self)
        return c

    def __sub__(self,other):
        c = self + (-1) * other
        return c

    def __kron__(self,other):
        if other is None: 
            return self
        a = self
        b = other
        c = tensor()
        c.d = a.d + b.d
        c.n = np.concatenate((a.n,b.n))
        c.r = np.concatenate((a.r[0:a.d],b.r[0:b.d+1]))
        c.get_ps()
        c.core = np.concatenate((a.core,b.core))
        return c

    def __dot__(self,other):
        r1 = self.r
        r2 = other.r
        d = self.d
        if ( np.iscomplex(self.core).any() or np.iscomplex(other.core).any()):
            dt = np.zeros(r1[0]*r2[0]*r1[d]*r2[d],dtype=np.complex)
            dt = tt_f90.tt_f90.ztt_dotprod(self.n,r1,r2,self.ps,other.ps,self.core+0j,other.core+0j,dt.size)
        else:
            dt = np.zeros(r1[0]*r2[0]*r1[d]*r2[d])
            dt = tt_f90.tt_f90.dtt_dotprod(self.n,r1,r2,self.ps,other.ps,np.real(self.core),np.real(other.core),dt.size)
        if dt.size is 1:
            dt = dt[0]
        return dt

    def __col__(self,k):
        c = tensor()
        d = self.d
        r = self.r.copy()
        n = self.n.copy()
        ps = self.ps.copy()
        core = self.core.copy()
        last_core = self.core[ps[d-1]-1:ps[d]-1]
        last_core = last_core.reshape((r[d-1]*n[d-1],r[d]),order='F')
        last_core = last_core[:,k]
        try: 
            r[d] = len(k)
        except:
            r[d] = 1
        ps[d] = ps[d-1] + r[d-1]*n[d-1]*r[d]
        core[ps[d-1]-1:ps[d]-1] = last_core.flatten('F')
        c.d = d
        c.n = n
        c.r = r
        c.ps = ps
        c.core = core
        return c

    def __diag__(self):
        cl = tensor.to_list(self)
        d = self.d
        r = self.r
        n = self.n
        res = []
        dtype = self.core.dtype 
        for i in xrange(d):
            cur_core = cl[i]
            res_core = np.zeros((r[i], n[i], n[i], r[i+1]), dtype = dtype)
            for s1 in xrange(r[i]):
                for s2 in xrange(r[i+1]):
                    res_core[s1, :, :, s2] = np.diag(cur_core[s1, :, s2].reshape(n[i], order='F'))
            res.append(res_core)
        return matrix.from_list(res)
    
    def __neg__(self):
       return self*(-1)

    def get_ps(self):
        self.ps = np.cumsum(np.concatenate(([1],self.n*self.r[0:self.d]*self.r[1:self.d+1]))).astype(np.int32)

    def alloc_core(self):
        self.core = np.zeros((self.ps[self.d]-1,),dtype=np.float)

    def copy(self):
        c = tensor()
        c.core = self.core.copy()
        c.d = self.d
        c.n = self.n.copy()
        c.r = self.r.copy()
        c.ps = self.ps.copy()
        return c
	
    def rmean(self):
        """ Calculates the mean rank of a TT-tensor"""
        if np.prod(self.n) == 0:
            return 0
        # Solving quadratic equation ar^2 + br + c = 0;
        a = np.sum(self.n[1:-1])
        b = self.n[0] + self.n[-1]
        c = - np.sum(self.n * self.r[1:] * self.r[:-1])
        D = b ** 2 - 4 * a * c
        r = 0.5 * (-b + sqrt(D)) / a
        return r




class matrix:
    def __init__(self,a=None,eps=1e-14, n=None, m=None):
        if a is None:
            self.n = 0 #Only two additional fields
            self.m = 0
            self.tt = tensor()
            return
        if isinstance(a,tensor): #Convert from a tt-tensor
            if ( n is None or m is None):
                n1 = np.sqrt(a.n).astype(np.int32)
                m1 = np.sqrt(a.n).astype(np.int32)
            else:
                n1 = np.array(n,dtype=int32)
                m1 = np.array(m,dtype=int32)
                self.n = n1
                self.m = m1
                self.tt = tensor()
                self.tt.core = a.core.copy()
                self.tt.ps = a.ps.copy()
                self.tt.r = a.r.copy()
                return
        try: 
            c = np.asarray(a,dtype=np.float64)
            d = c.ndim/2
            p = c.shape
            self.n = np.array(p[0:d],dtype=np.int32)
            self.m = np.array(p[d:2*d],dtype=np.int32)
            prm = np.arange(2*d)
            prm = prm.reshape((d,2),order='F')
            prm = prm.transpose()
            prm = prm.flatten('F')
            sz = self.n * self.m
            b = c.transpose(prm).reshape(sz,order='F')
            self.tt=tensor(b,eps)
            return
        except ValueError:
            pass
                
    @staticmethod
    def from_list(a):
        d = len(a) #Number of cores
        res = matrix()
        n = np.zeros(d,dtype=np.int32)
        r = np.zeros(d+1,dtype=np.int32)
        m = np.zeros(d,dtype=np.int32)
        cr = np.array([])
        for i in xrange(d):
            cr = np.concatenate((cr,a[i].flatten('F')))
            r[i] = a[i].shape[0]
            r[i+1] = a[i].shape[3]
            n[i] = a[i].shape[1]
            m[i] = a[i].shape[2]
        res.n = n
        res.m = m
        tt = tensor()
        tt.n = n * m 
        tt.core = cr
        tt.r = r
        tt.d = d
        tt.get_ps()
        res.tt = tt
        return res

    @staticmethod
    def to_list(ttmat):
        tt = ttmat.tt
        d = tt.d
        r = tt.r
        n = ttmat.n
        m = ttmat.m
        ps = tt.ps
        core = tt.core
        res = []
        for i in xrange(d):
            cur_core = core[ps[i]-1:ps[i+1]-1]
            cur_core = cur_core.reshape((r[i],n[i],m[i],r[i+1]),order='F')
            res.append(cur_core)
        return res

    def __repr__(self):
        res = "This is a %d-dimensional matrix \n" % self.tt.d
        r = self.tt.r
        d = self.tt.d
        n = self.n
        m = self.m
        for i in range(d):
            res = res + ("r(%d)=%d, n(%d)=%d, m(%d)=%d \n" % (i, r[i],i,n[i],i,m[i]))
        res = res + ("r(%d)=%d \n" % (d,r[d]))
        return res

    @property
    def is_complex(self):
        return np.iscomplex(self.tt.core).any()

    def __getitem__(self, index):
        if len(index) == 2:
            if isinstance(index[0], int) and index[1] == slice(None):
                # row requested
                row = index[0]
                mycrs = matrix.to_list(self)
                crs = []
                for i in xrange(self.tt.d):
                    crs.append(mycrs[i][:, row % self.n[i], :, :].copy())
                    row /= self.n[i]
                return tensor.from_list(crs)
            elif isinstance(index[1], int) and index[0] == slice(None):
                # col requested
                col = index[1]
                mycrs = matrix.to_list(self)
                crs = []
                for i in xrange(self.tt.d):
                    crs.append(mycrs[i][:, :, col % self.m[i], :].copy())
                    col /= self.m[i]
                return tensor.from_list(crs)
            elif isinstance(index[0], int) and isinstance(index[1], int):
                # element requested
                pass
            else:
                # complicated submatrix requested
                pass

    def __add__(self,other):
        if other is None:
            return self
        c = matrix()
        c.tt = self.tt + other.tt
        c.n = np.asanyarray(self.n,dtype=np.int32).copy()
        c.m = np.asanyarray(self.m,dtype=np.int32).copy()
        return c

    def __radd__(self, other):
        if other is None:
            return self
        return other + self

    def __sub__(self,other):
		c = matrix()
		c.tt = self.tt-other.tt
		c.n = np.asanyarray(self.n,dtype=np.int32).copy()
		c.m = np.asanyarray(self.m,dtype=np.int32).copy()
		return c
    
    def __neg__(self):
        return (-1)*self

    def __matmul__(self,other):
        if self.is_complex or other.is_complex:
            pass
        else:
            c = matrix()
            c.n = self.n.copy()
            c.m = other.m.copy()
            tt = tensor()
            tt.d = self.tt.d 
            tt.n = c.n * c.m
            tt.r = core_f90.core.dmat_mat(self.n,self.m,other.m,np.real(self.tt.core),np.real(other.tt.core),self.tt.r,other.tt.r)
            tt.core = core_f90.core.result_core.copy()
            core_f90.core.dealloc()
            tt.get_ps()
            c.tt = tt
            return c

    def __rmul__(self,other):
        if hasattr(other,'__matmul__'):
            return other.__matmul__(self)
        else:
            c = matrix()
            c.tt = other * self.tt
            c.n = self.n
            c.m = self.m
            return c

    def __mul__(self,other):
        if hasattr(other,'__matmul__'):
            return self.__matmul__(other)
        else:
            c = matrix()
            c.tt = self.tt * other
            c.n = self.n
            c.m = self.m
            return c
    
    def __kron__(self,other):
        """ Kronecker product of two TT-matrices """
        if other is None:
            return self
        a = self
        b = other
        c = matrix()
        c.n = np.concatenate((a.n,b.n))
        c.m = np.concatenate((a.m,b.m))
        c.tt = kron(a.tt,b.tt)
        return c

    def norm(self):
        return self.tt.norm()

    def round(self,eps):
        """ Computes an approximation to a 
	    TT-matrix in with accuracy EPS 
	"""
        c = matrix()
        c.tt = self.tt.round(eps)
        c.n = self.n.copy()
        c.m = self.m.copy()
        return c
	
    def copy(self):
        """ Creates a copy of the TT-matrix """
        c = matrix()
        c.tt = self.tt.copy()
        c.n = self.n.copy()
        c.m = self.m.copy()
        return c

    def __diag__(self):
        """ Computes the diagonal of the TT-matrix"""
        c = tensor()
        c.n = self.n.copy()
        c.r = self.tt.r.copy()
        c.d = self.tt.d #Number are NOT referenced
        c.get_ps()
        c.alloc_core()
        #Actually copy the data
        for i in xrange(c.d):
            cur_core1 = np.zeros((c.r[i],c.n[i],c.r[i+1]))
            cur_core = self.tt.core[self.tt.ps[i]-1:self.tt.ps[i+1]-1]
            cur_core = cur_core.reshape(c.r[i],self.n[i],self.m[i],c.r[i+1],order='F')
            for j in xrange(c.n[i]):
                cur_core1[:,j,:] = cur_core[:,j,j,:]
                c.core[c.ps[i]-1:c.ps[i+1]-1] = cur_core1.flatten('F')
        return c

    def full(self):
        """ Transforms a TT-matrix into a full matrix"""
        N = self.n.prod()
        M = self.m.prod()
        a = self.tt.full()
        d = self.tt.d
        sz = np.vstack((self.n,self.m)).flatten('F')
        a = a.reshape(sz,order='F')
        #Design a permutation
        prm = np.arange(2*d)
        prm = prm.reshape((d,2),order='F')
        prm = prm.transpose()
        prm = prm.flatten('F')
        #Get the inverse permutation
        iprm = [0]*(2*d)
        for i in xrange(2*d):
            iprm[prm[i]] = i
        a = a.transpose(iprm).reshape(N,M,order='F')
        a = a.reshape(N,M)
        return a


#Some binary operations (put aside to wrap something in future)
#TT-matrix by a TT-vector product
def matvec(a,b):
    acrs = tensor.to_list(a.tt)
    bcrs = tensor.to_list(b)
    ccrs = []
    d = b.d
    for i in xrange(d):
        acr = np.reshape(acrs[i], (a.tt.r[i], a.n[i], a.m[i], a.tt.r[i + 1]), order='F')
        acr = acr.transpose([3, 0, 1, 2]) # a(R_{i+1}, R_i, n_i, m_i)
        bcr = bcrs[i].transpose([1, 0, 2]) # b(m_i, r_i, r_{i+1})
        ccr = np.tensordot(acr, bcr, axes=(3, 0)) # c(R_{i+1}, R_i, n_i, r_i, r_{i+1})
        ccr = ccr.transpose([1, 3, 2, 0, 4]).reshape((a.tt.r[i] * b.r[i], a.n[i], a.tt.r[i+1] * b.r[i+1]), order='F')
        ccrs.append(ccr)
    return tensor.from_list(ccrs)
        



#TT-by-a-full matrix product (wrapped in Fortran 90, inspired by
#MATLAB prototype)
#def tt_full_mv(a,b):
#    mv = matrix_f90.matrix.tt_mv_full
#    if b.ndim is 1:
#        rb = 1
#    else:
#        rb = b.shape[1]
#    x1 = b.reshape(b.shape[0],rb)
#    y = np.zeros(a.n.prod(),dtype=np.float)
#    y = mv(a.n,a.m,a.tt.r,a.tt.ps,a.tt.core,x1,a.n.prod())
#    return y

def col(a,k):
    """Get the column of the block TT-tensor"""
    if hasattr(a,'__col__'):
        return a.__col__(k)
    else:
        raise ValueError('col is waiting for a TT-tensor or a TT-matrix')

def kron(a,b):
    """Kronecker product of two TT-matrices or two TT-tensors"""
    if  hasattr(a,'__kron__'):
        return a.__kron__(b)
    if a is None:
        return b
    else:
        raise ValueError('Kron is waiting for two TT-tensors or two TT-matrices')

def dot(a,b):
    """Dot product of two TT-matrices or two TT-tensors"""
    if  hasattr(a,'__dot__'):
        return a.__dot__(b)
    if a is None:
        return b
    else:
        raise ValueError('Dot is waiting for two TT-tensors or two TT-matrices')


def diag(a):
    """ Diagonal of a TT-matrix OR diagonal matrix from a TT-tensor """
    if hasattr(a,'__diag__'):
        return a.__diag__()
    else:
        raise ValueError('Can be called only on TT-tensor or a TT-matrix')


def mkron(a, *args):
    """Kronecker product of all the arguments"""
    if not isinstance(a, list):
        a = [a]
    a = list(a) # copy list
    for i in args:
        if isinstance(i, list):
            a.extend(i)
        else:
            a.append(i)
    
    c = tensor()
    c.d = 0
    c.n = np.array([], dtype=np.int32)
    c.r = np.array([], dtype=np.int32)
    c.core = []
    
    for t in a:
        c.d += t.d
        c.n = np.concatenate((c.n, t.n))
        c.r = np.concatenate((c.r[:-1], t.r))
        c.core = np.concatenate((c.core, t.core))
    
    c.get_ps()
    return c
                         

def _hdm (a,b):
    c = tensor()
    c.d = a.d
    c.n = a.n
    c.r = np.zeros((a.d+1,1),dtype=np.int32)
    c.ps = np.zeros((a.d+1,1),dtype=np.int32)
    c.r,c.ps = tt_f90.tt_f90.dtt_hdm(a.n,a.r,b.r,a.ps,b.ps,a.core,b.core)
    c.core = tt_f90.tt_f90.core.copy()
    tt_f90.tt_f90.tt_dealloc()
    return c



#Basic functions for the arrays creation


def ones(n,d=None):
	""" Creates a TT-tensor of all ones"""
	c = tensor()
	if d is None:
            c.n = np.array(n,dtype=np.int32)
            c.d = c.n.size
	else:
            c.n = np.array([n]*d,dtype=np.int32)
            c.d = d
	c.r = np.ones((c.d+1,),dtype=np.int32)
	c.get_ps()
	c.core = np.ones(c.ps[c.d]-1)
	return c


def rand(n,d,r):
	""" tt_rand(n,d,r) -- generate a random TT-tensor"""
	n0 = np.asanyarray(n,dtype=np.int32)
	r0 = np.asanyarray(r,dtype=np.int32)
	if n0.size is 1:
		n0 = np.ones((d,),dtype=np.int32)*n0
	if r0.size is 1:
		r0 = np.ones((d+1,),dtype=np.int32)*r0
		r0[0] = 1
		r0[d] = 1
	c = tensor()
	c.d = d
	c.n = n0
	c.r = r0
	c.get_ps()
	c.core = np.random.randn(c.ps[d]-1)
	return c


#Identity matrix
def eye(n,d=None):
	""" Creates an identity TT-matrix"""
	c = matrix()
	c.tt = tensor()
	if d is None:
		n0=np.asanyarray(n,dtype=np.int32)
		c.tt.d=n0.size
	else:
		n0 = np.asanyarray([n]*d,dtype=np.int32)
		c.tt.d = d
	c.n = n0.copy()
	c.m = n0.copy()
	c.tt.n = (c.n)*(c.m)
	c.tt.r = np.ones((c.tt.d+1,),dtype=np.int32)
	c.tt.get_ps()
	c.tt.alloc_core()
	for i in xrange(c.tt.d):
		c.tt.core[c.tt.ps[i]-1:c.tt.ps[i+1]-1] = np.eye(c.n[i]).flatten()
	return c

#Arbitrary multilevel Toeplitz matrix
def Toeplitz(x, d=None, D=None, kind='F'):
    """ Creates multilevel Toeplitz TT-matrix with D levels.
        
        Possible matrix types:
        'F' - full Toeplitz matrix,             size(x) = 2^{d+1}
        'C' - circulant matrix,                 size(x) = 2^d
        'L' - lower triangular Toeplitz matrix, size(x) = 2^d
        'U' - upper triangular Toeplitz matrix, size(x) = 2^d
        
        Sample call for one-level Toeplitz matrix:
          T = tt.Toeplitz(x)
        
        Sample call for one-level circulant matrix:
          T = tt.Toeplitz(x, kind='C')
        
        Sample call for three-level upper-triangular Toeplitz matrix:
          T = tt.Toeplitz(x, D=3, kind='U')
          
        Sample call for two-level mixed-type Toeplitz matrix:
          T = tt.Toeplitz(x, kind=['L', 'U'])
        
        Sample call for two-level mixed-size Toeplitz matrix:
          T = tt.Toeplitz(x, [3, 4], kind='C')
    """
    
    # checking for arguments consistency
    def check_kinds(D, kind):
        if D % len(kind) == 0:
            kind.extend(kind * (D / len(kind) - 1)) 
        if len(kind) != D:
            raise ValueError("Must give proper amount of matrix kinds (one or D, for example)")
    
    kind = list(kind)
    if not set(kind).issubset(['F', 'C', 'L', 'U']):
        raise ValueError("Toeplitz matrix kind must be one of F, C, L, U.")
    if d is None:
        if D is None:
            D = len(kind)
        if x.d % D:
            raise ValueError("x.d must be divisible by D when d is not specified!")
        if len(kind) == 1:
            d = np.array([x.d / D - (1 if kind[0] == 'F' else 0)] * D, dtype=np.int32)
            kind = kind * D
        else:
            check_kinds(D, kind)
            if set(kind).issubset(['F']):
                d = np.array([x.d / D - 1] * D, dtype=np.int32)
            elif set(kind).issubset(['C', 'L', 'U']):
                d = np.array([x.d / D] * D, dtype=np.int32)
            else:
                raise ValueError("Only similar matrix kinds (only F or only C, L and U) are accepted when d is not specified!")
    elif d is not None:
        d = np.asarray(d, dtype=np.int32)
        if D is None:
            D = len(d)
        if D != len(d):
            raise ValueError("D must be equal to len(d)")
        check_kinds(D, kind)
        if np.sum([d + (1 if knd == 'F' else 0) for knd in kind]) != x.d:
            raise ValueError("Dimensions inconsistency: x.d != d_1 + d_2 + ... + d_D")
    
    # predefined matrices and tensors:
    I = [[1, 0], [0, 1]]
    J = [[0, 1], [0, 0]]
    JT= [[0, 0], [1, 0]]
    H = [[0, 1], [1, 0]]
    S = np.array([[[0], [1]], [[1], [0]]]).transpose() # 2 x 2 x 1
    P = np.zeros((2, 2, 2, 2))
    P[:, :, 0, 0] = I; P[:, :, 1, 0] = H
    P[:, :, 0, 1] = H; P[:, :, 1, 1] = I
    P = np.transpose(P) # 2 x 2! x 2 x 2 x '1'
    Q = np.zeros((2, 2, 2, 2))
    Q[:, :, 0, 0] = I; Q[:, :, 1, 0] = JT
    Q[:, :, 0, 1] = JT
    Q = np.transpose(Q) # 2 x 2! x 2 x 2 x '1'
    R = np.zeros((2, 2, 2, 2))
    R[:, :, 1, 0] = J
    R[:, :, 0, 1] = J; R[:, :, 1, 1] = I;
    R = np.transpose(R) # 2 x 2! x 2 x 2 x '1'
    W = np.zeros([2] * 5) # 2 x 2! x 2 x 2 x 2
    W[0, :, :, 0, 0] = W[1, :, :, 1, 1] = I
    W[0, :, :, 1, 0] = W[0, :, :, 0, 1] = JT
    W[1, :, :, 1, 0] = W[1, :, :, 0, 1] = J
    W = np.transpose(W) # 2 x 2! x 2 x 2 x 2
    V = np.zeros((2, 2, 2, 2))
    V[0, :, :, 0] = I
    V[0, :, :, 1] = JT
    V[1, :, :, 1] = J
    V = np.transpose(V) # '1' x 2! x 2 x 2 x 2
    
    crs = []
    xcrs = tensor.to_list(x)
    dp = 0 # dimensions passed
    for j in xrange(D):
        currd = d[j]
        xcr = xcrs[dp]
        cr = np.tensordot(V, xcr, (0, 1)) # 
        cr = cr.transpose(3, 0, 1, 2, 4)  # <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
        cr = cr.reshape((x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <r_dp| x 2 x 2 x |2r_{dp+1}>
        dp += 1
        crs.append(cr)
        for i in xrange(1, currd - 1):
            xcr = xcrs[dp]
            cr = np.tensordot(W, xcr, (1, 1)) # (<2| x 2 x 2 x |2>) x <r_dp| x |r_{dp+1}>
            cr = cr.transpose([0, 4, 1, 2, 3, 5]) # <2| x <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            dp += 1
            crs.append(cr)
        if kind[j] == 'F':
            xcr = xcrs[dp] # r_dp x 2 x r_{dp+1}
            cr = np.tensordot(W, xcr, (1, 1)).transpose([0, 4, 1, 2, 3, 5])
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            dp += 1
            xcr = xcrs[dp] # r_dp x 2 x r_{dp+1}
            tmp = np.tensordot(S, xcr, (1, 1)) # <2| x |1> x <r_dp| x |r_{dp+1}>
            #tmp = tmp.transpose([0, 2, 1, 3]) # TODO: figure out WHY THE HELL this spoils everything
            tmp = tmp.reshape((2 * x.r[dp], x.r[dp+1]), order='F') # <2r_dp| x |r_{dp+1}>
            cr = np.tensordot(cr, tmp, (3, 0)) # <2r_{dp-1}| x 2 x 2 x |r_{dp+1}>
            dp += 1
            crs.append(cr)
        else:
            dotcore = None
            if kind[j] == 'C':
                dotcore = P
            elif kind[j] == 'L':
                dotcore = Q
            elif kind[j] == 'U':
                dotcore = R
            xcr = xcrs[dp] # r_dp x 2 x r_{dp+1}
            cr = np.tensordot(dotcore, xcr, (1, 1)) # <2| x 2 x 2 x |'1'> x <r_dp| x |r_{dp+1}>
            cr = cr.transpose([0, 3, 1, 2, 4]) # <2| x <r_dp| x 2 x 2 x |r_{dp+1}>
            cr = cr.reshape((2 * x.r[dp], 2, 2, x.r[dp+1]), order='F')
            dp += 1
            crs.append(cr)
    return matrix.from_list(crs)


#Laplace operator
def qlaplace_dd(d):
    """Creates a QTT representation of the Laplace operator"""
    res = matrix()
    d0 = d[::-1]
    D = len(d0)
    I = np.eye(2)
    J = np.array([[0,1],[0,0]])
    cr=[]
    if D is 1:
        for k in xrange(1,d0[0]+1):
            if k is 1:
                cur_core=np.zeros((1,2,2,3));
                cur_core[:,:,:,0]=2*I-J-J.T;
                cur_core[:,:,:,1]=-J;
                cur_core[:,:,:,2]=-J.T;
            elif k is d0[0]:
                cur_core=np.zeros((3,2,2,1));
                cur_core[0,:,:,0]=I;
                cur_core[1,:,:,0]=J.T;
                cur_core[2,:,:,0]=J;
            else:
                cur_core=np.zeros((3,2,2,3));
                cur_core[0,:,:,0]=I;
                cur_core[1,:,:,1]=J;
                cur_core[2,:,:,2]=J.T;
                cur_core[1,:,:,0]=J.T;
                cur_core[2,:,:,0]=J;
            cr.append(cur_core)
    else:
        for k in xrange(D):
            for kappa in xrange(1,d0[k]+1):
                if kappa is 1:
                    if k is 0:
                        cur_core=np.zeros((1,2,2,4));
                        cur_core[:,:,:,0]=2*I-J-J.T;
                        cur_core[:,:,:,1]=-J;
                        cur_core[:,:,:,2]=-J.T;
                        cur_core[:,:,:,3]=I;
                    elif k is D-1:
                        cur_core=np.zeros((2,2,2,3));
                        cur_core[0,:,:,0]=2*I-J-J.T;
                        cur_core[0,:,:,1]=-J;
                        cur_core[0,:,:,2]=-J.T;
                        cur_core[1,:,:,0]=I;
                    else:
                        cur_core=np.zeros((2,2,2,4));
                        cur_core[0,:,:,0]=2*I-J-J.T;
                        cur_core[0,:,:,1]=-J;
                        cur_core[0,:,:,2]=-J.T;
                        cur_core[0,:,:,3]=I;
                        cur_core[1,:,:,0]=I;
                elif kappa is d0[k]:
                    if k is D-1:
                        cur_core=np.zeros((3,2,2,1));
                        cur_core[0,:,:,0]=I;
                        cur_core[1,:,:,0]=J.T;
                        cur_core[2,:,:,0]=J;
                    else:
                        cur_core=np.zeros((4,2,2,2));
                        cur_core[3,:,:,0]=I;
                        cur_core[0,:,:,1]=I;
                        cur_core[1,:,:,1]=J.T
                        cur_core[2,:,:,1]=J;
                else:
                    if k is D-1:
                        cur_core=np.zeros((3,2,2,3));
                        cur_core[0,:,:,0]=I;
                        cur_core[1,:,:,1]=J;
                        cur_core[2,:,:,2]=J.T;
                        cur_core[1,:,:,0]=J.T;
                        cur_core[2,:,:,0]=J;
                    else:
                        cur_core=np.zeros((4,2,2,4));
                        cur_core[0,:,:,0]=I;
                        cur_core[1,:,:,1]=J;
                        cur_core[2,:,:,2]=J.T;
                        cur_core[1,:,:,0]=J.T;
                        cur_core[2,:,:,0]=J;
                        cur_core[3,:,:,3]=I;
                cr.append(cur_core)
    return matrix.from_list(cr)


def xfun(n,d=None):
    """ Create a QTT-representation of 0:prod(n) vector"""
    # call examples:
    #   tt.xfun(2, 5)         # create 2 x 2 x 2 x 2 x 2 TT-tensor
    #   tt.xfun(3)            # create [0, 1, 2] one-dimensional TT-tensor
    #   tt.xfun([3, 5, 7], 2) # create 3 x 5 x 7 x 3 x 5 x 7 TT-tensor
    if isinstance(n, (int, long)):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    if d == 1:
        return tensor.from_list([np.reshape(np.arange(n0[0]), (1, n0[0], 1))])
    cr=[]
    cur_core = np.ones((1,n0[0],2))
    cur_core[0,:,0] = np.arange(n0[0])
    cr.append(cur_core)
    ni = float(n0[0])
    for i in xrange(1, d - 1):
        cur_core = np.zeros((2,n0[i],2))
        for j in xrange(n0[i]):
            cur_core[:, j, :] = np.eye(2)
        cur_core[1, :, 0] = ni * np.arange(n0[i])
        ni *= n0[i]
        cr.append(cur_core)
    cur_core = np.ones((2, n0[d - 1], 1))
    cur_core[1,:,0] = ni*np.arange(n0[d - 1])
    cr.append(cur_core)
    return tensor.from_list(cr)


def sin(d, alpha=1.0, phase=0.0):
    """ Create TT-tensor for sin(alpha n + phi)"""
    cr = []
    cur_core = np.zeros([1, 2, 2], dtype=np.float)
    cur_core[0, 0, :] = [math.cos(phase)        , math.sin(phase)        ]
    cur_core[0, 1, :] = [math.cos(alpha + phase), math.sin(alpha + phase)]
    cr.append(cur_core)
    for i in xrange(1, d-1):
        cur_core = np.zeros([2, 2, 2], dtype=np.float)
        cur_core[0, 0, :] = [1.0                     , 0.0                      ]
        cur_core[1, 0, :] = [0.0                     , 1.0                      ]
        cur_core[0, 1, :] = [ math.cos(alpha * 2 ** i), math.sin(alpha * 2 ** i)]
        cur_core[1, 1, :] = [-math.sin(alpha * 2 ** i), math.cos(alpha * 2 ** i)]
        cr.append(cur_core)
    cur_core = np.zeros([2, 2, 1], dtype=np.float)
    cur_core[0, :, 0] = [0.0, math.sin(alpha * 2 ** (d-1))]
    cur_core[1, :, 0] = [1.0, math.cos(alpha * 2 ** (d-1))]
    cr.append(cur_core)
    return tensor.from_list(cr)


def cos(d, alpha=1.0, phase=0.0):
    """ Create TT-tensor for cos(alpha n + phi)"""
    return sin(d, alpha, phase + math.pi * 0.5)

def delta(n, d=None, center=0):
    """ Create TT-tensor for delta(x - x_0) """
    if isinstance(n, (int, long)):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    
    if center < 0:
        cind = [0] * d
    else:
        cind = []
        for i in xrange(d):
            cind.append(center % n0[i])
            center /= n0[i]
        if center > 0:
            cind = [0] * d
    
    cr=[]
    for i in xrange(d):
        cur_core = np.zeros((1, n0[i], 1))
        cur_core[0, cind[i], 0] = 1
        cr.append(cur_core)
    return tensor.from_list(cr)

def stepfun(n, d=None, center=1, direction=1):
    """ Create TT-tensor for Heaviside step function H(x - x_0)
    
    H(x) = 1 when x >= 0 and = 0 when x < 0.
    For negative direction H(x_0 - x) is approximated. """
    if isinstance(n, (int, long)):
        n = [n]
    if d is None:
        n0 = np.asanyarray(n, dtype=np.int32)
    else:
        n0 = np.array(n * d, dtype=np.int32)
    d = n0.size
    N = np.prod(n0)
    
    if center >= N and direction < 0 or center <= 0 and direction > 0:
        return ones(n0)
    
    if center <= 0 and direction < 0 or center >= N and direction > 0:
        raise ValueError("Heaviside function with specified center and direction gives zero tensor!")
    if direction > 0:   
        center = N - center
    cind = []
    for i in xrange(d):
        cind.append(center % n0[i])
        center /= n0[i]
    
    def gen_notx(currcind, currn):
        return [0.0] * (currn - currcind) + [1.0] * currcind
    def gen_notx_rev(currcind, currn):
        return [1.0] * currcind + [0.0] * (currn - currcind)
    def gen_x(currcind, currn):
        result = [0.0] * currn
        result[currn - currcind - 1] = 1.0
        return result
    def gen_x_rev(currcind, currn):
        result = [0.0] * currn
        result[currcind] = 1.0
        return result
    
    if direction > 0:
        x = gen_x
        notx = gen_notx
    else:
        x    = gen_x_rev
        notx = gen_notx_rev
    
    crs = []
    prevrank = 1
    for i in range(d)[::-1]:
        break_further = max([0] + cind[:i])
        nextrank = 2 if break_further else 1
        one = [1] * n0[i]
        cr = np.zeros([nextrank, n0[i], prevrank], dtype=np.float)
        tempx = x(cind[i], n0[i])
        tempnotx = notx(cind[i], n0[i])
        # high-conditional magic
        if not break_further:
            if cind[i]:
                if prevrank > 1:
                    cr[0, :, 0] = one
                    cr[0, :, 1] = tempnotx
                else:
                    cr[0, :, 0] = tempnotx
            else:
                cr[0, :, 0] = one
        else:
            if prevrank > 1:
                cr[0, :, 0] = one
                if cind[i]:
                    cr[0, :, 1] = tempnotx
                    cr[1, :, 1] = tempx
                else:
                    cr[1, :, 1] = tempx
            else:
                if cind[i]:
                    cr[0, :, 0] = tempnotx
                    cr[1, :, 0] = tempx
                else:
                    nextrank = 1
                    cr = cr[:1, :, :]
                    cr[0, :, 0] = tempx
        prevrank = nextrank
        crs.append(cr)
    return tensor.from_list(crs[::-1])        

