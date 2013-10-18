import numpy as np
from numpy import prod, reshape, nonzero, size, sqrt
import math
from math import sqrt
from numbers import Number
import tt_f90
import core_f90

from tensor import tensor

####################################################################################################
#############################################          #############################################
############################################   matrix   ############################################
#############################################          #############################################
####################################################################################################

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
                n1 = np.array(n,dtype=np.int32)
                m1 = np.array(m,dtype=np.int32)
            self.n = n1
            self.m = m1
            self.tt = tensor()
            self.tt.core = a.core.copy()
            self.tt.ps = a.ps.copy()
            self.tt.r = a.r.copy()
            self.tt.n = a.n.copy()
            self.tt.d = self.tt.n.size
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
    
    def write(self, fname):
        self.tt.write(fname)

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
        return self.tt.is_complex
        
    def real(self):
        """Return real part of a matrix."""
        return matrix(self.tt.real(), n=self.n, m=self.m)
    
    def imag(self):
        """Return imaginary part of a matrix."""
        return matrix(self.tt.imag(), n=self.n, m=self.m)
    
    def c2r(self):
        """Get real matrix from complex one suitable for solving complex linear system with real solver.
        
        For matrix :math:`M(i_1,j_1,\\ldots,i_d,j_d) = \\Re M + i\\Im M` returns (d+1)-dimensional matrix
        :math:`\\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,i_{d+1},j_{d+1})` of form 
        :math:`\\begin{bmatrix}\\Re M & -\\Im M \\\\ \\Im M &  \\Re M  \\end{bmatrix}`. This function 
        is useful for solving complex linear system :math:`\\mathcal{A}X = B` with real solver by 
        transforming it into
        
        .. math::
           \\begin{bmatrix}\\Re\\mathcal{A} & -\\Im\\mathcal{A} \\\\ 
                           \\Im\\mathcal{A} &  \\Re\\mathcal{A}  \\end{bmatrix}
           \\begin{bmatrix}\\Re X \\\\ \\Im X\\end{bmatrix} = 
           \\begin{bmatrix}\\Re B \\\\ \\Im B\\end{bmatrix}.
        
        """
        return matrix(a=self.tt.__complex_op('M'), n=np.concatenate((self.n, [2])), m=np.concatenate((self.m, [2])))
    
    def r2c(self):
        """Get complex matrix from real one made by ``matrix.c2r()``.
        
        For matrix :math:`\\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,i_{d+1},j_{d+1})` returns complex matrix
        
        .. math::
           M(i_1,j_1,\\ldots,i_d,j_d) = \\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,0,0) + i\\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,1,0).
        
        """
        tmp = self.tt.copy()
        newcore = np.array(tmp.core, dtype=np.complex)
        cr = newcore[tmp.ps[-2]-1:tmp.ps[-1]-1]
        cr = cr.reshape((tmp.r[-2], tmp.n[-1], tmp.r[-1]), order='F')
        cr[:, 1, :] *= 1j
        cr[:, 2:, :] = 0.0
        newcore[tmp.ps[-2]-1:tmp.ps[-1]-1] = cr.flatten('F')
        tmp.core = newcore
        return matrix(sum(tmp, axis=tmp.d-1), n=self.n, m=self.m)

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
        c = matrix()
        c.n = self.n.copy()
        c.m = other.m.copy()
        tt = tensor()
        tt.d = self.tt.d 
        tt.n = c.n * c.m
        if self.is_complex or other.is_complex:
            tt.r = core_f90.core.zmat_mat(self.n,self.m,other.m,np.array(self.tt.core, dtype=np.complex),np.array(other.tt.core, dtype=np.complex),self.tt.r,other.tt.r)
            tt.core = core_f90.core.zresult_core.copy()
        else:
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
    
    def rmean(self):
        return self.tt.rmean()
