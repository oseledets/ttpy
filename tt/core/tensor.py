import numpy as np
from numpy import prod, reshape, nonzero, size, sqrt
import math
from math import sqrt
from numbers import Number
import tt_f90
import core_f90



####################################################################################################
#############################################          #############################################
############################################   tensor   ############################################
#############################################          #############################################
####################################################################################################

#The main class for working with TT-tensors
class tensor:
    """Construct new TT-tensor.
        
    When called with no arguments, creates dummy object which can be filled from outside.
    
    When ``a`` is specified, computes approximate decomposition of array ``a`` with accuracy ``eps``:
    
    :param a: A tensor to approximate.
    :type a: ndarray
    
    >>> a = numpy.sin(numpy.arange(2 ** 10)).reshape([2] * 10, order='F')
    >>> a = tt.tensor(a)
    >>> a.r
    array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=int32)
    >>> # now let's try different accuracy
    >>> b = numpy.random.rand(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    >>> btt = tt.tensor(b, 1E-14)
    >>> btt.r
    array([ 1,  2,  4,  8, 16, 32, 16,  8,  4,  2,  1], dtype=int32)
    >>> btt = tt.tensor(b, 1E-1)
    >>> btt.r
    array([ 1,  2,  4,  8, 14, 20, 14,  8,  4,  2,  1], dtype=int32)
    
    Attributes:
    
    d : int
        Dimensionality of the tensor.
    n : ndarray of shape (d,)
        Mode sizes of the tensor: if :math:`n_i=\\texttt{n[i-1]}`, then the tensor has shape :math:`n_1\\times\ldots\\times n_d`.
    r : ndarray of shape (d+1,)
        TT-ranks of current TT decomposition of the tensor.
    core : ndarray
        Flatten (Fortran-ordered) TT cores stored sequentially in a one-dimensional array.
        To get a list of three-dimensional cores, use ``tt.tensor.to_list(my_tensor)``.
    """
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
        """Generate TT-tensor object from given TT cores.
        
        :param a: List of TT cores.
        :type a: list
        :returns: tensor -- TT-tensor constructed from the given cores.
        
        """
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
        """Return list of TT cores a TT decomposition consists of.
        
        :param tt: TT-tensor.
        :type a: tensor
        :returns: list -- list of ``tt.d`` three-dimensional cores, ``i``-th core is an ndarray of shape ``(tt.r[i], tt.n[i], tt.r[i+1])``.
        """
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
    
    @property
    def is_complex(self):
        return np.iscomplexobj(self.core)
    
    def _matrix__complex_op(self, op):
        return self.__complex_op(op)
    
    def __complex_op(self, op):
        crs = tensor.to_list(self)
        newcrs = []
        cr = crs[0]
        rl, n, rr = cr.shape
        newcr = np.zeros((rl, n, rr * 2), dtype=np.float)
        newcr[:, :, :rr] = np.real(cr)
        newcr[:, :, rr:] = np.imag(cr)
        newcrs.append(newcr)
        for i in xrange(1, self.d - 1):
            cr = crs[i]
            rl, n, rr = cr.shape
            newcr = np.zeros((rl * 2, n, rr * 2), dtype=np.float)
            newcr[:rl, :, :rr] = newcr[rl:, :, rr:] = np.real(cr)
            newcr[:rl, :, rr:] =  np.imag(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcrs.append(newcr)
        cr = crs[-1]
        rl, n, rr = cr.shape
        if op in ['R', 'r', 'Re']:
            # get real part
            newcr = np.zeros((rl * 2, n, rr), dtype=np.float)
            newcr[:rl, :, :] =  np.real(cr)
            newcr[rl:, :, :] = -np.imag(cr)
        elif op in ['I', 'i', 'Im']:
            # get imaginary part
            newcr = np.zeros((rl * 2, n, rr), dtype=np.float)
            newcr[:rl, :, :] =  np.imag(cr)
            newcr[rl:, :, :] =  np.real(cr)
        elif op in ['A', 'B', 'all', 'both']:
            # get both parts (increase dimensionality)
            newcr = np.zeros((rl * 2, n, 2 * rr), dtype=np.float)
            newcr[:rl, :, :rr] =  np.real(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcr[:rl, :, rr:] =  np.imag(cr)
            newcr[rl:, :, rr:] =  np.real(cr)
            newcrs.append(newcr)
            newcr = np.zeros((rr * 2, 2, 1), dtype=np.float)
            newcr[:rr, 0, :] = newcr[rr:, 1, :] = 1.0
        elif op in ['M']:
            # get matrix modificated for real-arithm. solver
            newcr = np.zeros((rl * 2, n, 2 * rr), dtype=np.float)
            newcr[:rl, :, :rr] =  np.real(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcr[:rl, :, rr:] =  np.imag(cr)
            newcr[rl:, :, rr:] =  np.real(cr)
            newcrs.append(newcr)
            newcr = np.zeros((rr * 2, 4, 1), dtype=np.float)
            newcr[:rr, [0, 3], :] = 1.0
            newcr[rr:, 1, :] =  1.0
            newcr[rr:, 2, :] = -1.0
        else:
            raise ValueError("Unexpected parameter " + op + " at tt.tensor.__complex_op")
        newcrs.append(newcr)
        return tensor.from_list(newcrs)
    
    def real(self):
        """Get real part of a TT-tensor."""
        return self.__complex_op('Re')
    
    def imag(self):
        """Get imaginary part of a TT-tensor."""
        return self.__complex_op('Im')
    
    def c2r(self):
        """Get real tensor from complex one suitable for solving complex linear system with real solver.
        
        For tensor :math:`X(i_1,\\ldots,i_d) = \\Re X + i\\Im X` returns (d+1)-dimensional tensor 
        of form :math:`[\\Re X\\ \\Im X]`. This function is useful for solving complex linear system
        :math:`\\mathcal{A}X = B` with real solver by transforming it into
        
        .. math::
           \\begin{bmatrix}\\Re\\mathcal{A} & -\\Im\\mathcal{A} \\\\ 
                           \\Im\\mathcal{A} &  \\Re\\mathcal{A}  \\end{bmatrix}
           \\begin{bmatrix}\\Re X \\\\ \\Im X\\end{bmatrix} = 
           \\begin{bmatrix}\\Re B \\\\ \\Im B\\end{bmatrix}.
        
        """
        return self.__complex_op('both')
    
    def r2c(self):
        """Get complex tensor from real one made by ``tensor.c2r()``.
        
        For tensor :math:`\\tilde{X}(i_1,\\ldots,i_d,i_{d+1})` returns complex tensor
        
        .. math::
           X(i_1,\\ldots,i_d) = \\tilde{X}(i_1,\\ldots,i_d,0) + i\\tilde{X}(i_1,\\ldots,i_d,1).
        
        >>> a = tt.rand(2,10,5) + 1j * tt.rand(2,10,5)
        >>> (a.c2r().r2c() - a).norm() / a.norm()
        7.310562016615692e-16
        
        """
        tmp = self.copy()
        newcore = np.array(tmp.core, dtype=np.complex)
        cr = newcore[tmp.ps[-2]-1:tmp.ps[-1]-1]
        cr = cr.reshape((tmp.r[-2], tmp.n[-1], tmp.r[-1]), order='F')
        cr[:, 1, :] *= 1j
        newcore[tmp.ps[-2]-1:tmp.ps[-1]-1] = cr.flatten('F')
        tmp.core = newcore
        return sum(tmp, axis=tmp.d-1)
    
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
        if np.iscomplexobj(self.core):
            tt_f90.tt_f90.ztt_write_wrapper(self.n,self.r,self.ps,self.core,fname)
        else:
            tt_f90.tt_f90.dtt_write_wrapper(self.n,self.r,self.ps,np.real(self.core),fname)

    def full(self):
        """Returns full array (uncompressed).
        
        .. warning::
           TT compression allows to keep in memory tensors much larger than ones PC can handle in 
           raw format. Therefore this function is quite unsafe; use it at your own risk.
       
       :returns: numpy.ndarray -- full tensor.
       
       """
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
       """Applies TT rounding procedure to the TT-tensor and **returns rounded tensor**.
       
       :param eps: Rounding accuracy.
       :type eps: float
       :returns: tensor -- rounded TT-tensor.
       
       Usage example:
       
       >>> a = tt.ones(2, 10)
       >>> b = a + a
       >>> print b.r
       array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=int32)
       >>> b = b.round(1E-14)
       >>> print b.r
       array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)
       
       """
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
        from tt import matrix
        return matrix.from_list(res)

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
        from tt import matrix
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
        """ Calculates the mean rank of a TT-tensor."""
        if not np.all(self.n):
            return 0
        # Solving quadratic equation ar^2 + br + c = 0;
        a = np.sum(self.n[1:-1])
        b = self.n[0] + self.n[-1]
        c = - np.sum(self.n * self.r[1:] * self.r[:-1])
        D = b ** 2 - 4 * a * c
        r = 0.5 * (-b + sqrt(D)) / a
        return r

