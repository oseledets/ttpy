import numpy as _np
import math as _math
import copy as _cp
import fractions as _fractions
from numbers import Number as _Number
import tt_f90 as _tt_f90
import core_f90 as _core_f90

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
    
    :param eps: Approximation accuracy
    :type a: float

    :param rmax: Maximal rank
    :type rmax: int
    
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
    def __init__(self, a=None, eps=1e-14, rmax=100000):
        
        if a is None:
            self.core = 0
            self.d = 0
            self.n = 0
            self.r = 0
            self.ps = 0
            return
        self.d = a.ndim
        self.n = _np.array(a.shape,dtype=_np.int32)
        r = _np.zeros((self.d+1,),dtype=_np.int32)
        ps = _np.zeros((self.d+1,),dtype=_np.int32)
        
        if ( _np.iscomplex(a).any() ):
            if rmax is not None:
                self.r, self.ps = _tt_f90.tt_f90.zfull_to_tt(a.flatten('F'), self.n, self.d, eps, rmax)
            else:
                self.r, self.ps = _tt_f90.tt_f90.zfull_to_tt(a.flatten('F'), self.n, self.d, eps)

            self.core = _tt_f90.tt_f90.zcore.copy()
        else:
            if rmax is not None:
                self.r,self.ps = _tt_f90.tt_f90.dfull_to_tt(_np.real(a).flatten('F'),self.n,self.d,eps,rmax)
            else:
                self.r,self.ps = _tt_f90.tt_f90.dfull_to_tt(_np.real(a).flatten('F'),self.n,self.d,eps)
            self.core = _tt_f90.tt_f90.core.copy()

        _tt_f90.tt_f90.tt_dealloc()        
    
    @staticmethod
    def from_list(a,order='F'):
        """Generate TT-tensor object from given TT cores.
        
        :param a: List of TT cores.
        :type a: list
        :returns: tensor -- TT-tensor constructed from the given cores.
        
        """
        d = len(a) #Number of cores
        res = tensor()
        n = _np.zeros(d,dtype=_np.int32)
        r = _np.zeros(d+1,dtype=_np.int32)
        cr = _np.array([])
        for i in xrange(d):
            cr = _np.concatenate((cr,a[i].flatten(order)))
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
    def erank(self):
        """ Effective rank of the TT-tensor """
        r = self.r
        n = self.n
        d = self.d
        if d <= 1:
            er = 0e0
        else:
            sz = _np.dot(n * r[0:d], r[1:])
            if sz == 0:
                er = 0e0
            else:
                b = r[0] * n[0] + n[d-1] * r[d]
                if d is 2:
                    er = sz * 1.0/b
                else:
                    a = _np.sum(n[1:d-1])
                    er = (_np.sqrt(b * b + 4 * a * sz) - b)/(2*a)
        return er
    
    def __getitem__(self, index):
        """Get element of the TT-tensor.

        :param index: array_like.
        :returns: number -- an element of the tensor."""
        if len(index) != self.d:
            print("Incorrect index length.")
            return
        prefix = 1
        for i in xrange(self.d):
            cur_core = self.core[self.ps[i]-1:self.ps[i+1]-1]
            cur_core = cur_core.reshape((self.r[i], self.n[i], self.r[i+1]), order='F')
            cur_core = cur_core[:, index[i], :]
            prefix = _np.dot(prefix, cur_core)
        return prefix[0][0]
    
    @property
    def is_complex(self):
        return _np.iscomplexobj(self.core)
    
    def _matrix__complex_op(self, op):
        return self.__complex_op(op)
    
    def __complex_op(self, op):
        crs = tensor.to_list(self)
        newcrs = []
        cr = crs[0]
        rl, n, rr = cr.shape
        newcr = _np.zeros((rl, n, rr * 2), dtype=_np.float)
        newcr[:, :, :rr] = _np.real(cr)
        newcr[:, :, rr:] = _np.imag(cr)
        newcrs.append(newcr)
        for i in xrange(1, self.d - 1):
            cr = crs[i]
            rl, n, rr = cr.shape
            newcr = _np.zeros((rl * 2, n, rr * 2), dtype=_np.float)
            newcr[:rl, :, :rr] = newcr[rl:, :, rr:] = _np.real(cr)
            newcr[:rl, :, rr:] =  _np.imag(cr)
            newcr[rl:, :, :rr] = -_np.imag(cr)
            newcrs.append(newcr)
        cr = crs[-1]
        rl, n, rr = cr.shape
        if op in ['R', 'r', 'Re']:
            # get real part
            newcr = _np.zeros((rl * 2, n, rr), dtype=_np.float)
            newcr[:rl, :, :] =  _np.real(cr)
            newcr[rl:, :, :] = -_np.imag(cr)
        elif op in ['I', 'i', 'Im']:
            # get imaginary part
            newcr = _np.zeros((rl * 2, n, rr), dtype=_np.float)
            newcr[:rl, :, :] =  _np.imag(cr)
            newcr[rl:, :, :] =  _np.real(cr)
        elif op in ['A', 'B', 'all', 'both']:
            # get both parts (increase dimensionality)
            newcr = _np.zeros((rl * 2, n, 2 * rr), dtype=_np.float)
            newcr[:rl, :, :rr] =  _np.real(cr)
            newcr[rl:, :, :rr] = -_np.imag(cr)
            newcr[:rl, :, rr:] =  _np.imag(cr)
            newcr[rl:, :, rr:] =  _np.real(cr)
            newcrs.append(newcr)
            newcr = _np.zeros((rr * 2, 2, 1), dtype=_np.float)
            newcr[:rr, 0, :] = newcr[rr:, 1, :] = 1.0
        elif op in ['M']:
            # get matrix modificated for real-arithm. solver
            newcr = _np.zeros((rl * 2, n, 2 * rr), dtype=_np.float)
            newcr[:rl, :, :rr] =  _np.real(cr)
            newcr[rl:, :, :rr] = -_np.imag(cr)
            newcr[:rl, :, rr:] =  _np.imag(cr)
            newcr[rl:, :, rr:] =  _np.real(cr)
            newcrs.append(newcr)
            newcr = _np.zeros((rr * 2, 4, 1), dtype=_np.float)
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
        newcore = _np.array(tmp.core, dtype=_np.complex)
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
        if _np.iscomplexobj(self.core):
            _tt_f90.tt_f90.ztt_write_wrapper(self.n,self.r,self.ps,self.core,fname)
        else:
            _tt_f90.tt_f90.dtt_write_wrapper(self.n,self.r,self.ps,_np.real(self.core),fname)

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
            sz = _np.concatenate(([self.r[0]],sz))
        if self.r[self.d] > 1:
            sz = _np.concatenate(([self.r[self.d]],sz))
        #a = _np.zeros(sz,order='F')
        if ( _np.iscomplex(self.core).any() ):
            a = _tt_f90.tt_f90.ztt_to_full(self.n,self.r,self.ps,self.core,_np.prod(sz)) 
        else:
            a = _tt_f90.tt_f90.dtt_to_full(self.n,self.r,self.ps,_np.real(self.core),_np.prod(sz)) 
        a = a.reshape(sz,order='F')
        #import ipdb; ipdb.set_trace()
        return a

    def __add__(self,other):
        if other is None:
            return self
        c = tensor()
        c.r = _np.zeros((self.d+1,),dtype=_np.int32)
        c.ps = _np.zeros((self.d+1,),dtype=_np.int32)
        c.n = self.n
        c.d = self.d
        if ( _np.iscomplex(self.core).any() or _np.iscomplex(other.core).any()):
            c.r,c.ps = _tt_f90.tt_f90.ztt_add(self.n,self.r,other.r,self.ps,other.ps,self.core+0j,other.core+0j)
            c.core = _tt_f90.tt_f90.zcore.copy()
        else:
             #This could be a real fix in the case we fell to the real world
            c.r,c.ps = _tt_f90.tt_f90.dtt_add(self.n,self.r,other.r,self.ps,other.ps,_np.real(self.core),_np.real(other.core))
            c.core = _tt_f90.tt_f90.core.copy()
        _tt_f90.tt_f90.tt_dealloc()
        return c

    def __radd__(self,other):
        if other is None:
            return self
        return other + self


    #@profile
    def round(self, eps, rmax = 1000000):
       """Applies TT rounding procedure to the TT-tensor and **returns rounded tensor**.
       
       :param eps: Rounding accuracy.
       :type eps: float
       :param rmax: Maximal rank
       :type rmax: int
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
       if ( _np.iscomplex(self.core).any() ):
           _tt_f90.tt_f90.ztt_compr2(c.n,c.r,c.ps,self.core,eps,rmax)
           c.core = _tt_f90.tt_f90.zcore.copy()
       else:
           _tt_f90.tt_f90.dtt_compr2(c.n,c.r,c.ps,self.core,eps,rmax)
           c.core=_tt_f90.tt_f90.core.copy()
       _tt_f90.tt_f90.tt_dealloc()
       return c

    #@profile
    def norm(self):
        if ( _np.iscomplex(self.core).any() ):
            nrm = _tt_f90.tt_f90.ztt_nrm(self.n,self.r,self.ps,self.core)
        else:
            nrm=_tt_f90.tt_f90.dtt_nrm(self.n,self.r,self.ps,_np.real(self.core))
        return nrm
	
    def __rmul__(self,other):
       c = tensor()
       c.d = self.d
       c.n = self.n
       if isinstance(other,_Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0]-1:c.ps[1]-1]
            new_core = new_core * other
            c.core = _np.array(c.core,dtype=new_core.dtype)
            c.core[c.ps[0]-1:c.ps[1]-1] = new_core
       else:
           c =_hdm(self,other)
       return c
	
    def __mul__(self,other):
        c = tensor()
        c.d = self.d
        c.n = self.n
        if isinstance(other,_Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0]-1:c.ps[1]-1]
            new_core = new_core * other
            c.core = _np.array(c.core,dtype=new_core.dtype)
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
        c.n = _np.concatenate((a.n,b.n))
        c.r = _np.concatenate((a.r[0:a.d],b.r[0:b.d+1]))
        c.get_ps()
        c.core = _np.concatenate((a.core,b.core))
        return c

    def __dot__(self,other):
        r1 = self.r
        r2 = other.r
        d = self.d
        if ( _np.iscomplex(self.core).any() or _np.iscomplex(other.core).any()):
            dt = _np.zeros(r1[0]*r2[0]*r1[d]*r2[d],dtype=_np.complex)
            dt = _tt_f90.tt_f90.ztt_dotprod(self.n,r1,r2,self.ps,other.ps,self.core+0j,other.core+0j,dt.size)
        else:
            dt = _np.zeros(r1[0]*r2[0]*r1[d]*r2[d])
            dt = _tt_f90.tt_f90.dtt_dotprod(self.n,r1,r2,self.ps,other.ps,_np.real(self.core),_np.real(other.core),dt.size)
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
            res_core = _np.zeros((r[i], n[i], n[i], r[i+1]), dtype = dtype)
            for s1 in xrange(r[i]):
                for s2 in xrange(r[i+1]):
                    res_core[s1, :, :, s2] = _np.diag(cur_core[s1, :, s2].reshape(n[i], order='F'))
            res.append(res_core)
        return matrix.from_list(res)
    
    def __neg__(self):
       return self*(-1)

    def get_ps(self):
        self.ps = _np.cumsum(_np.concatenate(([1],self.n*self.r[0:self.d]*self.r[1:self.d+1]))).astype(_np.int32)

    def alloc_core(self):
        self.core = _np.zeros((self.ps[self.d]-1,),dtype=_np.float)

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
        if not _np.all(self.n):
            return 0
        # Solving quadratic equation ar^2 + br + c = 0;
        a = _np.sum(self.n[1:-1])
        b = self.n[0] + self.n[-1]
        c = - _np.sum(self.n * self.r[1:] * self.r[:-1])
        D = b ** 2 - 4 * a * c
        r = 0.5 * (-b + _np.sqrt(D)) / a
        return r


####################################################################################################
#############################################          #############################################
############################################   matrix   ############################################
#############################################          #############################################
####################################################################################################

class matrix:
    def __init__(self, a=None, eps=1e-14, n=None, m=None, rmax = 100000):
       
        self.n = 0
        self.m = 0
        self.tt = tensor()
        
        if isinstance(a, tensor): #Convert from a tt-tensor
            if ( n is None or m is None):
                n1 = _np.sqrt(a.n).astype(_np.int32)
                m1 = _np.sqrt(a.n).astype(_np.int32)
            else:
                n1 = _np.array([n], dtype = _np.int32)
                m1 = _np.array([m], dtype = _np.int32)
            self.n = n1
            self.m = m1
            self.tt.core = a.core.copy()
            self.tt.ps = a.ps.copy()
            self.tt.r = a.r.copy()
            self.tt.n = a.n.copy()
            self.tt.d = self.tt.n.size
            return

        if isinstance(a, _np.ndarray): 
            d = a.ndim/2
            p = a.shape
            self.n = _np.array(p[:d], dtype = _np.int32)
            self.m = _np.array(p[d:], dtype = _np.int32)
            prm = _np.arange(2*d)
            prm = prm.reshape((d, 2), order='F')
            prm = prm.T
            prm = prm.flatten('F')
            sz = self.n * self.m
            b = a.transpose(prm).reshape(sz, order='F')
            self.tt = tensor(b, eps, rmax)
            return
        
        if isinstance(a, matrix):
            self.n = a.n.copy()
            self.m = a.m.copy()
            self.tt = a.tt.copy()
            return

    @staticmethod
    def from_list(a):
        d = len(a) #Number of cores
        res = matrix()
        n = _np.zeros(d,dtype=_np.int32)
        r = _np.zeros(d+1,dtype=_np.int32)
        m = _np.zeros(d,dtype=_np.int32)
        cr = _np.array([])
        for i in xrange(d):
            cr = _np.concatenate((cr,a[i].flatten('F')))
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
    def erank(self):
        return self.tt.erank
     
    @property
    def is_complex(self):  
        return self.tt.is_complex
        
    @property
    def T(self):
        """Transposed TT-matrix"""
        mycrs = matrix.to_list(self)
        trans_crs = []
        for cr in mycrs:
            trans_crs.append(_np.transpose(cr, [0, 2, 1, 3]))
        return matrix.from_list(trans_crs)

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
        return matrix(a=self.tt.__complex_op('M'), n=_np.concatenate((self.n, [2])), m=_np.concatenate((self.m, [2])))
    
    def r2c(self):
        """Get complex matrix from real one made by ``matrix.c2r()``.
        
        For matrix :math:`\\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,i_{d+1},j_{d+1})` returns complex matrix
        
        .. math::
           M(i_1,j_1,\\ldots,i_d,j_d) = \\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,0,0) + i\\tilde{M}(i_1,j_1,\\ldots,i_d,j_d,1,0).
        
        """
        tmp = self.tt.copy()
        newcore = _np.array(tmp.core, dtype=_np.complex)
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
        c.n = _np.asanyarray(self.n,dtype=_np.int32).copy()
        c.m = _np.asanyarray(self.m,dtype=_np.int32).copy()
        return c

    def __radd__(self, other):
        if other is None:
            return self
        return other + self

    def __sub__(self,other):
		c = matrix()
		c.tt = self.tt-other.tt
		c.n = _np.asanyarray(self.n,dtype=_np.int32).copy()
		c.m = _np.asanyarray(self.m,dtype=_np.int32).copy()
		return c
    
    def __neg__(self):
        return (-1)*self

    def __matmul__(self,other):
        
        diff = len(self.n) - len(other.m)
        L = self  if diff >= 0 else kron(self , matrix(ones(1, abs(diff))))
        R = other if diff <= 0 else kron(other, matrix(ones(1, abs(diff))))
        
        c = matrix()
        c.n = L.n.copy()
        c.m = R.m.copy()
        tt = tensor()
        tt.d = L.tt.d 
        tt.n = c.n * c.m
        if L.is_complex or R.is_complex:
            tt.r = _core_f90.core.zmat_mat(L.n, L.m, R.m, _np.array(L.tt.core, dtype=_np.complex), _np.array(R.tt.core, dtype=_np.complex), L.tt.r, R.tt.r)
            tt.core = _core_f90.core.zresult_core.copy()
        else:
            tt.r = _core_f90.core.dmat_mat(L.n, L.m, R.m,_np.real(L.tt.core), _np.real(R.tt.core), L.tt.r, R.tt.r)
            tt.core = _core_f90.core.result_core.copy()
        _core_f90.core.dealloc()
        tt.get_ps()
        c.tt = tt
        return c

    def __rmul__(self, other):
        if hasattr(other,'__matmul__'):
            return other.__matmul__(self)
        else:
            c = matrix()
            c.tt = other * self.tt
            c.n = self.n
            c.m = self.m
            return c

    def __mul__(self, other):
        if hasattr(other,'__matmul__'):
            return self.__matmul__(other)
        elif isinstance(other, (tensor, _Number)):
            c = matrix()
            c.tt = self.tt * other
            c.n = self.n
            c.m = self.m
            return c
        else:
            x = _np.asanyarray(other).flatten(order = 'F')
            N = _np.prod(self.m)
            if N != x.size:
                raise ValueError
            x = _np.reshape(x, _np.concatenate(([1], self.m)), order = 'F')
            cores = tensor.to_list(self.tt)
            curr = x.copy()
            for i in range(len(cores)):
                core = cores[i]
                core = _np.reshape(core, [self.tt.r[i], self.n[i], self.m[i], self.tt.r[i + 1]], order = 'F')
                #print curr.shape, core.shape
                curr = _np.tensordot(curr, core, axes = ([0, 1], [0, 2]))
                curr = _np.rollaxis(curr, -1)
            curr = _np.sum(curr, axis = 0)
            return curr.flatten(order = 'F')
            
    
    def __kron__(self,other):
        """ Kronecker product of two TT-matrices """
        if other is None:
            return self
        a = self
        b = other
        c = matrix()
        c.n = _np.concatenate((a.n,b.n))
        c.m = _np.concatenate((a.m,b.m))
        c.tt = kron(a.tt,b.tt)
        return c

    def norm(self):
        return self.tt.norm()

    def round(self, eps, rmax=100000):
        """ Computes an approximation to a 
	    TT-matrix in with accuracy EPS 
	"""
        c = matrix()
        c.tt = self.tt.round(eps, rmax)
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
            cur_core1 = _np.zeros((c.r[i],c.n[i],c.r[i+1]))
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
        sz = _np.vstack((self.n,self.m)).flatten('F')
        a = a.reshape(sz,order='F')
        #Design a permutation
        prm = _np.arange(2*d)
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

####################################################################################################
#############################################          #############################################
############################################   Tools    ############################################
#############################################          #############################################
####################################################################################################

#Some binary operations (put aside to wrap something in future)
#TT-matrix by a TT-vector product
def matvec(a, b, compression=False):
    """Matrix-vector product in TT format."""
    acrs = tensor.to_list(a.tt)
    bcrs = tensor.to_list(b)
    ccrs = []
    d = b.d
    
    def get_core(i):
        acr = _np.reshape(acrs[i], (a.tt.r[i], a.n[i], a.m[i], a.tt.r[i + 1]), order='F')
        acr = acr.transpose([3, 0, 1, 2]) # a(R_{i+1}, R_i, n_i, m_i)
        bcr = bcrs[i].transpose([1, 0, 2]) # b(m_i, r_i, r_{i+1})
        ccr = _np.tensordot(acr, bcr, axes=(3, 0)) # c(R_{i+1}, R_i, n_i, r_i, r_{i+1})
        ccr = ccr.transpose([1, 3, 2, 0, 4]).reshape((a.tt.r[i] * b.r[i], a.n[i], a.tt.r[i+1] * b.r[i+1]), order='F')
        return ccr
    
    if compression: # the compression is laaaaazy and one-directioned
        # calculate norm of resulting vector first
        nrm = _np.array([[1.0]]) # 1 x 1
        v = _np.array([[1.0]])
        for i in xrange(d):
            ccr = get_core(i)
            #print(str(ccr.shape) + " -> "),
            # minimal loss compression
            ccr = _np.tensordot(v, ccr, (1, 0))
            rl, n, rr = ccr.shape
            if i < d - 1:
                u, s, v = _np.linalg.svd(ccr.reshape((rl * n, rr), order='F'), full_matrices=False)
                newr = min(rl * n, rr)
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = _np.dot(_np.diag(s[:newr]), v[:newr, :])
            #print ccr.shape
            nrm = _np.tensordot(nrm, ccr, (0, 0)) # r x r . r x n x R -> r x n x R
            nrm = _np.tensordot(nrm, _np.conj(ccr), (0, 0)) # r x n x R . r x n x R -> n x R x n x R
            nrm = nrm.diagonal(axis1=0, axis2=2) # n x R x n x R -> R x R x n
            nrm = nrm.sum(axis=2) # R x R x n -> R x R
        if nrm.size > 1:
            raise Exception, 'too many numbers in norm'
        #print "Norm calculated:", nrm
        nrm = _np.sqrt(_np.linalg.norm(nrm))
        #print "Norm predicted:", nrm
        compression = compression * nrm / _np.sqrt(d - 1)
        v = _np.array([[1.0]])
    
    for i in xrange(d):
        ccr = get_core(i)
        rl, n, rr = ccr.shape
        if compression:
            ccr = _np.tensordot(v, ccr, (1, 0)) # c(s_i, n_i, r_i, r_{i+1})
            if i < d - 1:
                rl = v.shape[0]
                u, s, v = _np.linalg.svd(ccr.reshape((rl * n, rr), order='F'), full_matrices=False)
                ss = _np.cumsum(s[::-1])[::-1]
                newr = max(min([r for r in range(ss.size) if ss[r] <= compression] + [min(rl * n, rr)]), 1)
                #print "Rank % 4d replaced by % 4d" % (rr, newr)
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = _np.dot(_np.diag(s[:newr]), v[:newr, :])
        ccrs.append(ccr)
    result = tensor.from_list(ccrs)
    if compression:
        #print result
        print "Norm actual:", result.norm(), " mean rank:", result.rmean()
        #print "Norm very actual:", matvec(a,b).norm()
    return result


#TT-by-a-full matrix product (wrapped in Fortran 90, inspired by
#MATLAB prototype)
#def tt_full_mv(a,b):
#    mv = matrix_f90.matrix.tt_mv_full
#    if b.ndim is 1:
#        rb = 1
#    else:
#        rb = b.shape[1]
#    x1 = b.reshape(b.shape[0],rb)
#    y = _np.zeros(a.n.prod(),dtype=_np.float)
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
        raise ValueError('Dot is waiting for two TT-tensors or two TT-    matrices')


def diag(a):
    """ Diagonal of a TT-matrix OR diagonal matrix from a TT-tensor."""
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
    c.n = _np.array([], dtype=_np.int32)
    c.r = _np.array([], dtype=_np.int32)
    c.core = []
    
    for t in a:
        thetensor = t.tt if isinstance(t, matrix) else t
        c.d += thetensor.d
        c.n = _np.concatenate((c.n, thetensor.n))
        c.r = _np.concatenate((c.r[:-1], thetensor.r))
        c.core = _np.concatenate((c.core, thetensor.core))
            
    c.get_ps()
    return c

def concatenate(*args):
    """Concatenates given TT-tensors.
    
    For two tensors :math:`X(i_1,\\ldots,i_d),Y(i_1,\\ldots,i_d)` returns :math:`(d+1)`-dimensional
    tensor :math:`Z(i_0,i_1,\\ldots,i_d)`, :math:`i_0=\\overline{0,1}`, such that
    
    .. math::
       Z(0, i_1, \\ldots, i_d) = X(i_1, \\ldots, i_d),
       
       Z(1, i_1, \\ldots, i_d) = Y(i_1, \\ldots, i_d).
    
    """
    tmp = _np.array([[1] + [0] * (len(args) - 1)])
    result = kron(tensor(tmp), args[0])
    for i in range(1, len(args)):
        result += kron(tensor(_np.array([[0] * i + [1] + [0] * (len(args) - i - 1)])), args[i])
    return result
    
    

def _hdm (a,b):
    c = tensor()
    c.d = a.d
    c.n = a.n
    c.r = _np.zeros((a.d+1,1),dtype=_np.int32)
    c.ps = _np.zeros((a.d+1,1),dtype=_np.int32)
    if _np.iscomplexobj(a.core) or _np.iscomplexobj(b.core):
        c.r,c.ps = _tt_f90.tt_f90.ztt_hdm(a.n,a.r,b.r,a.ps,b.ps,a.core,b.core)
        c.core = _tt_f90.tt_f90.zcore.copy()
    else:
        c.r,c.ps = _tt_f90.tt_f90.dtt_hdm(a.n,a.r,b.r,a.ps,b.ps,a.core,b.core)
        c.core = _tt_f90.tt_f90.core.copy()
    _tt_f90.tt_f90.tt_dealloc()
    return c

def sum(a, axis=-1):    
    """Sum TT-tensor over specified axes"""
    d = a.d
    crs = tensor.to_list(a.tt if isinstance(a, matrix) else a)
    if axis < 0:
        axis = range(a.d)
    elif isinstance(axis, int):
        axis = [axis]
    axis = list(axis)[::-1]
    for ax in axis:
        crs[ax] = _np.sum(crs[ax], axis=1)
        rleft, rright = crs[ax].shape
        if (rleft >= rright or rleft < rright and ax + 1 >= d) and ax > 0:
            crs[ax-1] = _np.tensordot(crs[ax-1], crs[ax], axes=(2,0))
        elif ax + 1 < d:
            crs[ax+1] = _np.tensordot(crs[ax], crs[ax+1], axes=(1,0))
        else:
            return _np.sum(crs[ax])
        crs.pop(ax)
        d -= 1
    return tensor.from_list(crs)

#Basic functions for the arrays creation


def ones(n,d=None):
	""" Creates a TT-tensor of all ones"""
	c = tensor()
	if d is None:
            c.n = _np.array(n,dtype=_np.int32)
            c.d = c.n.size
	else:
            c.n = _np.array([n]*d,dtype=_np.int32)
            c.d = d
	c.r = _np.ones((c.d+1,),dtype=_np.int32)
	c.get_ps()
	c.core = _np.ones(c.ps[c.d]-1)
	return c


def rand(n, d=None, r=2):
	"""Generate a random d-dimensional TT-tensor with ranks ``r``."""
	n0 = _np.asanyarray(n, dtype=_np.int32)
	r0 = _np.asanyarray(r, dtype=_np.int32)
        if d is None:
            d = n.size
        if n0.size is 1:
	    n0 = _np.ones((d,),dtype=_np.int32)*n0
	if r0.size is 1:
	    r0 = _np.ones((d+1,),dtype=_np.int32)*r0
	    r0[0] = 1
	    r0[d] = 1
	c = tensor()
	c.d = d
	c.n = n0
	c.r = r0
	c.get_ps()
	c.core = _np.random.randn(c.ps[d]-1)
	return c


#Identity matrix
def eye(n,d=None):
	""" Creates an identity TT-matrix"""
	c = matrix()
	c.tt = tensor()
	if d is None:
		n0=_np.asanyarray(n,dtype=_np.int32)
		c.tt.d=n0.size
	else:
		n0 = _np.asanyarray([n]*d,dtype=_np.int32)
		c.tt.d = d
	c.n = n0.copy()
	c.m = n0.copy()
	c.tt.n = (c.n)*(c.m)
	c.tt.r = _np.ones((c.tt.d+1,),dtype=_np.int32)
	c.tt.get_ps()
	c.tt.alloc_core()
	for i in xrange(c.tt.d):
		c.tt.core[c.tt.ps[i]-1:c.tt.ps[i+1]-1] = _np.eye(c.n[i]).flatten()
	return c

#Arbitrary multilevel Toeplitz matrix
def Toeplitz(x, d=None, D=None, kind='F'):
    """ Creates multilevel Toeplitz TT-matrix with ``D`` levels.
        
        Possible matrix types:
        
        * 'F' - full Toeplitz matrix,             size(x) = 2^{d+1}
        * 'C' - circulant matrix,                 size(x) = 2^d
        * 'L' - lower triangular Toeplitz matrix, size(x) = 2^d
        * 'U' - upper triangular Toeplitz matrix, size(x) = 2^d
        
        Sample calls:
        
        >>> # one-level Toeplitz matrix:
        >>> T = tt.Toeplitz(x)
        >>> # one-level circulant matrix:
        >>> T = tt.Toeplitz(x, kind='C')
        >>> # three-level upper-triangular Toeplitz matrix:
        >>> T = tt.Toeplitz(x, D=3, kind='U')
        >>> # two-level mixed-type Toeplitz matrix:
        >>> T = tt.Toeplitz(x, kind=['L', 'U'])
        >>> # two-level mixed-size Toeplitz matrix:
        >>> T = tt.Toeplitz(x, [3, 4], kind='C')
        
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
            d = _np.array([x.d / D - (1 if kind[0] == 'F' else 0)] * D, dtype=_np.int32)
            kind = kind * D
        else:
            check_kinds(D, kind)
            if set(kind).issubset(['F']):
                d = _np.array([x.d / D - 1] * D, dtype=_np.int32)
            elif set(kind).issubset(['C', 'L', 'U']):
                d = _np.array([x.d / D] * D, dtype=_np.int32)
            else:
                raise ValueError("Only similar matrix kinds (only F or only C, L and U) are accepted when d is not specified!")
    elif d is not None:
        d = _np.asarray(d, dtype=_np.int32).flatten()
        if D is None:
            D = d.size
        elif d.size == 1:
            d = _np.array([d[0]] * D, dtype=_np.int32)
        if D != d.size:
            raise ValueError("D must be equal to len(d)")
        check_kinds(D, kind)
        if _np.sum(d) + _np.sum([(1 if knd == 'F' else 0) for knd in kind]) != x.d:
            raise ValueError("Dimensions inconsistency: x.d != d_1 + d_2 + ... + d_D")
    
    # predefined matrices and tensors:
    I = [[1, 0], [0, 1]]
    J = [[0, 1], [0, 0]]
    JT= [[0, 0], [1, 0]]
    H = [[0, 1], [1, 0]]
    S = _np.array([[[0], [1]], [[1], [0]]]).transpose() # 2 x 2 x 1
    P = _np.zeros((2, 2, 2, 2))
    P[:, :, 0, 0] = I; P[:, :, 1, 0] = H
    P[:, :, 0, 1] = H; P[:, :, 1, 1] = I
    P = _np.transpose(P) # 2 x 2! x 2 x 2 x '1'
    Q = _np.zeros((2, 2, 2, 2))
    Q[:, :, 0, 0] = I; Q[:, :, 1, 0] = JT
    Q[:, :, 0, 1] = JT
    Q = _np.transpose(Q) # 2 x 2! x 2 x 2 x '1'
    R = _np.zeros((2, 2, 2, 2))
    R[:, :, 1, 0] = J
    R[:, :, 0, 1] = J; R[:, :, 1, 1] = I;
    R = _np.transpose(R) # 2 x 2! x 2 x 2 x '1'
    W = _np.zeros([2] * 5) # 2 x 2! x 2 x 2 x 2
    W[0, :, :, 0, 0] = W[1, :, :, 1, 1] = I
    W[0, :, :, 1, 0] = W[0, :, :, 0, 1] = JT
    W[1, :, :, 1, 0] = W[1, :, :, 0, 1] = J
    W = _np.transpose(W) # 2 x 2! x 2 x 2 x 2
    V = _np.zeros((2, 2, 2, 2))
    V[0, :, :, 0] = I
    V[0, :, :, 1] = JT
    V[1, :, :, 1] = J
    V = _np.transpose(V) # '1' x 2! x 2 x 2 x 2
    
    crs = []
    xcrs = tensor.to_list(x)
    dp = 0 # dimensions passed
    for j in xrange(D):
        currd = d[j]
        xcr = xcrs[dp]
        cr = _np.tensordot(V, xcr, (0, 1)) # 
        cr = cr.transpose(3, 0, 1, 2, 4)  # <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
        cr = cr.reshape((x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <r_dp| x 2 x 2 x |2r_{dp+1}>
        dp += 1
        crs.append(cr)
        for i in xrange(1, currd - 1):
            xcr = xcrs[dp]
            cr = _np.tensordot(W, xcr, (1, 1)) # (<2| x 2 x 2 x |2>) x <r_dp| x |r_{dp+1}>
            cr = cr.transpose([0, 4, 1, 2, 3, 5]) # <2| x <r_dp| x 2 x 2 x |2> x |r_{dp+1}>
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            dp += 1
            crs.append(cr)
        if kind[j] == 'F':
            xcr = xcrs[dp] # r_dp x 2 x r_{dp+1}
            cr = _np.tensordot(W, xcr, (1, 1)).transpose([0, 4, 1, 2, 3, 5])
            cr = cr.reshape((2 * x.r[dp], 2, 2, 2 * x.r[dp+1]), order='F') # <2r_dp| x 2 x 2 x |2r_{dp+1}>
            dp += 1
            xcr = xcrs[dp] # r_dp x 2 x r_{dp+1}
            tmp = _np.tensordot(S, xcr, (1, 1)) # <2| x |1> x <r_dp| x |r_{dp+1}>
            #tmp = tmp.transpose([0, 2, 1, 3]) # TODO: figure out WHY THE HELL this spoils everything
            tmp = tmp.reshape((2 * x.r[dp], x.r[dp+1]), order='F') # <2r_dp| x |r_{dp+1}>
            cr = _np.tensordot(cr, tmp, (3, 0)) # <2r_{dp-1}| x 2 x 2 x |r_{dp+1}>
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
            cr = _np.tensordot(dotcore, xcr, (1, 1)) # <2| x 2 x 2 x |'1'> x <r_dp| x |r_{dp+1}>
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
    I = _np.eye(2)
    J = _np.array([[0,1],[0,0]])
    cr=[]
    if D is 1:
        for k in xrange(1,d0[0]+1):
            if k is 1:
                cur_core=_np.zeros((1,2,2,3));
                cur_core[:,:,:,0]=2*I-J-J.T;
                cur_core[:,:,:,1]=-J;
                cur_core[:,:,:,2]=-J.T;
            elif k is d0[0]:
                cur_core=_np.zeros((3,2,2,1));
                cur_core[0,:,:,0]=I;
                cur_core[1,:,:,0]=J.T;
                cur_core[2,:,:,0]=J;
            else:
                cur_core=_np.zeros((3,2,2,3));
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
                        cur_core=_np.zeros((1,2,2,4));
                        cur_core[:,:,:,0]=2*I-J-J.T;
                        cur_core[:,:,:,1]=-J;
                        cur_core[:,:,:,2]=-J.T;
                        cur_core[:,:,:,3]=I;
                    elif k is D-1:
                        cur_core=_np.zeros((2,2,2,3));
                        cur_core[0,:,:,0]=2*I-J-J.T;
                        cur_core[0,:,:,1]=-J;
                        cur_core[0,:,:,2]=-J.T;
                        cur_core[1,:,:,0]=I;
                    else:
                        cur_core=_np.zeros((2,2,2,4));
                        cur_core[0,:,:,0]=2*I-J-J.T;
                        cur_core[0,:,:,1]=-J;
                        cur_core[0,:,:,2]=-J.T;
                        cur_core[0,:,:,3]=I;
                        cur_core[1,:,:,0]=I;
                elif kappa is d0[k]:
                    if k is D-1:
                        cur_core=_np.zeros((3,2,2,1));
                        cur_core[0,:,:,0]=I;
                        cur_core[1,:,:,0]=J.T;
                        cur_core[2,:,:,0]=J;
                    else:
                        cur_core=_np.zeros((4,2,2,2));
                        cur_core[3,:,:,0]=I;
                        cur_core[0,:,:,1]=I;
                        cur_core[1,:,:,1]=J.T
                        cur_core[2,:,:,1]=J;
                else:
                    if k is D-1:
                        cur_core=_np.zeros((3,2,2,3));
                        cur_core[0,:,:,0]=I;
                        cur_core[1,:,:,1]=J;
                        cur_core[2,:,:,2]=J.T;
                        cur_core[1,:,:,0]=J.T;
                        cur_core[2,:,:,0]=J;
                    else:
                        cur_core=_np.zeros((4,2,2,4));
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
        n0 = _np.asanyarray(n, dtype=_np.int32)
    else:
        n0 = _np.array(n * d, dtype=_np.int32)
    d = n0.size
    if d == 1:
        return tensor.from_list([_np.reshape(_np.arange(n0[0]), (1, n0[0], 1))])
    cr=[]
    cur_core = _np.ones((1,n0[0],2))
    cur_core[0,:,0] = _np.arange(n0[0])
    cr.append(cur_core)
    ni = float(n0[0])
    for i in xrange(1, d - 1):
        cur_core = _np.zeros((2,n0[i],2))
        for j in xrange(n0[i]):
            cur_core[:, j, :] = _np.eye(2)
        cur_core[1, :, 0] = ni * _np.arange(n0[i])
        ni *= n0[i]
        cr.append(cur_core)
    cur_core = _np.ones((2, n0[d - 1], 1))
    cur_core[1,:,0] = ni*_np.arange(n0[d - 1])
    cr.append(cur_core)
    return tensor.from_list(cr)


def sin(d, alpha=1.0, phase=0.0):
    """ Create TT-tensor for :math:`\\sin(\\alpha n + \\varphi)`."""
    cr = []
    cur_core = _np.zeros([1, 2, 2], dtype=_np.float)
    cur_core[0, 0, :] = [_math.cos(phase)        , _math.sin(phase)        ]
    cur_core[0, 1, :] = [_math.cos(alpha + phase), _math.sin(alpha + phase)]
    cr.append(cur_core)
    for i in xrange(1, d-1):
        cur_core = _np.zeros([2, 2, 2], dtype=_np.float)
        cur_core[0, 0, :] = [1.0                     , 0.0                      ]
        cur_core[1, 0, :] = [0.0                     , 1.0                      ]
        cur_core[0, 1, :] = [ _math.cos(alpha * 2 ** i), _math.sin(alpha * 2 ** i)]
        cur_core[1, 1, :] = [-_math.sin(alpha * 2 ** i), _math.cos(alpha * 2 ** i)]
        cr.append(cur_core)
    cur_core = _np.zeros([2, 2, 1], dtype=_np.float)
    cur_core[0, :, 0] = [0.0, _math.sin(alpha * 2 ** (d-1))]
    cur_core[1, :, 0] = [1.0, _math.cos(alpha * 2 ** (d-1))]
    cr.append(cur_core)
    return tensor.from_list(cr)


def cos(d, alpha=1.0, phase=0.0):
    """ Create TT-tensor for :math:`\\cos(\\alpha n + \\varphi)`."""
    return sin(d, alpha, phase + _math.pi * 0.5)

def delta(n, d=None, center=0):
    """ Create TT-tensor for delta-function :math:`\\delta(x - x_0)`. """
    if isinstance(n, (int, long)):
        n = [n]
    if d is None:
        n0 = _np.asanyarray(n, dtype=_np.int32)
    else:
        n0 = _np.array(n * d, dtype=_np.int32)
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
        cur_core = _np.zeros((1, n0[i], 1))
        cur_core[0, cind[i], 0] = 1
        cr.append(cur_core)
    return tensor.from_list(cr)

def stepfun(n, d=None, center=1, direction=1):
    """ Create TT-tensor for Heaviside step function :math:`\chi(x - x_0)`.
    
    Heaviside step function is defined as
    
    .. math::
    
        \chi(x) = \\left\{ \\begin{array}{l} 1 \mbox{ when } x \ge 0, \\\\ 0 \mbox{ when } x < 0. \\end{array} \\right.
    
    For negative value of ``direction`` :math:`\chi(x_0 - x)` is approximated. """
    if isinstance(n, (int, long)):
        n = [n]
    if d is None:
        n0 = _np.asanyarray(n, dtype=_np.int32)
    else:
        n0 = _np.array(n * d, dtype=_np.int32)
    d = n0.size
    N = _np.prod(n0)
    
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
        cr = _np.zeros([nextrank, n0[i], prevrank], dtype=_np.float)
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


def qshift(d):
    x = []
    x.append(_np.array([0.0, 1.0]))
    for _ in xrange(1, d):
        x.append(_np.array([1.0, 0.0])) 
    return Toeplitz(tensor.from_list(x), kind='L')

####### Recent update #######
def ind2sub(siz, idx):
    '''
    Translates full-format index into tt.vector one's.
    ----------
    Parameters:
        siz - tt.vector modes
        idx - full-vector index
    Note: not vectorized.
    '''
    n = len(siz)
    subs = _np.empty((n))
    k = _np.cumprod(siz[:-1])
    k = _np.concatenate((_np.ones(1), k))
    for i in xrange(n-1, -1, -1):
        subs[i] = _np.floor(idx / k[i])
        idx = idx % k[i]
    return subs

def unit(n, d = None, j = None, tt_instance = True):
    ''' Generates e_j vector in tt.vector format
    ---------
    Parameters:
        n - modes (either integer or array)
        d - dimensionality (integer)
        j - position of 1 in full-format e_j (integer)
        tt_instance - if True, returns tt.vector;
                      if False, returns tt cores as a list
    '''
    if isinstance(n, int):
        if d is None:
            d = 1
        n = n * _np.ones(d)
    else:
        d = len(n)
    if j is None:
        j = 0
    rv = []
    
    j = ind2sub(n, j)
    
    for k in xrange(d):
        rv.append(_np.zeros((1, n[k], 1)))
        rv[-1][0, j[k], 0] = 1
    if tt_instance:
        rv = tensor.from_list(rv)
    return rv

def IpaS(d, a, tt_instance = True):
    '''A special bidiagonal matrix in the QTT-format
    M = IPAS(D, A)
    Generates I+a*S_{-1} matrix in the QTT-format:
    1 0 0 0
    a 1 0 0
    0 a 1 0
    0 0 a 1
    Convenient for Crank-Nicolson and time gradient matrices
    '''

    if d == 1:
        M = _np.array([[1, 0], [a, 1]]).reshape((1, 2, 2, 1), order = 'F')
    else:
        M = [None]*d
        M[0] = _np.zeros((1, 2, 2, 2))
        M[0][0, :, :, 0] = _np.array([[1, 0], [a, 1]])
        M[0][0, :, :, 1] = _np.array([[0, a], [0, 0]])
        for i in xrange(1, d-1):
            M[i] = _np.zeros((2,2,2,2))
            M[i][:, :, 0, 0] = _np.eye(2)
            M[i][:, :, 1, 0] = _np.array([[0, 0], [1, 0]])
            M[i][:, :, 1, 1] = _np.array([[0, 1], [0, 0]])
        M[d-1] = _np.zeros((2,2,2,1))
        M[d-1][:, :, 0, 0] = _np.eye(2)
        M[d-1][:, :, 1, 0] = _np.array([[0, 0], [1, 0]])
    if tt_instance:
        M = matrix.from_list(M)
    return M

def reshape(tt_array, shape, eps=1e-14, rl=1, rr=1):
    ''' Reshape of the TT-tensor
       [TT1]=TT_RESHAPE(TT,SZ) reshapes TT-tensor or TT-matrix into another 
       with mode sizes SZ, accuracy 1e-14

       [TT1]=TT_RESHAPE(TT,SZ,EPS) reshapes TT-tensor/matrix into another with
       mode sizes SZ and accuracy EPS
       
       [TT1]=TT_RESHAPE(TT,SZ,EPS, RL) reshapes TT-tensor/matrix into another 
       with mode size SZ and left tail rank RL

       [TT1]=TT_RESHAPE(TT,SZ,EPS, RL, RR) reshapes TT-tensor/matrix into 
       another with mode size SZ and tail ranks RL*RR
       Reshapes TT-tensor/matrix into a new one, with dimensions specified by SZ.

       If the i_nput is TT-matrix, SZ must have the sizes for both modes, 
       so it is a matrix if sizes d2-by-2.
       If the i_nput is TT-tensor, SZ may be either a column or a row vector.
    '''
    
    def gcd(a, b):
    	'''Greatest common divider'''
        f = _np.frompyfunc(_fractions.gcd, 2, 1)
        return f(a,b)
    
    def my_chop2(sv, eps): # from ttpy/multifuncr.py
        if eps <= 0.0:
            r = len(sv)
            return r
        sv0 = _np.cumsum(abs(sv[::-1]) ** 2)[::-1]
        ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
        if len(ff) == 0:
            return len(sv)
        else:
            return _np.amin(ff)
    
    tt1 = _cp.deepcopy(tt_array)
    sz = _cp.deepcopy(shape)
    ismatrix = False
    if isinstance(tt1, matrix):
        d1 = tt1.tt.d
        d2 = sz.shape[0]
        ismatrix = True
        # The size should be [n,m] in R^{d x 2}
        restn2_n = sz[:, 0]
        restn2_m = sz[:, 1]
        sz_n = _cp.copy(sz[:, 0])
        sz_m = _cp.copy(sz[:, 1])
        n1_n = tt1.n
        n1_m = tt1.m    
        sz = _np.prod(sz, axis = 1) # We will split/convolve using the vector form anyway
        tt1 = tt1.tt
    else:
        d1 = tt1.d
        d2 = len(sz)


    # Recompute sz to include r0,rd,
    # and the items of tt1

    sz[0] = sz[0] * rl
    sz[d2-1] = sz[d2-1] * rr
    tt1.n[0] = tt1.n[0] * tt1.r[0]
    tt1.n[d1-1] = tt1.n[d1-1] * tt1.r[d1]
    if ismatrix: # in matrix: 1st tail rank goes to the n-mode, last to the m-mode
        restn2_n[0] = restn2_n[0] * rl
        restn2_m[d2-1] = restn2_m[d2-1] * rr
        n1_n[0] = n1_n[0] * tt1.r[0]
        n1_m[d1-1] = n1_m[d1-1] * tt1.r[d1]

    tt1.r[0] = 1
    tt1.r[d1] = 1

    n1 = tt1.n

    assert _np.prod(n1) == _np.prod(sz), 'Reshape: incorrect sizes'

    needQRs = False
    if d2 > d1:
        needQRs = True

    if d2 <= d1:
        i2 = 0
        n2 = sz
        for i1 in xrange(d1):
            if n2[i2] == 1:
                i2 = i2 + 1
                if i2 > d2:
                    break
            if n2[i2] % n1[i1] == 0:
                n2[i2] = n2[i2] / n1[i1]
            else:
                needQRs = True
                break

    r1 = tt1.r
    tt1 = tt1.to_list(tt1)

    if needQRs: # We have to split some cores -> perform QRs
        for i in xrange(d1-1, 0, -1):
            cr = tt1[i]
            cr = _np.reshape(cr, (r1[i], n1[i]*r1[i+1]), order = 'F')
            [cr, rv] = _np.linalg.qr(cr.T) # Size n*r2, r1new - r1nwe,r1
            cr0 = tt1[i-1]
            cr0 = _np.reshape(cr0, (r1[i-1]*n1[i-1], r1[i]), order = 'F')
            cr0 = _np.dot(cr0, rv.T) # r0*n0, r1new
            r1[i] = cr.shape[1]        
            cr0 = _np.reshape(cr0, (r1[i-1], n1[i-1], r1[i]), order = 'F')
            cr = _np.reshape(cr.T, (r1[i], n1[i], r1[i+1]), order = 'F')
            tt1[i] = cr
            tt1[i-1] = cr0  

    r2 = _np.ones(d2 + 1)
        
    i1 = 0 # Working index in tt1
    i2 = 0 # Working index in tt2
    core2 = _np.zeros((0))
    curcr2 = 1
    restn2 = sz
    n2 = _np.ones(d2)
    if ismatrix:
        n2_n = _np.ones(d2)
        n2_m = _np.ones(d2)

    while i1 < d1:
        curcr1 = tt1[i1]    
        if gcd(restn2[i2], n1[i1]) == n1[i1]:
            # The whole core1 fits to core2. Convolve it
            if (i1 < d1-1) and (needQRs): # QR to the next core - for safety
                curcr1 = _np.reshape(curcr1, (r1[i1]*n1[i1], r1[i1+1]), order = 'F')
                [curcr1, rv] = _np.linalg.qr(curcr1)
                curcr12 = tt1[i1+1]
                curcr12 = _np.reshape(curcr12, (r1[i1+1], n1[i1+1]*r1[i1+2]), order = 'F')
                curcr12 = _np.dot(rv, curcr12)
                r1[i1+1] = curcr12.shape[0]
                tt1[i1+1] = _np.reshape(curcr12, (r1[i1+1], n1[i1+1], r1[i1+2]), order = 'F')
            # Actually merge is here
            curcr1 = _np.reshape(curcr1, (r1[i1], n1[i1]*r1[i1+1]), order = 'F')
            curcr2 = _np.dot(curcr2, curcr1) # size r21*nold, dn*r22        
            if ismatrix: # Permute if we are working with tt_matrix
                curcr2 = _np.reshape(curcr2, (r2[i2], n2_n[i2], n2_m[i2], n1_n[i1], n1_m[i1], r1[i1+1]), order = 'F')
                curcr2 = _np.transpose(curcr2, [0, 1, 3, 2, 4, 5])
                # Update the "matrix" sizes            
                n2_n[i2] = n2_n[i2]*n1_n[i1]
                n2_m[i2] = n2_m[i2]*n1_m[i1]
                restn2_n[i2] = restn2_n[i2] / n1_n[i1]
                restn2_m[i2] = restn2_m[i2] / n1_m[i1]
            r2[i2+1] = r1[i1+1]
            # Update the sizes of tt2
            n2[i2] = n2[i2]*n1[i1]
            restn2[i2] = restn2[i2] / n1[i1]
            curcr2 = _np.reshape(curcr2, (r2[i2]*n2[i2], r2[i2+1]), order = 'F')
            i1 = i1+1 # current core1 is over
        else:
            if (gcd(restn2[i2], n1[i1]) !=1 ) or (restn2[i2] == 1):
                # There exists a nontrivial divisor, or a singleton requested
                # Split it and convolve
                n12 = gcd(restn2[i2], n1[i1])
                if ismatrix: # Permute before the truncation
                    # Matrix sizes we are able to split
                    n12_n = gcd(restn2_n[i2], n1_n[i1])
                    n12_m = gcd(restn2_m[i2], n1_m[i1])
                    curcr1 = _np.reshape(curcr1, (r1[i1], n12_n, n1_n[i1] / n12_n, n12_m, n1_m[i1] / n12_m, r1[i1+1]), order = 'F')
                    curcr1 = _np.transpose(curcr1, [0, 1, 3, 2, 4, 5])
                    # Update the matrix sizes of tt2 and tt1
                    n2_n[i2] = n2_n[i2]*n12_n
                    n2_m[i2] = n2_m[i2]*n12_m
                    restn2_n[i2] = restn2_n[i2] / n12_n
                    restn2_m[i2] = restn2_m[i2] / n12_m
                    n1_n[i1] = n1_n[i1] / n12_n
                    n1_m[i1] = n1_m[i1] / n12_m
                
                curcr1 = _np.reshape(curcr1, (r1[i1]*n12, (n1[i1]/n12)*r1[i1+1]), order = 'F')
                [u,s,v] = _np.linalg.svd(curcr1, full_matrices = False)
                r = my_chop2(s, eps*_np.linalg.norm(s)/(d2-1)**0.5)
                u = u[:, :r]
                v = v.T
                v = v[:, :r]*s[:r]
                u = _np.reshape(u, (r1[i1], n12*r), order = 'F')
                # u is our admissible chunk, merge it to core2
                curcr2 = _np.dot(curcr2, u) # size r21*nold, dn*r22
                r2[i2+1] = r
                # Update the sizes of tt2
                n2[i2] = n2[i2]*n12
                restn2[i2] = restn2[i2] / n12
                curcr2 = _np.reshape(curcr2, (r2[i2]*n2[i2], r2[i2+1]), order = 'F')
                r1[i1] = r
                # and tt1
                n1[i1] = n1[i1] / n12
                # keep v in tt1 for next operations
                curcr1 = _np.reshape(v.T, (r1[i1], n1[i1], r1[i1+1]), order = 'F')
                tt1[i1] = curcr1
            else:
                # Bad case. We have to merge cores of tt1 until a common divisor appears
                i1new = i1+1
                curcr1 = _np.reshape(curcr1, (r1[i1]*n1[i1], r1[i1+1]), order = 'F')
                while (gcd(restn2[i2], n1[i1]) == 1) and (i1new < d1):
                    cr1new = tt1[i1new]
                    cr1new = _np.reshape(cr1new, (r1[i1new], n1[i1new]*r1[i1new+1]), order = 'F')
                    curcr1 = _np.dot(curcr1, cr1new) # size r1(i1)*n1(i1), n1new*r1new
                    if ismatrix: # Permutes and matrix size updates
                        curcr1 = _np.reshape(curcr1, (r1[i1], n1_n[i1], n1_m[i1], n1_n[i1new], n1_m[i1new], r1[i1new+1]), order = 'F')
                        curcr1 = _np.transpose(curcr1, [0, 1, 3, 2, 4, 5])
                        n1_n[i1] = n1_n[i1]*n1_n[i1new]
                        n1_m[i1] = n1_m[i1]*n1_m[i1new]
                    n1[i1] = n1[i1]*n1[i1new]
                    curcr1 = _np.reshape(curcr1, (r1[i1]*n1[i1], r1[i1new+1]), order = 'F')
                    i1new = i1new+1
                # Inner cores merged => squeeze tt1 data
                n1 = _np.concatenate((n1[:i1], n1[i1new:]))
                r1 = _np.concatenate((r1[:i1], r1[i1new:]))
                tt1[i] = _np.reshape(curcr1, (r1[i1], n1[i1], r1[i1new]), order = 'F')
                tt1 = tt1[:i1] + tt1[i1new:]
                d1 = len(n1)
        
        if (restn2[i2] == 1) and ((i1 >= d1) or ((i1 < d1) and (n1[i1] != 1))):
            # The core of tt2 is finished
            # The second condition prevents core2 from finishing until we 
            # squeeze all tailing singletons in tt1.
            curcr2 = curcr2.flatten(order = 'F')
            core2 = _np.concatenate((core2, curcr2))
            i2 = i2+1
            # Start new core2
            curcr2 = 1

    # If we have been asked for singletons - just add them
    while (i2 < d2):
        core2 = _np.concatenate((core2, _np.ones(1)))
        r2[i2] = 1
        i2 = i2+1

    tt2 = ones(2, 1) # dummy tensor
    tt2.d = d2
    tt2.n = n2
    tt2.r = r2
    tt2.core = core2
    tt2.ps = _np.cumsum(_np.concatenate((_np.ones(1), r2[:-1] * n2 * r2[1:])))


    tt2.n[0] = tt2.n[0] / rl
    tt2.n[d2-1] = tt2.n[d2-1] / rr
    tt2.r[0] = rl
    tt2.r[d2] = rr

    if ismatrix:
        ttt = eye(1,1) # dummy tt matrix
        ttt.n = sz_n
        ttt.m = sz_m
        ttt.tt = tt2
        return ttt
    else:
        return tt2
