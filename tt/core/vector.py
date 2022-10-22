from __future__ import absolute_import, division, print_function

from numbers import Number
from warnings import warn

import numpy as np
from six.moves import xrange

from .tt_f90 import tt_f90
from .utils import my_chop2


class vector(object):
    r"""The main class for working with vectors in the TT-format. It constructs
    new TT-vector. When called with no arguments, creates dummy object which
    can be filled from outside. When ``a`` is specified, computes approximate
    decomposition of array ``a`` with accuracy ``eps``:

    :param a: A tensor to approximate.
    :type a: ndarray

    :param eps: Approximation accuracy
    :type a: float

    :param rmax: Maximal rank
    :type rmax: int

    >>> a = numpy.sin(numpy.arange(2 ** 10)).reshape([2] * 10, order='F')
    >>> a = tt.vector(a)
    >>> a.r
    array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=int32)
    >>> # now let's try different accuracy
    >>> b = numpy.random.rand(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    >>> btt = tt.vector(b, 1E-14)
    >>> btt.r
    array([ 1,  2,  4,  8, 16, 32, 16,  8,  4,  2,  1], dtype=int32)
    >>> btt = tt.vector(b, 1E-1)
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
        To get a list of three-dimensional cores, use ``tt.vector.to_list(my_tensor)``.
    """

    def __init__(self, a=None, eps=1e-14, rmax=100000):
        if a is None:
            self.core = np.array([0.0])
            self.d = 0
            self.n = np.array([0], dtype=np.int32)
            self.r = np.array([1], dtype=np.int32)
            self.ps = np.array([0], dtype=np.int32)
            return
        self.d = a.ndim
        self.n = np.array(a.shape, dtype=np.int32)
        r = np.zeros((self.d + 1,), dtype=np.int32)
        ps = np.zeros((self.d + 1,), dtype=np.int32)
        if (np.iscomplex(a).any()):
            if rmax is not None:
                self.r, self.ps = tt_f90.zfull_to_tt(
                    a.flatten('F'), self.n, self.d, eps, rmax)
            else:
                self.r, self.ps = tt_f90.zfull_to_tt(
                    a.flatten('F'), self.n, self.d, eps)
            self.core = tt_f90.zcore.copy()
        else:
            if rmax is not None:
                self.r, self.ps = tt_f90.dfull_to_tt(
                    np.real(a).flatten('F'), self.n, self.d, eps, rmax)
            else:
                self.r, self.ps = tt_f90.dfull_to_tt(
                    np.real(a).flatten('F'), self.n, self.d, eps)
            self.core = tt_f90.core.copy()
        tt_f90.tt_dealloc()

    @staticmethod
    def from_list(a, order='F'):
        """Generate TT-vectorr object from given TT cores.

        :param a: List of TT cores.
        :type a: list
        :returns: vector -- TT-vector constructed from the given cores.

        """
        d = len(a)  # Number of cores
        res = vector()
        n = np.zeros(d, dtype=np.int32)
        r = np.zeros(d+1, dtype=np.int32)
        cr = np.array([])
        for i in xrange(d):
            cr = np.concatenate((cr, a[i].flatten(order)))
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
        :param tt: TT-vector.
        :type tt: vector
        :returns: list -- list of ``tt.d`` three-dimensional cores, ``i``-th core is an ndarray of shape ``(tt.r[i], tt.n[i], tt.r[i+1])``.
        """
        d = tt.d
        r = tt.r
        n = tt.n
        ps = tt.ps
        core = tt.core
        res = []
        for i in xrange(d):
            cur_core = core[ps[i] - 1:ps[i + 1] - 1]
            cur_core = cur_core.reshape((r[i], n[i], r[i + 1]), order='F')
            res.append(cur_core)
        return res

    @property
    def erank(self):
        """ Effective rank of the TT-vector """
        r = self.r
        n = self.n
        d = self.d
        if d <= 1:
            er = 0e0
        else:
            sz = np.dot(n * r[0:d], r[1:])
            if sz == 0:
                er = 0e0
            else:
                b = r[0] * n[0] + n[d - 1] * r[d]
                if d == 2:
                    er = sz * 1.0 / b
                else:
                    a = np.sum(n[1:d - 1])
                    er = (np.sqrt(b * b + 4 * a * sz) - b) / (2 * a)
        return er

    def __getitem__(self, index):
        """Get element of the TT-vector.

        :param index: array_like (it supports slicing).
        :returns: number -- an element of the tensor or a new tensor.

        Examples:
        Suppose that a is a 3-dimensional tt.vector of size 4 x 5 x 6
        a[1, 2, 3] returns the element with index (1, 2, 3)
        a[1, :, 1:3] returns a 2-dimensional tt.vector of size 5 x 2
        """
        if len(index) != self.d:
            print("Incorrect index length.")
            return
        # TODO: add tests.
        # TODO: in case of requesting one element this implementation is slower
        # than the old one.
        running_fact = None
        answ_cores = []
        for i in xrange(self.d):
            # r0, n, r1 = cores[i].shape
            cur_core = self.core[self.ps[i] - 1:self.ps[i + 1] - 1]
            cur_core = cur_core.reshape(
                (self.r[i], self.n[i], self.r[i + 1]), order='F')
            cur_core = cur_core[
                :, index[i], :].reshape(
                (self.r[i], -1), order='F')
            if running_fact is None:
                new_r0 = self.r[i]
                cur_core = cur_core.copy()
            else:
                new_r0 = running_fact.shape[0]
                cur_core = np.dot(running_fact, cur_core)
            cur_core = cur_core.reshape((new_r0, -1, self.r[i + 1]), order='F')
            if cur_core.shape[1] == 1:
                running_fact = cur_core.reshape((new_r0, -1), order='F')
            else:
                answ_cores.append(cur_core)
                running_fact = None
        if len(answ_cores) == 0:
            return running_fact[0, 0]
        if running_fact is not None:
            answ_cores[-1] = np.dot(answ_cores[-1], running_fact)
        return self.from_list(answ_cores)

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
        newcr = np.zeros((rl, n, rr * 2), dtype=np.float64)
        newcr[:, :, :rr] = np.real(cr)
        newcr[:, :, rr:] = np.imag(cr)
        newcrs.append(newcr)
        for i in xrange(1, self.d - 1):
            cr = crs[i]
            rl, n, rr = cr.shape
            newcr = np.zeros((rl * 2, n, rr * 2), dtype=np.float64)
            newcr[:rl, :, :rr] = newcr[rl:, :, rr:] = np.real(cr)
            newcr[:rl, :, rr:] = np.imag(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcrs.append(newcr)
        cr = crs[-1]
        rl, n, rr = cr.shape
        if op in ['R', 'r', 'Re']:
            # get real part
            newcr = np.zeros((rl * 2, n, rr), dtype=np.float64)
            newcr[:rl, :, :] = np.real(cr)
            newcr[rl:, :, :] = -np.imag(cr)
        elif op in ['I', 'i', 'Im']:
            # get imaginary part
            newcr = np.zeros((rl * 2, n, rr), dtype=np.float64)
            newcr[:rl, :, :] = np.imag(cr)
            newcr[rl:, :, :] = np.real(cr)
        elif op in ['A', 'B', 'all', 'both']:
            # get both parts (increase dimensionality)
            newcr = np.zeros((rl * 2, n, 2 * rr), dtype=np.float64)
            newcr[:rl, :, :rr] = np.real(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcr[:rl, :, rr:] = np.imag(cr)
            newcr[rl:, :, rr:] = np.real(cr)
            newcrs.append(newcr)
            newcr = np.zeros((rr * 2, 2, 1), dtype=np.float64)
            newcr[:rr, 0, :] = newcr[rr:, 1, :] = 1.0
        elif op in ['M']:
            # get matrix modificated for real-arithm. solver
            newcr = np.zeros((rl * 2, n, 2 * rr), dtype=np.float64)
            newcr[:rl, :, :rr] = np.real(cr)
            newcr[rl:, :, :rr] = -np.imag(cr)
            newcr[:rl, :, rr:] = np.imag(cr)
            newcr[rl:, :, rr:] = np.real(cr)
            newcrs.append(newcr)
            newcr = np.zeros((rr * 2, 4, 1), dtype=np.float64)
            newcr[:rr, [0, 3], :] = 1.0
            newcr[rr:, 1, :] = 1.0
            newcr[rr:, 2, :] = -1.0
        else:
            raise ValueError(
                "Unexpected parameter " +
                op +
                " at tt.vector.__complex_op")
        newcrs.append(newcr)
        return vector.from_list(newcrs)

    def real(self):
        """Get real part of a TT-vector."""
        return self.__complex_op('Re')

    def imag(self):
        """Get imaginary part of a TT-vector."""
        return self.__complex_op('Im')

    def c2r(self):
        """Get real vector.from complex one suitable for solving complex linear system with real solver.

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
        """Get complex vector.from real one made by ``tensor.c2r()``.

        For tensor :math:`\\tilde{X}(i_1,\\ldots,i_d,i_{d+1})` returns complex tensor

        .. math::
           X(i_1,\\ldots,i_d) = \\tilde{X}(i_1,\\ldots,i_d,0) + i\\tilde{X}(i_1,\\ldots,i_d,1).

        >>> a = tt.rand(2,10,5) + 1j * tt.rand(2,10,5)
        >>> (a.c2r().r2c() - a).norm() / a.norm()
        7.310562016615692e-16

        """
        tmp = self.copy()
        newcore = np.array(tmp.core, dtype=np.complex128)
        cr = newcore[tmp.ps[-2] - 1:tmp.ps[-1] - 1]
        cr = cr.reshape((tmp.r[-2], tmp.n[-1], tmp.r[-1]), order='F')
        cr[:, 1, :] *= 1j
        newcore[tmp.ps[-2] - 1:tmp.ps[-1] - 1] = cr.flatten('F')
        tmp.core = newcore
        return sum(tmp, axis=tmp.d - 1)

    # Print statement
    def __repr__(self):
        if self.d == 0:
            return "Empty tensor"
        res = "This is a %d-dimensional tensor \n" % self.d
        r = self.r
        d = self.d
        n = self.n
        for i in range(0, d):
            res = res + ("r(%d)=%d, n(%d)=%d \n" % (i, r[i], i, n[i]))
        res = res + ("r(%d)=%d \n" % (d, r[d]))
        return res

    def write(self, fname):
        if np.iscomplexobj(self.core):
            tt_f90.ztt_write_wrapper(
                self.n, self.r, self.ps, self.core, fname)
        else:
            tt_f90.dtt_write_wrapper(
                self.n, self.r, self.ps, np.real(
                    self.core), fname)

    def full(self, asvector=False):
        """Returns full array (uncompressed).

        .. warning::
           TT compression allows to keep in memory tensors much larger than ones PC can handle in
           raw format. Therefore this function is quite unsafe; use it at your own risk.

       :returns: numpy.ndarray -- full tensor.

       """
        # Generate correct size vector
        sz = self.n.copy()
        if self.r[0] > 1:
            sz = np.concatenate(([self.r[0]], sz))
        if self.r[self.d] > 1:
            sz = np.concatenate(([self.r[self.d]], sz))
        if (np.iscomplex(self.core).any()):
            a = tt_f90.ztt_to_full(
                self.n, self.r, self.ps, self.core, np.prod(sz))
        else:
            a = tt_f90.dtt_to_full(
                self.n, self.r, self.ps, np.real(
                    self.core), np.prod(sz))
        a = a.reshape(sz, order='F')
        if asvector:
            a=a.flatten(order='F')
        return a

    def __add__(self, other):
        if other is None:
            return self
        c = vector()
        c.r = np.zeros((self.d + 1,), dtype=np.int32)
        c.ps = np.zeros((self.d + 1,), dtype=np.int32)
        c.n = self.n
        c.d = self.d
        if (np.iscomplex(self.core).any() or np.iscomplex(other.core).any()):
            c.r, c.ps = tt_f90.ztt_add(
                self.n, self.r, other.r, self.ps, other.ps, self.core + 0j, other.core + 0j)
            c.core = tt_f90.zcore.copy()
        else:
            # This could be a real fix in the case we fell to the real world
            c.r, c.ps = tt_f90.dtt_add(
                self.n, self.r, other.r, self.ps, other.ps, np.real(
                    self.core), np.real(
                    other.core))
            c.core = tt_f90.core.copy()
        tt_f90.tt_dealloc()
        return c

    def __radd__(self, other):
        if other is None:
            return self
        return other + self

    def round(self, eps=1e-14, rmax=1000000):
        """Applies TT rounding procedure to the TT-vector and **returns rounded tensor**.

        :param eps: Rounding accuracy.
        :type eps: float
        :param rmax: Maximal rank
        :type rmax: int
        :returns: tensor -- rounded TT-vector.

        Usage example:

        >>> a = tt.ones(2, 10)
        >>> b = a + a
        >>> print b.r
        array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=int32)
        >>> b = b.round(1E-14)
        >>> print b.r
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

        """
        c = vector()
        c.n = np.copy(self.n)
        c.d = self.d
        c.r = np.copy(self.r)
        c.ps = np.copy(self.ps)
        if (np.iscomplex(self.core).any()):
            tt_f90.ztt_compr2(c.n, c.r, c.ps, self.core, eps, rmax)
            c.core = tt_f90.zcore.copy()
        else:
            tt_f90.dtt_compr2(c.n, c.r, c.ps, self.core, eps, rmax)
            c.core = tt_f90.core.copy()
        tt_f90.tt_dealloc()
        return c

    def norm(self):
        if (np.iscomplex(self.core).any()):
            nrm = tt_f90.ztt_nrm(self.n, self.r, self.ps, self.core)
        else:
            nrm = tt_f90.dtt_nrm(
                self.n, self.r, self.ps, np.real(self.core))
        return nrm

    def __rmul__(self, other):
        c = vector()
        c.d = self.d
        c.n = self.n
        if isinstance(other, Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0] - 1:c.ps[1] - 1]
            new_core = new_core * other
            c.core = np.array(c.core, dtype=new_core.dtype)
            c.core[c.ps[0] - 1:c.ps[1] - 1] = new_core
        else:
            c = _hdm(self, other)
        return c

    def __mul__(self, other):
        c = vector()
        c.d = self.d
        c.n = self.n
        if isinstance(other, Number):
            c.r = self.r.copy()
            c.ps = self.ps.copy()
            c.core = self.core.copy()
            new_core = c.core[c.ps[0] - 1:c.ps[1] - 1]
            new_core = new_core * other
            c.core = np.array(c.core, dtype=new_core.dtype)
            c.core[c.ps[0] - 1:c.ps[1] - 1] = new_core
        else:
            c = _hdm(other, self)
        return c

    def __sub__(self, other):
        c = self + (-1) * other
        return c

    def __kron__(self, other):
        if other is None:
            return self
        a = self
        b = other
        c = vector()
        c.d = a.d + b.d
        c.n = np.concatenate((a.n, b.n))
        c.r = np.concatenate((a.r[0:a.d], b.r[0:b.d + 1]))
        c.get_ps()
        c.core = np.concatenate((a.core, b.core))
        return c

    def __dot__(self, other):
        r1 = self.r
        r2 = other.r
        d = self.d
        if (np.iscomplex(self.core).any() or np.iscomplex(other.core).any()):
            dt = np.zeros(r1[0] * r2[0] * r1[d] * r2[d], dtype=np.complex128)
            dt = tt_f90.ztt_dotprod(
                self.n,
                r1,
                r2,
                self.ps,
                other.ps,
                self.core + 0j,
                other.core + 0j,
                dt.size)
        else:
            dt = np.zeros(r1[0] * r2[0] * r1[d] * r2[d])
            dt = tt_f90.dtt_dotprod(
                self.n, r1, r2, self.ps, other.ps, np.real(
                    self.core), np.real(
                    other.core), dt.size)
        if dt.size == 1:
            dt = dt[0]
        return dt

    def __col__(self, k):
        c = vector()
        d = self.d
        r = self.r.copy()
        n = self.n.copy()
        ps = self.ps.copy()
        core = self.core.copy()
        last_core = self.core[ps[d - 1] - 1:ps[d] - 1]
        last_core = last_core.reshape((r[d - 1] * n[d - 1], r[d]), order='F')
        last_core = last_core[:, k]
        try:
            r[d] = len(k)
        except TypeError:
            r[d] = 1
        ps[d] = ps[d - 1] + r[d - 1] * n[d - 1] * r[d]
        core[ps[d - 1] - 1:ps[d] - 1] = last_core.flatten('F')
        c.d = d
        c.n = n
        c.r = r
        c.ps = ps
        c.core = core
        return c

    def __diag__(self):
        from . import matrix as _matrix
        cl = tensor.to_list(self)
        d = self.d
        r = self.r
        n = self.n
        res = []
        dtype = self.core.dtype
        for i in xrange(d):
            cur_core = cl[i]
            res_core = np.zeros((r[i], n[i], n[i], r[i + 1]), dtype=dtype)
            for s1 in xrange(r[i]):
                for s2 in xrange(r[i + 1]):
                    res_core[
                        s1, :, :, s2] = np.diag(
                        cur_core[
                            s1, :, s2].reshape(
                            n[i], order='F'))
            res.append(res_core)
        return _matrix.matrix.from_list(res)

    def __neg__(self):
        return self * (-1)

    def get_ps(self):
        self.ps = np.cumsum(
            np.concatenate(
                ([1],
                 self.n *
                 self.r[
                    0:self.d] *
                    self.r[
                    1:self.d +
                    1]))).astype(
            np.int32)

    def alloc_core(self):
        self.core = np.zeros((self.ps[self.d] - 1, ), dtype=np.float64)

    def copy(self):
        c = vector()
        c.core = self.core.copy()
        c.d = self.d
        c.n = self.n.copy()
        c.r = self.r.copy()
        c.ps = self.ps.copy()
        return c

    def rmean(self):
        """ Calculates the mean rank of a TT-vector."""
        if not np.all(self.n):
            return 0
        # Solving quadratic equation ar^2 + br + c = 0;
        a = np.sum(self.n[1:-1])
        b = self.n[0] + self.n[-1]
        c = - np.sum(self.n * self.r[1:] * self.r[:-1])
        D = b ** 2 - 4 * a * c
        r = 0.5 * (-b + np.sqrt(D)) / a
        return r

    def qtt_fft1(self,tol,inverse=False, bitReverse=True): 
        """ Compute 1D (inverse) discrete Fourier Transform in the QTT format.
        :param tol: error tolerance.
        :type tol: float 
        
        :param inverse: whether do an inverse FFT or not.
        :type inverse: Boolean 
        
        :param bitReverse: whether do the bit reversion or not. If this function is used as a subroutine for multi-dimensional qtt-fft, this option
        need to be set False.
        :type bitReverse: Boolean.
        
        :returns: QTT-vector of FFT coefficients. 
 
        This is a python translation of the Matlab function "qtt_fft1" in Ivan Oseledets' project TT-Toolbox(https://github.com/oseledets/TT-Toolbox)
       
        See S. Dolgov, B. Khoromskij, D. Savostyanov, 
        Superfast Fourier transform using QTT approximation,
        J. Fourier Anal. Appl., 18(5), 2012.
        """
    
        d = self.d 
        r = self.r.copy()
        y = self.to_list(self)   
        
        if inverse:
            twiddle =-1+1.22e-16j # exp(pi*1j)
        else:
            twiddle =-1-1.22e-16j # exp(-pi*1j)
        
        for i in range(d-1, 0, -1):
            
            r1= y[i].shape[0]   # head r
            r2= y[i].shape[2]   # tail r
            crd2 = np.zeros((r1, 2, r2), order='F',  dtype=complex)
            # last block +-
            crd2[:,0,:]= (y[i][:,0,:] + y[i][:,1,:])/np.sqrt(2)
            crd2[:,1,:]= (y[i][:,0,:] - y[i][:,1,:])/np.sqrt(2)
            # last block twiddles
            y[i]= np.zeros((r1*2, 2, r2),order='F',dtype=complex)
            y[i][0:r1,    0, 0:r2]= crd2[:,0,:]
            y[i][r1:r1*2, 1, 0:r2]= crd2[:,1,:]
            #1..i-1 block twiddles and qr
            rv=1; 
            
            for j in range(0, i):
            
                cr=y[j]
                r1= cr.shape[0]   # head r
                r2= cr.shape[2]   # tail r
                if j==0:
                    r[j]=r1
                    r[j+1] = r2*2
                    y[j] = np.zeros((r[j], 2, r[j+1]),order='F',dtype=complex)
                    y[j][0:r1, :, 0:r2] = cr 
                    y[j][0:r1, 0, r2 :r[j+1]] = cr[:,0,:] 
                    y[j][0:r1, 1, r2 :r[j+1]] = twiddle**(1.0/(2**(i-j)))*cr[:,1,:]
                else:
                    r[j]=r1*2
                    r[j+1] = r2*2
                    y[j] = np.zeros((r[j], 2, r[j+1]),order='F',dtype=complex)
                    y[j][0:r1, :, 0:r2] = cr 
                    y[j][r1:r[j], 0, r2 :r[j+1]] = cr[:,0,:] 
                    y[j][r1:r[j], 1, r2 :r[j+1]] = twiddle**(1.0/(2**(i-j)))*cr[:,1,:]
                        
                    
                y[j] = np.reshape(y[j],( r[j], 2*r[j+1]),order='F')
                y[j] = np.dot(rv,y[j])
                r[j] = y[j].shape[0]
                y[j] = np.reshape(y[j],( 2*r[j],  r[j+1]),order='F')

                y[j], rv = np.linalg.qr(y[j])
                y[j] = np.reshape(y[j], (r[j], 2, rv.shape[0]),order='F')

            y[i] = np.reshape(y[i], (r[i], 2*r[i+1]),order='F')
            y[i] = np.dot(rv,y[i])
            r[i] = rv.shape[0]
            # backward svd
            for j in range(i, 0,-1):
                u,s,v = np.linalg.svd(y[j], full_matrices=False)
                rnew = my_chop2(s, np.linalg.norm(s)*tol/np.sqrt(i))
                u=np.dot(u[:, 0:rnew], np.diag(s[0:rnew]))
                v= v[0:rnew, :] 
                y[j] = np.reshape(v, (rnew, 2, r[j+1]),order='F' )
                y[j-1] = np.reshape(y[j-1], (r[j-1]*2,r[j] ),order='F' )
                y[j-1] = np.dot(y[j-1], u)
                r[j] = rnew
                y[j-1] = np.reshape(y[j-1], (r[j-1],r[j]*2 ),order='F' )
                
            y[0] = np.reshape(y[0], (r[0],2, r[1]), order='F' )
        
        # FFT on the first block
        y[0]=np.transpose(y[0],(1,0,2))
        y[0]=np.reshape(y[0],(2, r[0]*r[1]),order='F')
        y[0]= np.dot( np.array([[1,1],[1,-1]]), y[0])/np.sqrt(2)
        y[0]=np.reshape(y[0],(2, r[0], r[1]),order='F')
        y[0]=np.transpose(y[0],(1,0,2))

        if bitReverse:
            # Reverse the train
            y2=[None]*d
            for i in range(d):
                y2[d-i-1]= np.transpose(y[i],(2,1,0))

            y=self.from_list(y2)
        else: # for multi-dimensional qtt_fft
            y=self.from_list(y)
        return y        
        
def _hdm(a, b):
    c = vector()
    c.d = a.d
    c.n = a.n
    c.r = np.zeros((a.d + 1, 1), dtype=np.int32)
    c.ps = np.zeros((a.d + 1, 1), dtype=np.int32)
    if np.iscomplexobj(a.core) or np.iscomplexobj(b.core):
        c.r, c.ps = tt_f90.ztt_hdm(
            a.n, a.r, b.r, a.ps, b.ps, a.core, b.core)
        c.core = tt_f90.zcore.copy()
    else:
        c.r, c.ps = tt_f90.dtt_hdm(
            a.n, a.r, b.r, a.ps, b.ps, a.core, b.core)
        c.core = tt_f90.core.copy()
    tt_f90.tt_dealloc()
    return c


class tensor(vector):  # For combatibility issues

    def __init__(self, *args, **kwargs):
        super(tensor, self).__init__(*args, **kwargs)
        warn('Type `tt.tensor` is deprecated, use `tt.vector` instead.',
             DeprecationWarning)
