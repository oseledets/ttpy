#""" Basic subroutines for ttpy """  
#""" They still focus on the linear format for passing the data around, 
#    and still convert to list (and back for some simple tasks) """ 
import numpy as np
from numpy import prod, reshape, nonzero, size, sqrt
import math
from math import sqrt
from numbers import Number
import tt_f90
import core_f90

from tensor import tensor
from matrix import matrix

#Some binary operations (put aside to wrap something in future)
#TT-matrix by a TT-vector product
def matvec(a,b, compression=False):
    """Matrix-vector product in TT format."""
    acrs = tensor.to_list(a.tt)
    bcrs = tensor.to_list(b)
    ccrs = []
    d = b.d
    
    def get_core(i):
        acr = np.reshape(acrs[i], (a.tt.r[i], a.n[i], a.m[i], a.tt.r[i + 1]), order='F')
        acr = acr.transpose([3, 0, 1, 2]) # a(R_{i+1}, R_i, n_i, m_i)
        bcr = bcrs[i].transpose([1, 0, 2]) # b(m_i, r_i, r_{i+1})
        ccr = np.tensordot(acr, bcr, axes=(3, 0)) # c(R_{i+1}, R_i, n_i, r_i, r_{i+1})
        ccr = ccr.transpose([1, 3, 2, 0, 4]).reshape((a.tt.r[i] * b.r[i], a.n[i], a.tt.r[i+1] * b.r[i+1]), order='F')
        return ccr
    
    if compression: # the compression is laaaaazy and one-directioned
        # calculate norm of resulting vector first
        nrm = np.array([[1.0]]) # 1 x 1
        v = np.array([[1.0]])
        for i in xrange(d):
            ccr = get_core(i)
            #print(str(ccr.shape) + " -> "),
            # minimal loss compression
            ccr = np.tensordot(v, ccr, (1, 0))
            rl, n, rr = ccr.shape
            if i < d - 1:
                u, s, v = np.linalg.svd(ccr.reshape((rl * n, rr), order='F'), full_matrices=False)
                newr = min(rl * n, rr)
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = np.dot(np.diag(s[:newr]), v[:newr, :])
            #print ccr.shape
            nrm = np.tensordot(nrm, ccr, (0, 0)) # r x r . r x n x R -> r x n x R
            nrm = np.tensordot(nrm, np.conj(ccr), (0, 0)) # r x n x R . r x n x R -> n x R x n x R
            nrm = nrm.diagonal(axis1=0, axis2=2) # n x R x n x R -> R x R x n
            nrm = nrm.sum(axis=2) # R x R x n -> R x R
        if nrm.size > 1:
            raise Exception, 'too many numbers in norm'
        #print "Norm calculated:", nrm
        nrm = sqrt(np.linalg.norm(nrm))
        #print "Norm predicted:", nrm
        compression = compression * nrm / sqrt(d - 1)
        v = np.array([[1.0]])
    
    for i in xrange(d):
        ccr = get_core(i)
        rl, n, rr = ccr.shape
        if compression:
            ccr = np.tensordot(v, ccr, (1, 0)) # c(s_i, n_i, r_i, r_{i+1})
            if i < d - 1:
                rl = v.shape[0]
                u, s, v = np.linalg.svd(ccr.reshape((rl * n, rr), order='F'), full_matrices=False)
                ss = np.cumsum(s[::-1])[::-1]
                newr = max(min([r for r in range(ss.size) if ss[r] <= compression] + [min(rl * n, rr)]), 1)
                #print "Rank % 4d replaced by % 4d" % (rr, newr)
                ccr = u[:, :newr].reshape((rl, n, newr), order='F')
                v = np.dot(np.diag(s[:newr]), v[:newr, :])
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
    c.n = np.array([], dtype=np.int32)
    c.r = np.array([], dtype=np.int32)
    c.core = []
    
    for t in a:
        thetensor = t.tt if isinstance(t, matrix) else t
        c.d += thetensor.d
        c.n = np.concatenate((c.n, thetensor.n))
        c.r = np.concatenate((c.r[:-1], thetensor.r))
        c.core = np.concatenate((c.core, thetensor.core))
            
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
    tmp = [1] + [0] * (len(args) - 1)
    result = kron(tensor(tmp), args[0])
    for i in range(1, len(args)):
        result += kron(tensor([0] * i + [1] + [0] * (len(args) - i - 1)), args[i])
    return result
    
    

def _hdm (a,b):
    c = tensor()
    c.d = a.d
    c.n = a.n
    c.r = np.zeros((a.d+1,1),dtype=np.int32)
    c.ps = np.zeros((a.d+1,1),dtype=np.int32)
    if np.iscomplexobj(a.core) or np.iscomplexobj(b.core):
        c.r,c.ps = tt_f90.tt_f90.ztt_hdm(a.n,a.r,b.r,a.ps,b.ps,a.core,b.core)
        c.core = tt_f90.tt_f90.zcore.copy()
    else:
        c.r,c.ps = tt_f90.tt_f90.dtt_hdm(a.n,a.r,b.r,a.ps,b.ps,a.core,b.core)
        c.core = tt_f90.tt_f90.core.copy()
    tt_f90.tt_f90.tt_dealloc()
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
        crs[ax] = np.sum(crs[ax], axis=1)
        rleft, rright = crs[ax].shape
        if (rleft >= rright or rleft < rright and ax + 1 >= d) and ax > 0:
            crs[ax-1] = np.tensordot(crs[ax-1], crs[ax], axes=(2,0))
        elif ax + 1 < d:
            crs[ax+1] = np.tensordot(crs[ax], crs[ax+1], axes=(1,0))
        else:
            return np.sum(crs[ax])
        crs.pop(ax)
        d -= 1
    return tensor.from_list(crs)

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
	"""Generate a random d-dimensional TT-tensor with ranks ``r``."""
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
        d = np.asarray(d, dtype=np.int32).flatten()
        if D is None:
            D = d.size
        elif d.size == 1:
            d = np.array([d[0]] * D, dtype=np.int32)
        if D != d.size:
            raise ValueError("D must be equal to len(d)")
        check_kinds(D, kind)
        if np.sum(d) + np.sum([(1 if knd == 'F' else 0) for knd in kind]) != x.d:
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
    """ Create TT-tensor for :math:`\\sin(\\alpha n + \\varphi)`."""
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
    """ Create TT-tensor for :math:`\\cos(\\alpha n + \\varphi)`."""
    return sin(d, alpha, phase + math.pi * 0.5)

def delta(n, d=None, center=0):
    """ Create TT-tensor for delta-function :math:`\\delta(x - x_0)`. """
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
    """ Create TT-tensor for Heaviside step function :math:`\chi(x - x_0)`.
    
    Heaviside step function is defined as
    
    .. math::
    
        \chi(x) = \\left\{ \\begin{array}{l} 1 \mbox{ when } x \ge 0, \\\\ 0 \mbox{ when } x < 0. \\end{array} \\right.
    
    For negative value of ``direction`` :math:`\chi(x_0 - x)` is approximated. """
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

