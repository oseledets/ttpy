import numpy as _np
import fractions as _fractions

# Available functions:
# ind2sub, gcd, my_chop2

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
    for i in xrange(n - 1, -1, -1):
        subs[i] = _np.floor(idx / k[i])
        idx = idx % k[i]
    return subs
    
def gcd(a, b):
    '''Greatest common divider'''
    f = _np.frompyfunc(_fractions.gcd, 2, 1)
    return f(a, b)

def my_chop2(sv, eps):  # from ttpy/multifuncr.py
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = _np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return _np.amin(ff)
