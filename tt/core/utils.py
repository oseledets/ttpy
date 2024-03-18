from __future__ import absolute_import, division, print_function

import sys

import numpy as np
from six.moves import xrange

if sys.version_info < (3, 5):
    from fractions import gcd as _gcd
else:
    from math import gcd as _gcd


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
    subs = np.empty((n))
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))
    for i in xrange(n - 1, -1, -1):
        subs[i] = np.floor(idx / k[i])
        idx = idx % k[i]
    return subs.astype(np.int32)


def gcd(a, b):
    '''Greatest common divider'''
    f = np.frompyfunc(_gcd, 2, 1)
    return f(a, b)


def my_chop2(sv, eps):  # from ttpy/multifuncr.py
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return np.amin(ff)
