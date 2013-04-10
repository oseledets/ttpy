try: 
    from core.tt import *
except:
    import ctypes
    try:
        ctypes.CDLL("libmkl_rt.so", ctypes.RTLD_GLOBAL)
    except:
        try:
            ctypes.CDLL("liblapack.so.3", ctypes.RTLD_GLOBAL)
        except:
            print "Did not find MKL or LAPACK library"
    from core.tt import *

from multifuncrs import multifuncrs
from solvers import GMRES
