try: 
    from core import *
except:
    import ctypes
    try:
        ctypes.CDLL("libmkl_rt.so", ctypes.RTLD_GLOBAL)
    except:
        try:
            ctypes.CDLL("liblapack.so", ctypes.RTLD_GLOBAL)
        except:
            print "Did not find MKL or LAPACK library"
    from core import *

from multifuncrs import multifuncrs
from multifuncrs2 import multifuncrs2
from solvers import GMRES
