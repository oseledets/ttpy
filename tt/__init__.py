from __future__ import print_function, absolute_import, division
from sys import platform, stderr

# First of all, we should check context in which package is loaded. If there is
# such variable then we should to avoid loading of submodules.
try:
    __TTPY_SETUP__
except NameError:
    __TTPY_SETUP__ = False

if __TTPY_SETUP__:
    stderr.write('Building from ttpy source directory.\n')
else:
    try:
        from .core.tt import *
    except:
        import ctypes
        try:
            ctypes.CDLL("libmkl_rt.so", ctypes.RTLD_GLOBAL)
        except:
            try:
                if platform.startswith('linux'):
                    ctypes.CDLL("liblapack.so", ctypes.RTLD_GLOBAL)
                elif platform.startswith('darwin'):
                    ctypes.CDLL("liblapack.dylib", ctypes.RTLD_GLOBAL)
            except:
                print("Did not find MKL or LAPACK library")
        from .core.tt import *

    from .multifuncrs import multifuncrs
    from .multifuncrs2 import multifuncrs2
    from .solvers import GMRES
