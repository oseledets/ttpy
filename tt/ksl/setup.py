from setuptools import setup, find_packages
#from distutils.core import setup
from distutils.extension import Extension
import numpy as np

# we'd better have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)

surf3d_ext = Extension("surf3d", ["surf3d.pyx"], extra_compile_args = ["-O3", "-Wall", "-undefined,dynamic_lookup"], include_dirs = [np.get_include()])


extensions = [surf3d_ext]

# finally, we can pass all this to distutils
setup(
    name="ani3d",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass = {'build_ext': build_ext},
)
