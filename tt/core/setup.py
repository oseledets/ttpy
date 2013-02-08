#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('core', parent_package, top_path)
    tt_fort = '../tt-fort'
    #print_dir = '../tt-fort/print'
    tt_src = ['nan.f90', 'timef.f90', 'say.f90', 'rnd.f90', 'ptype.f90', 'sort.f90', 'trans.f90', 'ort.f90', 
              'mat.f90', 'check.f90', 'lr.f90', 'maxvol.f90', 'svd.f90', 'matrix_util.f90', 'tt.f90', 'ttaux.f90', 
              'ttop.f90', 'ttio.f90',  'tts.f90', 'python_conv.f90','tt_linalg.f90']
    print_src = ['putstrmodule.F90','dispmodule.f90']
    src = [tt_fort +'/'+ x for x in tt_src]
    #print_src = [print_dir + '/' + x for x in print_src]
    #config.add_library('print_lib',sources=print_src)
    #config.add_library("mytt",sources=)
    src += ['tt_f90.f90','tt_f90.pyf']
    #inc_dir = ['../tt-fort']
    #inc_dir = ['/Users/ivan/work/python/ttpy/tt/build/temp.macosx-10.5-x86_64-2.7']
    #inc_dir = ['build/temp.macosx-10.5-x86_64-2.7']
    
    
    #plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = None
    #inc_dir = ['build/temp%s' % plat_specifier]
    config.add_extension('tt_f90',sources=src,depends=['print_lib'],libraries=['print_lib'])#,libraries="mytt")#,include_dirs=inc_dir),#extra_objects="../tt-fort/mytt.a") 
    tt_src = ['matrix_util.f90','core.f90']
    src = [tt_fort + '/' + x for x in tt_src]
    src += ['core.pyf']
    config.add_extension('core_f90',sources=src) 
    return config
    
#from distutils.core import setup
#from numpy.distutils.core import setup, Extension
#src = ['tt_f90.f90','tt_f90.pyf']
#inc_dir = ['tt-fort']
#lib = ['tt-fort/mytt.a']
#ext = Extension('tt_f90', src, include_dirs=inc_dir)
#setup(ext_modules = [ext])


if __name__ == '__main__':
    print 'This is the wrong setup.py to run'


#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
