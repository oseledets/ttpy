#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('tt', parent_package, top_path) 
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False,     
    )
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = ['build/temp%s' % plat_specifier]
    
    tt_src = ['nan.f90', 'timef.f90', 'say.f90', 'rnd.f90', 'ptype.f90', 'sort.f90', 'trans.f90', 'ort.f90', 
              'mat.f90', 'check.f90', 'lr.f90', 'maxvol.f90', 'svd.f90', 'matrix_util.f90', 'tt.f90', 'ttaux.f90', 
              'ttop.f90', 'ttio.f90',  'tts.f90', 'python_conv.f90','tt_linalg.f90']
    
    
    print_dir = 'tt-fort/print'
    print_src = ['putstrmodule.F90','dispmodule.f90']
    print_src = [print_dir+'/'+x for x in print_src]
    config.add_include_dirs(inc_dir)
    config.add_library('print_lib',sources=print_src) 
    config.add_library('mytt',sources=tt_src)
    
    
    #config.add_library('mytt',sources = tt_src)
    #config.add_include_dirs(config.paths([tt_fort]))
    config.add_subpackage('core')
    #config.add_subpackage('amr')
    #config.add_subpackage('kls')
    #config.add_subpackage('eigb')
    return config
    
#from distutils.core import setup
#from numpy.distutils.core import setup, Extension
#src = ['tt_f90.f90','tt_f90.pyf']
#inc_dir = ['tt-fort']
#lib = ['tt-fort/mytt.a']
#ext = Extension('tt_f90', src, include_dirs=inc_dir)
#setup(ext_modules = [ext])


if __name__ == '__main__':
    from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())
    ss = configuration(top_path='')
    ss = ss.todict()
    setup(**ss)
    #ss.get_build_temp_dir()
#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
