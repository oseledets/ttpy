#This script will build the main subpackages  
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('core', parent_package, top_path)
    #import ipdb; ipdb.set_trace()
    #config.add_library()
    lib_tt = ['../tt-fort/mytt.a']
    src = ['tt_f90.f90','tt_f90.pyf']
    inc_dir = ['../tt-fort']
    config.add_extension('tt_f90',sources=src,depends=lib_tt,include_dirs=inc_dir,extra_objects="../tt-fort/mytt.a")
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
    setup(**configuration(top_path='').todict())


#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
