#This script will build the main subpackages  
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('amr', parent_package, top_path)
    tt_fort = '../tt-fort'
    src = ['putstrmodule.F90','dispmodule.f90','tt_linalg.f90','tt_adapt_als.f90']
    src = [ tt_fort + '/' + x for x in src]
    src.append('amr.pyf')
    #config.add_library('mytt2',my_src)
    #src = ['../tt-fort/tt_adapt_als.f90','../tt-fort/dispmodule.f90','amr.pyf']
    
    #src = ['../tt-fort/tt_linalg.f90','../tt-fort/tt_adapt_als.f90','amr.pyf']
    #tt_fort_dir = '../tt-fort'
    #src = ['tt-fort/' + x for x in src]
    config.add_extension('amr_f90',sources=src)#,include_dirs=inc_di)
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
    #from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())


#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
