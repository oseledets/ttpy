#This script will build the main subpackages  
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('kls', parent_package, top_path)
    #import ipdb; ipdb.set_trace()
    #config.add_library()
    tt_dir = '../tt-fort/'
    
    src = ['putstrmodule.F90','dispmodule.f90','matrix_util.f90','ttals.f90','dlacn1.f','dlapst.f','dlarpc.f',
           'zlacn1.f','expokit.f','explib.f90','normest.f90','tt_kls.f90']
    exp_src = ['explib.f90, expokit.f','normest.f90','dlancn1.f','dlapst.f','dlarpc.f'] + tt_dir
    config.add_library('expokit',sources=[primme_dir+]
    config.add_library('primme',sources=[join(primme_dir,'*.c')] + [join(primme_dir,'*.f')])#,macros=['F77UNDERSCORE'])
    src.append('tt_eigb.pyf')
    config.add_extension('tt_eigb',sources=src,depends=['primme'],libraries=['primme'])
    src = [tt_dir + x for x in src]
    src.append('tt_kls.pyf')
    config.add_extension('dyn_tt',sources=src,include_dirs=[tt_dir])
    #config.add_extension['
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
