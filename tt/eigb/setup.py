#This script will build the main subpackages  
from os.path import join
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('eigb', parent_package, top_path)
    tt_fort = '../tt-fort/'
    primme_dir = '../tt-fort/primme/'
    src = ['ttals.f90', 'tt_eigb.f90']
    src = [ tt_fort + '/' + x for x in src]
    config.add_library('primme',sources=[join(primme_dir,'*.c')] + [join(primme_dir,'*.f')], \
            include_dirs = ['.'])#,macros=['F77UNDERSCORE'])
    src.append('tt_eigb.pyf')
    config.add_extension('tt_eigb',sources=src,depends=['primme','mytt','print_lib'],libraries=['primme','mytt','print_lib'])
    return config
    


if __name__ == '__main__':
    print 'This is the wrong setup.py to run'


#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
