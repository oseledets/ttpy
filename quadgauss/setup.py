#This script will build the main subpackages  
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('tt', parent_package, top_path) 
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False,     
    )
    #config.add_include_dirs(tt_fort)
    config.add_extension('quadgauss',sources='hermite_rule.f90')#,include_dirs=inc_dir),#extra_objects="../tt-fort/mytt.a")
    return config
    

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())


#, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs
