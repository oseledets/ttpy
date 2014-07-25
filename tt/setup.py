#This script will build the main subpackages  
from distutils.util import get_platform 
from numpy.distutils.misc_util import Configuration, get_info
import sys

def configuration(parent_package='',top_path=None):
    sys.argv.extend ( ['config_fc', '--fcompiler=gnu95'])
    config = Configuration('tt', parent_package, top_path) 
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False,     
    )
    
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = ['build/temp%s' % plat_specifier]
    
    tt_dir = 'tt-fort'
    tt_src = ['nan.f90', 'default.f90', 'timef.f90', 'say.f90', 'rnd.f90', 'ptype.f90', 'sort.f90', 'trans.f90', 'ort.f90', 
              'mat.f90', 'check.f90', 'lr.f90', 'maxvol.f90', 'svd.f90', 'matrix_util.f90', 'tt.f90', 'ttaux.f90', 
              'ttop.f90', 'ttio.f90',  'tts.f90', 'python_conv.f90','tt_linalg.f90', 'ttlocsolve.f90', 'ttnodeop.f90', 'ttamen.f90']
    tt_src = [tt_dir + '/' + x for x in tt_src] 
    
    print_dir = 'tt-fort/print'
    print_src = ['putstrmodule.F90','dispmodule.f90']
    print_src = [print_dir + '/' + x for x in print_src]
    
    
    config.add_include_dirs(inc_dir)
    config.add_library('print_lib',sources=print_src)
    config.add_library('mytt',sources=tt_src)
    
    
    config.add_subpackage('core')
    config.add_subpackage('amen')
    config.add_subpackage('ksl')
    config.add_subpackage('eigb')
    config.add_subpackage('maxvol')
    config.add_subpackage('cross')
    return config
    


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
