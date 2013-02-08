#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('core', parent_package, top_path)
    tt_fort = '../tt-fort'
    
    src = ['tt_f90.f90','tt_f90.pyf']
    
    
    config.add_extension('tt_f90',sources=src,depends=['mytt','print_lib'],libraries=['mytt','print_lib'])
    tt_src = ['matrix_util.f90','core.f90']
    src = [tt_fort + '/' + x for x in tt_src]
    src += ['core.pyf']
    config.add_extension('core_f90',sources=src) 
    return config
    


if __name__ == '__main__':
    print 'This is the wrong setup.py to run'


