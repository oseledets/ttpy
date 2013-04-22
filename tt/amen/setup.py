#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('amen', parent_package, top_path)
    
    src = ['amen_f90.f90','amen_f90.pyf']
    
    
    config.add_extension('amen_f90',sources=src,depends=['mytt'],libraries=['mytt'])
    return config
    


if __name__ == '__main__':
    print 'This is the wrong setup.py to run'


