#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('cross', parent_package, top_path)
    config.add_subpackage('rectcross')
    
    return config
    


if __name__ == '__main__':
    print 'This is the wrong setup.py to run'


