#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from numpy.distutils.misc_util import Configuration
from os.path import join


TTFORT_DIR = '../tt-fort'

TT_SRC = [
    'tt_f90.f90',
    'tt_f90.pyf',
]

TTCORE_SRC = [
    'matrix_util.f90',
    'core.f90',
]


def configuration(parent_package='', top_path=None):
    ttcore_src = [join(TTFORT_DIR, x) for x in TTCORE_SRC]
    ttcore_src.append('core.pyf')
    
    config = Configuration('core', parent_package, top_path)
    config.add_extension(
        'tt_f90',
        sources=TT_SRC,
        depends=[
            'mytt',
            'print_lib',
        ],
        libraries=[
            'mytt',
            'print_lib'
        ],
    )
    config.add_extension(
        'core_f90',
        sources=ttcore_src,
    )

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
