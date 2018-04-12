#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from numpy.distutils.misc_util import Configuration
from os.path import join


TTFORT_DIR = '../tt-fort/'
PRIMME_DIR = '../tt-fort/primme'

TTEIGB_SRC = [
    'ttals.f90',
    'tt_eigb.f90',
]


def configuration(parent_package='', top_path=None):
    tteigb_src = [join(TTFORT_DIR, x) for x in TTEIGB_SRC]
    tteigb_src.append('tt_eigb.pyf')

    config = Configuration('eigb', parent_package, top_path)
    config.add_library(
        'primme',
        sources=[join(PRIMME_DIR, '*.c')] + [join(PRIMME_DIR, '*.f')],
        include_dirs=['.'],
    )
    config.add_extension(
        'tt_eigb',
        sources=tteigb_src,
        depends=[
            'primme',
            'mytt',
            'print_lib',
        ],
        libraries=[
            'primme',
            'mytt',
            'print_lib',
        ],
    )

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
