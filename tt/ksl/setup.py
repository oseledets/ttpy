#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from numpy.distutils.misc_util import Configuration
from os.path import join


TTFORT_DIR = '../tt-fort'
EXPM_DIR = '../tt-fort/expm'

EXPOKIT_SRC = [
    'explib.f90',
    'normest.f90',
    'expokit.f',
    'dlacn1.f',
    'dlapst.f',
    'dlarpc.f',
    'zlacn1.f',
]

TTKSL_SRC = [
    'ttals.f90',
    'tt_ksl.f90',
    'tt_diag_ksl.f90'
]


def configuration(parent_package='', top_path=None):
    expokit_src = [join(EXPM_DIR, x) for x in EXPOKIT_SRC]

    ttksl_src = [join(TTFORT_DIR, x) for x in TTKSL_SRC]
    ttksl_src.append('tt_ksl.pyf')

    config = Configuration('ksl', parent_package, top_path)
    config.add_library(
        'expokit',
        sources=expokit_src,
    )
    config.add_extension(
        'dyn_tt',
        sources=ttksl_src,
        depends=[
            'print_lib',
            'expokit',
            'mytt',
        ],
        libraries=[
            'print_lib',
            'expokit',
            'mytt',
        ],
    )

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
