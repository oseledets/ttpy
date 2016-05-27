#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from distutils.util import get_platform
from numpy.distutils.misc_util import Configuration, get_info
from numpy.distutils.core import setup
from os.path import join

import sys


TTFORT_DIR = 'tt-fort'
TTFORT_SRC = [
    'nan.f90',
    'default.f90',
    'timef.f90',
    'say.f90',
    'rnd.f90',
    'ptype.f90',
    'sort.f90',
    'trans.f90',
    'ort.f90',
    'mat.f90',
    'check.f90',
    'lr.f90',
    'maxvol.f90',
    'svd.f90',
    'matrix_util.f90',
    'tt.f90',
    'ttaux.f90',
    'ttop.f90',
    'ttio.f90',
    'tts.f90',
    'python_conv.f90',
    'tt_linalg.f90',
    'ttlocsolve.f90',
    'ttnodeop.f90',
    'ttamen.f90',
]

PRINT_DIR = 'tt-fort/print'
PRINT_SRC = [
    'putstrmodule.F90',
    'dispmodule.f90',
]


def configuration(parent_package='', top_path=None):
    plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
    inc_dir = ['build/temp%s' % plat_specifier]

    config = Configuration('tt', parent_package, top_path)
    config.add_include_dirs(inc_dir)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=False,
    )

    config.add_library('print_lib', sources=[join(PRINT_DIR, x) for x in PRINT_SRC])
    config.add_library('mytt', sources=[join(TTFORT_DIR, x) for x in TTFORT_SRC])

    config.add_subpackage('core')
    config.add_subpackage('amen')
    config.add_subpackage('ksl')
    config.add_subpackage('eigb')
    config.add_subpackage('maxvol')
    config.add_subpackage('cross')
    config.add_subpackage('optimize')
    config.add_subpackage('utils')
    config.add_subpackage('riemannian')

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
