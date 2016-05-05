#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from numpy.distutils.misc_util import Configuration


MAXVOL_SRC = [
    'maxvol.f90',
    'maxvol.pyf',
]


def configuration(parent_package='', top_path=None):
    config = Configuration('maxvol', parent_package, top_path)
    config.add_extension('maxvol', sources=MAXVOL_SRC)
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
