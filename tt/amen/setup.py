#   setup.py
#   This script will build the main subpackages
#   See LICENSE for details

from __future__ import print_function, absolute_import
from numpy.distutils.misc_util import Configuration


AMEN_SRC = [
    'amen_f90.f90',
    'amen_f90.pyf',
]


def configuration(parent_package='', top_path=None):
    config = Configuration('amen', parent_package, top_path)
    config.add_extension(
        'amen_f90',
        sources=AMEN_SRC,
        depends=['mytt'],
        libraries=['mytt'],
    )

    return config


if __name__ == '__main__':
    print('This is the wrong setup.py to run')
