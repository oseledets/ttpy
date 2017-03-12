#!/usr/bin/env python
"""TTPY: Python implementation of the Tensor Train (TT) - Toolbox.

Python implementation of the Tensor Train (TT) - Toolbox. It contains several important packages for working with the
TT-format in Python. It is able to do TT-interpolation, solve linear systems, eigenproblems, solve dynamical problems.
Several computational routines are done in Fortran (which can be used separatedly), and are wrapped with the f2py tool.
"""

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

DOCLINES = (__doc__ or '').split('\n')

PLATFORMS = [
    'Windows',
    'Linux',
    'Solaris',
    'Mac OS-X',
    'Unix',
]

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Fortran
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

MAJOR = 1
MINOR = 1
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )
    config.add_subpackage('tt')
    return config

def setup_package():
    metadata = dict(
        name='ttpy',
        version=VERSION,
        description = DOCLINES[0],
        long_description = '\n'.join(DOCLINES[2:]),
        url='https://github.com/oseledets/ttpy',
        download_url='https://github.com/oseledets/ttpy/tarball/v' + VERSION,
        author='Ivan Oseledets',
        maintainer='Ivan Oseledets',
        author_email='ivan.oseledets@gmail.com',
        platforms=PLATFORMS,
        classifiers=[line for line in CLASSIFIERS.split('\n') if line],
        configuration=configuration,
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
