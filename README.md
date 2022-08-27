[![Build Status](https://api.travis-ci.org/oseledets/ttpy.svg?style=flat-square)](https://travis-ci.org/oseledets/ttpy)
[![Coverage Status](https://img.shields.io/coveralls/oseledets/ttpy/master.svg?style=flat-square)](https://coveralls.io/r/oseledets/ttpy)
[![PyPi](https://img.shields.io/pypi/pyversions/ttpy.svg?style=flat-square)](https://pypi.python.org/pypi/ttpy/)
[![Downloads](https://img.shields.io/pypi/dm/ttpy.svg?style=flat-square)](https://pypi.python.org/pypi/ttpy/)
[![Downloads](https://img.shields.io/docker/pulls/daskol/ttpy)](https://hub.docker.com/r/daskol/ttpy)

ttpy
====
Python implementation of the Tensor Train (TT) -Toolbox. It contains several
important packages for working with the TT-format
in Python. It is able to do TT-interpolation, solve linear systems, eigenproblems, solve dynamical problems. 
Several computational routines are done in Fortran (which can be used separatedly), and are wrapped with the f2py tool.

Installation
============
## Prerequisites

**It is recommended** that you use [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/) 
  which has MKL library built-in. Anaconda Python is used for the development  of ttpy.

## Pip install 
Install dependencies (numpy and cython)
```
   conda install numpy cython
```
Install ttpy
```
   pip install ttpy
```

## Installing from source code

To install the development version, you need to install from the source.
First, clone the repository with all submodules:
```
   git clone --recursive git://github.com/oseledets/ttpy.git
```
```
   python setup.py install
```
To update to the latest version (with all submodules) run
```
git pull
git submodule update --init --recursive *
```

## Pre-built Docker image

The easiest way to play with TT Toolbox is through using the Docker (see
[Docker Hub page](https://hub.docker.com/r/daskol/ttpy) for details).
In order to use it one needs just pull an image and run a container as in
snippet below.

```shell
$ docker pull daskol/ttpy
    ...
Status: Image is up to date for daskol/ttpy:latest
docker.io/daskol/ttpy:latest
$ docker run -it --rm -v "$PWD":/workspace daskol/ttpy
Python 3.10.6 (main, Aug  3 2022, 10:13:24) [GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tt
```

What those packages do
======================

They have the following functionality

- `tt` : The main package, with tt.vector and tt.matrix classes, basic arithmetic,
       norms, scalar products, rounding full -> tt and tt -> full conversion routines, and many others
- `tt.amen` : AMEN solver for linear systems (Python wrapper for Fortran code written by S. V. Dolgov and D. V. Savostyanov) 
                it can be also used for fast matrix-by-vector products. 
- `tt.eigb` : Block eigenvalue solver in the TT-format 
- `tt.ksl` :  Solution of the linear dynamic problems in the TT-format, using the projector-splitting 
                KSL scheme. A Python wrapper for a Fortran code (I. V. Oseledets)
- `tt.cross` : Has a working implementation of the black-box cross method. 


Documentation and examples
==========================

The package provides Sphinx-generated documentation. To build HTML version, just do
```
cd tt/doc
make html
```

A few examples are available right now under [examples](examples/) directory


For any questions, please create an issue on Github.


Contributor policy
====================
This project is now following the _git flow_ approach. Namely:

- branch `master` is only for stable versions and releases;
- branch `develop` is main working branch;
- contributor should create new branch for certain feature and then merge with `develop` branch as feature was done;
- each release on `master` branch should correspond to package on PyPI;
- A maintainer checks all the pull request 

A pull request should satisfy the following requirements:
- style and quality description of pull request;
- new changes should be tested and shouldn't break anything;
- pull request for one fix or one feature(could be several commits);
- try to keep the code style of the project;

Current maintainer is [Ivan Oseledets](oseledets.github.io).




