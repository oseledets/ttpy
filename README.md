ttpy
====

Python implementation of the TT-Toolbox. It contains several
important packages for working with the Tensor Train (TT) format
in Python. It is able to do interpolation, solve linear systems, eigenproblems, solve dynamical problems. 
Several computational routines are done in Fortran (which can be used separatedly), and are wrapped with the f2py tool.

Main contributors
============
- Ivan Oseledets (version 0.1, build system, **CROSS**, **KSL** and **EIGB** modules)
- Tigran Saluev (made the code usable; implemented **multifucrs** and **TT-GMRES** algorithms and many more; a lot of bugfixes; initial documentation)
- Dmitry Savostyanov ( **tt-fort** submodule code, **AMEN** module)
- Sergey Dolgov ( **AMEN** module)

Installation
============

##From a binstar repository
If you use 64 bit linux, you can install the module from the binstar repository:
```
sudo apt-get install libgfortran3
conda install -c https://conda.binstar.org/bihaqo ttpy
```

##Downloading the code
This installation works with git submodules, so you should be sure, that you got them all right.
The best way is to work with this repository is to use git with a version >= 1.6.5.
Then, to clone the repository, you can simply run
```
git clone --recursive git://github.com/oseledets/ttpy.git

```
To update to the latest version, run
```
git pull
git submodule update --init --recursive *
python setup.py install
```
This command will update the submodules as well.

##Prerequisites
**It is highly recommended** that you use either

- [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/) 
  which has MKL in built in for the [academics](https://store.continuum.io/cshop/academicanaconda)
  Anaconda Python is the version for which the development of ttpy is done

- [Enthought Python distribution](https://www.enthought.com/products/epd/) -- should work as well with 
  the non-free version, but not tested



##Installing the package
The installation of the package is done via **setup.py** scripts.

The installation script is a very newbie one, but it seems to work.
Go to the **tt** directory and run
```
python setup.py install
```
This builds the package tt with submodules amr, eigb, kls. 
For some of the examples the quadgauss package is required, so go to 
**quadgauss** directory and run setup.py there

## BLAS

Almost all of the packages depend on the BLAS/LAPACK libraries, but the code 
is **not explicitly linked** against them, so if you do not have the BLAS/LAPACK
in the global namespace of your Python interpreter, you will encounter "undefined symbols"
error during the import of the **tt** package. If you have Anaconda or EPD with MKL installed, it should
work ``as it is''. The  [initialization file of **ttpy**](/tt/__init__.py) tries to dynamically load runtime MKL or lapack libraries 
(should work on Linux if those libraries are in LD_LIBRARY_PATH).

What those packages do
======================

They have the following functionality

- **tt** : The main package, with tt.tensor and tt.matrix classes, basic arithmetic,
       norms, scalar products, rounding full -> tt and tt -> full conversion routines, and so on

- **tt.amen** : AMEN solver for linear systems (Python wrapper for Fortran code written by S. V. Dolgov and D. V. Savostyanov) 
                it can be also used for fast matrix-by-vector products. 

- **tt.eigb** : Block eigenvalue solver in the TT-format 
            (subroutine **tt.eigb.eigb**) 

- **tt.ksl** :  Solution of the linear dynamic problems in the TT-format, using the projector-splitting 
                KSL scheme. A Python wrapper for a Fortran code (I. V. Oseledets)

- **tt.cross** : Has a working implementation of the black-box cross method. For now, please use the rect_cross function.

Documentation and examples
==========================

The package provides Sphinx-generated documentation. To build HTML version, just do
```
cd tt/doc
make html
```

A few examples are available right now under examples/ directory


For any questions, please feel free to contact me by email or create an issue on Github.






