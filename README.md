ttpy
====

Python implementation of the TT-Toolbox. It contains several
important packages for working with the Tensor Train (TT) format
in Python. It is able to do interpolation, solve linear systems, eigenproblems, solve dynamical problems. 
Many computational routines are done in Fortran (which can be used separatedly), and are wrapped with the f2py tool.

Main contributors
============
- Ivan Oseledets (version 0.1, build system, **KSL** and **EIGB** modules)
- Tigran Saluev (made the code usable; implemented **multifucrs** and **TT-GMRES** algorithms and many more; a lot of bugfixes; initial documentation)
- Dmitry Savostyanov ( **tt-fort** submodule code, **AMEN** module)
- Sergey Dolgov ( **AMEN** module)

Installation
============

##Downloading the code
This installation works with git submodules, so you should be sure, that you got them all right.
The best way is to work with this repository is to use git with a version >= 1.6.5.
Then, to clone the repository, you can simply run
```
git clone --recursive git://github.com/oseledets/ttpy.git
cd ttpy
git submodule foreach git checkout master 
```
To update to a newer version, run
```
git submodule update --recursive --init
```
This command will update the submodules as well.


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

##BLAS and so on

Almost all of the packages depend on the BLAS/LAPACK libraries, but the code 
is not explicitly linked against them, so if you do not have the BLAS/LAPACK
in the global namespace of your Python interpreter, you will encounter "undefined symbols"
error during the import of the tt package. There are two possibilities how to deal with 
this situation

- Use the Enthought Python Distribution (EPD) (non-Free version, but it is free for academics).
It loads the well-tuned MKL library into the global namespace, and you do not have to do anything.

- Use ctypes to load the BLAS (preferrably MKL) dynamic library. If you have MKL installed somewhere on 
your PATH, you can simply use
```python
import ctypes
ctypes.CDLL("libmkl_rt.so", ctypes.RTLD_GLOBAL)
```
before the tt import 

What those packages do
======================

They have the following functionality

- **tt** : The main package, with tt.tensor and tt.matrix classes, basic arithmetic,
       norms, scalar products, rounding full -> tt and tt -> full conversion routines, and so on

- **tt.amen** : AMEN solver for linear systems (Python wrapper for Fortran code written by S. V. Dolgov and D. V. Savostyanov) 
                it can be also used for fast matrix-by-vector products. 

- **tt.eigb** : Contains a test version of the ALS block eigenvalue solver in the TT-format 
            (subroutine **tt.eigb.eigb**) 

- **tt.ksl** :  Solution of the linear dynamic problems in the TT-format, using the KSL scheme. A Python wrapper for a Fortran code
                written by I. V. Oseledets

Documentation and examples
==========================

The package provides Sphinx-generated documentation. To build HTML version, just do
```
cd tt/doc
make html
```

A few examples are available right now under examples/ directory







