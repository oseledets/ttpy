ttpy
====

Python implementation of the TT-Toolbox. It contains several
important packages for working with the Tensor Train (TT) format
in Python. Most of the computational routines are done in Fortran, 
and are wrapped with f2py tool.


Installation
============
The installation script is a very newbie one, but it seems to work.
Go to the *tt* directory and run
```
python setup.py build_ext --inplace
```
This builds the package tt with submodules amr, eigb, kls. 
For some of the examples the quadgauss package is required, so go to 
*quadgauss* directory and run setup.py there


What those packages do
======================

They have the following functionality

- tt : The main package, with tt.tensor and tt.matrix classes, basic arithmetic,
       norms, scalar products, rounding full -> tt and tt -> full conversion routines, and so on

- tt.amr : Contains the AMR/DMRG fast matrix-by-vector product (subroutine *tt.amr.mvk4*) and 
           AMR/DMRG approximate linear system solver (subroutine *tt.amr.amr_solve*). The matrices
           and vectors should be given in the TT-format

- tt.eigb : Contains a test version of the ALS block eigenvalue solver in the TT-format 
            (subroutine *tt.eigb.eigb*) 

- tt.kls : A very experimental implementation of the dynamical low-rank approximation in the TT-format
           using the *KLS* scheme, which is much faster, than other known approaches. The scheme 
           has the second order. Both real and complex versions are available.

Examples
========

Right now the examples are located in the top of the directory, which is not correct, of course. 
When I will figure out, how it should be done in a right way (via unittests or something?) I will fix that.
Right now you just take a look at files starting with *test_* to see what happens. Also, the files 
*hh_hermite.py* and *hh_hermite2.py* contain experiments for the molecular Schrodinger equation.




