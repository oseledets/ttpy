gfortran -c zlacn1.f dlacn1.f    dlapst.f    dlarpc.f    expokit.f
ar rc tmp.a zlacn1.o dlacn1.o   dlapst.o   dlarpc.o  expokit.o
gfortran -fno-automatic --free-line-length-none -fcheck=all -std=f2008 dispmodule.f90 explib.f90  normest.f90 matrix_util.f90 ttals.f90 tt_kls.f90 test_complex.f90 dispmodule.a tmp.a -llapack -lblas

