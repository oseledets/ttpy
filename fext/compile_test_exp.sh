gfortran -c -fcheck=all zdotc.f expokit.f
#ifort -c  expokit.f
ar rc tmp.a expokit.o zdotc.o
#gfortran -c zlacn1.f dlacn1.f    dlapst.f    dlarpc.f    expokit.f
#ar rc tmp.a zlacn1.o dlacn1.o   dlapst.o   dlarpc.o  expokit.o

gfortran -fno-automatic --free-line-length-none -fcheck=all  explib.f90 test_exp.f90 tmp.a -llapack -lblas
#ifort  -mkl  -fno-automatic --free-line-length-none -fcheck=all -std=f2008 explib.f90 test_exp.f90 tmp.a


