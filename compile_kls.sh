#gfortran -O3 -c -ffree-line-length-none matrix_util.f90 ttals.f90
f2py --build-dir f2py_temp --fcompiler=gnu95 -c tt_kls.pyf  dispmodule.f90 expokit.f zlacn1.f dlacn1.f dlapst.f dlarpc.f normest.f90 explib.f90 matrix_util.f90 ttals.f90 tt_kls.f90  --f90flags="-ffree-line-length-none -fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fcheck=all  -std=f2008  -pedantic  -fbacktrace" 


