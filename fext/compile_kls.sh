#gfortran -O3 -c -ffree-line-length-none matrix_util.f90 ttals.f90
f2py --build-dir f2py_temp --fcompiler=gnu95 -c tt_kls.pyf  tt-fort/dispmodule.f90 tt-fort/expokit.f tt-fort/zlacn1.f tt-fort/dlacn1.f tt-fort/dlapst.f tt-fort/dlarpc.f tt-fort/normest.f90 tt-fort/explib.f90 tt-fort/matrix_util.f90 tt-fort/ttals.f90 tt-fort/tt_kls.f90  --f90flags="-ffree-line-length-none -fimplicit-none  -Wall  -Wline-truncation  -Wcharacter-truncation  -Wsurprising  -Waliasing  -Wimplicit-interface  -Wunused-parameter  -fcheck=all  -std=f2008  -pedantic  -fbacktrace" 


